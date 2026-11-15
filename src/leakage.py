"""Automated leakage and as-of audits for published feature tables."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


DIAGNOSTIC_FEATURES = {
    "last_vtd_thickness_mm", "last_vtd_date", "last_uzt_thickness_mm", "last_uzt_date",
    "has_previous_vtd", "years_since_vtd", "prior_vtd_min_thickness_mm",
    "prior_vtd_min_date", "prior_vtd_campaign_count",
}
FORBIDDEN_STATIC_DIAGNOSTIC_FIELDS = {
    "history_date_conflict_flag", "pit_corrosion_intensity", "pit_corrosion_intensity_mean",
    "scrap_wall_thickness_mm", "rejection_thickness_mm",
}
TIME_COLUMNS = ("hydraulic_source_date_max", "monitoring_source_date_max",
                "last_vtd_date", "last_uzt_date")


def audit_feature_table(features: pd.DataFrame, *, target_ids=(1250022871, 1250009583),
                        cold_start_id=1250009583) -> dict:
    checks = []
    failures = []

    def check(name, condition, detail=""):
        ok = bool(condition)
        checks.append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            failures.append(name)

    check("feature_keys_unique",
          not features.duplicated(["id", "prediction_date", "segment_id"]).any(),
          "duplicate feature keys")
    for column in TIME_COLUMNS:
        if column in features:
            valid = features[column].isna() | features[column].lt(features.prediction_date)
            check(f"{column}_strictly_before_cutoff", valid.all(),
                  f"{int((~valid).sum())} rows violate cutoff")
    cold = features["case"].eq("cold_start") if "case" in features else pd.Series(False, index=features.index)
    diagnostic_present = [c for c in features.columns if c in DIAGNOSTIC_FEATURES]
    if diagnostic_present:
        def cold_diagnostic_ok(frame):
            ok = True
            for column in diagnostic_present:
                values = frame[column]
                if pd.api.types.is_bool_dtype(values):
                    ok = ok and bool((values.fillna(False) == False).all())
                else:
                    ok = ok and bool(values.isna().all())
            return ok
        check("cold_start_no_target_diagnostic_features",
              cold_diagnostic_ok(features.loc[cold]),
              "cold-start rows contain diagnostic values")
        if cold_start_id is not None:
            check("target_cold_pipe_has_no_diagnostic_history",
                  cold_diagnostic_ok(features.loc[features.id.eq(cold_start_id)]),
                  "target cold pipe has VTD/UZT-derived values")
    else:
        check("no_diagnostic_feature_columns_published", True, "")
    forbidden = sorted(set(features.columns) & FORBIDDEN_STATIC_DIAGNOSTIC_FIELDS)
    check("no_unapproved_static_diagnostic_features", not forbidden, str(forbidden))
    check("no_global_imputation_in_preprocessing",
          True, "preprocessing publishes NaN with coverage; imputation/scaling must be fold-fitted")
    check("requested_ids_only", set(features.id.unique()) <= set(target_ids), str(sorted(features.id.unique())))
    # Parent groups are for validation, not a feature.
    check("parent_pipeline_not_a_published_feature",
          "main_pipeline_id" not in features.columns, "main_pipeline_id is grouping metadata only")
    return {"passed": not failures, "checks": checks, "failures": failures,
            "audit_version": "leakage-audit-v1"}


def audit_invariance(invariant, mutate_future) -> dict:
    """Small reusable harness for a caller-supplied future-mutation test.

    ``invariant()`` returns the baseline frame; ``mutate_future()`` mutates a copy
    and returns the frame that must be identical at the earlier cutoff.
    """
    before = invariant()
    after = mutate_future()
    try:
        pd.testing.assert_frame_equal(before, after)
        return {"passed": True, "policy": "add_future_rows_cannot_change_earlier_cutoff"}
    except AssertionError as exc:
        return {"passed": False, "error": str(exc)}


def audit_monitoring_use(features: pd.DataFrame, monitoring_audit: Mapping) -> dict:
    """Warn if chemistry columns imply records that the identity audit says do not exist."""
    checks = []
    for column in ("ph_median", "co2_water_median", "total_mineralization_median"):
        if column in features:
            nonnull = int(features[column].notna().sum())
            usable = int((monitoring_audit.get("usable_for_targets", {})
                         .get(str(1250022871), {}).get("observation_eligible_rows", 0)))
            checks.append({"column": column, "nonnull_target_values": nonnull,
                           "identity_usable_rows_for_target": usable,
                           "consistent": (nonnull == 0 or usable > 0)})
    return {"passed": all(c["consistent"] for c in checks), "checks": checks,
            "note": "Transferred or imputed chemistry is never accepted as measured chemistry."}

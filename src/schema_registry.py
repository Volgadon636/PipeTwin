"""Versioned field contracts and a complete feature inventory.

The inventory is intentionally exhaustive: every supplied raw column gets a row,
even if it is archival and not used by a model.  Field-level provenance status is
kept separate from record-level validity flags.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import CANONICAL_COL_MAPPINGS, TECH_FILE
from .source_registry import (CONTRACTS, HYDRAULIC, MONITORING, PASSPORT, PROFILE,
                              TEMPLATE, UZT, VTD, LEGACY_TO_SOURCE, source_contract)

STATUSES = ("measured", "source_calculated", "derived_verified", "assumed_scenario",
            "available_but_excluded", "unresolved_semantics", "not_provided")


@dataclass(frozen=True)
class FieldSpec:
    source_type: str
    canonical: str
    scope_grain: str
    raw_unit: str = ""
    normalized_unit: str = ""
    temporal_availability: str = "unknown"
    transformation: str = ""
    status: str = "available_but_excluded"
    exclusion_reason: str = ""
    mentor_question_id: str = ""
    unit_verified: str = "not_applicable"
    source_calculation: str = ""


def _spec(source_type, canonical, scope, unit="", norm_unit="", availability="", transformation="",
          status="available_but_excluded", reason="", question="", unit_verified="not_applicable",
          source_calculation=""):
    return FieldSpec(source_type, canonical, scope, unit, norm_unit, availability,
                     transformation, status, reason, question, unit_verified,
                     source_calculation)


# Explicit contracts for the fields that matter scientifically.  Raw fields not
# listed fall back to a source-specific archival status below.
SPECS: dict[tuple[str, str], FieldSpec] = {}


def _add(source_type, canonical, scope, unit="", norm_unit="", availability="",
         transformation="", status="available_but_excluded", reason="", question="",
         unit_verified="not_applicable", source_calculation=""):
    SPECS[(source_type, canonical)] = _spec(
        source_type, canonical, scope, unit, norm_unit, availability, transformation,
        status, reason, question, unit_verified, source_calculation)


# Passport / simple section -------------------------------------------------
_add(PASSPORT, "id", "simple_section", "OIS-PIPE ID", "integer ID", "static",
     "exact identifier parse", "measured")
_add(PASSPORT, "length_km", "simple_section", "km", "km", "static",
     "length_m=length_km*1000; grid uses actual length", "measured",
     unit_verified="verified by grid reconciliation")
_add(PASSPORT, "diameter_mm", "simple_section", "mm", "mm", "static",
     "external diameter; no ID conversion", "measured")
_add(PASSPORT, "nominal_thickness_mm", "simple_section", "mm", "mm", "static",
     "used as cold-start prior and observation ceiling", "measured")
_add(PASSPORT, "commissioning_date", "simple_section", "date", "date", "static",
     "date-only; no timezone inferred", "measured")
_add(PASSPORT, "main_pipeline_id", "parent_pipeline", "ID", "ID", "static",
     "validation grouping only; not a model factor", "measured",
     reason="Mentor meeting 2: use simple sections; parent/direction not required predictors")
_add(PASSPORT, "direction_id", "direction", "ID", "ID", "static",
     "identity audit only; not a model factor", "measured",
     reason="Mentor meeting 2: do not add direction as a factor")
_add(PASSPORT, "direction_name", "direction", "text", "text", "static",
     "identity audit only", "available_but_excluded",
     reason="Hierarchy is not a required predictor; retained for conflict audit")
_add(PASSPORT, "pipeline_name", "simple_section", "text", "text", "static",
     "identity audit only", "available_but_excluded", reason="Unverified as-of history")
_add(PASSPORT, "steel_grade", "simple_section", "text", "category", "static",
     "archival; no approved categorical encoding", "available_but_excluded",
     reason="Static steel grade is not yet in the approved feature set; unknown != absent")
_add(PASSPORT, "internal_coating", "simple_section", "text", "category", "static",
     "archival; no approved categorical encoding", "available_but_excluded",
     reason="Coating type/condition has no dated history in the supplied archive")
_add(PASSPORT, "scrap_wall_thickness_mm", "simple_section", "mm", "mm", "static",
     "retained for audit; no critical-threshold policy confirmed", "unresolved_semantics",
     question="Q-0006", reason="Critical/rejection thickness policy and applicability at forecast date unresolved")
_add(PASSPORT, "length_accounting_m", "simple_section", "m", "m", "static",
     "cross-check only; do not substitute for grid length", "available_but_excluded",
     reason="Accounting length may differ from route/grid length")

# Profile -------------------------------------------------------------------
_add(PROFILE, "id", "simple_section", "ID", "integer ID", "static", "exact ID", "measured")
_add(PROFILE, "total_distance", "segment", "m", "m", "static",
     "piecewise-linear interpolation; no coordinate clipping", "measured")
_add(PROFILE, "z", "segment", "m", "m", "static",
     "length-weighted mean and boundary slope", "measured")
_add(PROFILE, "lon", "point", "degrees", "degrees", "static", "archival", "available_but_excluded",
     reason="Coordinates are not in the approved feature set")
_add(PROFILE, "lat", "point", "degrees", "degrees", "static", "archival", "available_but_excluded",
     reason="Coordinates are not in the approved feature set")

# VTD -----------------------------------------------------------------------
_add(VTD, "id", "simple_section", "ID", "integer ID", "inspection", "exact ID", "measured")
_add(VTD, "date", "campaign", "date", "date", "inspection", "date-only unless timestamp supplied", "measured")
_add(VTD, "distance", "point", "m", "m", "inspection",
     "Distance wins over Distance_raw; out-of-pipe quarantined", "measured")
_add(VTD, "distance_raw", "point", "m", "m", "inspection", "preserved, not used for mapping",
     "available_but_excluded", reason="Mentor instruction: use Distance, not Distance_raw")
_add(VTD, "row_id", "record", "ID", "ID", "inspection", "provenance", "measured")
_add(VTD, "residual_wall_thickness", "segment/campaign", "mm", "mm", "inspection",
     "observation-level candidate -> eligible minimum", "measured",
     source_calculation="min over eligible measured values; missing stays missing")
_add(VTD, "wall_thickness", "point", "mm", "mm", "inspection",
     "local VTD wall; informational comparison only", "measured",
     reason="Local wall reference differs from passport nominal; not a second ceiling")
_add(VTD, "defect_depth_pct", "point", "%", "%", "inspection",
     "2018 campaign fraction->percent only after measured check", "source_calculated",
     source_calculation="documented campaign-specific conversion")
_add(VTD, "defect_depth", "point", "mm", "mm", "inspection", "preserved", "source_calculated",
     reason="Defect depth is not the target; reconstruction of missing residual is disabled")
_add(VTD, "angle", "point", "degrees", "degrees", "inspection", "preserved", "available_but_excluded",
     reason="Mentor meeting 2: angle should not be used/matched in the current solution")
_add(VTD, "defect_length", "point", "mm", "mm", "inspection", "preserved", "available_but_excluded",
     reason="No approved defect start/centre/length overlap rule; do not spread minima")
_add(VTD, "defect_width", "point", "mm", "mm", "inspection", "preserved", "available_but_excluded",
     reason="No approved defect matching/transfer rule")
_add(VTD, "defect_type", "point", "category", "category", "inspection", "preserved",
     "available_but_excluded", reason="Categorical defect matching not approved")
_add(VTD, "section_number", "campaign", "ID", "ID", "inspection", "audit", "available_but_excluded",
     reason="Section numbering not proven equivalent to the 10 m grid")
_add(VTD, "feature_number", "record", "ID", "ID", "inspection", "audit", "available_but_excluded",
     reason="Feature identity changes between campaigns")

# UZT -----------------------------------------------------------------------
_add(UZT, "id", "simple_section", "ID", "integer ID", "inspection", "exact ID", "measured")
_add(UZT, "date", "campaign", "date", "date", "inspection", "date parsing", "measured")
_add(UZT, "distance", "point", "m", "m", "inspection",
     "control-point distance; separate from VTD", "measured",
     reason="All current UZT measurements are above passport nominal and excluded pending method")
_add(UZT, "start_distance", "control_section", "m", "m", "inspection", "audit", "available_but_excluded",
     reason="UZT point identity/wall reference not reconciled with VTD")
_add(UZT, "residual_wall_thickness", "point", "mm", "mm", "inspection",
     "preserved; policy excludes above-nominal values", "available_but_excluded",
     question="Q-0005", reason="All 67 supplied values exceed passport nominal; methodology unconfirmed")
_add(UZT, "angle", "point", "degrees", "degrees", "inspection", "preserved", "available_but_excluded",
     reason="Instrument orientation not established; angle is not a current feature")
_add(UZT, "control_section_num", "control_section", "ID", "ID", "inspection", "audit",
     "available_but_excluded", reason="Control-point identity not joined to VTD")
_add(UZT, "control_point_id", "point", "ID", "ID", "inspection", "audit",
     "available_but_excluded", reason="Control-point identity not joined to VTD")
_add(UZT, "measurement_id", "record", "ID", "ID", "inspection", "audit",
     "available_but_excluded", reason="Measurement identity retained for audit")

# Hydraulic / Techmod -------------------------------------------------------
_add(HYDRAULIC, "id", "simple_section", "ID", "integer ID", "daily calculation", "exact ID", "measured")
_add(HYDRAULIC, "date", "daily calculation", "timestamp", "timestamp", "daily calculation",
     "calc_date != observation/availability; cutoff strict", "source_calculated",
     source_calculation="calculation timestamp")
_add(HYDRAULIC, "p_start_fact", "section", "pressure", "pressure", "daily calculation", "preserved",
     "available_but_excluded", reason="Factual pressure not used; units/basis unconfirmed", question="Q-0002")
_add(HYDRAULIC, "p_end_fact", "section", "pressure", "pressure", "daily calculation", "preserved",
     "available_but_excluded", reason="Factual pressure not used; units/basis unconfirmed", question="Q-0002")
_add(HYDRAULIC, "p_start", "section", "pressure", "pressure", "daily calculation",
     "mean/std; do not interpret as drop", "source_calculated",
     reason="Gauge/absolute and exact unit unconfirmed; level/variability only", question="Q-0002",
     unit_verified="unverified (observed scale only)")
_add(HYDRAULIC, "p_end", "section", "pressure", "pressure", "daily calculation",
     "mean/std; do not interpret as drop", "source_calculated",
     reason="Gauge/absolute and exact unit unconfirmed; level/variability only", question="Q-0002",
     unit_verified="unverified (observed scale only)")
_add(HYDRAULIC, "q_liquid", "section", "volume/time", "volume/time", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="operating/standard conditions unconfirmed")
_add(HYDRAULIC, "q_oil", "section", "volume/time", "volume/time", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     reason="Phase conditions unconfirmed; masked with suspect liquid flow")
_add(HYDRAULIC, "q_gas", "section", "volume/time", "volume/time", "daily calculation", "preserved",
     "available_but_excluded", reason="Phase conditions and relation to gas_content_rate unverified",
     question="Q-0002")
_add(HYDRAULIC, "gas_content_rate", "section", "ratio", "ratio", "daily calculation", "preserved",
     "available_but_excluded", reason="Do not equate with q_gas without a definition", question="Q-0002")
_add(HYDRAULIC, "watercut_pct", "section", "%", "%", "daily calculation",
     "pct retained; water_fraction=watercut/100 only as named derived field", "source_calculated",
     unit_verified="percent convention confirmed in data range")
_add(HYDRAULIC, "v_liquid_true", "section", "m/s", "m/s", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="observed scale consistent; exact definition unconfirmed")
_add(HYDRAULIC, "v_mix", "section", "m/s", "m/s", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="observed scale consistent; mixture definition unconfirmed")
_add(HYDRAULIC, "flow_regime", "section", "category", "category", "daily calculation", "preserved",
     "available_but_excluded", reason="Source regime category is not a Reynolds classification")
_add(HYDRAULIC, "fill_coefficient", "section", "fraction/percent", "unresolved", "daily calculation",
     "preserved", "unresolved_semantics", reason="Fraction/percent and phase are unverified")
_add(HYDRAULIC, "density_liquid_work", "section", "kg/m3", "kg/m3", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="observed scale consistent; reference conditions unconfirmed")
_add(HYDRAULIC, "density_oil", "section", "kg/m3", "kg/m3", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="observed scale consistent; reference conditions unconfirmed")
_add(HYDRAULIC, "viscosity_liquid_work", "section", "Pa.s or m2/s", "unresolved", "daily calculation",
     "preserved; not used for Reynolds", "unresolved_semantics", question="Q-0002",
     reason="Dynamic vs kinematic viscosity and units are not established")
_add(HYDRAULIC, "pit_corrosion_intensity", "section", "unknown", "unresolved", "daily calculation",
     "archived; excluded", "available_but_excluded",
     reason="Unverified source calculation; leakage/provenance audit not passed")
_add(HYDRAULIC, "t_start", "section", "degC", "degC", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="observed scale consistent; measurement/source unconfirmed")
_add(HYDRAULIC, "t_end", "section", "degC", "degC", "daily calculation",
     "daily median then observed-day mean", "source_calculated", question="Q-0002",
     unit_verified="observed scale consistent; measurement/source unconfirmed")
_add(HYDRAULIC, "t_mix", "section", "degC", "degC", "daily calculation",
     "daily median then observed-day mean", "unresolved_semantics", question="Q-0002",
     reason="Physical meaning unverified; model v2 use is a documented proxy, not a measured mixed temperature")
_add(HYDRAULIC, "length", "hydraulic_section", "m", "m", "daily calculation", "reconciliation only",
     "available_but_excluded", reason="Hydraulic section length is not the passport 10 m grid length")
_add(HYDRAULIC, "diameter_mm", "section", "m", "mm", "daily calculation",
     "multiply by 1000 only after passport comparison", "derived_verified",
     reason="Converted value retained for audit; passport diameter is the feature source",
     unit_verified="verified against passport")

# Monitoring / RepCorr ------------------------------------------------------
_add(MONITORING, "id", "simple_section", "ID", "integer ID", "monitoring interval",
     "identity audit; no automatic join", "unresolved_semantics", question="Q-0004",
     reason="412 unknown IDs and 83 direction conflicts; no usable monitoring rows")
_add(MONITORING, "direction_name", "direction", "text", "text", "monitoring interval",
     "identity audit only", "unresolved_semantics", question="Q-0004",
     reason="Direction conflicts block joins")
_add(MONITORING, "sample_installation_date", "coupon", "date", "date", "monitoring interval",
     "exposure start; not availability", "unresolved_semantics", question="Q-0004")
_add(MONITORING, "sample_removal_date", "coupon", "date", "date", "monitoring interval",
     "availability proxy only after removal; no pre-removal use", "unresolved_semantics",
     question="Q-0004")
_add(MONITORING, "exposure_days", "coupon", "days", "days", "monitoring interval",
     "declared and calculated; mismatches flagged", "source_calculated", question="Q-0004")
_add(MONITORING, "ph", "coupon", "pH", "pH", "monitoring interval", "preserved", "measured",
     reason="No usable mapped record for either target; no imputation", question="Q-0004")
_add(MONITORING, "co2_water", "coupon", "concentration", "unresolved", "monitoring interval",
     "preserved", "unresolved_semantics", question="Q-0003",
     reason="Dissolved CO2 vs gas fraction/partial pressure not established")
_add(MONITORING, "total_mineralization", "coupon", "concentration", "unresolved", "monitoring interval",
     "preserved", "measured", reason="No usable mapped record for either target", question="Q-0004")
_add(MONITORING, "fluid_density", "coupon", "unknown", "unresolved", "monitoring interval",
     "preserved; suspicious scale flagged", "unresolved_semantics", question="Q-0002",
     reason="Values such as 39/76/81 suggest a unit problem")
_add(MONITORING, "flow_liquid", "coupon", "volume/time", "unresolved", "monitoring interval",
     "preserved", "available_but_excluded", reason="Monitoring-associated rate is not a daily Techmod replacement",
     question="Q-0004")
_add(MONITORING, "watercut_pct", "coupon", "%", "%", "monitoring interval", "preserved",
     "available_but_excluded", reason="Not a silent replacement for daily Techmod watercut", question="Q-0004")
_add(MONITORING, "ukk_number", "monitoring_station", "ID", "ID", "monitoring interval",
     "identity audit", "available_but_excluded", reason="Station identity not joinable without crosswalk",
     question="Q-0004")
_add(MONITORING, "ukk_distance", "monitoring_station", "m", "m", "monitoring interval",
     "point location only", "available_but_excluded", reason="Point measurement not propagated across pipe",
     question="Q-0004")
_add(MONITORING, "protective_effect_pct", "coupon", "%", "%", "monitoring interval", "preserved",
     "available_but_excluded", reason="Protective effect is not inhibitor dose", question="Q-0003")
_add(MONITORING, "diameter", "simple_section", "mm", "mm", "monitoring interval", "audit",
     "available_but_excluded", reason="Identity/unit cross-check only")
_add(MONITORING, "wall_thickness", "simple_section", "mm", "mm", "monitoring interval", "audit",
     "available_but_excluded", reason="Not the target residual measurement")


DEFAULT_REASON = {
    PASSPORT: "Archival passport field; no dated as-of history or approved feature contract",
    PROFILE: "Archival profile field; not used by the current feature contract",
    VTD: "Archival VTD field; no approved defect-matching or predictor contract",
    UZT: "Archival UZT field; measurement identity/method unresolved",
    HYDRAULIC: "Archival hydraulic field; not in the approved feature contract",
    MONITORING: "Archival monitoring field; identity/coverage unresolved",
    TEMPLATE: "Submission-only field; not a feature",
}

DEFAULT_AVAILABILITY = {
    PASSPORT: "static (present-day passport; no change history)",
    PROFILE: "static route survey",
    VTD: "inspection event",
    UZT: "inspection event",
    HYDRAULIC: "daily calculation (calc_date; availability unconfirmed)",
    MONITORING: "coupon exposure interval (removal-date availability proxy)",
    TEMPLATE: "evaluation only",
}


def source_type_for_file(filename: str) -> str:
    if filename in LEGACY_TO_SOURCE:
        return LEGACY_TO_SOURCE[filename]
    for source_type, contract in CONTRACTS.items():
        if filename == contract.source_type:
            return source_type
    return filename


def raw_headers_for_inventory(summary: Mapping[str, dict]) -> list[tuple[str, str, str]]:
    """Return ``(legacy_file, raw_column, canonical_column)`` for every raw column."""
    rows = []
    for filename, info in summary.items():
        source_type = info.get("source_type") or source_type_for_file(filename)
        mapping = info.get("column_mapping", {})
        for raw, canonical in mapping.items():
            rows.append((filename, raw, canonical))
    return rows


def _infer_unit(header: str) -> str:
    text = str(header).lower()
    if "км" in text or re.search(r"\bkm\b", text):
        return "km"
    if "мм" in text or re.search(r"\bmm\b", text):
        return "mm"
    if re.search(r"(?:^|[_\s])m(?:$|[_\s,])", text) and "mm" not in text:
        return "m"
    if "%" in text or "процент" in text:
        return "%"
    if "°c" in text or "град" in text or re.search(r"\bc\b", text):
        return "degC"
    if "кг/м" in text or "kg/m" in text:
        return "kg/m3"
    if "мпа" in text or "бар" in text:
        return "pressure"
    if "дней" in text or "days" in text:
        return "days"
    return ""


def build_feature_inventory(cleaned: Mapping[str, pd.DataFrame], summary: Mapping[str, dict],
                            target_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per raw supplied column plus one row per published feature.

    Missing percentages are computed globally.  Per-target values are computed only
    when the field is represented in ``target_features``; otherwise the value is
    explicitly ``not_in_target_features`` rather than silently zero.
    """
    rows = []
    targets = {}
    if target_features is not None and not target_features.empty and "id" in target_features:
        targets = {int(pid): part for pid, part in target_features.groupby("id")}
    for filename, raw, canonical in raw_headers_for_inventory(summary):
        source_type = source_type_for_file(filename)
        spec = SPECS.get((source_type, canonical))
        if spec is None:
            spec = _spec(source_type, canonical,
                         scope="archival",
                         unit=_infer_unit(raw),
                         norm_unit="",
                         availability=DEFAULT_AVAILABILITY.get(source_type, "unknown"),
                         transformation="",
                         status="available_but_excluded",
                         reason=DEFAULT_REASON.get(source_type, "Archival field not approved for features"))
        frame = cleaned.get(filename)
        missing_pct = None
        if frame is not None and canonical in frame.columns:
            missing_pct = round(100.0 * float(frame[canonical].isna().mean()), 6)
        per_target = {}
        for pid, part in targets.items():
            candidates = [canonical] + [c for c in part.columns if c.startswith(canonical + "_")]
            column = next((c for c in candidates if c in part.columns), None)
            per_target[pid] = (None if column is None else
                               round(100.0 * float(part[column].isna().mean()), 6))
        rows.append({
            "field_id": f"{source_type}:{raw}",
            "source_type": source_type,
            "source_file": filename,
            "source_column": raw,
            "canonical_field": canonical,
            "scope_grain": spec.scope_grain,
            "raw_unit": spec.raw_unit or _infer_unit(raw),
            "normalized_unit": spec.normalized_unit,
            "unit_verified": spec.unit_verified,
            "temporal_availability": spec.temporal_availability or DEFAULT_AVAILABILITY.get(source_type, "unknown"),
            "missing_pct_global": missing_pct,
            "missing_pct_target_1250022871": per_target.get(1250022871),
            "missing_pct_target_1250009583": per_target.get(1250009583),
            "missing_pct_folds": "not_computed_in_preprocessing; see reports/feature_inventory_folds.csv after evaluation",
            "transformation": spec.transformation,
            "feature_status": spec.status,
            "exclusion_reason": spec.exclusion_reason,
            "mentor_question_id": spec.mentor_question_id,
            "source_calculation": spec.source_calculation,
        })
    # Derived/published feature columns; these are not raw cells and are labelled.
    if target_features is not None:
        raw_fields = {(r["canonical_field"], r["source_type"]) for r in rows}
        for column in target_features.columns:
            if any(column == c or column.startswith(c + "_") for c, _ in raw_fields):
                # Still keep an explicit derived row for aggregates and flags.
                pass
            if column in {"id", "segment_id", "prediction_date", "case", "start_m", "end_m", "length_m"}:
                status = "measured" if column in {"id", "segment_id"} else "derived_verified"
                reason = ""
            elif column in {"water_fraction_mean", "d_inner_assumed_mm", "age_years",
                            "years_since_vtd", "has_previous_vtd", "hydraulic_coverage_fraction",
                            "hydraulic_max_gap_days", "profile_extrapolated_flag",
                            "max_profile_gap_m", "slope_m_per_m"}:
                status = ("assumed_scenario" if column == "d_inner_assumed_mm"
                          else "derived_verified")
                reason = ("D_outer - 2*nominal is an explicit approximation" if column == "d_inner_assumed_mm" else "")
            elif column.startswith(("last_vtd", "last_uzt", "prior_vtd")):
                status, reason = "measured", ""
            elif column.startswith(("monitoring_", "ph_", "co2_", "total_mineralization_", "hydraulic_")):
                status, reason = "source_calculated", "Aggregate/coverage diagnostic; verify coverage before use"
            elif column.endswith(("_mean", "_std", "_valid_days")):
                status, reason = "source_calculated", ""
            elif column.endswith("_flag"):
                status, reason = "derived_verified", ""
            else:
                status, reason = "derived_verified", ""
            for pid, part in targets.items():
                pass
            rows.append({
                "field_id": f"derived:{column}",
                "source_type": "derived",
                "source_file": "processed/target_features.csv",
                "source_column": column,
                "canonical_field": column,
                "scope_grain": "segment/request" if "segment_id" in target_features.columns else "request",
                "raw_unit": "",
                "normalized_unit": "",
                "unit_verified": "",
                "temporal_availability": "cutoff-specific derived table",
                "missing_pct_global": round(100.0 * float(target_features[column].isna().mean()), 6),
                "missing_pct_target_1250022871": (
                    round(100.0 * float(target_features.loc[target_features.id.eq(1250022871), column].isna().mean()), 6)
                    if target_features.id.eq(1250022871).any() else None),
                "missing_pct_target_1250009583": (
                    round(100.0 * float(target_features.loc[target_features.id.eq(1250009583), column].isna().mean()), 6)
                    if target_features.id.eq(1250009583).any() else None),
                "missing_pct_folds": "not_computed_in_preprocessing; see reports/feature_inventory_folds.csv after evaluation",
                "transformation": "published derived/aggregated column",
                "feature_status": status,
                "exclusion_reason": reason,
                "mentor_question_id": "",
                "source_calculation": "",
            })
    inventory = pd.DataFrame(rows)
    if not inventory.empty:
        inventory = inventory.sort_values(["source_type", "source_column", "field_id"],
                                          kind="stable").reset_index(drop=True)
    return inventory


def feature_inventory_summary(inventory: pd.DataFrame) -> dict:
    if inventory.empty:
        return {"rows": 0, "by_status": {}}
    return {
        "rows": int(len(inventory)),
        "by_status": {str(k): int(v) for k, v in inventory.feature_status.value_counts().items()},
        "by_source": {str(k): int(v) for k, v in inventory.source_type.value_counts().items()},
    }

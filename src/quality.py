"""Quality, schema-drift, coverage and row-accounting reports."""
from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np
import pandas as pd

from .config import CANONICAL_COL_MAPPINGS
from .data_cleaner import monitoring_join_audit, row_dispositions
from .schema_registry import (DEFAULT_REASON, SPECS, build_feature_inventory,
                              feature_inventory_summary, source_type_for_file)
from .source_registry import CONTRACTS, DISPLAY_NAME


def schema_drift(cleaned: Mapping[str, pd.DataFrame], summary: Mapping[str, dict]) -> dict:
    sources = {}
    for filename, frame in cleaned.items():
        source_type = source_type_for_file(filename)
        contract = CONTRACTS.get(source_type)
        required = set(contract.required_fields) if contract else set()
        optional = set(contract.optional_fields) if contract else set()
        mapping = summary.get(filename, {}).get("column_mapping", {})
        canonical_known = set(required) | set(optional)
        archival, unknown = [], []
        for raw, canonical in mapping.items():
            if canonical in canonical_known:
                continue
            spec = SPECS.get((source_type, canonical))
            row = {"raw_column": raw, "canonical_column": canonical,
                   "reason": (spec.exclusion_reason if spec else
                              DEFAULT_REASON.get(source_type, "archival field"))}
            if spec is None:
                unknown.append(row)
            else:
                archival.append(row)
        sources[filename] = {
            "source_type": source_type,
            "required_fields_present": sorted(required & set(mapping.values())),
            "required_fields_missing": sorted(required - set(mapping.values())),
            "archival_known_but_excluded": archival,
            "unknown_columns": unknown,
            "unknown_column_count": len(unknown),
            "duplicate_raw_headers": summary.get(filename, {}).get("duplicate_headers", []),
            "empty_raw_headers": summary.get(filename, {}).get("empty_headers", []),
            "parse_issue_records": summary.get(filename, {}).get("parse_issue_records", 0),
            "raw_preserved": True,
            "policy": "Unrecognized columns remain in the archival table; they are not model features.",
        }
    return sources


def date_coverage(cleaned: Mapping[str, pd.DataFrame]) -> dict:
    report = {}
    for filename, frame in cleaned.items():
        for column in frame.columns:
            if not pd.api.types.is_datetime64_any_dtype(frame[column]):
                continue
            valid = frame[column].dropna()
            report[f"{filename}:{column}"] = {
                "rows": int(len(frame)),
                "valid_dates": int(len(valid)),
                "missing_or_invalid": int(len(frame) - len(valid)),
                "min": None if valid.empty else str(valid.min()),
                "max": None if valid.empty else str(valid.max()),
                "date_only_rows": (int(frame[column + "_date_only"].fillna(False).sum())
                                   if column + "_date_only" in frame else None),
                "declared_timezone_values": (sorted(set(frame[column + "_tz_offset"].dropna().astype(str)))
                                             if column + "_tz_offset" in frame else []),
            }
    return report


def spatial_coverage(cleaned: Mapping[str, pd.DataFrame], grid: pd.DataFrame,
                     quarantine: Mapping[str, pd.DataFrame], summary: Mapping[str, dict] | None = None) -> dict:
    passport = cleaned["passport.csv"].set_index("id")
    lengths = passport.length_km * 1000
    payload = {
        "grid_segments": int(len(grid)),
        "grid_pipes": int(grid.id.nunique()),
        "grid_length_m": float(grid.length_m.sum()),
        "short_final_segments": int((grid.length_m < 10 - 1e-9).sum()),
        "zero_or_negative_segments": int((grid.length_m <= 0).sum()),
        "endpoint_rule": "a point exactly at pipe length maps to the final segment; no extra endpoint segment",
        "out_of_pipe": {},
        "profile": {},
    }
    for key, frame in quarantine.items():
        if "distance" in frame:
            payload["out_of_pipe"][key] = {
                "rows": int(len(frame)),
                "negative_coordinates": int((frame.distance < 0).sum()),
                "above_pipe_length": int((frame.distance > frame.id.map(lengths)).sum()),
            }
    profile = cleaned.get("profile.csv", pd.DataFrame())
    if profile is not None and not profile.empty:
        profile_info = (summary or {}).get("profile.csv", {})
        payload["profile"] = {
            "rows": int(len(profile)),
            "pipes": int(profile.id.nunique()),
            "duplicate_rows_removed": max(0, int(profile_info.get("record_count", len(profile))) - len(profile)),
            "max_gap_m": float(profile.sort_values(["id", "total_distance"]).groupby("id")
                               .total_distance.diff().max()),
            "pipes_missing_profile": sorted(set(grid.id) - set(profile.id)),
            "short_pipes_under_10m": int((lengths < 10).sum()),
        }
    return payload


def vtd_campaign_report(cleaned: Mapping[str, pd.DataFrame], vtd_by_segment: pd.DataFrame) -> dict:
    vtd = cleaned.get("vtd.csv", pd.DataFrame())
    if vtd is None or vtd.empty:
        return {"status": "no_vtd_rows"}
    rows = []
    for (pid, date), part in vtd.groupby(["id", "date"], dropna=False):
        seg = vtd_by_segment[vtd_by_segment.id.eq(pid) & vtd_by_segment.date.eq(date)]
        rows.append({
            "id": int(pid), "date": str(date) if pd.notna(date) else None,
            "observations": int(len(part)),
            "segments_with_observation": int(len(seg)),
            "segments_with_eligible_measurement": int((seg.thickness_measurement_count > 0).sum()),
            "above_nominal_observations": int(part.above_nominal_flag.fillna(False).sum()),
            "missing_residual_observations": int(part.residual_wall_thickness.isna().sum()),
            "precommissioning_observations": int(part.before_commissioning_flag.fillna(False).sum()),
            "min_residual_mm": float(part.residual_for_target_mm.min()) if part.residual_for_target_mm.notna().any() else None,
        })
    return {"campaigns": rows,
            "note": "Sparse campaigns have different ascertainment; observation counts are not independent defect counts."}


def feature_coverage(target_features: pd.DataFrame) -> dict:
    if target_features is None or target_features.empty:
        return {"rows": 0}
    result = {}
    for (pid, case), part in target_features.groupby(["id", "case"], sort=True):
        missing = {c: int(part[c].isna().sum()) for c in part.columns}
        result[f"{int(pid)}:{case}"] = {
            "rows": int(len(part)),
            "missing_by_column": missing,
            "columns_with_zero_coverage": sorted([c for c, n in missing.items() if n == len(part)]),
            "columns_full_coverage": sorted([c for c, n in missing.items() if n == 0]),
        }
    return result


def persistence_report(hydraulic_quality: pd.DataFrame) -> dict:
    if hydraulic_quality is None or hydraulic_quality.empty:
        return {"rows": 0, "by_action": {}, "by_rule": {}}
    return {
        "rows": int(len(hydraulic_quality)),
        "by_action": {str(k): int(v) for k, v in hydraulic_quality.action.value_counts().items()},
        "by_rule": {str(k): int(v) for k, v in hydraulic_quality.rule.value_counts().items()},
        "policy": "Persistence is evaluated after cutoff and before aggregation; three distinct consecutive observed calendar days required.",
    }


def rejected_records(quarantine: Mapping[str, pd.DataFrame], quality_issues: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in quarantine.items():
        for row in frame.itertuples(index=False):
            data = row._asdict() if hasattr(row, "_asdict") else dict(zip(frame.columns, row))
            rows.append({
                "source_file": name,
                "source_row": data.get("source_row", ""),
                "source_record_number": data.get("source_record_number", ""),
                "source_physical_line": data.get("source_physical_line", ""),
                "error_code": "quarantine",
                "action": "exclude_from_analytical_table_keep_source",
                "id": data.get("id", ""),
                "field": "distance" if "distance" in data else "",
                "original_value": data.get("distance", ""),
                "source_reference": f"{name} record {data.get('source_row', '')}",
            })
    if quality_issues is not None and not quality_issues.empty:
        for row in quality_issues.itertuples(index=False):
            if row.action in ("quarantine", "exclude_duplicate", "quarantine_record", "reject_record"):
                rows.append({
                    "source_file": row.file,
                    "source_row": row.source_row,
                    "source_record_number": "",
                    "source_physical_line": "",
                    "error_code": row.rule,
                    "action": row.action,
                    "id": row.id,
                    "field": row.field,
                    "original_value": row.original_value,
                    "source_reference": f"{row.file} record {row.source_row}",
                })
    result = pd.DataFrame(rows)
    columns = ["source_file", "source_row", "source_record_number", "source_physical_line",
               "error_code", "action", "id", "field", "original_value", "source_reference"]
    if result.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        result[column] = result[column].astype('string')
    return result.drop_duplicates().sort_values(
        ["source_file", "source_row", "error_code"], kind="stable").reset_index(drop=True)


def build_quality_bundle(cleaned, summary, quarantine, quality_issues, grid, tables,
                         discovery=None) -> dict:
    source_inventory = []
    if discovery is not None:
        source_inventory = discovery.inventory()
    unit_report = {}
    try:
        from .units import build_unit_report
        unit_report = build_unit_report(cleaned, {})
    except Exception as exc:  # keep quality reporting available even if an optional source is absent
        unit_report = {"status": "unavailable", "error": str(exc)}
    return {
        "source_inventory": source_inventory,
        "schema_drift": schema_drift(cleaned, summary),
        "unit_resolution": unit_report,
        "row_reconciliation": row_dispositions(cleaned, quarantine, summary).to_dict("records"),
        "monitoring_join": monitoring_join_audit(cleaned.get("regime.csv"), cleaned.get("passport.csv")),
        "date_coverage": date_coverage(cleaned),
        "spatial_coverage": spatial_coverage(cleaned, grid, quarantine, summary),
        "vtd_campaigns": vtd_campaign_report(cleaned, tables.get("vtd_by_segment")),
        "feature_coverage": feature_coverage(tables.get("target_features")),
        "persistence_filter": persistence_report(tables.get("hydraulic_quality")),
        "rejected_records": rejected_records(quarantine, quality_issues),
        "policy": "Counts describe the supplied snapshot; they are not invariants and were not forced to old report numbers.",
    }

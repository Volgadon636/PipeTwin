"""Aggregate-to-source lineage without copying enormous row lists."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd

from .config import TECH_FILE
from .grid_builder import assign_segments


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json(value) -> str:
    return _hash_text(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _ref(source_file: str, row_min, row_max, count, note=""):
    if row_min is None or pd.isna(row_min):
        return None
    return {
        "source_file": source_file,
        "source_row_min": int(row_min),
        "source_row_max": int(row_max),
        "record_count": int(count),
        "note": note,
    }


def build_lineage(cleaned, summary, tables, config, grid) -> tuple[dict, pd.DataFrame]:
    source_hashes = {name: (info or {}).get("sha256") for name, info in summary.items()}
    config_hash = _hash_json(config)
    logical = {
        "passport_static": {"source_file": "passport.csv", "source_columns": ["id", "length_km",
                            "diameter_mm", "nominal_thickness_mm", "commissioning_date"],
                            "rule": "static passport; no dated replacement history"},
        "profile_geometry": {"source_file": "profile.csv", "source_columns": ["total_distance", "z"],
                             "rule": "piecewise-linear elevation and boundary slope"},
        "vtd_minimum": {"source_file": "vtd.csv", "source_columns": ["Distance", "date",
                         "Residual Wall Thickness"], "rule": "observation-level eligibility then segment minimum"},
        "uzt_point": {"source_file": "uzt.csv", "source_columns": ["distance", "date",
                       "residual_wall_thickness"], "rule": "point measurement; current policy excludes all supplied values"},
        "hydraulic_asof": {"source_file": TECH_FILE, "source_columns": ["calc_date", "q", "watercut",
                            "v_liquid_true", "v_mix", "t_mix"], "rule": "cutoff first, then daily median and observed-day mean"},
        "monitoring_asof": {"source_file": "regime.csv", "source_columns": ["pH", "CO2 in Water Phase",
                             "Total Mineralization"], "rule": "identity-validated coupon history only"},
        "grid": {"source_file": "processed/segments_grid.csv", "source_columns": ["id", "segment_id"],
                 "rule": "ceil(length_m/10); endpoint at L belongs to final segment"},
    }
    feature_inventory = tables.get("feature_inventory")
    column_lineage = {}
    for table_name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            continue
        for column in table.columns:
            if column.startswith(("z_", "slope_", "profile_", "max_profile_")) or column in {
                    "start_m", "end_m", "length_m", "points_count"}:
                family = "profile_geometry"
            elif column.startswith(("last_vtd", "has_previous_vtd", "years_since_vtd",
                                    "above_nominal", "nominal_unresolved")):
                family = "vtd_minimum"
            elif column.startswith("last_uzt"):
                family = "uzt_point"
            elif column.startswith(("ph_", "co2_", "monitoring_", "total_mineralization_")):
                family = "monitoring_asof"
            elif column.startswith(("hydraulic_", "water_fraction", "q_", "p_", "v_", "t_",
                                    "density_", "watercut_", "pit_")):
                family = "hydraulic_asof"
            elif column in {"id", "diameter_mm", "nominal_thickness_mm", "commissioning_date",
                            "age_years", "age_days", "d_inner_assumed_mm"}:
                family = "passport_static"
            else:
                family = "grid"
            column_lineage[f"{table_name}.{column}"] = family
    # Safe source-row references for the target rows. Ranges point back to the
    # immutable source; full original rows are not copied into cells.
    vtd = cleaned.get("vtd.csv")
    vtd_refs = {}
    if vtd is not None and not vtd.empty:
        mapped = assign_segments(vtd, "distance", grid)
        for (pid, date, segment), part in mapped.groupby(["id", "date", "segment_id"]):
            vtd_refs[(int(pid), pd.Timestamp(date), int(segment))] = _ref(
                "vtd.csv", part.source_row.min(), part.source_row.max(), len(part))
    profile = cleaned.get("profile.csv")
    prof_refs = {}
    if profile is not None and not profile.empty:
        for pid, part in profile.groupby("id"):
            prof_refs[int(pid)] = _ref("profile.csv", part.source_row.min(), part.source_row.max(), len(part))
    passport = cleaned.get("passport.csv")
    pass_refs = {}
    if passport is not None and not passport.empty:
        for pid, part in passport.groupby("id"):
            pass_refs[int(pid)] = _ref("passport.csv", part.source_row.min(), part.source_row.max(), len(part))
    tech = cleaned.get(TECH_FILE)
    tech_refs = {}
    if tech is not None and not tech.empty:
        for pid, part in tech.groupby("id"):
            tech_refs[int(pid)] = _ref(TECH_FILE, part.source_row.min(), part.source_row.max(), len(part))
    keys = []
    features = tables.get("target_features")
    if features is not None and not features.empty:
        for row in features.itertuples(index=False):
            refs = {"passport": pass_refs.get(int(row.id)), "profile": prof_refs.get(int(row.id))}
            if "last_vtd_date" in features and pd.notna(row.last_vtd_date):
                refs["vtd_last"] = vtd_refs.get((int(row.id), pd.Timestamp(row.last_vtd_date), int(row.segment_id)))
            if int(row.id) in tech_refs:
                refs["hydraulic_archive"] = tech_refs[int(row.id)]
            fingerprint = _hash_text(json.dumps({
                "id": int(row.id), "segment_id": int(row.segment_id),
                "cutoff": str(row.prediction_date), "case": str(row.case),
                "config": config_hash, "sources": source_hashes,
            }, sort_keys=True, default=str))
            keys.append({
                "id": int(row.id), "segment_id": int(row.segment_id),
                "prediction_date": row.prediction_date, "case": row.case,
                "lineage_id": "LIN-" + fingerprint[:20],
                "passport_ref": json.dumps(refs.get("passport"), ensure_ascii=False, sort_keys=True),
                "profile_ref": json.dumps(refs.get("profile"), ensure_ascii=False, sort_keys=True),
                "vtd_ref": json.dumps(refs.get("vtd_last"), ensure_ascii=False, sort_keys=True),
                "hydraulic_ref": json.dumps(refs.get("hydraulic_archive"), ensure_ascii=False, sort_keys=True),
                "config_hash": config_hash,
            })
    report = {
        "version": "lineage-v1",
        "config_hash": config_hash,
        "source_hashes": source_hashes,
        "logical_lineages": logical,
        "column_lineage": column_lineage,
        "record_reference_policy": "JSON refs give source_file + source_row range + count; "
                                   "they never claim a full row copy or contaminate analytical values.",
        "aggregation_policy": "Minima resolve to VTD source-row ranges; hydraulic means resolve to the "
                              "as-of window and source hash; profile values resolve to route-profile ranges.",
        "feature_inventory_rows": 0 if feature_inventory is None else int(len(feature_inventory)),
    }
    return report, pd.DataFrame(keys)

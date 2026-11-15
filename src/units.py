"""Explicit unit-resolution policy.

Only conversions with a source-verifiable rule are reported as verified.  Unknown
or ambiguous unit semantics remain unresolved; no pressure, viscosity, CO2 or
roughness conversion is guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .config import TECH_FILE
from .source_registry import HYDRAULIC, MONITORING, PASSPORT, PROFILE, UZT, VTD


@dataclass(frozen=True)
class UnitRule:
    rule_id: str
    source_type: str
    field: str
    raw_unit: str
    normalized_unit: str
    status: str
    verification: str
    conversion: str = ""


RULES = [
    UnitRule("UNIT-0001", PASSPORT, "length_km", "km", "m", "verified",
             "grid length = round(length_km*1000, 6); last segment truncated", "multiply by 1000"),
    UnitRule("UNIT-0002", PASSPORT, "diameter_mm", "mm", "mm", "verified",
             "passport header states mm; values 159/219 match Techmod after conversion"),
    UnitRule("UNIT-0003", PASSPORT, "nominal_thickness_mm", "mm", "mm", "verified",
             "passport header states mm; positive for all 16 IDs"),
    UnitRule("UNIT-0004", PROFILE, "total_distance", "m", "m", "verified",
             "profile coordinates are metres on the same nominal pipe scale"),
    UnitRule("UNIT-0005", PROFILE, "z", "m", "m", "verified",
             "elevation profile header/description; no datum transformation applied"),
    UnitRule("UNIT-0006", VTD, "distance", "m", "m", "verified",
             "Distance is used directly; out-of-[0,L] quarantined, never clipped"),
    UnitRule("UNIT-0007", VTD, "residual_wall_thickness", "mm", "mm", "verified",
             "source values compared with passport nominal; no unit scaling"),
    UnitRule("UNIT-0008", VTD, "defect_depth_pct", "%", "%", "conditional",
             "2018-09-23 campaign fraction->percent only after measured residual check",
             "multiply by 100 for the configured campaign"),
    UnitRule("UNIT-0009", UZT, "residual_wall_thickness", "mm", "mm", "unresolved",
             "values above passport nominal; method/reference unknown; excluded"),
    UnitRule("UNIT-0010", HYDRAULIC, "diameter_mm", "m", "mm", "verified",
             "convert only when d*1000 matches passport diameter", "multiply by 1000"),
    UnitRule("UNIT-0011", HYDRAULIC, "watercut_pct", "%", "%", "verified",
             "observed range 0..100; fraction only as named water_fraction_mean", "divide by 100 for derived field"),
    UnitRule("UNIT-0012", HYDRAULIC, "p_start/p_end", "pressure", "unresolved", "unresolved",
             "gauge/absolute basis and exact unit require confirmation; no conversion"),
    UnitRule("UNIT-0013", HYDRAULIC, "viscosity_liquid_work", "Pa.s or m2/s", "unresolved", "unresolved",
             "dynamic vs kinematic not established; Reynolds not computed"),
    UnitRule("UNIT-0014", HYDRAULIC, "q/q_oil/q_gas", "volume/time", "unresolved", "unresolved",
             "operating vs standard conditions not established; no conversion"),
    UnitRule("UNIT-0015", HYDRAULIC, "t_start/t_end/t_mix", "degC", "degC", "observed_scale_only",
             "no transformation; t_mix meaning unresolved"),
    UnitRule("UNIT-0016", MONITORING, "ph", "pH", "pH", "verified",
             "dimensionless pH; no mapping to target pipe"),
    UnitRule("UNIT-0017", MONITORING, "co2_water", "concentration", "unresolved", "unresolved",
             "dissolved concentration vs gas fraction/partial pressure unresolved"),
    UnitRule("UNIT-0018", MONITORING, "fluid_density", "unknown", "unresolved", "unresolved",
             "suspicious values 39/76/81; no scale conversion"),
    UnitRule("UNIT-0019", HYDRAULIC, "length", "m", "m", "unresolved_scope",
             "hydraulic section length is not the passport 10 m grid length; reconciliation only"),
]


def build_unit_report(cleaned: Mapping[str, pd.DataFrame], config: Mapping) -> dict:
    tech = cleaned.get(TECH_FILE, pd.DataFrame())
    vtd = cleaned.get('vtd.csv', pd.DataFrame())
    passport = cleaned.get('passport.csv', pd.DataFrame())
    monitoring = cleaned.get('regime.csv', pd.DataFrame())

    def count(mask):
        try:
            return int(pd.Series(mask).fillna(False).sum())
        except Exception:
            return 0

    conversions = [
        {"rule_id": "UNIT-0001", "field": "length_km -> length_m", "rows": int(len(passport)),
         "status": "verified"},
        {"rule_id": "UNIT-0008", "field": "defect_depth_pct fraction -> percent",
         "rows": count(vtd.get('depth_fraction_normalized', pd.Series(dtype=bool))),
         "status": "conditional_campaign"},
        {"rule_id": "UNIT-0010", "field": "Techmod d_mm m -> mm",
         "rows": count(tech.get('diameter_mm_raw', pd.Series(dtype=float)).notna()),
         "status": "verified"},
        {"rule_id": "UNIT-0011", "field": "watercut_pct -> water_fraction",
         "rows": 0, "status": "derived_only"},
    ]
    unresolved = [
        {"rule_id": rule.rule_id, "source_type": rule.source_type, "field": rule.field,
         "raw_unit": rule.raw_unit, "status": rule.status, "verification": rule.verification}
        for rule in RULES if rule.status in ("unresolved", "observed_scale_only", "unresolved_scope")
    ]
    return {
        "rules": [rule.__dict__ for rule in RULES],
        "conversions": conversions,
        "unresolved": unresolved,
        "tech_length_reconciliation": _tech_length_reconciliation(tech, passport),
        "monitoring_unit_anomalies": {
            "fluid_density_suspect_rows": count(monitoring.get('density_suspect_flag', pd.Series(dtype=bool)))
            if monitoring is not None and len(monitoring) else 0,
        },
        "policy": "No conversion is applied from a filename, a single character count, or an inferred locale.",
        "config_version": config.get("unit_policy_version", "unit-policy-v1"),
    }


def _tech_length_reconciliation(tech: pd.DataFrame, passport: pd.DataFrame) -> dict:
    if tech is None or tech.empty or passport is None or passport.empty:
        return {"available": False, "reason": "missing source"}
    if 'length' not in tech or 'id' not in tech or 'length_km' not in passport:
        return {"available": False, "reason": "missing column"}
    expected = passport.set_index('id').length_km * 1000
    actual = tech.id.map(expected)
    comparable = tech.length.notna() & actual.notna()
    if not comparable.any():
        return {"available": False, "reason": "no comparable rows"}
    diff = (tech.loc[comparable, 'length'] - actual[comparable]).abs()
    return {
        "available": True,
        "rows": int(comparable.sum()),
        "exact_or_within_0_1m": int((diff <= 0.1).sum()),
        "mismatch_over_0_1m": int((diff > 0.1).sum()),
        "median_abs_diff_m": float(diff.median()),
        "max_abs_diff_m": float(diff.max()),
        "policy": "Source hydraulic length is retained; passport length defines the segment grid.",
    }

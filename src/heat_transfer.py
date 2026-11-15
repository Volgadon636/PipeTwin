"""Explicit steady-state heat-loss scenario.

This module is intentionally isolated from the verified grid and target features.
The mentor's supplied structure is used only as a scenario after its parameter
values are declared; it is not a measured 10 m temperature or a calibrated model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import pandas as pd


SCENARIO_VERSION = "heat-loss-scenario-v1"
DEFAULT_PARAMETERS = {
    "ground_temperature_c": 5.0,
    "overall_heat_transfer_w_m2_k": 1.0,
    "heat_capacity_j_kg_k": 3500.0,
    "roughness_m": 0.00005,
    "perimeter_basis": "pi*(D_inner-2*roughness) as supplied by mentor; same area basis for U",
    "flow_basis": "liquid mass flow rho_liquid*v_liquid*pi*D_inner^2/4; mixture flow unverified",
}


@dataclass(frozen=True)
class HeatTransferParameters:
    ground_temperature_c: float = 5.0
    overall_heat_transfer_w_m2_k: float = 1.0
    heat_capacity_j_kg_k: float = 3500.0
    roughness_m: float = 0.00005
    perimeter_basis: str = DEFAULT_PARAMETERS["perimeter_basis"]
    flow_basis: str = DEFAULT_PARAMETERS["flow_basis"]

    def assumptions(self) -> dict:
        return asdict(self)


def validate_parameters(params: HeatTransferParameters, d_inner_m: float,
                        mass_flow_kg_s: float) -> list[str]:
    issues = []
    if not math.isfinite(d_inner_m) or d_inner_m <= 0:
        issues.append("nonpositive_or_nonfinite_inner_diameter")
    if not math.isfinite(mass_flow_kg_s) or mass_flow_kg_s <= 0:
        issues.append("nonpositive_or_missing_mass_flow")
    if params.heat_capacity_j_kg_k <= 0:
        issues.append("nonpositive_heat_capacity")
    if params.overall_heat_transfer_w_m2_k < 0:
        issues.append("negative_heat_transfer_coefficient")
    if params.roughness_m < 0:
        issues.append("negative_roughness")
    if d_inner_m <= 2 * params.roughness_m:
        issues.append("diameter_not_larger_than_twice_roughness")
    return issues


def segment_outlet_temperature(t_in_c: float, length_m: float, d_inner_m: float,
                               mass_flow_kg_s: float,
                               params: HeatTransferParameters) -> dict:
    """Return one segment's outlet and delta under the declared steady-state model.

    ``T_out = T_ground + (T_in - T_ground)*exp(-U*P*L/(m_dot*cp))``.
    Negative delta means cooling under this convention.
    """
    issues = validate_parameters(params, d_inner_m, mass_flow_kg_s)
    if not math.isfinite(t_in_c):
        issues.append("nonfinite_inlet_temperature")
    if not math.isfinite(length_m) or length_m <= 0:
        issues.append("nonpositive_or_nonfinite_segment_length")
    if issues:
        return {"valid": False, "issues": issues, "t_in_c": t_in_c, "t_out_c": None,
                "delta_temperature_c": None, "heat_transfer_perimeter_m": None,
                "exponent": None, "parameters": json.dumps(params.assumptions(), sort_keys=True)}
    perimeter = math.pi * (d_inner_m - 2 * params.roughness_m)
    exponent = -(params.overall_heat_transfer_w_m2_k * perimeter * length_m
                 / (mass_flow_kg_s * params.heat_capacity_j_kg_k))
    t_out = params.ground_temperature_c + (t_in_c - params.ground_temperature_c) * math.exp(exponent)
    return {"valid": True, "issues": [], "t_in_c": float(t_in_c), "t_out_c": float(t_out),
            "delta_temperature_c": float(t_out - t_in_c),
            "heat_transfer_perimeter_m": float(perimeter), "exponent": float(exponent),
            "parameters": json.dumps(params.assumptions(), sort_keys=True)}


def simulate_pipe(segments: Iterable[Mapping], inlet_temperature_c: float,
                  d_inner_m: float, mass_flow_kg_s: float,
                  params: HeatTransferParameters) -> list[dict]:
    """Apply continuity from segment to segment using actual segment lengths."""
    ordered = sorted(segments, key=lambda r: (float(r["start_m"]), float(r["end_m"])))
    current = float(inlet_temperature_c)
    results = []
    for segment in ordered:
        result = segment_outlet_temperature(current, float(segment["length_m"]), d_inner_m,
                                            mass_flow_kg_s, params)
        result.update({"segment_id": int(segment["segment_id"]),
                       "start_m": float(segment["start_m"]),
                       "end_m": float(segment["end_m"]),
                       "length_m": float(segment["length_m"])})
        results.append(result)
        if not result["valid"]:
            break
        current = result["t_out_c"]
    return results


def mass_flow_from_liquid(density_kg_m3, velocity_m_s, d_inner_m) -> dict:
    values = (density_kg_m3, velocity_m_s, d_inner_m)
    if any(v is None for v in values):
        return {"valid": False, "mass_flow_kg_s": None, "reason": "missing_input"}
    try:
        rho, velocity, diameter = (float(v) for v in values)
    except (TypeError, ValueError):
        return {"valid": False, "mass_flow_kg_s": None, "reason": "non_numeric_input"}
    if not all(math.isfinite(v) for v in (rho, velocity, diameter)):
        return {"valid": False, "mass_flow_kg_s": None, "reason": "nonfinite_input"}
    if rho <= 0 or velocity <= 0 or diameter <= 0:
        return {"valid": False, "mass_flow_kg_s": None,
                "reason": "zero_or_nonpositive_flow_density_or_diameter"}
    area = math.pi * diameter ** 2 / 4.0
    return {"valid": True, "mass_flow_kg_s": float(rho * velocity * area),
            "reason": "", "area_m2": float(area),
            "basis": "liquid mass flow from rho_liquid * v_liquid * area; not a verified mixture flow"}


def build_scenario(target_features: pd.DataFrame, parameters: HeatTransferParameters,
                   scenario_id: str = "BASE") -> pd.DataFrame:
    """Build scenario rows for one pipe/request at a time.

    All required inputs must be present and usable.  Missing inlet temperature or
    nonpositive flow yields a row with ``valid=False`` and a reason; it is never
    silently filled from another pipe/date.
    """
    rows = []
    for (pid, cutoff, case), part in target_features.groupby(["id", "prediction_date", "case"], sort=True):
        part = part.sort_values("segment_id")
        first = part.iloc[0]
        t_in = first.get("t_start_mean")
        d_inner_mm = first.get("d_inner_assumed_mm")
        density = first.get("density_liquid_work_mean")
        velocity = first.get("v_liquid_true_mean")
        d_inner_m = None if pd.isna(d_inner_mm) else float(d_inner_mm) / 1000.0
        flow = mass_flow_from_liquid(density, velocity, d_inner_m)
        if not flow["valid"]:
            rows.append({"id": int(pid), "prediction_date": cutoff, "case": case,
                         "scenario_id": scenario_id, "segment_id": None,
                         "start_m": None, "end_m": None, "length_m": None,
                         "t_in_c": None, "t_out_c": None, "delta_temperature_c": None,
                         "valid": False, "reason": flow["reason"],
                         "parameters": json.dumps(parameters.assumptions(), sort_keys=True)})
            continue
        if pd.isna(t_in):
            rows.append({"id": int(pid), "prediction_date": cutoff, "case": case,
                         "scenario_id": scenario_id, "segment_id": None,
                         "start_m": None, "end_m": None, "length_m": None,
                         "t_in_c": None, "t_out_c": None, "delta_temperature_c": None,
                         "valid": False, "reason": "missing_inlet_temperature",
                         "parameters": json.dumps(parameters.assumptions(), sort_keys=True)})
            continue
        for item in simulate_pipe(part[["segment_id", "start_m", "end_m", "length_m"]].to_dict("records"),
                                  t_in, d_inner_m, flow["mass_flow_kg_s"], parameters):
            rows.append({"id": int(pid), "prediction_date": cutoff, "case": case,
                         "scenario_id": scenario_id, "segment_id": item["segment_id"],
                         "start_m": item["start_m"], "end_m": item["end_m"],
                         "length_m": item["length_m"], "t_in_c": item["t_in_c"],
                         "t_out_c": item["t_out_c"],
                         "delta_temperature_c": item["delta_temperature_c"],
                         "valid": item["valid"], "reason": ";".join(item["issues"]),
                         "mass_flow_kg_s": flow["mass_flow_kg_s"],
                         "d_inner_m": d_inner_m,
                         "parameters": json.dumps(parameters.assumptions(), sort_keys=True)})
    return pd.DataFrame(rows)


def sensitivity_parameters() -> list[HeatTransferParameters]:
    """Declared scenario grid, not a calibrated confidence interval."""
    result = []
    for ground in (0.0, 5.0, 10.0):
        for u in (0.5, 1.0, 2.0):
            for cp in (3000.0, 3500.0, 4000.0):
                result.append(HeatTransferParameters(ground_temperature_c=ground,
                                                     overall_heat_transfer_w_m2_k=u,
                                                     heat_capacity_j_kg_k=cp,
                                                     roughness_m=0.00005))
    return result

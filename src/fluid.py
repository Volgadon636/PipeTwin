"""Strict fluid-mechanics helpers with explicit unit prerequisites."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReynoldsResult:
    reynolds: float | None
    status: str
    reason: str
    formula: str
    viscosity_kind: str | None
    units: str


def liquid_reynolds(diameter_internal_m, velocity_m_s, density_kg_m3,
                    viscosity, viscosity_kind: str | None = None,
                    viscosity_unit: str | None = None) -> ReynoldsResult:
    """Compute liquid Reynolds only with a declared dynamic/kinematic viscosity.

    The caller must not mix Pa.s and m2/s.  A missing or zero viscosity is
    unavailable, never zero Reynolds.  Multiphase interpretation is not implied.
    """
    values = (diameter_internal_m, velocity_m_s, density_kg_m3, viscosity)
    if any(v is None for v in values):
        return ReynoldsResult(None, "unavailable", "missing_input", "", viscosity_kind, "SI")
    try:
        d, v, rho, mu = (float(x) for x in values)
    except (TypeError, ValueError):
        return ReynoldsResult(None, "invalid", "non_numeric_input", "", viscosity_kind, "SI")
    if not all(math.isfinite(x) for x in (d, v, rho, mu)):
        return ReynoldsResult(None, "invalid", "nonfinite_input", "", viscosity_kind, "SI")
    if d <= 0 or rho <= 0 or v < 0:
        return ReynoldsResult(None, "invalid", "nonpositive_diameter_density_or_velocity", "", viscosity_kind, "SI")
    if mu <= 0:
        return ReynoldsResult(None, "unavailable", "zero_or_negative_viscosity", "", viscosity_kind, "SI")
    if viscosity_kind not in ("dynamic", "kinematic"):
        return ReynoldsResult(None, "unresolved", "viscosity_kind_not_declared_dynamic_or_kinematic",
                              "", viscosity_kind, "SI")
    if viscosity_kind == "dynamic":
        if viscosity_unit not in (None, "Pa.s", "Pa*s", "kg/(m*s)"):
            return ReynoldsResult(None, "unresolved", "dynamic_viscosity_unit_not_Pa_s",
                                  "Re=rho*v*D/mu", viscosity_kind, "SI")
        re = rho * v * d / mu
        formula = "Re=rho_liquid*v_liquid*D_internal/mu_dynamic"
    else:
        if viscosity_unit not in (None, "m2/s", "m^2/s", "St", "cSt"):
            return ReynoldsResult(None, "unresolved", "kinematic_viscosity_unit_not_m2_s",
                                  "Re=v*D/nu", viscosity_kind, "SI")
        # St/cSt are accepted only after explicit conversion by the caller; this
        # helper expects SI m2/s when a unit is supplied.
        if viscosity_unit in ("St", "cSt"):
            return ReynoldsResult(None, "unresolved", "convert_St_or_cSt_to_m2_s_first",
                                  "Re=v*D/nu", viscosity_kind, "SI")
        re = v * d / mu
        formula = "Re=v_liquid*D_internal/nu_kinematic"
    if not math.isfinite(re):
        return ReynoldsResult(None, "invalid", "nonfinite_result", formula, viscosity_kind, "SI")
    return ReynoldsResult(float(re), "defined", "", formula, viscosity_kind, "SI")


def single_phase_regime_qualifier(reynolds: float | None) -> dict:
    """Return a clearly qualified threshold, never a multiphase claim."""
    if reynolds is None:
        return {"category": None, "qualifier": "Reynolds unavailable; no flow-regime claim"}
    category = "laminar_like" if reynolds < 2300 else ("transitional_like" if reynolds < 4000 else "turbulent_like")
    return {"category": category,
            "qualifier": "single-phase threshold only; do not reduce multiphase flow to this category"}

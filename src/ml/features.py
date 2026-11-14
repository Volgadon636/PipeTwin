"""Pluggable Feature Registry for pipeline corrosion modeling.

Each feature is defined with explicit provenance:
- feature_name: Unique semantic identifier
- formula_version: Algorithm and parameterization version
- source_columns: Raw/intermediate columns required for computation
- coverage: Ratio of valid, non-null, finite values across the dataset
"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class FeatureDescriptor:
    """Formal specification of an engineered feature."""
    feature_name: str
    formula_version: str
    source_columns: List[str]
    category: str  # 'profile', 'thermal', 'flow', 'degradation'
    description: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]


@dataclass
class FeatureReport:
    """Audit report for an engineered feature evaluated on a concrete dataset."""
    feature_name: str
    formula_version: str
    source_columns: List[str]
    category: str
    coverage: float
    non_null_count: int
    total_count: int
    mean: float
    std: float
    min_val: float
    max_val: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature Computation Functions (Pipeline-isolated to prevent lookahead)
# ---------------------------------------------------------------------------

def _compute_water_trap_sag(df: pd.DataFrame, window_segments: int = 50) -> pd.Series:
    """Depth (meters) of local topographic depressions/sags below surrounding crests.
    
    Formula v1.0: rolling bilateral crest envelope sag = max(0, min(left_max, right_max) - z).
    """
    parts = []
    group_cols = ['id', 'prediction_date'] if 'prediction_date' in df.columns else ['id']
    for _, g in df.groupby(group_cols, sort=False):
        g_sorted = g.sort_values('segment_id')
        z = g_sorted['z_mean'].fillna(0.0)
        left_max = z.rolling(window_segments, min_periods=1).max()
        right_max = z.iloc[::-1].rolling(window_segments, min_periods=1).max().iloc[::-1]
        sag = np.maximum(0.0, np.minimum(left_max, right_max) - z)
        parts.append(sag.reindex(g.index))
    return pd.concat(parts).astype(float) if parts else pd.Series(dtype=float, index=df.index)


def _compute_abs_slope(df: pd.DataFrame) -> pd.Series:
    """Absolute elevation gradient |dZ/dL| (m/m)."""
    return df['slope_m_per_m'].abs().fillna(0.0).astype(float)


def _compute_curvature(df: pd.DataFrame) -> pd.Series:
    """Second spatial derivative of elevation profile (slope change between adjacent segments)."""
    parts = []
    group_cols = ['id', 'prediction_date'] if 'prediction_date' in df.columns else ['id']
    for _, g in df.groupby(group_cols, sort=False):
        g_sorted = g.sort_values('segment_id')
        curv = g_sorted['slope_m_per_m'].diff().fillna(0.0)
        parts.append(curv.reindex(g.index))
    return pd.concat(parts).astype(float) if parts else pd.Series(dtype=float, index=df.index)


def _compute_relative_distance(df: pd.DataFrame) -> pd.Series:
    """Normalized position along pipeline length [0, 1] from inlet to outlet."""
    parts = []
    group_cols = ['id', 'prediction_date'] if 'prediction_date' in df.columns else ['id']
    for _, g in df.groupby(group_cols, sort=False):
        max_seg = g['segment_id'].max()
        rel_dist = (g['segment_id'] / max_seg).fillna(0.0) if max_seg > 0 else pd.Series(0.0, index=g.index)
        parts.append(rel_dist)
    return pd.concat(parts).astype(float) if parts else pd.Series(dtype=float, index=df.index)


def _compute_relative_elevation(df: pd.DataFrame) -> pd.Series:
    """Normalized elevation [0, 1] within the pipeline profile."""
    parts = []
    group_cols = ['id', 'prediction_date'] if 'prediction_date' in df.columns else ['id']
    for _, g in df.groupby(group_cols, sort=False):
        z = g['z_mean']
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min
        rel_z = ((z - z_min) / z_range).fillna(0.0) if z_range > 0 else pd.Series(0.0, index=g.index)
        parts.append(rel_z)
    return pd.concat(parts).astype(float) if parts else pd.Series(dtype=float, index=df.index)


def _compute_temperature_c(df: pd.DataFrame, ground_temp_c: float = 4.0) -> pd.Series:
    """Segment-resolved operating fluid temperature in Celsius via calibrated Shukhov thermal decay.
    
    Formula v2.0:
    When both inlet (t_start) and outlet (t_end) temperatures are available:
        T(u) = T_soil + (T_start - T_soil) * ((T_end - T_soil) / (T_start - T_soil))^u
    where u = segment_id / max(segment_id) in [0, 1].
    If boundary conditions do not permit exponential ratio, gracefully uses linear thermal gradient.
    If boundary measurements are absent, falls back to pipe-level mean mixture temperature.
    """
    parts = []
    group_cols = ['id', 'prediction_date'] if 'prediction_date' in df.columns else ['id']
    for _, g in df.groupby(group_cols, sort=False):
        max_seg = g['segment_id'].max() if 'segment_id' in g.columns else 0
        u = (g['segment_id'] / max_seg).fillna(0.0) if max_seg > 0 else pd.Series(0.0, index=g.index)
        
        has_start = 't_start_mean' in g.columns and g['t_start_mean'].notna().any()
        has_end = 't_end_mean' in g.columns and g['t_end_mean'].notna().any()
        
        if has_start and has_end:
            t_start = float(g['t_start_mean'].dropna().iloc[0])
            t_end = float(g['t_end_mean'].dropna().iloc[0])
            delta_start = t_start - ground_temp_c
            delta_end = t_end - ground_temp_c
            
            if delta_start > 0 and delta_end > 0 and abs(delta_start) > 1e-4:
                ratio = delta_end / delta_start
                t_profile = ground_temp_c + delta_start * (ratio ** u)
            else:
                t_profile = t_start - (t_start - t_end) * u
        elif has_start:
            t_profile = g['t_start_mean'].fillna(20.0)
        elif 't_mix_mean' in g.columns:
            t_profile = g['t_mix_mean'].fillna(20.0)
        else:
            t_profile = pd.Series(20.0, index=g.index)
            
        parts.append(t_profile.astype(float))
        
    return pd.concat(parts).astype(float) if parts else pd.Series(20.0, index=df.index, dtype=float)


def _compute_arrhenius_factor(df: pd.DataFrame) -> pd.Series:
    """Dimensionless Arrhenius thermal kinetic acceleration factor exp(-Ea / RT).
    
    Formula v1.0: Ea/R ~ 2500 K (mild steel CO2 corrosion activation energy ~ 20.8 kJ/mol).
    Normalized relative to 20 deg C (293.15 K).
    """
    temp_c = _compute_temperature_c(df)
    temp_k = temp_c + 273.15
    t_ref_k = 293.15
    theta = 2500.0  # Ea / R in Kelvin
    factor = np.exp(theta * (1.0 / t_ref_k - 1.0 / temp_k.clip(lower=253.15)))
    return pd.Series(factor, index=df.index, dtype=float)


def _compute_kinetic_energy(df: pd.DataFrame) -> pd.Series:
    """Dynamic fluid kinetic energy density 0.5 * rho * v^2 (J/m3)."""
    rho = df['density_liquid_work_mean'].fillna(850.0) if 'density_liquid_work_mean' in df.columns else 850.0
    v = df['v_liquid_true_mean'].fillna(1.0) if 'v_liquid_true_mean' in df.columns else 1.0
    return (0.5 * rho * (v ** 2)).astype(float)


def _compute_water_velocity_flux(df: pd.DataFrame) -> pd.Series:
    """Superficial water velocity flux (watercut_fraction * v_liquid) in m/s."""
    wc = (df['watercut_pct_mean'] / 100.0).fillna(0.5) if 'watercut_pct_mean' in df.columns else 0.5
    v = df['v_liquid_true_mean'].fillna(1.0) if 'v_liquid_true_mean' in df.columns else 1.0
    return (wc * v).astype(float)


def _compute_pressure_drop_proxy(df: pd.DataFrame) -> pd.Series:
    """Total pressure drop along the pipeline segment (P_start - P_end) in MPa."""
    p_start = df['p_start_mean'].fillna(0.0) if 'p_start_mean' in df.columns else 0.0
    p_end = df['p_end_mean'].fillna(0.0) if 'p_end_mean' in df.columns else 0.0
    return (p_start - p_end).astype(float)


def _compute_prior_thinning(df: pd.DataFrame) -> pd.Series:
    """Historical baseline metal loss max(0, nominal - h0) in mm."""
    h0 = df['h0_mm'] if 'h0_mm' in df.columns else df['nominal_thickness_mm']
    nom = df['nominal_thickness_mm'].fillna(h0)
    return np.maximum(0.0, nom - h0).astype(float)


def _compute_prior_annual_rate(df: pd.DataFrame) -> pd.Series:
    """Historical baseline thinning rate (loss / age_years) in mm/year."""
    loss = _compute_prior_thinning(df)
    age = df['age_years'].replace(0, np.nan).fillna(1.0) if 'age_years' in df.columns else 1.0
    return (loss / age).astype(float)


# ---------------------------------------------------------------------------
# Feature Registry Implementation
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """Central registry of auditable, pluggable feature extractors."""

    def __init__(self):
        self._registry: Dict[str, FeatureDescriptor] = {}
        self._register_default_features()

    def register(self, descriptor: FeatureDescriptor):
        self._registry[descriptor.feature_name] = descriptor

    def get(self, name: str) -> FeatureDescriptor:
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' not found in registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def list_features(self, category: Optional[str] = None) -> List[str]:
        if category:
            return [k for k, v in self._registry.items() if v.category == category]
        return list(self._registry.keys())

    def _register_default_features(self):
        # 1. Profile / Topography Category
        self.register(FeatureDescriptor(
            feature_name='water_trap_sag_m',
            formula_version='v1.0_rolling_sag_500m',
            source_columns=['z_mean'],
            category='profile',
            description='Depth (m) of local terrain depression below surrounding crests',
            compute_fn=_compute_water_trap_sag
        ))
        self.register(FeatureDescriptor(
            feature_name='slope_magnitude',
            formula_version='v1.0_abs_slope',
            source_columns=['slope_m_per_m'],
            category='profile',
            description='Magnitude of pipe inclination gradient |dZ/dL| (m/m)',
            compute_fn=_compute_abs_slope
        ))
        self.register(FeatureDescriptor(
            feature_name='profile_curvature',
            formula_version='v1.0_slope_diff',
            source_columns=['slope_m_per_m'],
            category='profile',
            description='Spatial rate of grade change (d2Z/dL2)',
            compute_fn=_compute_curvature
        ))
        self.register(FeatureDescriptor(
            feature_name='relative_distance',
            formula_version='v1.0_normalized_length',
            source_columns=['segment_id'],
            category='profile',
            description='Relative distance from inlet (0 to 1)',
            compute_fn=_compute_relative_distance
        ))
        self.register(FeatureDescriptor(
            feature_name='relative_elevation',
            formula_version='v1.0_normalized_z',
            source_columns=['z_mean'],
            category='profile',
            description='Relative elevation profile normalized to [0, 1]',
            compute_fn=_compute_relative_elevation
        ))

        # 2. Thermal / Kinetic Category
        self.register(FeatureDescriptor(
            feature_name='temperature_c_mean',
            formula_version='v2.0_shukhov_thermal_gradient',
            source_columns=['t_start_mean', 't_end_mean', 't_mix_mean', 'segment_id'],
            category='thermal',
            description='Segment-resolved fluid temperature (°C) via calibrated Shukhov thermal decay profile',
            compute_fn=_compute_temperature_c
        ))
        self.register(FeatureDescriptor(
            feature_name='arrhenius_thermal_factor',
            formula_version='v2.0_segment_arrhenius_factor',
            source_columns=['t_start_mean', 't_end_mean', 't_mix_mean', 'segment_id'],
            category='thermal',
            description='Segment-resolved Arrhenius corrosion reaction rate acceleration factor',
            compute_fn=_compute_arrhenius_factor
        ))

        # 3. Flow / Hydrodynamics Category
        self.register(FeatureDescriptor(
            feature_name='liquid_kinetic_energy',
            formula_version='v1.0_half_rho_v2',
            source_columns=['density_liquid_work_mean', 'v_liquid_true_mean'],
            category='flow',
            description='Dynamic kinetic energy density 0.5 * rho * v^2 (J/m3)',
            compute_fn=_compute_kinetic_energy
        ))
        self.register(FeatureDescriptor(
            feature_name='water_velocity_flux',
            formula_version='v1.0_wc_times_v',
            source_columns=['watercut_pct_mean', 'v_liquid_true_mean'],
            category='flow',
            description='Superficial water velocity flux (watercut * v) in m/s',
            compute_fn=_compute_water_velocity_flux
        ))
        self.register(FeatureDescriptor(
            feature_name='pressure_drop_proxy',
            formula_version='v1.0_p_diff',
            source_columns=['p_start_mean', 'p_end_mean'],
            category='flow',
            description='Differential pressure drop across pipeline (MPa)',
            compute_fn=_compute_pressure_drop_proxy
        ))

        # 4. Degradation / Inspection History Category
        self.register(FeatureDescriptor(
            feature_name='prior_thinning_mm',
            formula_version='v1.0_nom_minus_h0',
            source_columns=['nominal_thickness_mm', 'h0_mm'],
            category='degradation',
            description='Historical metal loss prior to forecast window (mm)',
            compute_fn=_compute_prior_thinning
        ))
        self.register(FeatureDescriptor(
            feature_name='prior_annual_rate',
            formula_version='v1.0_loss_over_age',
            source_columns=['nominal_thickness_mm', 'h0_mm', 'age_years'],
            category='degradation',
            description='Historical annual corrosion thinning rate (mm/year)',
            compute_fn=_compute_prior_annual_rate
        ))

    def apply(self, df: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[FeatureReport]]:
        """Compute selected features and generate an audit report with exact coverage."""
        out = df.copy()
        names = feature_names or list(self._registry.keys())
        reports = []

        total_rows = len(df)
        for name in names:
            descriptor = self.get(name)
            series = descriptor.compute_fn(out)
            out[name] = series

            valid_mask = series.notna() & np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
            valid_count = int(valid_mask.sum())
            coverage = float(valid_count / total_rows) if total_rows > 0 else 0.0

            valid_values = series[valid_mask].to_numpy(float)
            if len(valid_values) > 0:
                mean_v = float(valid_values.mean())
                std_v = float(valid_values.std())
                min_v = float(valid_values.min())
                max_v = float(valid_values.max())
            else:
                mean_v = std_v = min_v = max_v = 0.0

            report = FeatureReport(
                feature_name=descriptor.feature_name,
                formula_version=descriptor.formula_version,
                source_columns=descriptor.source_columns,
                category=descriptor.category,
                coverage=coverage,
                non_null_count=valid_count,
                total_count=total_rows,
                mean=mean_v,
                std=std_v,
                min_val=min_v,
                max_val=max_v
            )
            reports.append(report)

        return out, reports


# Singleton default registry instance
REGISTRY = FeatureRegistry()


def enrich_features_with_audit(df: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[FeatureReport]]:
    """Convenience wrapper for feature enrichment with auditable metadata."""
    return REGISTRY.apply(df, feature_names)

"""
Segment mapping module.
Aggregates and attaches VTD defects, UZT measurements, elevation profile,
and operational hydraulic regime to the 10-meter segments.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

from .config import SEGMENT_LENGTH_M


def map_vtd_to_segments(vtd_df: pd.DataFrame, passport_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps in-line inspection (VTD) defects to 10m segments.
    Aggregates per (id, date, segment_id):
      - defect_count
      - max_defect_depth_pct
      - min_residual_thickness_mm = nominal * (1 - max_depth / 100)
      - max_defect_length_mm, max_defect_width_mm
      - defect_types
    Segments without defects are kept empty (no default nominal thickness injected).
    """
    pipe_nominals = dict(zip(passport_df['id'].astype(int), passport_df['nominal_thickness_mm'].astype(float)))

    # Filter defects with recorded wall loss depth
    vtd_real = vtd_df[vtd_df['defect_depth_pct'].notnull()].copy()
    vtd_real['segment_id'] = np.floor(vtd_real['distance'] / SEGMENT_LENGTH_M).astype(int)

    vtd_by_segment = vtd_real.groupby(['id', 'date', 'segment_id']).agg(
        defect_count=('defect_depth_pct', 'count'),
        max_defect_depth_pct=('defect_depth_pct', 'max'),
        max_defect_length_mm=('defect_length', 'max'),
        max_defect_width_mm=('defect_width', 'max'),
        defect_types=('defect_type', lambda s: ', '.join(sorted(set(str(x) for x in s.dropna()))))
    ).reset_index()

    vtd_by_segment['nominal_thickness_mm'] = vtd_by_segment['id'].map(pipe_nominals)
    vtd_by_segment['min_residual_thickness_mm'] = (
        vtd_by_segment['nominal_thickness_mm'] * (1.0 - vtd_by_segment['max_defect_depth_pct'] / 100.0)
    ).round(3)

    return vtd_by_segment


def map_uzt_to_segments(uzt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps ultrasonic thickness (UZT) measurements to 10m segments.
    Aggregates per (id, date, segment_id):
      - measurement_count
      - min_residual_thickness_mm
      - mean_residual_thickness_mm
      - angles
    """
    uzt_copy = uzt_df.copy()
    uzt_copy['segment_id'] = np.floor(uzt_copy['distance'] / SEGMENT_LENGTH_M).astype(int)

    uzt_by_segment = uzt_copy.groupby(['id', 'date', 'segment_id']).agg(
        measurement_count=('residual_wall_thickness', 'count'),
        min_residual_thickness_mm=('residual_wall_thickness', 'min'),
        mean_residual_thickness_mm=('residual_wall_thickness', 'mean'),
        angles=('angle', lambda s: ', '.join(sorted(set(str(int(x)) for x in s.dropna()))))
    ).reset_index()
    uzt_by_segment['mean_residual_thickness_mm'] = uzt_by_segment['mean_residual_thickness_mm'].round(3)

    return uzt_by_segment


def map_profile_to_segments(profile_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps elevation profile coordinates to 10m segments.
    Computes mean, min, max Z coordinates, and slope between adjacent segments.
    """
    prof_copy = profile_df.copy()
    prof_copy['segment_id'] = np.floor(prof_copy['total_distance'] / SEGMENT_LENGTH_M).astype(int)

    prof_agg = prof_copy.groupby(['id', 'segment_id']).agg(
        points_count=('z', 'count'),
        z_mean=('z', 'mean'),
        z_min=('z', 'min'),
        z_max=('z', 'max')
    ).reset_index()

    profile_by_segment = pd.merge(grid_df, prof_agg, on=['id', 'segment_id'], how='left')

    # Linear interpolation for rare segments lacking direct GPS waypoints
    profile_by_segment['z_mean'] = profile_by_segment.groupby('id')['z_mean'].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both')
    ).round(3)
    profile_by_segment['z_min'] = profile_by_segment['z_min'].fillna(profile_by_segment['z_mean']).round(3)
    profile_by_segment['z_max'] = profile_by_segment['z_max'].fillna(profile_by_segment['z_mean']).round(3)

    # Slope between adjacent segments: delta(z_mean) / segment_length (m/m)
    profile_by_segment['slope_m_per_m'] = (
        profile_by_segment.groupby('id')['z_mean'].diff() / profile_by_segment['length_m']
    ).fillna(0.0).round(5)

    return profile_by_segment


def map_regime_to_segments(tech_df: pd.DataFrame, passport_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps daily hydraulic simulation regime to 10m segments.
    Calculates pipeline-level operating statistics and applies linear hydrodynamic
    pressure interpolation between p_start and p_end along the distance.
    """
    if tech_df is None or tech_df.empty:
        return pd.DataFrame()

    regime_stats = tech_df.groupby('id').agg(
        records_count=('date', 'count'),
        calc_date_min=('date', 'min'),
        calc_date_max=('date', 'max'),
        p_start_mean=('p_start', 'mean'),
        p_end_mean=('p_end', 'mean'),
        q_liquid_mean=('q_liquid', 'mean'),
        watercut_pct_mean=('watercut_pct', 'mean'),
        v_liquid_mean=('v_liquid_true', 'mean'),
        t_mix_mean=('t_mix', 'mean'),
        pit_corrosion_mean=('pit_corrosion_intensity', 'mean')
    ).reset_index()

    regime_mapped = pd.merge(grid_df, regime_stats, on='id', how='left')

    # Hydrodynamic pressure linear interpolation at the midpoint of each segment
    x_center = regime_mapped['start_m'] + regime_mapped['length_m'] / 2.0
    pipe_lens = dict(zip(passport_df['id'].astype(int), passport_df['length_km'] * 1000.0))
    total_len = regime_mapped['id'].map(pipe_lens)
    rel_pos = (x_center / total_len).clip(0.0, 1.0)

    regime_mapped['p_segment_mean'] = (
        regime_mapped['p_start_mean'] + rel_pos * (regime_mapped['p_end_mean'] - regime_mapped['p_start_mean'])
    ).round(3)

    return regime_mapped


def map_all_to_segments(cleaned_dfs: Dict[str, pd.DataFrame], grid_df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates mapping of VTD, UZT, Profile, and Regime datasets to 10m segments
    and saves output files in CSV and Parquet formats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_df = cleaned_dfs['passport.csv']

    vtd_by_segment = map_vtd_to_segments(cleaned_dfs['vtd.csv'], pass_df)
    uzt_by_segment = map_uzt_to_segments(cleaned_dfs['uzt.csv'])
    profile_by_segment = map_profile_to_segments(cleaned_dfs['profile.csv'], grid_df)

    tech_key = 'tech_mode_for_embedings_with_length_with_temperatures_ids16.csv'
    tech_df = cleaned_dfs.get(tech_key, pd.DataFrame())
    regime_mapped = map_regime_to_segments(tech_df, pass_df, grid_df)

    tables = {
        'vtd_by_segment': vtd_by_segment,
        'uzt_by_segment': uzt_by_segment,
        'profile_by_segment': profile_by_segment,
        'regime_mapped': regime_mapped
    }

    for name, df_tab in tables.items():
        if not df_tab.empty:
            df_tab.to_csv(output_dir / f"{name}.csv", index=False, encoding='utf-8')
            df_tab.to_parquet(output_dir / f"{name}.parquet", index=False)

    return tables

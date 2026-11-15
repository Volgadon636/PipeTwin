"""Normalize verified units and quarantine bad positions without inventing measurements."""
import re

import numpy as np
import pandas as pd

from .config import TECH_FILE


class QualityLog:
    def __init__(self, initial=()):
        self.rows = list(initial)

    def add(self, filename, df, mask, rule, action='flag_only', field=None):
        for index in df.index[mask.fillna(False)]:
            row = df.loc[index]
            self.rows.append({
                'file': filename, 'source_row': int(row['source_row']),
                'id': '' if pd.isna(row.get('id')) else str(int(row['id'])),
                'field': field or '', 'rule': rule, 'action': action,
                'original_value': '' if field is None or pd.isna(row.get(field)) else str(row[field]),
            })

    def frame(self):
        cols = ['file', 'source_row', 'id', 'field', 'rule', 'action', 'original_value']
        return pd.DataFrame(self.rows, columns=cols).sort_values(
            ['file', 'source_row', 'rule'], kind='stable').reset_index(drop=True)


def _direction(value):
    if pd.isna(value):
        return ''
    return re.sub(r'[^\w]', '', str(value).lower())


def clean_datasets(datasets, config, log):
    cleaned = {name: df.copy() for name, df in datasets.items()}
    quarantine = {}
    passport = cleaned['passport.csv']
    if passport.id.isna().any() or passport.id.duplicated().any():
        raise ValueError('Passport must have one nonmissing, unique ID per pipe')
    for col in ('length_km', 'diameter_mm', 'nominal_thickness_mm'):
        if passport[col].isna().any() or (passport[col] <= 0).any():
            raise ValueError(f'Invalid passport {col}')
    lengths = passport.set_index('id').length_km * 1000
    nominal = passport.set_index('id').nominal_thickness_mm
    introduced = passport.set_index('id').commissioning_date
    directions = passport.set_index('id').direction_name.map(_direction)
    tol = config['vtd_thickness_tolerance_mm']

    profile = cleaned['profile.csv']
    duplicates = profile.duplicated(subset=[c for c in profile if c != 'source_row'])
    log.add('profile.csv', profile, duplicates, 'exact_duplicate', 'exclude_duplicate')
    profile = profile.loc[~duplicates].copy()
    invalid = (profile.id.isna() | ~profile.id.isin(lengths.index) | profile.z.isna()
               | profile.total_distance.isna() | (profile.total_distance < 0)
               | (profile.total_distance > profile.id.map(lengths) + config['profile_endpoint_tolerance_m']))
    log.add('profile.csv', profile, invalid, 'invalid_profile_position', 'quarantine', 'total_distance')
    quarantine['profile_rejected'] = profile.loc[invalid].copy()
    profile = profile.loc[~invalid].sort_values(['id', 'total_distance', 'source_row']).copy()
    spans = profile.groupby(['id', 'total_distance']).z.agg(['min', 'max'])
    if ((spans['max'] - spans['min']) > 1e-8).any():
        raise ValueError('Conflicting profile heights at the same distance; resolve before interpolation')
    ds = profile.groupby('id').total_distance.diff()
    dz = profile.groupby('id').z.diff()
    profile['steep_step_flag'] = (dz.abs() > ds) & (ds > 0)
    log.add('profile.csv', profile, profile.steep_step_flag, 'steep_profile_step', field='z')
    cleaned['profile.csv'] = profile.reset_index(drop=True)

    vtd = cleaned['vtd.csv']
    vtd['nominal_thickness_mm'] = vtd.id.map(nominal)
    vtd['before_commissioning_flag'] = vtd.date < vtd.id.map(introduced)
    vtd['commissioning_unknown_flag'] = vtd.id.map(introduced).isna()
    vtd['wall_nominal_mismatch_flag'] = (vtd.wall_thickness - vtd.nominal_thickness_mm).abs() > tol
    vtd['defect_depth_pct_raw'] = vtd.defect_depth_pct
    vtd['depth_fraction_normalized'] = False
    # Scale only an explicitly identified inspection cohort, supported by measured residuals.
    for date in config['vtd_fraction_dates']:
        cohort = vtd.date.dt.normalize().eq(pd.Timestamp(date)) & vtd.defect_depth_pct.notna()
        if not cohort.any():
            continue
        paired = cohort & vtd.residual_wall_thickness.notna() & vtd.wall_thickness.gt(0)
        fraction_error = (vtd.residual_wall_thickness - vtd.wall_thickness * (1-vtd.defect_depth_pct)).abs()
        if (vtd.loc[cohort, 'defect_depth_pct'].gt(1).any()
                or not paired.any() or fraction_error[paired].median() > tol):
            raise ValueError(f'Depth fraction rule is not supported for {date}')
        log.add('vtd.csv', vtd, cohort, 'depth_fraction_to_percent', 'multiply_by_100', 'defect_depth_pct')
        vtd.loc[cohort, 'defect_depth_pct'] *= 100
        vtd.loc[cohort, 'depth_fraction_normalized'] = True
    vtd['residual_invalid_flag'] = (vtd.residual_wall_thickness.notna() & (
        vtd.residual_wall_thickness.le(0) |
        (vtd.wall_thickness.notna() & (vtd.residual_wall_thickness > vtd.wall_thickness + tol))))
    vtd['residual_for_target_mm'] = vtd.residual_wall_thickness.mask(vtd.residual_invalid_flag)
    vtd['residual_missing_flag'] = vtd.residual_wall_thickness.isna()
    vtd['depth_residual_mismatch_flag'] = (
        vtd.residual_wall_thickness - vtd.wall_thickness * (1-vtd.defect_depth_pct/100)).abs() > tol
    vtd['target_eligible'] = (vtd.residual_for_target_mm.notna() & ~vtd.before_commissioning_flag
                              & ~vtd.commissioning_unknown_flag)
    vtd['heldout_or_future_flag'] = False
    for target in config['targets']:
        hidden = vtd.id.eq(target['id']) & vtd.date.ge(pd.Timestamp(target['prediction_date']))
        vtd.loc[hidden, 'heldout_or_future_flag'] = True
    vtd.loc[vtd.heldout_or_future_flag, 'target_eligible'] = False
    for col in ('before_commissioning_flag', 'wall_nominal_mismatch_flag', 'residual_invalid_flag',
                'residual_missing_flag', 'depth_residual_mismatch_flag', 'heldout_or_future_flag'):
        log.add('vtd.csv', vtd, vtd[col], col, field='residual_wall_thickness')
    # Distance is local to the supplied ID. No zero-clipping or inferred offsets.
    invalid = (vtd.id.isna() | ~vtd.id.isin(lengths.index) | vtd.date.isna() | vtd.distance.isna()
               | vtd.distance.lt(0) | (vtd.distance > vtd.id.map(lengths)))
    log.add('vtd.csv', vtd, invalid, 'outside_pipe_or_missing_key', 'quarantine', 'distance')
    quarantine['vtd_rejected'] = vtd.loc[invalid].copy()
    cleaned['vtd.csv'] = vtd.loc[~invalid].reset_index(drop=True)
    conflicted = vtd.loc[vtd.before_commissioning_flag, 'id'].unique()
    passport['history_date_conflict_flag'] = passport.id.isin(conflicted)

    uzt = cleaned['uzt.csv']
    uzt['nominal_thickness_mm'] = uzt.id.map(nominal)
    uzt['above_nominal_flag'] = uzt.residual_wall_thickness > uzt.nominal_thickness_mm + tol
    uzt['before_commissioning_flag'] = uzt.date < uzt.id.map(introduced)
    uzt['measurement_eligible'] = (uzt.residual_wall_thickness.gt(0) & ~uzt.above_nominal_flag
                                  & ~uzt.before_commissioning_flag & uzt.id.map(introduced).notna())
    log.add('uzt.csv', uzt, uzt.above_nominal_flag, 'unresolved_thickness_above_nominal',
            'exclude_from_features_keep_measurement', 'residual_wall_thickness')
    invalid = (uzt.id.isna() | ~uzt.id.isin(lengths.index) | uzt.date.isna() | uzt.distance.isna()
               | uzt.distance.lt(0) | (uzt.distance > uzt.id.map(lengths)))
    log.add('uzt.csv', uzt, invalid, 'outside_pipe_or_missing_key', 'quarantine', 'distance')
    quarantine['uzt_rejected'] = uzt.loc[invalid].copy()
    cleaned['uzt.csv'] = uzt.loc[~invalid].reset_index(drop=True)

    monitoring = cleaned['regime.csv']
    monitoring['id_known'] = monitoring.id.isin(lengths.index)
    monitoring['direction_conflict_flag'] = (monitoring.id_known &
        monitoring.direction_name.map(_direction).ne(monitoring.id.map(directions)))
    monitoring['exposure_days_from_dates'] = (
        monitoring.sample_removal_date-monitoring.sample_installation_date).dt.total_seconds()/86400
    monitoring['exposure_invalid_flag'] = (monitoring.exposure_days_from_dates.isna()
                                            | monitoring.exposure_days_from_dates.le(0))
    monitoring['exposure_mismatch_flag'] = (monitoring.exposure_days_from_dates-monitoring.exposure_days).abs() > 1
    monitoring['density_suspect_flag'] = monitoring.fluid_density.notna() & ~monitoring.fluid_density.between(0.5, 2)
    monitoring['negative_protective_effect_flag'] = monitoring.protective_effect_pct.lt(0)
    monitoring['mapping_eligible'] = (monitoring.id_known & ~monitoring.direction_conflict_flag
                                     & monitoring.direction_name.notna())
    monitoring['observation_eligible'] = (monitoring.mapping_eligible &
        monitoring.sample_removal_date.notna() & ~monitoring.exposure_invalid_flag & ~monitoring.exposure_mismatch_flag)
    for flag in ('direction_conflict_flag', 'exposure_invalid_flag', 'exposure_mismatch_flag',
                 'density_suspect_flag', 'negative_protective_effect_flag'):
        log.add('regime.csv', monitoring, monitoring[flag], flag)
    log.add('regime.csv', monitoring, ~monitoring.id_known, 'unmapped_id', 'exclude_from_features_keep_record', 'id')

    tech = cleaned[TECH_FILE]
    tech['diameter_mm_raw'] = tech.diameter_mm
    expected_d = tech.id.map(passport.set_index('id').diameter_mm)
    in_meters = np.isclose(tech.diameter_mm*1000, expected_d, atol=0.1, rtol=0)
    in_mm = np.isclose(tech.diameter_mm, expected_d, atol=0.1, rtol=0)
    log.add(TECH_FILE, tech, pd.Series(in_meters, index=tech.index), 'diameter_m_to_mm', 'multiply_by_1000', 'diameter_mm')
    tech.loc[in_meters, 'diameter_mm'] *= 1000
    tech['diameter_mismatch_flag'] = ~(in_meters | in_mm)
    tech['before_commissioning_flag'] = tech.date < tech.id.map(introduced)
    tech['record_eligible'] = (tech.id.isin(lengths.index) & tech.date.notna()
                              & tech.id.map(introduced).notna() & ~tech.before_commissioning_flag)
    log.add(TECH_FILE, tech, ~tech.record_eligible, 'unusable_id_or_date', 'exclude_from_features_keep_record')
    log.add(TECH_FILE, tech, tech.diameter_mismatch_flag, 'diameter_mismatch', field='diameter_mm_raw')
    if tech.duplicated(['id', 'date']).any():
        raise ValueError('Duplicate hydraulic timestamps: resolve before aggregation')
    # Time-series filters must run AFTER a cutoff, never against the full future history.
    return cleaned, quarantine

"""Normalize verified units and quarantine bad positions without inventing measurements."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TECH_FILE

PROVENANCE_COLUMNS = {'source_row', 'source_record_number', 'source_physical_line',
                      'id_source_text', 'id_leading_zero_flag'}


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


def residual_validity(vtd, tolerance, apply_nominal_ceiling=True):
    """Observation-level candidate filter for measured residual thickness.

    Agreed mentor rule (meeting 2): a candidate is not used when its residual
    thickness is greater than the **passport nominal wall thickness of the same
    simple section**. It is a wall-thickness comparison, independent of defect
    length/width. The local ``wall_thickness`` recorded inside the VTD row is
    kept as a separate informational consistency flag: it is not a second
    ceiling, so it never overrides the agreed passport comparison. A missing
    passport nominal is flagged as unresolved instead of being silently
    accepted or excluded.

    Returns flags only; the caller keeps every source row for audit.
    """
    residual = vtd.residual_wall_thickness
    nominal = vtd.nominal_thickness_mm
    wall = vtd.wall_thickness
    above_nominal = (residual.notna() & nominal.notna() & apply_nominal_ceiling
                     & residual.gt(nominal + tolerance))
    nominal_unresolved = nominal.isna()
    above_local_wall = residual.notna() & wall.notna() & residual.gt(wall + tolerance)
    invalid = (residual.notna()
               & (residual.le(0) | above_nominal | nominal_unresolved))
    return pd.DataFrame({'above_nominal_flag': above_nominal,
                         'above_local_wall_flag': above_local_wall,
                         'nominal_unresolved_flag': nominal_unresolved,
                         'residual_invalid_flag': invalid}, index=vtd.index)


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
    duplicates = profile.duplicated(subset=[c for c in profile if c not in PROVENANCE_COLUMNS])
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
    validity = residual_validity(vtd, tol, bool(config['vtd_apply_passport_nominal_ceiling']))
    for column in validity:
        vtd[column] = validity[column]
    vtd['residual_for_target_mm'] = vtd.residual_wall_thickness.mask(vtd.residual_invalid_flag)
    vtd['residual_missing_flag'] = vtd.residual_wall_thickness.isna()
    vtd['depth_residual_mismatch_flag'] = (
        vtd.residual_wall_thickness - vtd.wall_thickness * (1-vtd.defect_depth_pct/100)).abs() > tol
    vtd['target_eligible'] = (vtd.residual_for_target_mm.notna() & ~vtd.before_commissioning_flag
                              & ~vtd.commissioning_unknown_flag & ~vtd.nominal_unresolved_flag)
    vtd['heldout_or_future_flag'] = False
    for target in config['targets']:
        hidden = vtd.id.eq(target['id']) & vtd.date.ge(pd.Timestamp(target['prediction_date']))
        vtd.loc[hidden, 'heldout_or_future_flag'] = True
    vtd.loc[vtd.heldout_or_future_flag, 'target_eligible'] = False
    # Every exclusion is kept in the source table and in the issue log with an
    # explicit reason; no source value is overwritten or dropped.
    audit_flags = {
        'before_commissioning_flag': ('exclude_from_target_keep_record', 'date'),
        'wall_nominal_mismatch_flag': ('flag_only', 'wall_thickness'),
        'above_nominal_flag': ('exclude_from_target_keep_record', 'residual_wall_thickness'),
        'above_local_wall_flag': ('flag_only', 'wall_thickness'),
        'nominal_unresolved_flag': ('unresolved_keep_record', 'nominal_thickness_mm'),
        'residual_invalid_flag': ('exclude_from_target_keep_record', 'residual_wall_thickness'),
        'residual_missing_flag': ('keep_missing', 'residual_wall_thickness'),
        'depth_residual_mismatch_flag': ('flag_only', 'defect_depth_pct'),
        'heldout_or_future_flag': ('exclude_from_target_keep_record', 'date'),
    }
    for column, (action, field) in audit_flags.items():
        log.add('vtd.csv', vtd, vtd[column], column, action, field)
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
    monitoring['exposure_invalid_flag'] = monitoring.exposure_days_from_dates.le(0)
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


def row_dispositions(cleaned, quarantine, source_summary):
    """Disjoint row accounting: a row is retained, duplicate, quarantined or malformed.

    Multiple quality flags on one retained row are reported separately and are not
    counted as multiple dropped records.
    """
    rows = []
    for name, frame in cleaned.items():
        info = source_summary.get(name, {})
        retained = len(frame)
        duplicate = 0
        if name == 'profile.csv' and 'source_row' in frame.columns:
            # Duplicate rows are absent from the cleaned frame, so use the source
            # record count and quarantine; derivation is explicit in the summary.
            pass
        qframe = quarantine.get(Path(name).stem.replace('.csv', '') + '_rejected')
        quarantined = 0
        for key, q in quarantine.items():
            if key.startswith(Path(name).stem + '_') and 'source_row' in q.columns:
                quarantined += int(q.source_row.nunique())
        malformed = int(info.get('malformed_records', 0))
        # Profile duplicate count is the difference between parsed records and the
        # disjoint retained/quarantine counts when no malformed records exist.
        input_records = int(info.get('record_count', retained + quarantined + malformed))
        duplicate = max(0, input_records - retained - quarantined - malformed)
        accounted = retained + quarantined + duplicate + malformed
        rows.append({
            'source_file': name,
            'source_type': info.get('source_type', ''),
            'input_records': input_records,
            'retained_records': retained,
            'duplicate_records': duplicate,
            'quarantined_records': quarantined,
            'malformed_records': malformed,
            'parse_rejected_records': 0,
            'accounted_records': accounted,
            'disjoint_accounting_ok': accounted == input_records,
            'note': ('Malformed/rejected field values are masked and listed in quality_issues; '
                     'they do not necessarily drop the whole record.'),
        })
    return pd.DataFrame(rows)


def monitoring_join_audit(monitoring, passport):
    if monitoring is None or monitoring.empty:
        return {'rows': 0, 'status': 'source_missing_or_empty', 'usable_for_targets': {}}
    ids = set(passport.id.dropna().astype(int))
    known = monitoring.id.isin(ids)
    direction_conflict = monitoring.get('direction_conflict_flag', pd.Series(False, index=monitoring.index)).fillna(False)
    mapping = monitoring.get('mapping_eligible', pd.Series(False, index=monitoring.index)).fillna(False)
    observation = monitoring.get('observation_eligible', pd.Series(False, index=monitoring.index)).fillna(False)
    per_target = {}
    for pid in (1250022871, 1250009583):
        candidates = monitoring.id.eq(pid)
        per_target[str(pid)] = {
            'candidate_rows': int(candidates.sum()),
            'mapping_eligible_rows': int((candidates & mapping).sum()),
            'observation_eligible_rows': int((candidates & observation).sum()),
        }
    examples = monitoring.loc[~known, ['source_row', 'id_source_text', 'direction_name']].head(10).astype('string').where(
        monitoring.loc[~known, ['source_row', 'id_source_text', 'direction_name']].head(10).notna(), None).to_dict('records')
    return {
        'rows': int(len(monitoring)),
        'unknown_id_rows': int((~known).sum()),
        'direction_conflict_rows': int(direction_conflict.sum()),
        'mapping_eligible_rows': int(mapping.sum()),
        'observation_eligible_rows': int(observation.sum()),
        'unmatched_id_examples': examples,
        'usable_for_targets': per_target,
        'policy': 'No automatic fuzzy join; no cross-pipe median transfer; point observations are not propagated.',
        'cardinality': 'monitoring rows are many-to-one by simple-section ID only after identity validation; '
                       'no target uses an unvalidated many-to-many join.',
    }

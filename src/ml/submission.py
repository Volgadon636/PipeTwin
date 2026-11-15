"""Submission validation, template inspection and UZT priority export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['id', 'segment_id', 'thickness_pred']
TEMPLATE_COLUMNS = ['id', 'segment_id', 'thickness_pred']


def inspect_template(path):
    """Return template status; filename alone is never sufficient."""
    if path is None:
        return {'available': False, 'status': 'no_template_supplied'}
    path = Path(path)
    if not path.is_file():
        return {'available': False, 'status': 'template_path_missing', 'path': str(path)}
    try:
        header = pd.read_csv(path, nrows=0, dtype='string')
    except Exception as exc:
        return {'available': True, 'status': 'unreadable_template', 'path': str(path),
                'error': str(exc)}
    headers = list(header.columns)
    if headers != TEMPLATE_COLUMNS:
        return {'available': True, 'status': 'not_a_submission_template',
                'path': str(path), 'headers': headers,
                'detail': 'Columns do not match exactly id,segment_id,thickness_pred'}
    frame = pd.read_csv(path, dtype='string', keep_default_na=False, na_values=[''])
    return {'available': True, 'status': 'authoritative_schema_match',
            'path': str(path), 'headers': headers, 'rows': len(frame)}


def _check_keys(frame, expected, label):
    if frame[['id', 'segment_id']].isna().any().any() or frame.duplicated(['id', 'segment_id']).any():
        raise ValueError(f'{label}: missing or duplicate keys')
    for key in ('id', 'segment_id'):
        if not pd.api.types.is_numeric_dtype(frame[key]) or not np.isfinite(frame[key]).all() or (frame[key] % 1 != 0).any():
            raise ValueError(f'{label}: keys must be finite integers')
    actual = pd.MultiIndex.from_frame(frame[['id', 'segment_id']])
    target = pd.MultiIndex.from_frame(expected[['id', 'segment_id']])
    if len(actual) != len(target) or len(target.difference(actual)) or len(actual.difference(target)):
        raise ValueError(f'{label}: key set does not match target grid')


def validate_submission(result, target_grid, bounds, template_path=None):
    if list(result.columns) != COLUMNS:
        raise ValueError('Submission columns must be id,segment_id,thickness_pred in that order')
    for frame, label in ((result, 'result'), (target_grid, 'target_grid'), (bounds, 'bounds')):
        _check_keys(frame, target_grid, label)
    _check_keys(bounds, target_grid, 'bounds')
    if not pd.api.types.is_numeric_dtype(result.thickness_pred):
        raise ValueError('Predictions must be numeric')
    if not np.isfinite(result.thickness_pred.to_numpy(float)).all():
        raise ValueError('Predictions must be finite')
    aligned = result.merge(bounds[['id', 'segment_id', 'h0_mm']], on=['id', 'segment_id'], validate='one_to_one')
    if aligned.h0_mm.le(0).any() or aligned.thickness_pred.lt(0).any() or (aligned.thickness_pred > aligned.h0_mm + 1e-10).any():
        raise ValueError('Prediction outside physical/assumed baseline bounds')
    template = inspect_template(template_path)
    if template.get('status') == 'authoritative_schema_match':
        template_frame = pd.read_csv(template_path, dtype={'id': 'int64', 'segment_id': 'int64'},
                                     usecols=['id', 'segment_id'])
        expected = pd.MultiIndex.from_frame(template_frame[['id', 'segment_id']])
        actual = pd.MultiIndex.from_frame(result[['id', 'segment_id']])
        if len(actual) != len(expected) or len(expected.difference(actual)) or len(actual.difference(expected)):
            raise ValueError('Submission keys do not match the authoritative template')
        template['key_set_compared'] = True
    else:
        template['key_set_compared'] = False
        template['validation_grid'] = 'derived_target_grid'
    return {'rows': len(result), 'unique_keys': True, 'finite_mm': True, 'bounded': True,
            'template': template}


def export_result(result, path, target_grid, bounds, template_path=None):
    """Write the exact three-column submission after validation; no index column."""
    validation = validate_submission(result, target_grid, bounds, template_path)
    frame = result[COLUMNS].sort_values(['id', 'segment_id'], kind='stable').reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8', lineterminator='\n', float_format='%.10g')
    return validation, {'path': str(path), 'rows': len(frame),
                        'columns': COLUMNS, 'header': ','.join(COLUMNS)}


def risk_ranking(diagnostics):
    """Lowest predicted thickness per pipe; expose ties and deterministic tie-breaks."""
    parts = []
    for _, part in diagnostics.groupby('id'):
        part = part.sort_values(['thickness_pred', 'segment_id']).copy()
        count = int(np.ceil(.1 * len(part)))
        part['priority_rank'] = np.arange(1, len(part)+1)
        part['top_10pct'] = part.priority_rank.le(count)
        part['prediction_tie_count'] = part.groupby('thickness_pred').segment_id.transform('size')
        part['ranking_basis'] = 'thickness_ascending_segment_id_tiebreak'
        nominal = (pd.to_numeric(part['nominal_thickness_mm'], errors='coerce')
                   if 'nominal_thickness_mm' in part else pd.Series(np.nan, index=part.index))
        part['relative_thinning'] = np.where(nominal > 0, 1.0 - part.thickness_pred / nominal, np.nan)
        part['priority_percentile'] = 100.0 * part.priority_rank / len(part)
        part['priority_reason'] = np.where(
            part.prediction_tie_count.ge(len(part)),
            'constant prediction: deterministic order only; no spatial discrimination',
            'lowest predicted minimum residual thickness; no critical threshold supplied')
        if 'baseline_missing' in part:
            assumption = part.baseline_missing.fillna(False)
        else:
            assumption = pd.Series(False, index=part.index)
        case_values = part['case'].astype('string') if 'case' in part else pd.Series('', index=part.index)
        source_values = part['baseline_source'].astype('string') if 'baseline_source' in part else pd.Series('', index=part.index)
        evidence = np.where(
            part.prediction_tie_count.ge(len(part)),
            'low: constant prediction; spatial ranking unsupported',
            np.where(case_values.eq('cold_start') | source_values.eq('nominal_at_commissioning'),
                     'low: cold-start nominal initial condition; no target inspection history',
                     np.where(assumption,
                              'medium/low: baseline uses a documented nearest-prior or nominal assumption',
                              'medium: measured prior baseline where available; forecast still uncalibrated')))
        part['evidence_quality'] = evidence
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def uzt_priority_table(diagnostics, forecast_date=None):
    """Separate inspection-priority table; never merged into result.csv."""
    required = {'id', 'segment_id', 'start_m', 'end_m', 'thickness_pred',
                'nominal_thickness_mm', 'relative_thinning', 'priority_rank',
                'evidence_quality', 'priority_reason'}
    missing = required - set(diagnostics.columns)
    if missing:
        raise ValueError(f'Priority diagnostics missing columns: {sorted(missing)}')
    frame = diagnostics.copy()
    frame['forecast_date'] = (forecast_date if forecast_date is not None else
                              frame.get('prediction_date'))
    columns = ['id', 'forecast_date', 'segment_id', 'start_m', 'end_m',
               'thickness_pred', 'nominal_thickness_mm', 'relative_thinning',
               'priority_rank', 'top_10pct', 'evidence_quality', 'priority_reason',
               'prediction_tie_count', 'ranking_basis']
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns].sort_values(['id', 'priority_rank'], kind='stable').reset_index(drop=True)

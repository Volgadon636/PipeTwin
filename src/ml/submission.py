"""Submission validation is key-based and fails closed."""
import numpy as np
import pandas as pd

COLUMNS = ['id', 'segment_id', 'thickness_pred']


def validate_submission(result, target_grid, bounds):
    if list(result.columns) != COLUMNS:
        raise ValueError('Submission columns must be id,segment_id,thickness_pred in that order')
    for frame in (result, target_grid, bounds):
        if frame[['id', 'segment_id']].isna().any().any() or frame.duplicated(['id', 'segment_id']).any():
            raise ValueError('Missing or duplicate keys')
        for key in ('id', 'segment_id'):
            if not pd.api.types.is_numeric_dtype(frame[key]) or not np.isfinite(frame[key]).all() or (frame[key] % 1 != 0).any():
                raise ValueError('Keys must be finite integers')
    expected = pd.MultiIndex.from_frame(target_grid[['id', 'segment_id']])
    actual = pd.MultiIndex.from_frame(result[['id', 'segment_id']])
    bound_keys = pd.MultiIndex.from_frame(bounds[['id', 'segment_id']])
    if len(actual) != len(expected) or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Submission keys do not match the target grid')
    if len(bound_keys) != len(expected) or len(expected.difference(bound_keys)):
        raise ValueError('Bounds do not match the target grid')
    if not pd.api.types.is_numeric_dtype(result.thickness_pred):
        raise ValueError('Predictions must be numeric')
    aligned = result.merge(bounds[['id', 'segment_id', 'h0_mm']], on=['id', 'segment_id'], validate='one_to_one')
    if not np.isfinite(aligned[['thickness_pred', 'h0_mm']].to_numpy(float)).all():
        raise ValueError('Nonfinite prediction or bound')
    if aligned.h0_mm.le(0).any() or aligned.thickness_pred.lt(0).any() or (aligned.thickness_pred > aligned.h0_mm + 1e-10).any():
        raise ValueError('Prediction outside physical/assumed baseline bounds')
    return {'rows': len(result), 'unique_keys': True, 'finite_mm': True, 'bounded': True}


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
        parts.append(part)
    return pd.concat(parts, ignore_index=True)

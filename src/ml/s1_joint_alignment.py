"""Validate a variable S1 coordinate shift using matching joint starts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.grid_builder import assign_segments
from src.storage import sha256, write_json
from .data import RepositoryData
from .incremental import _score
from .next_common import json_safe, write_csv
from .s1_alignment import KEYS, PEERS, S2_SOURCE, predict_section, section_snapshot
from .s1_contract import S1_CUTOFF, S1_ID, S2_ID, nominal_frame, target_key_grid
from .submission import COLUMNS, export_result
from .targeted_s2 import S2_CUTOFF

PRIOR_DATE = pd.Timestamp('2022-07-06')
BUFFER_M = 10
N_BLOCKS = 5
CANDIDATES = ('joint_min', 'joint_p05', 'joint_blend')


def joint_anchors(vtd, before, reference, section_ids):
    """Pair like-numbered pipes within each simple section, using local Distance."""
    fields = ['id', 'date', 'section_number', 'distance', 'relative_distance']
    rows = vtd.loc[vtd.id.isin(section_ids) & vtd.date.isin([before, reference]), fields].copy()
    rows['relative_distance'] = pd.to_numeric(rows.relative_distance, errors='coerce')
    rows['section_number'] = pd.to_numeric(rows.section_number, errors='coerce')
    rows = rows.dropna()
    rows = rows[rows.relative_distance.ge(0)]
    rows['start'] = rows.distance - rows.relative_distance
    starts = rows.groupby(['id', 'section_number', 'date']).start.agg(
        center='median', count='size', low='min', high='max').reset_index()
    starts = starts[(starts['count'] >= 2) & (starts.high - starts.low <= 0.5)]
    a = starts[starts.date.eq(before)]
    b = starts[starts.date.eq(reference)]
    pairs = a.merge(b, on=['id', 'section_number'], suffixes=('_source', '_reference'),
                    validate='one_to_one')
    pairs = pairs.rename(columns={'center_source': 'source_m', 'center_reference': 'reference_m'})
    pairs['shift_m'] = pairs.reference_m - pairs.source_m
    pairs = pairs.loc[pairs.shift_m.abs().le(20),
                      ['id', 'section_number', 'source_m', 'reference_m', 'shift_m',
                       'count_source', 'count_reference']]
    return pairs.sort_values(['id', 'source_m']).reset_index(drop=True)


def map_coordinates(points, anchors, outside='shift'):
    ordered = anchors.sort_values('source_m')
    x = ordered.source_m.to_numpy(float)
    y = ordered.reference_m.to_numpy(float)
    if (len(x) < 2 or not np.isfinite(np.r_[x, y]).all()
            or (np.diff(x) <= 0).any() or (np.diff(y) <= 0).any()):
        raise ValueError('Need at least two unique, ordered anchors in both coordinate frames')
    points = np.asarray(points, float)
    if outside == 'reject' and ((points < x[0]).any() or (points > x[-1]).any()):
        raise ValueError('Verified anchors must bracket the entire target section')
    return points + np.interp(points, x, y - x)


def heldout_anchors(anchors, lo, hi):
    """Exclude the whole holdout and its buffer in both coordinate frames."""
    return anchors.loc[~(anchors.source_m.between(lo, hi, inclusive='left')
                         | anchors.reference_m.between(lo, hi, inclusive='left'))]


def predict_aligned(vtd, grid, fallback, anchors, outside='shift'):
    prior = vtd.loc[vtd.id.eq(grid.id.iloc[0]) & vtd.date.eq(PRIOR_DATE)
                    & vtd.target_eligible & vtd.residual_for_target_mm.notna()].copy()
    prior['aligned_m'] = map_coordinates(prior.distance, anchors, outside=outside)
    inside = prior.aligned_m.between(0, grid.end_m.max())
    moved = assign_segments(prior.loc[inside], 'aligned_m', grid)
    grouped = moved.groupby(KEYS).residual_for_target_mm
    base = fallback.set_index(KEYS)
    minimum = grouped.min().reindex(base.index).fillna(base.thickness_pred)
    quantile = grouped.quantile(0.05).reindex(base.index).fillna(base.thickness_pred)
    # Standardize on full-grid predictions for this fold. No held-out truth is used.
    q_std, minimum_std = quantile.std(ddof=0), minimum.std(ddof=0)
    rescaled_q = (minimum.mean() + (quantile - quantile.mean()) * minimum_std / q_std
                  if q_std > 0 else minimum)
    combined = (minimum + rescaled_q) / 2
    out = base.assign(joint_min=minimum, joint_p05=quantile,
                      joint_blend=combined.clip(0, upper=base.nominal_thickness_mm),
                      aligned_measurement=base.index.isin(grouped.size().index))
    return out.reset_index(), int((~inside).sum())


def evaluate(data, raw, geometry, anchors):
    labels = data.read('processed/vtd_by_segment')
    labels = labels.loc[labels.id.isin(PEERS) & labels.date.eq(S1_CUTOFF) & labels.target_eligible,
                        KEYS + ['min_residual_thickness_mm']].rename(
                            columns={'min_residual_thickness_mm': 'next_mm'})
    parts, folds = [], []
    for pid, truth in labels.groupby('id', sort=True):
        grid = geometry[geometry.id.eq(pid)].sort_values('segment_id')
        snapshot = section_snapshot(data, int(pid))
        raw_baseline, _ = predict_section(snapshot, raw, grid, S1_CUTOFF, 0)
        fixed, _ = predict_section(snapshot, raw, grid, S1_CUTOFF, 8)
        base = raw_baseline.merge(fixed[KEYS + ['thickness_pred']].rename(
            columns={'thickness_pred': 'fixed_8'}), on=KEYS, validate='one_to_one')
        base['unshifted'] = base.thickness_pred
        section_anchors = anchors[anchors.id.eq(pid)]
        for fold, segments in enumerate(np.array_split(grid.segment_id.to_numpy(), N_BLOCKS)):
            block = grid[grid.segment_id.isin(segments)]
            if block.empty:
                continue
            lo, hi = block.start_m.min() - BUFFER_M, block.end_m.max() + BUFFER_M
            train = heldout_anchors(section_anchors, lo, hi)
            entry = {'id': int(pid), 'fold': fold, 'lo_with_buffer_m': lo,
                     'hi_with_buffer_m': hi, 'anchors_used': len(train), 'supported': True}
            try:
                predictions, _ = predict_aligned(raw, grid, base, train)
            except ValueError as error:
                predictions = base.assign(**{name: np.nan for name in CANDIDATES})
                entry.update(supported=False, reason=str(error))
            score_rows = truth[truth.segment_id.isin(segments)].merge(
                predictions[KEYS + ['unshifted', 'fixed_8', *CANDIDATES]],
                on=KEYS, how='left', validate='one_to_one')
            parts.append(score_rows.assign(fold=f'{pid}:{fold}', supported=entry['supported']))
            folds.append(entry)
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != len(labels) or oof.duplicated(KEYS).any():
        raise ValueError('Validation keys changed')
    common = oof.dropna(subset=['unshifted', 'fixed_8', *CANDIDATES])
    rows = []
    for scheme, frame in [('common', common), *[
            (str(pid), f) for pid, f in common.groupby('id', sort=True)]]:
        rows.extend(_score(frame, name, 'S1', scheme, 'joint_alignment', frame.fold.nunique())
                    for name in ('unshifted', 'fixed_8', *CANDIDATES))
    return oof, pd.DataFrame(rows), folds


def s2_inventory(data):
    """Count measured evidence under the unchanged S2 cutoff and target rules."""
    vtd = data.read('processed/vtd_by_segment')
    rows = vtd.loc[vtd.date.lt(S2_CUTOFF) & ~vtd.id.eq(S2_ID)]
    history = rows[rows.target_eligible]
    target = vtd[vtd.id.eq(S2_ID)]
    uzt = data.read('processed/uzt_by_segment')
    return {'cutoff': str(S2_CUTOFF.date()), 'eligible_rows': len(history),
            'eligible_ids': sorted(history.id.unique().tolist()),
            'eligible_parent_count': int(history.id.map(data.parent_groups()).nunique()),
            'eligible_dates': sorted(history.date.dt.strftime('%Y-%m-%d').unique().tolist()),
            'before_commissioning_rows': int(rows.before_commissioning_flag.sum()),
            'target_vtd_rows': len(target),
            'target_eligible_uzt_before_cutoff': int(uzt.loc[
                uzt.id.eq(S2_ID) & uzt.date.lt(S2_CUTOFF), 'eligible_measurement_count'].sum())}


def verified_target_anchors(path):
    anchors = pd.read_csv(path)
    required = {'id', 'source_m', 'reference_m', 'source_date', 'available_date', 'evidence'}
    if not required.issubset(anchors.columns) or not anchors.id.eq(S1_ID).all():
        raise ValueError('Expected verified anchors for S1 target only')
    if (not pd.to_datetime(anchors.source_date).eq(PRIOR_DATE).all()
            or not pd.to_datetime(anchors.available_date).lt(S1_CUTOFF).all()
            or anchors.evidence.fillna('').str.strip().eq('').any()):
        raise ValueError('Anchors must carry provenance and be known strictly before the cutoff')
    return anchors


def export_target(data, root, raw, geometry, output, anchors_path, candidate):
    anchors = verified_target_anchors(anchors_path)
    grid = geometry[geometry.id.eq(S1_ID)]
    map_coordinates([0, grid.end_m.max()], anchors, outside='reject')
    snapshot = section_snapshot(data, S1_ID)
    fallback, _ = predict_section(snapshot, raw, grid, S1_CUTOFF, 0)
    forecasts, dropped = predict_aligned(raw, grid, fallback, anchors, outside='reject')
    s1 = forecasts[KEYS + [candidate]].rename(columns={candidate: 'thickness_pred'})
    s2_source = root / S2_SOURCE
    prior = pd.read_csv(s2_source)
    s2 = prior.loc[prior.id.eq(S2_ID), COLUMNS].reset_index(drop=True)
    result = output / 'result.csv'
    bounds = nominal_frame(data).rename(columns={'nominal_thickness_mm': 'h0_mm'})
    validation, _ = export_result(pd.concat([s1, s2]), result, target_key_grid(data), bounds)
    written = pd.read_csv(result)
    pd.testing.assert_frame_equal(written[written.id.eq(S2_ID)].reset_index(drop=True), s2)
    return {'candidate': candidate, 'anchors_sha256': sha256(anchors_path),
            'outputs_sha256': sha256(result), 's2_source_sha256': sha256(s2_source),
            'out_of_range_observations': dropped, 'validation': validation}


def run(data, root, output, anchors_path=None, candidate='joint_blend'):
    output.mkdir(parents=True, exist_ok=False)
    raw = data.read('data/cleaned/vtd')
    geometry = data.read('processed/segments_grid')
    anchors = joint_anchors(raw, PRIOR_DATE, S1_CUTOFF, PEERS)
    oof, metrics, folds = evaluate(data, raw, geometry, anchors)
    write_csv(anchors, output / 'peer_anchors.csv', float_format='%.10f')
    write_csv(oof, output / 'peer_oof.csv', float_format='%.10f')
    write_csv(metrics, output / 'metrics.csv', float_format='%.10f')
    write_json(output / 's2_data_inventory.json', s2_inventory(data))
    target = {'status': 'awaiting verified pre-cutoff anchors'}
    if anchors_path is not None:
        target = export_target(data, root, raw, geometry, output, anchors_path, candidate)
    write_json(output / 'manifest.json', json_safe({
        'track': 'retrospective; same-date peer geometry only',
        'source_date': str(PRIOR_DATE.date()), 'reference_date': str(S1_CUTOFF.date()),
        'available_label_rows': len(oof), 'scored_common_rows': int(oof.supported.sum()),
        'n_peer_anchors': len(anchors), 'folds': folds, 'target_export': target,
        'code_sha256': sha256(Path(__file__)), 'input_sha256': data.hashes,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--verified-target-anchors', type=Path)
    parser.add_argument('--candidate', choices=CANDIDATES, default='joint_blend')
    args = parser.parse_args()
    run(RepositoryData(args.data_root), args.root, args.output_root,
        args.verified_target_anchors, args.candidate)


if __name__ == '__main__':
    main()

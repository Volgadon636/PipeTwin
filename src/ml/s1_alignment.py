"""Evaluate and export an opt-in odometer shift for S1 persistence."""
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
from .incremental_features import attach_defect_stats, prior_inspection_stats
from .next_common import json_safe, load_artifact, predict_ridge, write_csv
from .s1_contract import S1_CASE, S1_CUTOFF, S1_ID, enrich_s1, nominal_frame, target_key_grid
from .submission import COLUMNS, export_result
from .windows import make_snapshot

KEYS = ['id', 'segment_id']
PEERS = (1250009490, 1250011604)
OFFSETS = tuple(range(11))
PEER_HYPOTHESIS_M = 8
S2_SOURCE = Path('experiments/incremental_cycle/s1_challenger/result.csv')
RIDGE_SOURCE = Path('experiments/incremental_cycle/models/defect_stats_support/model_artifact.joblib')


def shifted_minimum(vtd, grid, cutoff, offset_m):
    """Translate the latest eligible campaign before assigning its 10 m keys."""
    if grid.id.nunique() != 1:
        raise ValueError('Alignment requires the complete grid of one section')
    if not np.isfinite(offset_m):
        raise ValueError('Offset must be finite')
    keep = (vtd.id.eq(grid.id.iloc[0]) & vtd.date.lt(cutoff) & vtd.target_eligible
            & vtd.residual_for_target_mm.notna())
    prior = vtd.loc[keep]
    source_date = prior.date.max()
    moved = prior.loc[prior.date.eq(source_date)].assign(
        aligned_distance=lambda frame: frame.distance + offset_m)
    inside = moved.aligned_distance.between(0, grid.end_m.max())
    assigned = assign_segments(moved.loc[inside], 'aligned_distance', grid)
    measured = assigned.groupby(KEYS).residual_for_target_mm.min()
    metadata = {'source_date': None if pd.isna(source_date) else str(source_date.date()),
                'observations_outside_grid': int((~inside).sum()),
                'direct_segments': len(measured)}
    return measured, metadata


def predict_section(snapshot, vtd, grid, cutoff, offset_m):
    key = snapshot.set_index(KEYS)
    shifted, metadata = shifted_minimum(vtd, grid, cutoff, offset_m)
    output = key[['h0_mm', 'nominal_thickness_mm', 'last_vtd_thickness_mm']].copy()
    output['shifted_measured_mm'] = shifted
    output['fallback'] = output.shifted_measured_mm.isna()
    output['thickness_pred'] = output.shifted_measured_mm.fillna(output.h0_mm).clip(
        lower=0, upper=output.nominal_thickness_mm)
    metadata['fallback_segments'] = int(output.fallback.sum())
    return output.reset_index(), metadata


def build_peer_table(data, vtd, grids, ridge):
    dates = pd.to_datetime(ridge['metadata']['training_dates'])
    if not len(dates) or not (dates < S1_CUTOFF).all():
        raise ValueError('Ridge reference must be trained before the S1 cutoff')
    known = data.read('processed/vtd_by_segment')
    profile = data.read('processed/profile_by_segment')
    measured = known.loc[
        known.id.isin(PEERS) & known.date.eq(S1_CUTOFF) & known.target_eligible,
        KEYS + ['min_residual_thickness_mm']
    ].rename(columns={'min_residual_thickness_mm': 'next_mm'})
    parts = []
    for pid, truth in measured.groupby('id', sort=True):
        grid = grids[grids.id.eq(pid)]
        snapshot = section_snapshot(data, int(pid))
        base, _ = predict_section(snapshot, vtd, grid, S1_CUTOFF, 0)
        frame = truth.merge(base[KEYS + ['thickness_pred', 'shifted_measured_mm']],
                            on=KEYS, validate='one_to_one').rename(columns={
                                'thickness_pred': 'shift_0',
                                'shifted_measured_mm': 'direct_0'})
        for offset in OFFSETS[1:]:
            shifted, _ = shifted_minimum(vtd, grid, S1_CUTOFF, offset)
            frame = frame.join(shifted.rename(f'direct_{offset}'), on=KEYS, validate='one_to_one')
            frame[f'shift_{offset}'] = frame[f'direct_{offset}'].fillna(frame.shift_0)
        features = enrich_s1(snapshot, known, profile, S1_CUTOFF)
        features['last_mm'] = features.s1a_prediction
        features = attach_defect_stats(features, prior_inspection_stats(vtd, grid, features))
        own = features.has_own_prior_vtd
        values = features.s1b_prediction.copy()
        values.loc[own] = predict_ridge(ridge, features.loc[own])
        frame = frame.join(features.assign(ridge=values).set_index(KEYS).ridge,
                           on=KEYS, validate='one_to_one')
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def section_snapshot(data, pid):
    request = {'id': pid, 'case': S1_CASE, 'prediction_date': str(S1_CUTOFF.date())}
    snapshot, _ = make_snapshot(data, request)
    return snapshot


def cross_section_offsets(peer):
    """Select on one labelled section and apply the offset to the other."""
    sections = set(peer.id)
    if sections != set(PEERS):
        raise ValueError('Expected both historical S1 reference sections')
    parts, choices = [], []
    for held, test in peer.groupby('id', sort=True):
        train = peer[~peer.id.eq(held)]
        candidates = []
        for offset in OFFSETS:
            r = train.next_mm.corr(train[f'shift_{offset}'])
            if np.isfinite(r) and r > 0:
                candidates.append((r * r, -offset))
        offset = -max(candidates, default=(0, 0))[1]
        choices.append({'held_section': int(held), 'train_section': int(train.id.iloc[0]),
                        'calibration_rows': len(train), 'test_rows': len(test),
                        'selected_offset_m': offset})
        parts.append(test.assign(cross_section=test[f'shift_{offset}']))
    return pd.concat(parts, ignore_index=True), choices


def run(data, root, output):
    output.mkdir(parents=True, exist_ok=False)
    vtd = data.read('data/cleaned/vtd')
    grids = data.read('processed/segments_grid')
    ridge_path = root / RIDGE_SOURCE
    peer, choices = cross_section_offsets(build_peer_table(data, vtd, grids, load_artifact(ridge_path)))
    columns = ['shift_0', f'shift_{PEER_HYPOTHESIS_M}', 'cross_section', 'ridge']
    scores = [_score(peer, col, 'S1', 'peer_pooled', 'alignment', len(PEERS))
              for col in columns]
    for pid, part in peer.groupby('id', sort=True):
        scores.extend(_score(part, col, 'S1', f'peer_{pid}', 'alignment', 1)
                      for col in columns)
    direct = peer.dropna(subset=['direct_0', f'direct_{PEER_HYPOTHESIS_M}'])
    scores.extend(_score(direct, col, 'S1', 'peer_direct_common', 'alignment', len(PEERS))
                  for col in ('direct_0', f'direct_{PEER_HYPOTHESIS_M}'))
    write_csv(peer, output / 'peer_predictions.csv', float_format='%.10f')
    write_csv(pd.DataFrame(scores), output / 'metrics.csv', float_format='%.10f')

    prediction, metadata = predict_section(
        section_snapshot(data, S1_ID), vtd, grids[grids.id.eq(S1_ID)], S1_CUTOFF, PEER_HYPOTHESIS_M)
    write_csv(prediction, output / 'target_diagnostics.csv', float_format='%.10f')
    baseline = root / S2_SOURCE
    previous = pd.read_csv(baseline)
    s2 = previous.loc[~previous.id.eq(S1_ID), COLUMNS]
    result = pd.concat([prediction[COLUMNS], s2], ignore_index=True)
    path = output / 'result.csv'
    bounds = nominal_frame(data).rename(columns={'nominal_thickness_mm': 'h0_mm'})
    validation, _ = export_result(result, path, target_key_grid(data), bounds)
    written_s2 = pd.read_csv(path).loc[lambda frame: ~frame.id.eq(S1_ID), COLUMNS]
    pd.testing.assert_frame_equal(written_s2.reset_index(drop=True), s2.reset_index(drop=True))
    write_json(output / 'manifest.json', json_safe({
        'target_id': S1_ID, 'target_date': str(S1_CUTOFF.date()),
        'offset_m': PEER_HYPOTHESIS_M, **metadata,
        'offset_provenance': 'Peer README hypothesis previously assessed on same-date reference labels',
        'strict_asof_validated': False, 'status': 'candidate for external comparison',
        'cross_section_calibration': choices,
        'peer_labels_date': str(S1_CUTOFF.date()),
        'result_sha256': sha256(path), 's2_source_sha256': sha256(baseline),
        'ridge_reference_sha256': sha256(ridge_path), 'code_sha256': sha256(Path(__file__)),
        'input_sha256': data.hashes, 'submission_validation': validation,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(RepositoryData(args.data_root), args.root, args.output_root)


if __name__ == '__main__':
    main()

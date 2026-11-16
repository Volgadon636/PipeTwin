"""Check the production H18/H31 code against frozen development schemes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.experiments.r2_pool_stage1_strong import mapped_labels, split_frames
from src.experiments.r2_pool_stage2 import read_baseline
from src.experiments.r2_pool_stage4 import KEY_DTYPES, assert_replay, keys_match
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.next_common import json_safe, write_csv
from src.ml.production import DEFAULT_CONFIG, DEFAULT_DATA, load_config, new_output
from src.ml.production_models import (
    FORECAST_KEYS as KEY, cold_training, fit_s1_models, fit_s2_models,
    predict_s1_models, predict_s2_models, s1_prediction_frame, s1_training,
)
from src.ml.s1_alignment import section_snapshot
from src.ml.s1_contract import S1_CASE, S1_CUTOFF, S1_ID, S2_ID, enrich_s1
from src.ml.s1_joint_alignment import PEERS, PRIOR_DATE, heldout_anchors, joint_anchors
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s2 import S2_CUTOFF
from src.ml.windows import make_snapshot
from src.storage import sha256, write_json

POOL = Path('experiments/r2_pool')
ARCHIVE = POOL / 'revisions/r2-pool-v3/baseline'
INCREMENTAL = Path('experiments/incremental_cycle')
PEER, RETRO, LOSO = 'S1_peer_2023', 'S2_retro_2023', 'S2_LOSO_pooled'
BLOCKS = ('S1_100m_20m', 'S1_500m_100m')
KP = ['id', 'segment_id']


def check_replay(base, frame, column, tolerance=1e-8, *, rounded_legacy_labels=False):
    current = frame[KEY + ['next_mm', column]].rename(columns={column: 'prediction'})
    if rounded_legacy_labels:
        keys_match(base, current, KEY)
        labels = base[KEY + ['next_mm']].merge(current[KEY + ['next_mm']], on=KEY, validate='one_to_one')
        # The old E0 files write labels to ten decimal places. Score only the
        # reconstructed, full-precision labels after checking that rounding.
        np.testing.assert_allclose(labels.next_mm_x, labels.next_mm_y, atol=5.01e-11, rtol=0.)
        base = base.drop(columns='next_mm').merge(current[KEY + ['next_mm']], on=KEY, validate='one_to_one')
    return assert_replay(base, current,
                         tolerance)


def s1_checks(data, specs):
    raw = data.read('data/cleaned/vtd')
    grids = data.read('processed/segments_grid')
    mapped = data.read('processed/vtd_by_segment')
    profile = data.read('processed/profile_by_segment')
    history = s1_training(data, raw, grids)
    path = ARCHIVE / PEER
    baseline = read_baseline(path, specs[PEER])
    membership = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
    truth = mapped_labels(mapped)
    truth = truth.loc[truth.id.isin(PEERS) & truth.prediction_date.eq(S1_CUTOFF)]
    keys_match(membership, truth, KEY)
    anchors = joint_anchors(raw, PRIOR_DATE, S1_CUTOFF, PEERS)
    folds = json.loads((path / 'anchor_folds.json').read_bytes())
    parts = []
    for pid in PEERS:
        train = history.loc[history.id.ne(pid)]
        models = fit_s1_models(train)
        grid = grids.loc[grids.id.eq(pid)].sort_values('segment_id')
        original = enrich_s1(section_snapshot(data, pid), mapped, profile, S1_CUTOFF)
        frame, _ = s1_prediction_frame(original.assign(last_mm=original.s1a_prediction), raw, grid)
        unshifted = predict_s1_models(models, frame).rename(columns={
            'H18': 'H18_distance', 'baseline': 'E0', 'joint_blend': 'joint_blend_distance'})
        unshifted = unshifted[KEY + ['H18_distance', 'E0', 'joint_blend_distance']]
        for entry in folds:
            if entry['id'] != pid or not entry['supported']:
                continue
            fold = f"{pid}:{entry['fold']}"
            refs = heldout_anchors(anchors.loc[anchors.id.eq(pid)], entry['lo_with_buffer_m'],
                                   entry['hi_with_buffer_m'])
            if len(refs) != entry['anchors_used']:
                raise ValueError(f'Changed retrospective alignment membership: {fold}')
            frame, _ = s1_prediction_frame(original.assign(last_mm=original.s1a_prediction),
                                           raw, grid, refs, anchor_extrapolation='shift')
            aligned = predict_s1_models(models, frame)[KEY + ['joint_blend', 'H18']].rename(columns={
                'H18': 'H18_aligned', 'joint_blend': 'joint_blend_aligned'})
            supported = membership.loc[membership.fold.eq(fold) & membership.supported, KEY + ['fold']]
            part = (supported.merge(truth, on=KEY, validate='one_to_one')
                    .merge(unshifted, on=KEY, validate='one_to_one')
                    .merge(aligned, on=KEY, validate='one_to_one'))
            parts.append(part)
    peer = pd.concat(parts, ignore_index=True).astype(KEY_DTYPES)
    keys_match(membership.loc[membership.supported], peer, KEY + ['fold'])
    comparison = POOL / 'batches/batch_04/hypotheses/H18/S1_peer_2023/geometry_type_distribution/validation_predictions.csv'
    h18 = pd.read_csv(comparison, parse_dates=['prediction_date']).astype(KEY_DTYPES)
    checks = {
        'joint_blend_aligned_max_abs_mm': check_replay(baseline, peer, 'joint_blend_aligned'),
        'H18_aligned_max_abs_mm': check_replay(h18.rename(columns={'candidate': 'prediction'}), peer, 'H18_aligned'),
    }
    yield PEER, peer, {'E0': 'H18_distance', 'joint_blend_distance': 'H18_distance',
                       'joint_blend_aligned': 'H18_aligned'}, checks

    grid = grids.loc[grids.id.eq(S1_ID)].sort_values('segment_id')
    cutoff = history.prediction_date.max()
    snapshot, _ = make_snapshot(data, {'id': S1_ID, 'prediction_date': str(cutoff.date()), 'case': S1_CASE})
    original = enrich_s1(snapshot, mapped, profile, cutoff)
    frame, _ = s1_prediction_frame(original.assign(last_mm=original.s1a_prediction), raw, grid)
    for scheme in BLOCKS:
        archived = pd.read_csv(INCREMENTAL / f'{scheme.lower()}_oof.csv',
                               parse_dates=['prediction_date']).astype(KEY_DTYPES).rename(columns={
                                   'global_ridge_geometry': 'prediction'})
        membership = pd.read_csv(ARCHIVE / scheme / 'split_keys.csv', parse_dates=['prediction_date'])
        all_keys, parts = [], []
        for _, block, train, held in split_frames(history, specs[scheme]):
            output_frame = predict_s1_models(fit_s1_models(train), frame).rename(columns={
                'H18': 'H18_distance', 'baseline': 'E0', 'joint_blend': 'joint_blend_distance'})
            part = held[KEY + ['next_mm']].merge(
                output_frame[KEY + ['E0', 'joint_blend_distance', 'H18_distance']],
                on=KEY, validate='one_to_one')
            parts.append(part.assign(block=block))
            all_keys.extend([train[KEY].assign(block=block, role='train'),
                             held[KEY].assign(block=block, role='test')])
        actual = pd.concat(parts, ignore_index=True).astype(KEY_DTYPES)
        keys_match(membership, pd.concat(all_keys, ignore_index=True), KEY + ['block', 'role'])
        keys_match(archived, actual, KEY + ['block'])
        checks = {'E0_max_abs_mm': check_replay(archived, actual, 'E0', rounded_legacy_labels=True)}
        yield scheme, actual, {'E0': 'H18_distance', 'joint_blend_distance': 'H18_distance'}, checks


def s2_checks(data, specs, config):
    all_rows, grids, intervals = cold_training(data, assessment=True)
    train = all_rows.loc[all_rows.prediction_date.lt(S2_CUTOFF)]
    post = all_rows.loc[all_rows.prediction_date.gt(S2_CUTOFF)]
    ranks = VARIANTS['basin_ranks']
    ranked = {key: terrain_features(frame) for key, frame in grids.items()}
    lookup = pd.concat([frame[KEY + ranks] for frame in ranked.values()])
    training_ranked = train.merge(lookup, on=KEY, how='left', validate='one_to_one')
    for scheme in (RETRO, LOSO):
        path = ARCHIVE / scheme
        baseline = read_baseline(path, specs[scheme])
        membership = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        folds = ([(None, train, post)] if scheme == RETRO else
                 [(int(pid), train.loc[train.id.ne(pid)], held) for pid, held in train.groupby('id')])
        all_keys, parts = [], []
        for fold, fit, held in folds:
            rows = pd.concat([fit[KEY].assign(role='train'), held[KEY].assign(role='test')])
            all_keys.append(rows if fold is None else rows.assign(fold=fold))
            model = fit_s2_models(fit, intervals, config['h31'])
            if scheme == LOSO:
                model_ranked = fit_s2_models(training_ranked.loc[training_ranked.id.ne(fold)],
                                             intervals, config['h31'], ranks)
            for key, labels in held.groupby(['id', 'prediction_date']):
                pred = predict_s2_models(model, grids[key], intervals).rename(columns={
                    'baseline': 'E0', 'H31': 'H31_spatial_features'})
                result = labels[KEY + ['next_mm']].merge(
                    pred[KEY + ['E0', 'H31_spatial_features']], on=KEY, validate='one_to_one')
                if scheme == LOSO:
                    audit = predict_s2_models(model_ranked, ranked[key], intervals).rename(columns={
                        'baseline': 'basin_ranks', 'H31': 'H31_basin_ranks'})
                    result = result.merge(audit[KEY + ['basin_ranks', 'H31_basin_ranks']],
                                          on=KEY, validate='one_to_one')
                parts.append(result)
        keys_match(membership, pd.concat(all_keys, ignore_index=True),
                   KEY + ['role'] + (['fold'] if scheme == LOSO else []))
        actual = pd.concat(parts, ignore_index=True).astype(KEY_DTYPES)
        key = 'E0' if scheme == RETRO else 'basin_ranks'
        candidate = 'H31_spatial_features' if scheme == RETRO else 'H31_basin_ranks'
        recorded = POOL / f'batches/batch_07/hypotheses/H31/{scheme}/spatial_gaussian_process/validation_predictions.csv'
        saved = pd.read_csv(recorded, parse_dates=['prediction_date']).astype(KEY_DTYPES)
        checks = {f'{key}_max_abs_mm': check_replay(baseline, actual, key),
                  f'{candidate}_max_abs_mm': check_replay(saved.rename(columns={'candidate': 'prediction'}),
                                                         actual, candidate)}
        if scheme == LOSO:
            previous = pd.read_csv(INCREMENTAL / 's2_section_oof.csv',
                                   parse_dates=['prediction_date']).astype(KEY_DTYPES).rename(columns={
                                       'spatial_no_temperature_centered': 'prediction'})
            checks['E0_max_abs_mm'] = check_replay(previous, actual, 'E0', rounded_legacy_labels=True)
        methods = {'E0': 'H31_spatial_features'}
        if scheme == LOSO:
            methods['basin_ranks'] = 'H31_basin_ranks'
        yield scheme, actual, methods, checks


def report_metrics(scheme, frame, comparisons):
    grouped = [('pooled', frame), *[(str(pid), part) for pid, part in frame.groupby('id')]]
    if 'block' in frame:
        grouped.extend((f'block_{bid}', part) for bid, part in frame.groupby('block'))
    rows = []
    for unit, part in grouped:
        for baseline, method in comparisons.items():
            data = {name: metrics(part.next_mm, part[metric]) for name, metric in
                    [('baseline', baseline), ('candidate', method)]}
            result = {'scheme': scheme, 'unit': unit, 'n': len(part),
                      'baseline': baseline, 'candidate': method}
            for name, value in data.items():
                r = value['pearson_r'] if len(part) >= 3 else None
                result.update({f'{name}_r': r, f'{name}_r2': None if r is None else r * r,
                               **{f'{name}_{metric}': value[metric] for metric in ('mae_mm', 'rmse_mm', 'bias_mm')}})
            first, second = result['baseline_r2'], result['candidate_r2']
            result['delta_r_squared'] = None if first is None or second is None else second - first
            rows.append(result)
    return rows


def verify_exports(result_path):
    folder = result_path.parent
    combined = pd.read_csv(result_path)
    exports = {name: pd.read_csv(folder / 'ablation' / f'{name}.csv')
               for name in ('baseline', 'h18_only', 'h31_only', 'h18_h31')}
    if sha256(folder / 'previous_result.csv') != sha256(folder / 'ablation/baseline.csv'):
        raise ValueError('The previous E0 result was not reproduced byte for byte')
    pd.testing.assert_frame_equal(combined, exports['h18_h31'])
    for only, held_id in [('h18_only', S2_ID), ('h31_only', S1_ID)]:
        preserve = combined.id.eq(held_id)
        pd.testing.assert_frame_equal(exports[only].loc[preserve], exports['baseline'].loc[preserve])
        pd.testing.assert_frame_equal(exports[only].loc[~preserve], combined.loc[~preserve])
    return {'pipeline_ablation_cases_unchanged': True, 'previous_result_reproduced_bytewise': True,
            'result_sha256': sha256(result_path), 'rows': len(combined)}


def run(data, config, result, output):
    specs = json.loads((POOL / 'protocol_photo_v3.json').read_bytes())['schemes']
    with new_output(output) as staging:
        checks, rows = {}, []
        for scheme, frame, pairs, audit in [*s1_checks(data, specs), *s2_checks(data, specs, config)]:
            write_csv(frame, staging / scheme / 'validation_predictions.csv', float_format='%.15g')
            rows.extend(report_metrics(scheme, frame, pairs))
            checks[scheme] = audit
        scores = pd.DataFrame(rows)
        write_csv(scores, staging / 'metrics.csv', float_format='%.15g')
        write_json(staging / 'validation.json', json_safe({
            'replays': checks, 'export_checks': verify_exports(result),
            'qualification': 'exploratory preserved development splits; no independent final labels',
            'pooled_metrics': scores.loc[scores.unit.eq('pooled')].to_dict('records'),
        }))
    print(scores.loc[scores.unit.eq('pooled'), ['scheme', 'baseline', 'candidate', 'n',
                                               'baseline_r2', 'candidate_r2', 'delta_r_squared']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(RepositoryData(args.data_root), load_config(args.config), args.result, args.output_root)

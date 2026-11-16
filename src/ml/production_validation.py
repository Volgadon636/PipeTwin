"""Replay the selected production models on the frozen research splits.

This is development validation already used for method selection, not a new
holdout. Research helpers supply historical geometry only; fitting and inference
use the production implementations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR
from src.experiments.r2_pool_stage3 import outer_train, peer_grids
from src.experiments.r2_pool_stage4 import keys_match
from src.ml import s1_joint_alignment as joint
from src.storage import sha256, write_json
from .data import RepositoryData
from .evaluation import metrics
from .next_common import json_safe, write_csv
from .production import DEFAULT_CONFIG, DEFAULT_DATA, _provenance, load_config, new_output
from .production_models import (
    FORECAST_KEYS, cold_training, fit_s1_models, fit_s2_models,
    predict_s1_models, predict_s2_models, s1_prediction_frame, s1_training,
)
from .s1_contract import S1_CUTOFF
from .s2_terrain import VARIANTS, terrain_features
from .targeted_s2 import SPATIAL_FEATURES, S2_CUTOFF

POOL = BASE_DIR / 'experiments/r2_pool'
ARCHIVE = POOL / 'revisions/r2-pool-v3/baseline'
REFERENCES = {
    'S1_peer_2023': POOL / 'batches/batch_04/hypotheses/H18/S1_peer_2023/geometry_type_distribution/validation_predictions.csv',
    'S2_retro_2023': POOL / 'batches/batch_07/hypotheses/H31/S2_retro_2023/spatial_gaussian_process/validation_predictions.csv',
    'S2_LOSO_pooled': POOL / 'batches/batch_07/hypotheses/H31/S2_LOSO_pooled/spatial_gaussian_process/validation_predictions.csv',
}


def s1_replay(data):
    raw = data.read('data/cleaned/vtd')
    geometry = data.read('processed/segments_grid')
    history = s1_training(data, raw, geometry)
    path = ARCHIVE / 'S1_peer_2023'
    specs = {f"{row['id']}:{row['fold']}": row
             for row in json.loads((path / 'anchor_folds.json').read_bytes())}
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    fitted, parts, splits = {}, [], []
    for pid, full, held in peer_grids(data, raw, geometry, path):
        full = full.assign(last_mm=full.s1a_prediction)
        if pid not in fitted:
            train = outer_train(history, pid, S1_CUTOFF)
            fitted[pid] = fit_s1_models(train)
            splits.append(train[FORECAST_KEYS].assign(held_id=pid, role='train'))
        spec = specs[full.fold.iloc[0]]
        retained = joint.heldout_anchors(anchors.loc[anchors.id.eq(pid)],
                                        spec['lo_with_buffer_m'], spec['hi_with_buffer_m'])
        # Historical research used the endpoint shift outside the retained
        # anchors. Production target inference requires verified full coverage.
        frame, _ = s1_prediction_frame(
            full.drop(columns='joint_blend'), raw, geometry.loc[geometry.id.eq(pid)],
            retained, anchor_extrapolation='shift')
        np.testing.assert_allclose(frame.joint_blend, full.joint_blend, atol=1e-9, rtol=0.)
        predicted = predict_s1_models(fitted[pid], frame)
        parts.append(predicted.loc[held.to_numpy()].rename(
            columns={'baseline': 'current_baseline', 'joint_blend': 'baseline', 'H18': 'candidate'}))
        splits.append(predicted.loc[held.to_numpy(), FORECAST_KEYS].assign(held_id=pid, role='test'))
    return pd.concat(parts, ignore_index=True), pd.concat(splits, ignore_index=True)


def s2_replay(data, config):
    measured, grids, intervals = cold_training(data, assessment=True)
    strict = measured.loc[measured.prediction_date.lt(S2_CUTOFF)].copy()
    post = measured.loc[measured.prediction_date.ge(S2_CUTOFF)].copy()
    grids = {key: terrain_features(full) for key, full in grids.items()}
    ranks = VARIANTS['basin_ranks']
    lookup = pd.concat([full[FORECAST_KEYS + ranks] for full in grids.values()], ignore_index=True)
    strict = strict.merge(lookup, on=FORECAST_KEYS, how='left', validate='one_to_one')
    for scheme, features in [('S2_retro_2023', SPATIAL_FEATURES), ('S2_LOSO_pooled', ranks)]:
        folds = ([(None, strict, post)] if scheme == 'S2_retro_2023' else
                 [(int(pid), strict.loc[strict.id.ne(pid)], held)
                  for pid, held in strict.groupby('id', sort=True)])
        parts, memberships = [], []
        for fold, train, held in folds:
            membership = pd.concat([train[FORECAST_KEYS].assign(role='train'),
                                    held[FORECAST_KEYS].assign(role='test')])
            memberships.append(membership if fold is None else membership.assign(fold=fold))
            models = fit_s2_models(train, intervals, config['h31'], features)
            for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
                predicted = predict_s2_models(models, grids[key], intervals)
                parts.append(labels[FORECAST_KEYS + ['next_mm']].merge(
                    predicted.rename(columns={'H31': 'candidate'}),
                    on=FORECAST_KEYS, validate='one_to_one'))
        membership = pd.concat(memberships, ignore_index=True)
        frozen = pd.read_csv(ARCHIVE / scheme / 'split_keys.csv', parse_dates=['prediction_date'])
        keys_match(frozen, membership, FORECAST_KEYS + ['role'] +
                   (['fold'] if scheme == 'S2_LOSO_pooled' else []))
        yield scheme, pd.concat(parts, ignore_index=True), membership


def compare_reference(scheme, predicted, tolerance=1e-8):
    reference = pd.read_csv(REFERENCES[scheme], parse_dates=['prediction_date'])
    keys_match(reference, predicted, FORECAST_KEYS)
    left = reference.set_index(FORECAST_KEYS).sort_index()
    right = predicted.set_index(FORECAST_KEYS).sort_index()
    if 'next_mm' in right:
        np.testing.assert_array_equal(left.next_mm, right.next_mm)
    for column in ['baseline', 'candidate']:
        np.testing.assert_allclose(left[column], right[column], atol=tolerance, rtol=0.)
    errors = {column: float((left[column] - right[column]).abs().max())
              for column in ['baseline', 'candidate']}
    joined = predicted.drop(columns='next_mm', errors='ignore').merge(
        reference[FORECAST_KEYS + ['next_mm']], on=FORECAST_KEYS, validate='one_to_one')
    return joined, errors


def score_rows(scheme, predicted):
    case = 'S1' if scheme.startswith('S1') else 'S2'
    candidate = 'H18' if case == 'S1' else 'H31'
    # A combination of disjoint case branches has the same values as the
    # selected single method. Its ablation is the unchanged baseline branch.
    methods = {'baseline': 'baseline', candidate.lower() + '_only': 'candidate',
               'h18_h31': 'candidate', 'without_' + candidate.lower(): 'baseline'}
    if 'current_baseline' in predicted:
        methods['previous_pipeline'] = 'current_baseline'
    groups = [('pooled_replay_only', None, None, predicted),
              *[('profile', int(pid), str(date.date()), part)
                for (pid, date), part in predicted.groupby(['id', 'prediction_date'], sort=True)]]
    for scope, pid, date, frame in groups:
        baseline = metrics(frame.next_mm, frame.baseline)
        if len(frame) < 3:
            baseline['pearson_squared'] = None
        for method, column in methods.items():
            score = metrics(frame.next_mm, frame[column])
            if len(frame) < 3:
                score.update(pearson_r=None, pearson_squared=None,
                             pearson_status='undefined_too_few_rows', pearson_reason='fewer_than_three_rows')
            base_r2, r2 = baseline['pearson_squared'], score['pearson_squared']
            yield {'scheme': scheme, 'case': case, 'scope': scope, 'id': pid,
                   'prediction_date': date, 'method': method, **score,
                   'prediction_std_mm': float(frame[column].std(ddof=0)),
                   'baseline_pearson_squared': base_r2,
                   'delta_pearson_squared': r2 - base_r2 if r2 is not None and base_r2 is not None else None,
                   'evidence': 'exploratory_reused_development_validation'}


def run(data, config, output):
    with new_output(output) as staging:
        first, split = s1_replay(data)
        runs = [('S1_peer_2023', first, split), *s2_replay(data, config)]
        rows, replays = [], {}
        for scheme, predicted, membership in runs:
            predicted, replays[scheme] = compare_reference(scheme, predicted)
            write_csv(predicted, staging / scheme / 'validation_predictions.csv', float_format='%.15g')
            write_csv(membership, staging / scheme / 'split_keys.csv')
            rows.extend(score_rows(scheme, predicted))
        write_csv(pd.DataFrame(rows), staging / 'metrics.csv', float_format='%.15g')
        write_json(staging / 'manifest.json', json_safe({
            'status': 'exploratory_reused_development_validation', 'config': config,
            'independent_final_holdout': False, 'replay_max_abs_mm': replays,
            'reference_sha256': {scheme: sha256(path) for scheme, path in REFERENCES.items()},
            'validation_code_sha256': sha256(Path(__file__)), **_provenance(data),
        }))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(RepositoryData(args.data_root), load_config(args.config), args.output_root)


if __name__ == '__main__':
    main()

"""Run H31–H35 on the frozen PipeTwin benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.experiments.r2_pool_stage1 import KEY
from src.experiments.r2_pool_stage1_strong import freeze_checkout, mapped_labels, split_frames
from src.experiments.r2_pool_stage2 import PEER, read_baseline
from src.experiments.r2_pool_stage3 import clipped, outer_train, peer_grids
from src.experiments.r2_pool_stage4 import (
    KEY_DTYPES, LOSO, RETRO, assert_replay, keys_match, prior_point_summaries, s1_historical, save_results,
)
from src.experiments.r2_pool_stage6 import S1_BLOCKS, demean_training_profile, read_comparator, save_block_scores
from src.experiments.r2_pool_stage7_models import (
    CENTERED, GP, NORMALIZED, PEARSON, fit_independent, fit_joint_s1,
    independent_predictions, predict_joint_s1,
)
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.incremental import _fit_s1, _s1_stats
from src.ml.incremental_features import DEFECT_STATS, attach_defect_stats
from src.ml.next_common import (
    fit_catboost, fit_ridge, json_safe, make_weights, predict_catboost, predict_ridge,
    s1_fit_rows, s1_training_table, versions, write_csv,
)
from src.ml.s1_contract import RIDGE_FEATURES, S1_CUTOFF, S1_ID, S2_ID
from src.ml.s1_model import fit_s1_ridge, predict_s1_ridge
from src.ml.s2_next import fit_spatial_residual
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s1 import CORE_FEATURES
from src.ml.targeted_s2 import (
    SPATIAL_FEATURES, S2_CUTOFF, _apply_candidate, _fit_physics, _predict_physics, prepare_full_grid,
)
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_07.json')
SHARED = 'shared_terrain_multitask'
ENSEMBLE = 'nested_profile_ensemble'
ENDPOINT = 'challenger_endpoint'
EQUAL = 'equal_weight_ensemble'
INDEPENDENT = {'H31': [GP], 'H32': [PEARSON], 'H33': [NORMALIZED, CENTERED]}


def blend(base, challenger, weight):
    # Preserve endpoints exactly, including their saved rounding.
    return base + weight * (challenger - base) if 0. < weight < 1. else (challenger if weight == 1. else base)


def choose_weight(inner, config):
    scores, audits = {}, {}
    for weight in config['weights']:
        profile_scores = []
        for (pid, date), held in inner.groupby(['id', 'prediction_date'], sort=True):
            prediction = blend(held.baseline, held.challenger, weight)
            r2 = (metrics(held.next_mm, prediction)['pearson_squared']
                  if len(held) >= config['minimum_inner_profile_rows'] else None)
            profile_scores.append({'id': int(pid), 'date': str(date.date()), 'rows': len(held), 'r_squared': r2})
        values = [profile['r_squared'] for profile in profile_scores]
        scores[weight] = None if any(value is None for value in values) else float(np.mean(values))
        audits[str(weight)] = {'mean_r_squared': scores[weight], 'profiles': profile_scores}
    eligible = {weight: value for weight, value in scores.items() if value is not None}
    if not eligible:
        return 0., {'status': 'fallback_no_defined_inner_scores', 'scores': audits}
    maximum = max(eligible.values())
    selected = min(weight for weight, score in eligible.items() if maximum - score <= 1e-12)
    return selected, {'status': 'selected_on_inner_oof', 'scores': audits, 'chosen_challenger_weight': selected}


def h18_fit(train, config):
    return (fit_s1_ridge(train), fit_s1_ridge(train, feature_order=RIDGE_FEATURES
            + config['geometry_distribution_features'] + config['type_features']))


def h18_prediction(pair, frame, base, direct):
    original, extension = pair
    correction = (predict_s1_ridge(extension, frame).ridge_delta_mm
                  - predict_s1_ridge(original, frame).ridge_delta_mm)
    return clipped(base + np.where(direct, correction, 0.), frame)


def s1_inner_predictions(train, config):
    parts = []
    for pid, held in train.groupby('id', sort=True):
        fit = train.loc[train.id.ne(pid)]
        prediction = h18_prediction(h18_fit(fit, config), held, held.proxy, held.has_own_prior_vtd)
        parts.append(held[KEY + ['next_mm']].assign(baseline=held.proxy, challenger=prediction,
                                                  inner_training_ids=','.join(map(str, sorted(fit.id.unique()))),
                                                  inner_training_rows=len(fit)))
    return pd.concat(parts, ignore_index=True)


def s1_predict_shape(frame, correction, direct):
    offset = correction[direct].mean() if direct.any() else 0.
    return clipped(frame.joint_blend + np.where(direct, correction - offset, 0.), frame)


def s1_peer(data, path, spec, prepared_s2, setup, b4, b6, output):
    raw = data.read('data/cleaned/vtd')
    raw = raw.loc[raw.id.isin([S1_ID, *joint.PEERS])]
    grids = data.read('processed/segments_grid')
    support = b4['hypotheses']['H17']
    h18 = b4['hypotheses']['H18']
    history = s1_historical(data, raw, grids, support)
    strict = prepared_s2['strict']
    late = history.loc[history.prediction_date.ge(S2_CUTOFF)]
    n_before = int(history.prediction_date.lt(S2_CUTOFF).sum())
    if n_before:
        raise ValueError('H34 applicability changed: inspect newly available S1 labels')
    h34_block = {'implementation': 'blocked_validation', 'result': 'not_evaluated',
                 's2_frozen_training_boundary': str(S2_CUTOFF.date()),
                 's1_repeat_labels_strictly_before_boundary': n_before,
                 'excluded_s1_repeat_labels_at_or_after_boundary': len(late),
                 'earliest_excluded_s1_repeat_label': str(late.prediction_date.min().date())}
    write_json(output / 'hypotheses/H34/s2_cutoff_audit.json', h34_block)
    split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
    labels = mapped_labels(data.read('processed/vtd_by_segment'))
    labels = labels.loc[labels.id.isin(joint.PEERS) & labels.prediction_date.eq(S1_CUTOFF)]
    keys_match(split, labels, KEY)
    base = read_baseline(path, spec)
    keys_match(base, split.loc[split.supported], KEY)
    anchor_specs = {f"{row['id']}:{row['fold']}": row
                    for row in json.loads((path / 'anchor_folds.json').read_bytes())}
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    fitted, audits, fits, inner_parts, parts = {}, {}, [], [], []
    features = b6['s1_peer_features']
    for pid, full, held in peer_grids(data, raw, grids, path):
        if pid not in fitted:
            train = outer_train(history, pid, S1_CUTOFF)
            linear = fit_ridge(train, features, 'proxy_residual', make_weights(train, 'row_uniform'),
                               alpha=10., model_name='H11_linear_control')
            independent = fit_independent(train, features, train.proxy, 'proxy_residual',
                                          linear, 'relative_position', setup)
            # Keep both task training sets outside the held section, and S2
            # labels strictly before the first S1 training date.
            donors = strict.loc[strict.id.ne(pid) & strict.prediction_date.lt(train.prediction_date.min())]
            intervals = prepared_s2['interval_map']
            physics = _predict_physics(_fit_physics(donors, intervals), donors, intervals)
            multitask = fit_joint_s1(train, donors, donors.next_mm.to_numpy(float) - physics,
                                     linear, setup['H34'])
            inner = s1_inner_predictions(train, h18)
            alpha, selection = choose_weight(inner, setup['H35'])
            inner_parts.append(inner.assign(outer_held_id=pid))
            fitted[pid] = independent, multitask, h18_fit(train, h18), alpha
            audits[pid] = {**independent['audit'], SHARED: multitask[-1], ENSEMBLE: selection}
            fits.extend([train[KEY].assign(task='S1', outer_held_id=pid),
                         donors[KEY].assign(task='S2', outer_held_id=pid)])
        independent, multitask, endpoint, alpha = fitted[pid]
        full = full.assign(last_mm=full.joint_min)
        direct = full.aligned_measurement.to_numpy(bool)
        outcomes = independent_predictions(independent, full, setup)
        outcomes.update(matched_linear=predict_ridge(independent['linear'], full),
                        **{SHARED: predict_joint_s1(multitask, full)})
        pred = full[KEY + ['fold']].assign(prediction=full.joint_blend)
        for name, correction in outcomes.items():
            pred[name] = s1_predict_shape(full, correction, direct)
        fold = anchor_specs[full.fold.iloc[0]]
        refs = joint.heldout_anchors(anchors.loc[anchors.id.eq(pid)],
                                    fold['lo_with_buffer_m'], fold['hi_with_buffer_m'])
        stats = prior_point_summaries(raw, full, S1_CUTOFF, joint.PRIOR_DATE, support, refs)
        features_h18 = full.merge(stats, on=joint.KEYS, validate='one_to_one')
        np.testing.assert_array_equal(direct, features_h18.prior_direct_mm.notna())
        candidate = h18_prediction(endpoint, features_h18, full.joint_blend, direct)
        pred[ENDPOINT] = candidate
        pred[ENSEMBLE] = blend(pred.prediction, candidate, alpha)
        pred[EQUAL] = blend(pred.prediction, candidate, .5)
        parts.append(pred.loc[held])
    predictions = pd.concat(parts, ignore_index=True).merge(
        labels, on=KEY, how='left', validate='one_to_one').astype(KEY_DTYPES)
    keys_match(split.loc[split.supported], predictions, KEY + ['fold'])
    write_csv(split, output / PEER / 'split_keys.csv')
    write_csv(pd.concat(fits), output / PEER / 'training_keys_by_task.csv')
    write_csv(pd.concat(inner_parts), output / PEER / 'ensemble_inner_predictions.csv', float_format='%.15g')
    return base, predictions, audits


def s1_blocks(data, archive, specs, setup, output):
    train = s1_fit_rows(s1_training_table(data))
    train = attach_defect_stats(train, _s1_stats(data, [train]))
    features = CORE_FEATURES + DEFECT_STATS['support']
    for scheme in S1_BLOCKS:
        path, spec = archive / scheme, specs[scheme]
        base = read_baseline(path, spec)
        split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        parts, splits, records = [], [], {}
        for _, fold, fit, held in split_frames(train, spec):
            linear = _fit_s1(fit, features, spec['model'])
            models = fit_independent(fit, features, fit.last_mm, 'target_delta_mm',
                                     linear, 'relative_position', setup)
            pred = held[KEY + ['next_mm']].assign(prediction=predict_ridge(linear, held), block=fold)
            for name, delta in independent_predictions(models, held, setup).items():
                pred[name] = clipped(held.last_mm + delta, held)
            parts.append(pred)
            records[fold] = models['audit']
            splits.extend([fit[KEY].assign(role='train', block=fold),
                           held[KEY].assign(role='test', block=fold)])
        actual = pd.concat(splits, ignore_index=True)
        keys_match(split, actual, KEY + ['block', 'role'])
        write_csv(actual, output / scheme / 'split_keys.csv')
        predictions = pd.concat(parts, ignore_index=True).astype(KEY_DTYPES)
        save_block_scores(output / scheme, base, predictions, [GP, PEARSON, NORMALIZED, CENTERED])
        yield scheme, base, predictions, records


def s2_ensemble_models(train, intervals, features, catboost, seed):
    physics = _fit_physics(train, intervals)
    fitted_level = _predict_physics(physics, train, intervals)
    linear = fit_spatial_residual(train, fitted_level, features)
    response = demean_training_profile(train.next_mm.to_numpy(float) - fitted_level, train).ravel()
    training = train.assign(centered_residual=response)
    model = fit_catboost(training, features, 'centered_residual', make_weights(train, 'row_uniform'),
                         model_name='H30_catboost_endpoint', params={**catboost, 'random_seed': seed})
    return physics, fitted_level, linear, model, training


def s2_inner_predictions(train, intervals, grids, features, catboost, seed):
    parts = []
    for pid, held in train.groupby('id', sort=True):
        fit = train.loc[train.id.ne(pid)]
        physics, _, linear, model, _ = s2_ensemble_models(fit, intervals, features, catboost, seed)
        for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
            full = grids[key]
            level = _predict_physics(physics, full, intervals)
            base = _apply_candidate(level, predict_ridge(linear, full), full, True)['h_clipped']
            challenger = _apply_candidate(level, predict_catboost(model, full), full, True)['h_clipped']
            pred = full[KEY].assign(baseline=base, challenger=challenger)
            part = labels[KEY + ['next_mm']].merge(pred, on=KEY, how='left', validate='one_to_one')
            parts.append(part.assign(inner_training_ids=','.join(map(str, sorted(fit.id.unique()))),
                                     inner_training_rows=len(fit)))
    return pd.concat(parts, ignore_index=True)


def replay_s2(prepared, archive, specs, protocol, b6, output):
    setup = protocol['hypotheses']
    strict, post = prepared['strict'], prepared['post']
    if (not strict.prediction_date.lt(S2_CUTOFF).all() or not post.prediction_date.gt(S2_CUTOFF).all()
            or prepared['measured'].id.eq(S2_ID).any()):
        raise ValueError('S2 training dates or target exclusion changed')
    intervals = prepared['interval_map']
    grids = {key: terrain_features(full) for key, full in prepared['full_frames'].items()}
    for full in grids.values():
        if full[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
            raise ValueError('S2 inference may not use inspection features')
    ranks = VARIANTS['basin_ranks']
    lookup = pd.concat([full[KEY + ranks] for full in grids.values()], ignore_index=True)
    strict = strict.merge(lookup, on=KEY, how='left', validate='one_to_one')
    catboost = b6['hypotheses']['H30']['catboost']
    seed = b6['seed']
    for scheme in (RETRO, LOSO):
        path, spec = archive / scheme, specs[scheme]
        base = read_baseline(path, spec)
        split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        features = SPATIAL_FEATURES if scheme == RETRO else ranks
        folds = ([(None, strict, post)] if scheme == RETRO else
                 [(int(pid), strict.loc[strict.id.ne(pid)], held) for pid, held in strict.groupby('id', sort=True)])
        parts, splits, records, inner_parts = [], [], {}, []
        for fold, train, held in folds:
            membership = pd.concat([train[KEY].assign(role='train'), held[KEY].assign(role='test')])
            splits.append(membership if fold is None else membership.assign(fold=fold))
            physics, fitted_level, linear, challenger, training = s2_ensemble_models(
                train, intervals, features, catboost, seed)
            models = fit_independent(training, features, fitted_level, 'centered_residual',
                                     linear, 'position', setup)
            inner = s2_inner_predictions(train, intervals, grids, features, catboost, seed)
            alpha, selection = choose_weight(inner, setup['H35'])
            fold_name = str(fold) if fold is not None else 'strict_before_2022_07_01'
            inner_parts.append(inner.assign(outer_fold=fold_name))
            records[fold_name] = {**models['audit'], ENSEMBLE: selection}
            for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
                full = grids[key]
                level = _predict_physics(physics, full, intervals)
                residuals = independent_predictions(models, full, setup)
                residuals.update(prediction=predict_ridge(linear, full),
                                 **{ENDPOINT: predict_catboost(challenger, full)})
                outcomes = {name: _apply_candidate(level, delta, full, True)['h_clipped']
                            for name, delta in residuals.items()}
                outcomes[ENSEMBLE] = blend(outcomes['prediction'], outcomes[ENDPOINT], alpha)
                outcomes[EQUAL] = blend(outcomes['prediction'], outcomes[ENDPOINT], .5)
                preds = full[KEY].assign(**outcomes)
                parts.append(labels[KEY + ['next_mm']].merge(preds, on=KEY, how='left', validate='one_to_one'))
        keys = pd.concat(splits, ignore_index=True)
        keys_match(split, keys, KEY + ['role'] + (['fold'] if scheme == LOSO else []))
        write_csv(keys, output / scheme / 'split_keys.csv')
        write_csv(pd.concat(inner_parts), output / scheme / 'ensemble_inner_predictions.csv', float_format='%.15g')
        yield scheme, base, pd.concat(parts, ignore_index=True).astype(KEY_DTYPES), records


def replay_dependency(path, base, predictions, column, tolerance):
    archive, r2 = read_comparator(path, base, tolerance)
    current = predictions[KEY + ['next_mm', column]].rename(columns={column: 'prediction'})
    difference = assert_replay(archive.rename(columns={'candidate': 'prediction'}), current, tolerance)
    return {'r2': r2, 'max_abs_difference_mm': difference, 'path': path.as_posix()}


def run(protocol_path, output):
    tick = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    baseline_protocol = Path(protocol['baseline_protocol'])
    baseline = json.loads(baseline_protocol.read_bytes())
    specs, tolerance = baseline['schemes'], baseline['archive_tolerance_mm']
    deps = {key: Path(value) for key, value in protocol['dependencies'].items()}
    b4 = json.loads(deps['batch_04_config'].read_bytes())
    b6 = json.loads(deps['batch_06_config'].read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source_snapshot.json', freeze_checkout(output))
    data = RepositoryData(Path(baseline['build']))
    archive = Path(protocol['baseline_root'])
    setup = protocol['hypotheses']
    prepared = prepare_full_grid(data)
    sources = [baseline_protocol, Path(__file__), Path(__file__).with_name('r2_pool_stage7_models.py'),
               *deps.values()]
    peer = s1_peer(data, archive / PEER, specs[PEER], prepared, setup, b4, b6, output)
    _, h12 = read_comparator(deps['H12_S1'], peer[0], tolerance)
    endpoints = {PEER: deps['H18_S1'], RETRO: deps['H30_S2_retro'], LOSO: deps['H30_S2_loso']}
    references, results, replays, audits, dependencies = {}, {}, {}, {}, {}
    listeners = [(PEER, *peer), *s1_blocks(data, archive, specs, setup, output),
                 *replay_s2(prepared, archive, specs, protocol, b6, output)]
    parents = data.parent_groups()
    for scheme, base, predictions, audit in listeners:
        spec = specs[scheme]
        error = assert_replay(base, predictions, tolerance)
        variants = dict(INDEPENDENT)
        if scheme in endpoints:
            variants['H35'] = [ENSEMBLE, ENDPOINT, EQUAL]
            dependencies[scheme] = {'ensemble_challenger': replay_dependency(
                endpoints[scheme], base, predictions, ENDPOINT, tolerance)}
        if scheme == PEER:
            variants['H34'] = [SHARED]
            dependencies[scheme]['H11'] = replay_dependency(
                deps['H11_S1'], base, predictions, 'matched_linear', tolerance)
        declared = protocol['reference_candidates'].get(scheme)
        if declared is None:
            reference = {'name': spec['model'],
                         'r2': metrics(base.next_mm, base.prediction)['pearson_squared']}
        else:
            path = Path(declared['predictions'])
            _, value = read_comparator(path, base, tolerance)
            reference = {**declared, 'r2': value}
            sources.append(path)
        references[scheme] = reference
        results[scheme] = save_results(output, scheme, base, predictions, variants, spec,
                                       reference, int(base.id.map(parents).nunique()))
        if scheme == PEER:
            linear_r2 = dependencies[PEER]['H11']['r2']
            for h, methods in results[scheme].items():
                for variant, score in methods.items():
                    value = score['candidate']['pearson_squared']
                    if value is not None:
                        score.update(delta_vs_H12=value - h12, delta_vs_matched_H11=value - linear_r2)
                    write_json(output / 'hypotheses' / h / scheme / variant / 'summary.json', score)
        audits[scheme] = audit
        replays[scheme] = {'rows': len(base), 'max_abs_prediction_difference_mm': error,
                           'same_keys_labels_and_splits': True}
        sources.extend(path for path in (archive / scheme).iterdir() if path.is_file())
        print(f'{scheme}: {len(base)} frozen keys, max replay difference {error:.3g} mm', flush=True)
        for h, methods in results[scheme].items():
            r2 = methods[setup[h]['primary']]['candidate']['pearson_squared']
            print(f'{h} {scheme}: {r2}', flush=True)
    for h, config in setup.items():
        folder = output / 'hypotheses' / h
        write_json(folder / 'config.json', config)
        summary = {'primary': config['primary'],
                   'schemes': {scheme: values[h] for scheme, values in results.items() if h in values}}
        if h == 'H34':
            summary['S2'] = {'implementation': 'blocked_validation', 'audit': 's2_cutoff_audit.json'}
        write_json(folder / 'summary.json', summary)
    write_json(output / 'fit_audit.json', json_safe(audits))
    write_json(output / 'stage_results.json', {'results': results, 'replays': replays,
                                              'references': references, 'H12_r2': h12,
                                              'dependencies': dependencies})
    write_json(output / 'run_manifest.json', {
        'protocol_version': protocol['protocol_version'], 'protocol_sha256': sha256(protocol_path),
        'code_sha256': sha256(Path(__file__)), 'source_snapshot_sha256': sha256(output / 'source_snapshot.json'),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'data_root': data.root.as_posix(), 'data_sha256': data.hashes,
        'input_sha256': {path.as_posix(): sha256(path) for path in sources},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - tick,
        'linear_algebra_threads': 1,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage7 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'exclusions': ['target S2 diagnostics', 'held labels for weight selection and model fits',
                       'S1 repeat labels at or after S2 cutoff for joint S2 evaluation',
                       'outside outer train data for H35 inner fitting'],
        'status': 'completed', 'replays': replays,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.protocol, args.output)

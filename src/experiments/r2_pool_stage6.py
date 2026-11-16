"""Evaluate H26–H30 on the original time and spatial PipeTwin splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from src.experiments.r2_pool_stage1 import KEY, scores
from src.experiments.r2_pool_stage1_strong import freeze_checkout, mapped_labels, split_frames
from src.experiments.r2_pool_stage2 import PEER, read_baseline
from src.experiments.r2_pool_stage3 import clipped, historical_training, outer_train, peer_grids
from src.experiments.r2_pool_stage4 import KEY_DTYPES, LOSO, RETRO, assert_replay, keys_match, save_results
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.incremental import _fit_s1, _s1_stats
from src.ml.incremental_features import DEFECT_STATS, attach_defect_stats
from src.ml.next_common import (
    NextPreprocessor, fit_catboost, fit_ridge, json_safe, make_weights, predict_catboost,
    predict_ridge, s1_fit_rows, s1_training_table, versions, write_csv,
)
from src.ml.s1_contract import S1_CUTOFF, S1_ID, S2_ID
from src.ml.s2_next import fit_spatial_residual
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s1 import CORE_FEATURES
from src.ml.targeted_s2 import (
    SPATIAL_FEATURES, S2_CUTOFF, _apply_candidate, _fit_physics, _predict_physics, prepare_full_grid,
)
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_06.json')
S1_BLOCKS = ('S1_100m_20m', 'S1_500m_100m')
SPLINES = 'additive_splines'
BOOSTING = 'shallow_catboost'
WITHIN = 'within_profile_spatial_ridge'
NEAREST = 'nearest_section_donor'
LARGEST = 'largest_section_donor'
CHEMISTRY = ['ph_median', 'co2_water_median', 'total_mineralization_median']


def demean_training_profile(values, frame):
    groups = [frame.id.to_numpy(), frame.prediction_date.to_numpy()]
    matrix = pd.DataFrame(values)
    return matrix.to_numpy(float) - matrix.groupby(groups).transform('mean').to_numpy(float)


def fit_within_profile(train, response, baseline):
    prep = baseline['preprocessor']
    design = demean_training_profile(prep.transform(train), train)
    y = demean_training_profile(response, train).ravel()
    ridge = Ridge(alpha=10., solver='cholesky').fit(
        design, y, sample_weight=make_weights(train, 'row_uniform'))
    return ridge, {'features': prep.used_features_, 'coefficients': ridge.coef_.tolist(),
                   'alpha': 10., 'intercept': ridge.intercept_,
                   'same_preprocessor_as_frozen_ridge': True, 'centered_design_and_response': True}


def fit_splines(train, features, target, weights, config):
    preprocessor = NextPreprocessor(features).fit(train)
    transformer = SplineTransformer(**config['spline'])
    design = transformer.fit_transform(preprocessor.transform(train))
    model = Ridge(alpha=config['ridge_alpha'], solver='cholesky')
    model.fit(design, train[target].to_numpy(float), sample_weight=weights)
    return {'preprocessor': preprocessor, 'transformer': transformer, 'estimator': model,
            'metadata': {'target': target, 'training_rows': len(train),
                         'preprocessing': preprocessor.metadata(), 'spline_parameters': config['spline'],
                         'basis_columns': design.shape[1], 'alpha': config['ridge_alpha'],
                         'knots_in_standardized_units': {column: spline.t.tolist()
                                                        for column, spline in zip(preprocessor.used_features_,
                                                                                  transformer.bsplines_)},
                         'coefficients': model.coef_.tolist(), 'intercept': model.intercept_}}


def fit_families(train, features, target, protocol):
    weights = make_weights(train, 'row_uniform')
    setup = protocol['hypotheses']
    return {SPLINES: fit_splines(train, features, target, weights, setup['H29']),
            BOOSTING: fit_catboost(train, features, target, weights, model_name=BOOSTING,
                                   params={**setup['H30']['catboost'], 'random_seed': protocol['seed']})}


def family_predictions(models, frame):
    spline = models[SPLINES]
    x = spline['transformer'].transform(spline['preprocessor'].transform(frame))
    return {SPLINES: spline['estimator'].predict(x),
            BOOSTING: predict_catboost(models[BOOSTING], frame)}


def replay_s1_peer(data, archive, spec, protocol, output):
    raw = data.read('data/cleaned/vtd')
    raw = raw.loc[raw.id.isin([S1_ID, *joint.PEERS])]
    grids = data.read('processed/segments_grid')
    history = historical_training(data, raw, grids, ())
    split = pd.read_csv(archive / 'split_keys.csv', parse_dates=['prediction_date'])
    labels = mapped_labels(data.read('processed/vtd_by_segment'))
    labels = labels.loc[labels.id.isin(joint.PEERS) & labels.prediction_date.eq(S1_CUTOFF)]
    keys_match(split, labels, KEY)
    supported = split.loc[split.supported]
    base = read_baseline(archive, spec)
    keys_match(base, supported, KEY)
    features = protocol['s1_peer_features']
    fitted, audits, train_rows, parts = {}, {}, [], []
    for pid, frame, held in peer_grids(data, raw, grids, archive):
        if pid not in fitted:
            train = outer_train(history, pid, S1_CUTOFF)
            fit = fit_families(train, features, 'proxy_residual', protocol)
            fit['matched_linear'] = fit_ridge(
                train, features, 'proxy_residual', make_weights(train, 'row_uniform'),
                alpha=10., model_name='matched_linear_for_H29_H30')
            fitted[pid] = fit
            audits[pid] = {name: model['metadata'] for name, model in fit.items()}
            train_rows.append(train[KEY + ['last_vtd_date', 'next_mm', 'proxy']].assign(held_id=pid))
        fit = fitted[pid]
        frame = frame.assign(last_mm=frame.joint_min)
        direct = frame.aligned_measurement.to_numpy(bool)
        predictions = family_predictions(fit, frame)
        predictions['matched_linear'] = predict_ridge(fit['matched_linear'], frame)
        part = frame[KEY + ['fold']].assign(prediction=frame.joint_blend)
        for name, delta in predictions.items():
            # The matched H11 control and both challenger families share the
            # full-grid centering and direct-measurement policy.
            mean = float(delta[direct].mean()) if direct.any() else 0.
            part[name] = clipped(frame.joint_blend + np.where(direct, delta - mean, 0.), frame)
        parts.append(part.loc[held])
    predictions = pd.concat(parts, ignore_index=True).merge(
        labels, on=KEY, how='left', validate='one_to_one').astype(KEY_DTYPES)
    keys_match(supported, predictions, KEY + ['fold'])
    write_csv(pd.concat(train_rows), output / PEER / 'training_keys.csv', float_format='%.15g')
    write_csv(split, output / PEER / 'split_keys.csv')
    return base, predictions, audits


def save_block_scores(output, base, pred, variants):
    paired = base[KEY + ['prediction']].rename(columns={'prediction': 'baseline'}).merge(
        pred, on=KEY, validate='one_to_one')
    details = []
    for block, rows in paired.groupby('block', sort=True):
        for name in variants:
            for entry in scores(rows.rename(columns={name: 'candidate'}), 'S1'):
                details.append({'block': block, 'variant': name, **entry})
    write_csv(pd.DataFrame(details), output / 'metrics_by_block.csv', float_format='%.15g')


def replay_s1_blocks(data, archive, specs, protocol, output):
    historical = s1_fit_rows(s1_training_table(data))
    historical = attach_defect_stats(historical, _s1_stats(data, [historical]))
    features = CORE_FEATURES + DEFECT_STATS['support']
    for scheme in S1_BLOCKS:
        spec = specs[scheme]
        path = archive / scheme
        base = read_baseline(path, spec)
        split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        records, splits, parts = {}, [], []
        for _, block, train, held in split_frames(historical, spec):
            splits.extend([train[KEY].assign(block=block, role='train'),
                           held[KEY].assign(block=block, role='test')])
            linear = _fit_s1(train, features, spec['model'])
            fitted = fit_families(train, features, 'target_delta_mm', protocol)
            part = held[KEY + ['next_mm']].assign(
                prediction=predict_ridge(linear, held), block=block)
            for name, delta in family_predictions(fitted, held).items():
                part[name] = clipped(held.last_mm + delta, held)
            records[block] = {name: model['metadata'] for name, model in fitted.items()}
            parts.append(part)
        membership = pd.concat(splits, ignore_index=True)
        keys_match(split, membership, KEY + ['block', 'role'])
        write_csv(membership, output / scheme / 'split_keys.csv')
        predictions = pd.concat(parts, ignore_index=True).astype(KEY_DTYPES)
        save_block_scores(output / scheme, base, predictions, [SPLINES, BOOSTING])
        yield scheme, base, predictions, records


def s2_descriptor(full, fields):
    if (full.hydraulic_source_date_max.notna()
            & full.hydraulic_source_date_max.ge(full.prediction_date)).any():
        raise ValueError('Donor descriptors must predate the prediction')
    weights = full.length_m.to_numpy(float)
    derived = {'log_section_length': float(np.log(weights.sum())),
               'mean_abs_slope': float(np.average(full.abs_slope, weights=weights))}
    descriptors = {key: derived[key] if key in derived else float(full[key].iloc[0])
                   for key in fields}
    if not np.isfinite(list(descriptors.values())).all():
        raise ValueError('Expected finite donor-matching descriptors')
    return descriptors


def donor_ranking(donors, held_descriptor, scales):
    if not donors or any(scale <= 0. for scale in scales.values()):
        raise ValueError('Missing eligible donor or nonpositive distance scale')
    ranking = []
    for (pid, date), (descriptor, count, _) in donors.items():
        distance = float(np.linalg.norm([(descriptor[c] - held_descriptor[c]) / scales[c]
                                         for c in scales]))
        ranking.append(((pid, date), distance, count))
    return sorted(ranking, key=lambda entry: (entry[1], -entry[2], entry[0]))


def fit_donors(train, physics, grids, features, config):
    frame = train.assign(physics_mm=physics)
    donors = {}
    for key, rows in frame.groupby(['id', 'prediction_date'], sort=True):
        if len(rows) >= config['min_donor_labels']:
            donor = fit_spatial_residual(rows, rows.physics_mm.to_numpy(float), features)
            donors[key] = (s2_descriptor(grids[key], config['distance_scales']), len(rows), donor)
    return donors


def donor_prediction(key, full, physics, donors, config):
    scales = config['distance_scales']
    descriptor = s2_descriptor(full, scales)
    ranking = donor_ranking(donors, descriptor, scales)
    nearest = ranking[0]
    largest = min(ranking, key=lambda x: (-x[2], x[0]))
    audit = []
    for candidate, distance, count in ranking:
        audit.append({'id': int(key[0]), 'prediction_date': str(key[1].date()),
                      'donor_id': int(candidate[0]), 'donor_date': str(candidate[1].date()),
                      'distance': distance, 'training_rows': count,
                      'nearest': candidate == nearest[0], 'largest': candidate == largest[0],
                      **{f'held_{k}': v for k, v in descriptor.items()},
                      **{f'donor_{k}': v for k, v in donors[candidate][0].items()}})
    predictions = {name: _apply_candidate(
        physics, predict_ridge(donors[entry[0]][2], full), full, True)['h_clipped']
        for name, entry in [(NEAREST, nearest), (LARGEST, largest)]}
    return predictions, audit


def chemistry_audit(data, grids, labels):
    mapped = data.read('processed/monitoring_mapped')
    raw = data.read('data/cleaned/regime')
    coverage = []
    for (pid, date), full in sorted(grids.items()):
        acceptable = full.monitoring_source_date_max.isna() | full.monitoring_source_date_max.lt(date)
        if not acceptable.all():
            raise ValueError('Future monitoring date in chemistry audit')
        count = int((labels.id.eq(pid) & labels.prediction_date.eq(date)).sum())
        coverage.append({'id': int(pid), 'prediction_date': str(date.date()), 'grid_rows': len(full),
                         'label_rows': count, 'linked_monitoring_rows': int(full.monitoring_records_count.gt(0).sum()),
                         'nonmissing_grid_rows': {name: int(full[name].notna().sum()) for name in CHEMISTRY}})
    return {'implementation': 'blocked_data', 'result': 'not_evaluated', 's2_grids': coverage,
            'target_monitoring_grid_rows': len(mapped),
            'raw_monitoring_rows': len(raw), 'raw_records_with_known_id': int(raw.id_known.sum()),
            'raw_mapping_eligible': int(raw.mapping_eligible.sum()),
            'raw_observation_eligible': int(raw.observation_eligible.sum()),
            'raw_nonmissing': {column: int(raw[column].notna().sum()) for column in
                               ('ph', 'co2_water', 'total_mineralization', 'protective_effect_pct',
                                'corrosion_inhibitor_grade')},
            'target_s2_raw_rows_with_exact_id': int(raw.id.eq(S2_ID).sum()),
            'missing': 'Verified pre-cutoff location and date for local chemistry and protection on S2 grids.'}


def replay_s2(data, archive, specs, protocol, output):
    prepared = prepare_full_grid(data)
    strict, post = prepared['strict'], prepared['post']
    if (not strict.prediction_date.lt(S2_CUTOFF).all() or not post.prediction_date.gt(S2_CUTOFF).all()
            or prepared['measured'].id.eq(S2_ID).any()):
        raise ValueError('S2 date or diagnostic exclusion failed')
    intervals = prepared['interval_map']
    target_key = (S2_ID, S2_CUTOFF)
    grids = {key: terrain_features(frame) for key, frame in
             {**prepared['full_frames'], target_key: prepared['target']}.items()}
    for full in grids.values():
        if full[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
            raise ValueError('Unexpected diagnostics in a cold-start grid')
    write_json(output / 'hypotheses/H28/input_audit.json',
               chemistry_audit(data, grids, prepared['measured']))
    ranks = VARIANTS['basin_ranks']
    lookup = pd.concat([full[KEY + ranks] for full in grids.values()], ignore_index=True)
    strict = strict.merge(lookup, on=KEY, how='left', validate='one_to_one')
    config = protocol['hypotheses']['H27']
    for scheme in (RETRO, LOSO):
        feature_order = SPATIAL_FEATURES if scheme == RETRO else ranks
        path, spec = archive / scheme, specs[scheme]
        base = read_baseline(path, spec)
        frozen_split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        folds = ([(None, strict, post)] if scheme == RETRO else
                 [(int(pid), strict.loc[strict.id.ne(pid)], held) for pid, held in strict.groupby('id', sort=True)])
        splits, fitted, parts, choices = [], {}, [], []
        for fold, train, held in folds:
            if fold is None and not train.prediction_date.lt(held.prediction_date.min()).all():
                raise ValueError('Retrospective donor labels must strictly precede held date')
            split = pd.concat([train[KEY].assign(role='train'), held[KEY].assign(role='test')])
            splits.append(split if fold is None else split.assign(fold=fold))
            physics_model = _fit_physics(train, intervals)
            physics_train = _predict_physics(physics_model, train, intervals)
            baseline = fit_spatial_residual(train, physics_train, feature_order)
            residual = train.next_mm.to_numpy(float) - physics_train
            within, within_audit = fit_within_profile(train, residual, baseline)
            train = train.assign(centered_residual=demean_training_profile(residual, train).ravel())
            families = fit_families(train, feature_order, 'centered_residual', protocol)
            donors = fit_donors(train, physics_train, grids, feature_order, config)
            fitted[str(fold) if fold is not None else 'strict_before_2022_07_01'] = {
                'physics': physics_model.metadata(), 'baseline': baseline['metadata'],
                WITHIN: within_audit,
                **{name: model['metadata'] for name, model in families.items()},
                'donors': {f'{key[0]}@{key[1].date()}': model['metadata']
                           for key, (_, _, model) in donors.items()},
            }
            for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
                full = grids[key]
                physics_test = _predict_physics(physics_model, full, intervals)
                residuals = family_predictions(families, full)
                residuals.update(prediction=predict_ridge(baseline, full),
                                 **{WITHIN: within.predict(baseline['preprocessor'].transform(full))})
                predictions = {name: _apply_candidate(physics_test, shape, full, True)['h_clipped']
                               for name, shape in residuals.items()}
                transferred, audit = donor_prediction(key, full, physics_test, donors, config)
                choices.extend(audit)
                forecasts = full[KEY].assign(**predictions, **transferred)
                parts.append(labels[KEY + ['next_mm']].merge(forecasts, on=KEY, how='left',
                                                             validate='one_to_one'))
            if scheme == RETRO:
                full = grids[target_key]
                if not train.prediction_date.lt(S2_CUTOFF).all():
                    raise ValueError('Target donor audit has data past cutoff')
                ranking = donor_ranking(donors, s2_descriptor(full, config['distance_scales']),
                                        config['distance_scales'])
                fitted['target_analogue_distances'] = [
                    {'donor_id': key[0], 'donor_date': str(key[1].date()),
                     'distance': distance, 'training_rows': count}
                    for key, distance, count in ranking]
        membership = pd.concat(splits, ignore_index=True)
        keys_match(frozen_split, membership, KEY + ['role'] + (['fold'] if scheme == LOSO else []))
        write_csv(membership, output / scheme / 'split_keys.csv')
        write_csv(pd.DataFrame(choices), output / scheme / 'donors_by_profile.csv', float_format='%.15g')
        yield scheme, base, pd.concat(parts, ignore_index=True).astype(KEY_DTYPES), fitted


def read_comparator(path, base, tolerance):
    pair = pd.read_csv(path, parse_dates=['prediction_date']).astype(KEY_DTYPES)
    assert_replay(base, pair.rename(columns={'baseline': 'prediction'}), tolerance)
    return pair, metrics(pair.next_mm, pair.candidate)['pearson_squared']


def run(protocol_path, output):
    start = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    baseline_protocol = Path(protocol['baseline_protocol'])
    baseline = json.loads(baseline_protocol.read_bytes())
    archive = Path(protocol['baseline_root'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source_snapshot.json', freeze_checkout(output))
    data = RepositoryData(Path(baseline['build']))
    parents, specs = data.parent_groups(), baseline['schemes']
    tolerance = baseline['archive_tolerance_mm']
    references, results, replays, audits = {}, {}, {}, {}
    sources = [baseline_protocol, Path(__file__)]
    peer = replay_s1_peer(data, archive / PEER, specs[PEER], protocol, output)
    h12_file, linear_file = Path(protocol['s1_h12_reference']), Path(protocol['s1_linear_dependency'])
    _, h12_r2 = read_comparator(h12_file, peer[0], tolerance)
    archived_control, linear_r2 = read_comparator(linear_file, peer[0], tolerance)
    linear_replay = peer[1][KEY + ['next_mm', 'matched_linear']].rename(
        columns={'matched_linear': 'prediction'})
    linear_error = assert_replay(archived_control.rename(columns={'candidate': 'prediction'}),
                                 linear_replay, tolerance)
    sources.extend([h12_file, linear_file])
    families = {'H29': [SPLINES], 'H30': [BOOSTING]}
    runs = [
        (PEER, *peer), *replay_s1_blocks(data, archive, specs, protocol, output),
        *replay_s2(data, archive, specs, protocol, output),
    ]
    for scheme, base, predictions, audit in runs:
        spec = specs[scheme]
        error = assert_replay(base, predictions, tolerance)
        variants = {**families}
        if spec['case'] == 'S2':
            variants.update(H26=[WITHIN], H27=[NEAREST, LARGEST])
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
            write_csv(predictions[KEY + ['fold', 'next_mm', 'prediction', 'matched_linear',
                                         SPLINES, BOOSTING]],
                      output / PEER / 'matched_linear_and_family_predictions.csv', float_format='%.15g')
            for name, methods in results[scheme].items():
                score = next(iter(methods.values()))
                r2 = score['candidate']['pearson_squared']
                score['matched_linear_r2'] = linear_r2
                score['delta_vs_matched_linear'] = None if r2 is None else r2 - linear_r2
                score['h12_r2'] = h12_r2
                score['delta_vs_h12'] = None if r2 is None else r2 - h12_r2
                variant = next(iter(methods))
                write_json(output / 'hypotheses' / name / PEER / variant / 'summary.json', score)
        audits[scheme] = audit
        replays[scheme] = {'rows': len(base), 'max_abs_prediction_difference_mm': error,
                           'same_keys_labels_and_splits': True}
        sources.extend(path for path in (archive / scheme).iterdir() if path.is_file())
        print(f'{scheme}: {len(base)} frozen keys, max replay difference {error:.3g} mm', flush=True)
        for h, methods in results[scheme].items():
            score = methods[protocol['hypotheses'][h]['primary']]
            print(f'{h} {scheme}: {score["candidate"]["pearson_squared"]}', flush=True)
    for h, spec in protocol['hypotheses'].items():
        folder = output / 'hypotheses' / h
        write_json(folder / 'config.json', spec)
        summary = {'primary': spec['primary'],
                   'schemes': {name: group[h] for name, group in results.items() if h in group}}
        if h == 'H28':
            summary.update(implementation='blocked_data', result='not_evaluated', audit='input_audit.json')
        write_json(folder / 'summary.json', summary)
    write_json(output / 'fit_audit.json', json_safe(audits))
    write_json(output / 'stage_results.json', {'results': results, 'replays': replays, 'references': references,
                                              'H12_r2': h12_r2, 'H11_linear_r2': linear_r2,
                                              'H11_replay_difference_mm': linear_error,
                                              'blocked': {'H28': 'blocked_data'}})
    write_json(output / 'run_manifest.json', {
        'protocol_version': protocol['protocol_version'], 'protocol_sha256': sha256(protocol_path),
        'code_sha256': sha256(Path(__file__)),
        'source_snapshot_sha256': sha256(output / 'source_snapshot.json'),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'data_root': data.root.as_posix(), 'data_sha256': data.hashes,
        'input_sha256': {path.as_posix(): sha256(path) for path in sources},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - start,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage6 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'exclusions': ['target S2 diagnostic data', 'cutoff-day S1 thickness inputs',
                       'held labels for analogue ranking, preprocessing, and parameter choices',
                       'unverified chemistry and protection data'],
        'replays': replays, 'status': 'completed',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)

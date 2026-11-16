"""Run independent hypotheses H16–H20 against the frozen PipeTwin benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge

from src.experiments.r2_pool_stage1 import KEY
from src.experiments.r2_pool_stage1_photo import score_comparison
from src.experiments.r2_pool_stage1_strong import freeze_checkout, mapped_labels
from src.experiments.r2_pool_stage2 import PEER, read_baseline
from src.experiments.r2_pool_stage3 import (
    axis, clipped, historical_training, outer_train, peer_grids, window_features,
)
from src.grid_builder import assign_segments
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.incremental import _fit_s2
from src.ml.incremental_features import multiscale_elevation
from src.ml.next_common import json_safe, versions, write_csv
from src.ml.s1_contract import RIDGE_FEATURES, S1_CUTOFF, S1_ID, S2_ID
from src.ml.s1_model import (
    S1Preprocessor, equal_simple_pipe_weights, fit_s1_ridge, predict_s1_ridge,
)
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s2 import (
    SPATIAL_FEATURES, S2_CUTOFF, _predict_candidate, prepare_full_grid,
)
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_04.json')
RETRO = 'S2_retro_2023'
LOSO = 'S2_LOSO_pooled'
KS = joint.KEYS
TEMPERATURE = 'prior_centered_fluid_temperature_c'
KEY_DTYPES = {'id': 'int64', 'segment_id': 'int64', 'prediction_date': 'datetime64[ns]'}


def keys_match(expected, actual, columns):
    left, right = (frame.astype(KEY_DTYPES).set_index(columns, verify_integrity=True).sort_index().index
                   for frame in (expected, actual))
    pd.testing.assert_index_equal(left, right)


def assert_replay(base, replay, tolerance):
    keys_match(base, replay, KEY)
    left, right = (frame.set_index(KEY).sort_index() for frame in (base, replay))
    np.testing.assert_array_equal(left.next_mm, right.next_mm)
    np.testing.assert_allclose(left.prediction, right.prediction, atol=tolerance, rtol=0.)
    return float((left.prediction - right.prediction).abs().max())


def prior_point_summaries(raw, grid, cutoff, prior_date, support, anchors=None):
    """Aggregate eligible earlier measurements on a single simple-section grid."""
    (pid,) = grid.id.unique()
    if not prior_date < cutoff or pid == S2_ID:
        raise ValueError('Prior point summaries require a non-S2 section and an earlier campaign')
    prior = raw.loc[raw.id.eq(pid) & raw.date.eq(prior_date) & raw.target_eligible
                    & raw.residual_for_target_mm.notna()].copy()
    distance = prior.distance if anchors is None else joint.map_coordinates(prior.distance, anchors)
    prior['point_m'] = distance
    prior = assign_segments(prior.loc[prior.point_m.between(0, grid.end_m.max())], 'point_m', grid)
    description = prior.feature_description.fillna('').str.strip().str.casefold()
    side = prior.defect_type.fillna('').str.strip().str.casefold()
    prior = prior.assign(
        length_log=np.log1p(prior.defect_length.where(prior.defect_length.gt(0))),
        width_log=np.log1p(prior.defect_width.where(prior.defect_width.gt(0))),
        reported_metal_loss=description.str.startswith(('потеря металла', 'корроз')),
        external=side.str.startswith('внеш').where(side.ne('')),
    )
    groups = prior.groupby(KS)
    values = groups.residual_for_target_mm
    points = groups.agg(
        prior_direct_mm=('residual_for_target_mm', 'min'),
        prior_measurement_count=('residual_for_target_mm', 'size'),
        prior_log_median_length=('length_log', 'median'),
        prior_log_median_width=('width_log', 'median'),
        prior_reported_metal_loss_fraction=('reported_metal_loss', 'mean'),
        prior_external_fraction=('external', 'mean'),
    ).assign(prior_median_minus_min_mm=values.median() - values.min(),
             prior_p90_minus_p10_mm=values.quantile(.9) - values.quantile(.1))
    aligned = grid.merge(points.reset_index(), on=KS, how='left', validate='one_to_one')
    width = support['neighbor_radius_m']
    nearby = window_features(aligned, aligned.prior_direct_mm, [width])[f'prior_count_{width}m']
    high = (aligned.prior_measurement_count.ge(support['minimum_own_measurements'])
            & (nearby >= support['minimum_measured_neighbors']))
    return aligned[[*KS, *points.columns]].assign(
        prior_neighbor_measurement_segments=nearby, high_support=high,
        prior_feature_date=prior_date)


def s1_historical(data, raw, grids, support):
    historical = historical_training(data, raw, grids, ())
    parts = []
    for (pid, date, prior), labels in historical.groupby(
            ['id', 'prediction_date', 'last_vtd_date'], sort=True):
        grid = grids.loc[grids.id.eq(pid)].sort_values('start_m')
        stats = prior_point_summaries(raw, grid, date, prior, support)
        part = labels.merge(stats, on=KS, how='left', validate='one_to_one')
        np.testing.assert_allclose(part.prior_direct_mm, part.last_mm, rtol=0., atol=1e-9)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit_measurement_model(train, spec):
    prep = S1Preprocessor(spec['features']).fit(train)
    design = prep.transform(train)
    weights = equal_simple_pipe_weights(train)
    models = {
        spec['primary']: HuberRegressor(epsilon=spec['huber_epsilon'], alpha=spec['l2_alpha'],
                                        max_iter=2000, tol=1e-7),
        'squared_loss_matched': Ridge(alpha=spec['l2_alpha'], solver='cholesky'),
    }
    fits = {name: model.fit(design, train.proxy_residual, sample_weight=weights)
            for name, model in models.items()}
    if fits[spec['primary']].n_iter_ >= models[spec['primary']].max_iter:
        raise ValueError('Huber fit reached its iteration limit')
    return prep, fits


def correction_weight(high_support, low, high):
    return np.where(high_support, high, low)


def select_support_weights(train, weights):
    """The inner holdout and the outer section never fit their own corrections."""
    pairs = [(low, high) for low in weights for high in weights if high <= low]
    scores = {pair: {} for pair in pairs}
    for pid, held in train.groupby('id', sort=True):
        model = fit_s1_ridge(train.loc[train.id.ne(pid)])
        delta = predict_s1_ridge(model, held).ridge_delta_mm
        for low, high in pairs:
            pred = clipped(held.proxy + correction_weight(held.high_support, low, high) * delta, held)
            scores[low, high][int(pid)] = (metrics(held.next_mm, pred)['pearson_squared']
                                           if len(held) >= 3 else None)
    means = {pair: float(np.mean(list(values.values()))) for pair, values in scores.items()
             if all(value is not None for value in values.values())}
    if not any(low == high for low, high in means):
        raise ValueError('No defined inner score for the uniform control')
    ranking = sorted(means, key=lambda pair: (-means[pair], sum(pair), pair))
    table = [{'low_support_weight': low, 'high_support_weight': high,
              'mean_r2': means.get((low, high)), 'by_section': values}
             for (low, high), values in scores.items()]
    return ranking[0], next(pair for pair in ranking if pair[0] == pair[1]), table


def measurement_shape(model, prep, frame, direct):
    correction = model.predict(prep.transform(frame.loc[direct])) if direct.any() else np.array([])
    out = np.zeros(len(frame))
    if direct.any():
        out[direct] = correction - correction.mean()
    return out


def run_s1(data, archive, spec, setup, output, tolerance):
    raw = data.read('data/cleaned/vtd')
    raw = raw.loc[raw.id.isin([S1_ID, *joint.PEERS])]
    grids = data.read('processed/segments_grid')
    historical = s1_historical(data, raw, grids, setup['H17'])
    base = read_baseline(archive, spec)
    split = pd.read_csv(archive / 'split_keys.csv', parse_dates=['prediction_date'])
    anchor_specs = {f"{row['id']}:{row['fold']}": row
                    for row in json.loads((archive / 'anchor_folds.json').read_bytes())}
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    labels = mapped_labels(data.read('processed/vtd_by_segment'))
    labels = labels.loc[labels.id.isin(joint.PEERS) & labels.prediction_date.eq(S1_CUTOFF)]
    keys_match(split, labels, KEY)
    tests = split.loc[split.supported, KEY + ['fold']]
    keys_match(base, tests, KEY)
    s1_variants = {
        h: [cfg['primary'], *cfg['controls']] for h, cfg in setup.items() if cfg['case'] == 'S1'
    }
    outcomes, training, fits, audit = [], [], {}, {}
    geometry = setup['H18']['geometry_distribution_features']
    extensions = {setup['H18']['primary']: geometry + setup['H18']['type_features'],
                  'geometry_distribution': geometry}
    source_counts = []
    for pid, full, held in peer_grids(data, raw, grids, archive):
        if pid not in fits:
            train = outer_train(historical, pid, S1_CUTOFF)
            training.append(train[KEY + ['prior_feature_date', 'next_mm', 'proxy', 'high_support']]
                            .assign(held_id=pid))
            prep, robust = fit_measurement_model(train, setup['H16'])
            s1c = fit_s1_ridge(train)
            gate, uniform, inner = select_support_weights(train, setup['H17']['weights'])
            expanded = {name: fit_s1_ridge(train, feature_order=RIDGE_FEATURES + features)
                        for name, features in extensions.items()}
            fits[pid] = prep, robust, s1c, gate, uniform, expanded
            audit[pid] = {
                'training_rows': len(train), 'training_high_support': int(train.high_support.sum()),
                'H16': {'preprocessing': prep.metadata(), 'models': {
                    name: {'intercept': model.intercept_, 'coef': model.coef_.tolist(),
                           'scale': getattr(model, 'scale_', None),
                           'outlier_count': int(model.outliers_.sum()) if hasattr(model, 'outliers_') else None}
                    for name, model in robust.items()}},
                'H17': {'weights_low_high': gate, 'uniform_control_low_high': uniform,
                        'inner_scores': inner, 'auxiliary': s1c['metadata']},
                'H18': {name: model['metadata'] for name, model in expanded.items()},
            }
        prep, robust, s1c, gate, uniform, expanded = fits[pid]
        fold = anchor_specs[full.fold.iloc[0]]
        retained = joint.heldout_anchors(anchors.loc[anchors.id.eq(pid)],
                                        fold['lo_with_buffer_m'], fold['hi_with_buffer_m'])
        stats = prior_point_summaries(raw, full, S1_CUTOFF, joint.PRIOR_DATE,
                                      setup['H17'], retained)
        enriched = full.merge(stats, on=KS, validate='one_to_one')
        direct = enriched.aligned_measurement.to_numpy(bool)
        np.testing.assert_array_equal(direct, enriched.prior_direct_mm.notna())
        np.testing.assert_allclose(enriched.loc[direct, 'prior_direct_mm'],
                                   enriched.loc[direct, 'joint_min'], rtol=0., atol=1e-9)
        enriched['proxy'] = enriched.joint_blend
        baseline = enriched.joint_blend
        pred = enriched[KEY + ['fold']].assign(prediction=baseline)
        for name, model in robust.items():
            pred[name] = clipped(baseline + measurement_shape(model, prep, enriched, direct), enriched)
        delta = predict_s1_ridge(s1c, enriched).ridge_delta_mm
        for name, pair in [(setup['H17']['primary'], gate), ('best_uniform_gate', uniform),
                           ('full_correction', (1., 1.))]:
            weight = correction_weight(enriched.high_support, *pair)
            pred[name] = clipped(baseline + np.where(direct, delta * weight, 0.), enriched)
        for name, model in expanded.items():
            extra = predict_s1_ridge(model, enriched).ridge_delta_mm - delta
            pred[name] = clipped(baseline + np.where(direct, extra, 0.), enriched)
        outcomes.append(pred.loc[held.to_numpy()])
        source_counts.append(enriched.loc[held.to_numpy(), KEY + ['fold', *stats.columns.difference(KS)]])
    predictions = pd.concat(outcomes, ignore_index=True)
    keys_match(tests, predictions, KEY + ['fold'])
    predictions = predictions.merge(labels, on=KEY, how='left', validate='one_to_one').astype(KEY_DTYPES)
    replay_error = assert_replay(base, predictions, tolerance)
    write_csv(pd.concat(training), output / 's1_training_keys.csv')
    write_csv(pd.concat(source_counts), output / 's1_prior_measurement_features.csv', float_format='%.15g')
    return base, predictions, s1_variants, audit, replay_error


def s2_covariates(full, widths):
    if full.id.nunique() != 1 or full.prediction_date.nunique() != 1:
        raise ValueError('Each terrain and thermal grid must belong to one section and date')
    frame = terrain_features(full)
    if frame[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
        raise ValueError('S2 grid contains diagnostic features')
    terrain = multiscale_elevation(frame, widths)
    additions = []
    for width in widths:
        relative, slope = f'elevation_relative_{width}m', f'elevation_slope_{width}m'
        curvature = f'elevation_curvature_{width}m'
        terrain[curvature] = (np.gradient(terrain[slope].to_numpy(float), axis(frame))
                              if len(frame) >= 3 else np.nan)
        additions.extend([relative, slope, curvature])
    frame = frame.merge(terrain[KS + additions], on=KS, validate='one_to_one')
    midpoint = (frame.t_in_segment_c + frame.t_out_segment_c) / 2.
    valid = frame.scenario_valid.fillna(False) & np.isfinite(midpoint)
    frame[TEMPERATURE] = (midpoint - np.average(midpoint, weights=frame.length_m)
                          if valid.all() else np.nan)
    return frame, additions


def s2_predictions(test, grids, intervals, artifacts, control_name):
    outputs = []
    for key, labels in test.groupby(['id', 'prediction_date'], sort=True):
        full = grids[key]
        forecasts = full[KEY].copy()
        forecasts['prediction'] = _predict_candidate(artifacts[control_name], full, intervals)['h_clipped']
        for name, model in artifacts.items():
            if name == control_name:
                continue
            temperature_variant = name in ('centered_fluid_temperature', 'local_cooling')
            if temperature_variant and not full[TEMPERATURE].notna().all():
                forecasts[name] = forecasts.prediction
            else:
                forecasts[name] = _predict_candidate(model, full, intervals)['h_clipped']
        outputs.append(labels[KEY + ['next_mm']].merge(forecasts, on=KEY, how='left',
                                                       validate='one_to_one'))
    return pd.concat(outputs, ignore_index=True)


def s2_variation(frame, n_labels, columns):
    return {'id': int(frame.id.iloc[0]), 'prediction_date': str(frame.prediction_date.iloc[0].date()),
            'grid_rows': len(frame), 'label_rows': n_labels,
            'temperature_valid_rows': int(frame[TEMPERATURE].notna().sum()),
            'temperature_missing_reason': '|'.join(sorted(set(frame.scenario_reason.dropna()) - {''})),
            'temperature_vs_position_r': frame[TEMPERATURE].corr(frame.position)
            if frame[TEMPERATURE].nunique() > 1 and frame.position.nunique() > 1 else None,
            **{f'{column}_std': frame[column].std(ddof=0) for column in columns}}


def run_s2(data, archive, schemes, setup, output, tolerance):
    prepared = prepare_full_grid(data)
    strict, post = prepared['strict'], prepared['post']
    if not strict.prediction_date.lt(S2_CUTOFF).all() or post.prediction_date.min() <= S2_CUTOFF:
        raise ValueError('S2 retrospective training dates changed')
    if any(frame.id.eq(S2_ID).any() for frame in (strict, post)):
        raise ValueError('Target S2 pipe cannot supply diagnostic labels')
    intervals = prepared['interval_map']
    widths = setup['H19']['window_widths_m']
    grids = {}
    for key, full in prepared['full_frames'].items():
        grids[key], terrain = s2_covariates(full, widths)
    target, _ = s2_covariates(prepared['target'], widths)
    covariates = [*terrain, TEMPERATURE, 'scenario_cooling_c']
    label_counts = prepared['measured'].groupby(['id', 'prediction_date']).size()
    variation = [s2_variation(frame, int(label_counts.get(key, 0)), covariates)
                 for key, frame in grids.items()]
    variation.append(s2_variation(target, 0, covariates))
    write_csv(pd.DataFrame(variation), output / 's2_within_pipe_variation.csv', float_format='%.15g')
    summaries = pd.concat([full[KEY + covariates + VARIANTS['basin_ranks']]
                           for full in grids.values()], ignore_index=True)
    train = strict.drop(columns='scenario_cooling_c').merge(summaries, on=KEY, how='left',
                                                            validate='one_to_one')

    for scheme, pairs in [(RETRO, [(None, train, post)]),
                          (LOSO, [(int(pid), train.loc[train.id.ne(pid)], held)
                                  for pid, held in train.groupby('id', sort=True)])]:
        spec = schemes[scheme]
        control = spec['model']
        features = SPATIAL_FEATURES if scheme == RETRO else VARIANTS['basin_ranks']
        configs = {
            control: features,
            setup['H19']['primary']: features + terrain,
            setup['H20']['primary']: features + [TEMPERATURE],
            'local_cooling': features + ['scenario_cooling_c'],
        }
        s2_variants = {'H19': [setup['H19']['primary']],
                       'H20': [setup['H20']['primary'], 'local_cooling']}
        if scheme == RETRO:
            configs.update(coordinate_only=['position'], terrain_without_coordinate=[
                name for name in features if name != 'position'])
            s2_variants['H20'] += ['coordinate_only', 'terrain_without_coordinate']
        frozen = read_baseline(archive / scheme, spec)
        frozen_split = pd.read_csv(archive / scheme / 'split_keys.csv', parse_dates=['prediction_date'])
        holdout, splits, audit = [], [], {}
        for fold, fit, test in pairs:
            if fold is not None and fit.id.eq(fold).any():
                raise ValueError('S2 LOSO training includes the holdout section')
            labels = [fit[KEY + ['next_mm']].assign(role='train'), test[KEY + ['next_mm']].assign(role='test')]
            split = pd.concat(labels, ignore_index=True)
            splits.append(split if fold is None else split.assign(fold=fold))
            artifacts = _fit_s2(fit, intervals, configs)
            holdout.append(s2_predictions(test, grids, intervals, artifacts, control))
            audit[str(fold) if fold is not None else 'strict_before_2022_07_01'] = {
                'train_rows': len(fit), 'test_rows': len(test),
                'train_dates': sorted(str(date.date()) for date in fit.prediction_date.unique()),
                'valid_temperature_train_rows': int(fit[TEMPERATURE].notna().sum()),
                'models': {name: value['metadata'] for name, value in artifacts.items()},
            }
        reconstructed = pd.concat(splits, ignore_index=True)
        keys_match(frozen_split, reconstructed, KEY + ['role'] + (['fold'] if scheme == LOSO else []))
        write_csv(reconstructed, output / scheme / 'split_keys.csv')
        predictions = pd.concat(holdout, ignore_index=True).astype(KEY_DTYPES)
        error = assert_replay(frozen, predictions, tolerance)
        yield scheme, frozen, predictions, s2_variants, audit, error


def save_results(output, scheme, base, predictions, variants, spec, reference, parent_count):
    results = {}
    for hypothesis, names in variants.items():
        results[hypothesis] = {}
        for name in names:
            folder = output / 'hypotheses' / hypothesis / scheme / name
            proposed = predictions[KEY + ['next_mm', name]].rename(columns={name: 'prediction'})
            summary = score_comparison(base, proposed, spec, parent_count, folder)
            r2 = summary['candidate']['pearson_squared']
            summary['reference_candidate'] = reference
            summary['delta_vs_reference_candidate'] = None if r2 is None else r2 - reference['r2']
            write_json(folder / 'summary.json', summary)
            results[hypothesis][name] = summary
    return results


def run(protocol_path, output):
    start = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    baseline_protocol = Path(protocol['baseline_protocol'])
    baseline = json.loads(baseline_protocol.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source_snapshot.json', freeze_checkout(output))
    data = RepositoryData(Path(baseline['build']))
    archive = Path(protocol['baseline_root'])
    setup = protocol['hypotheses']
    tolerance = baseline['archive_tolerance_mm']
    schemes = baseline['schemes']
    references = protocol['reference_candidates']
    parents = data.parent_groups()
    results, audits, replays = {}, {}, {}
    s1 = run_s1(data, archive / PEER, schemes[PEER], setup, output, tolerance)
    h12_path = Path(references[PEER]['predictions'])
    h12 = pd.read_csv(h12_path, parse_dates=['prediction_date']).astype(KEY_DTYPES)
    reference_error = assert_replay(s1[0], h12.rename(columns={'baseline': 'prediction'}), tolerance)
    np.testing.assert_allclose(metrics(h12.next_mm, h12.candidate)['pearson_squared'],
                               references[PEER]['r2'], atol=1e-10, rtol=0.)
    s2 = run_s2(data, archive, schemes, setup, output, tolerance)
    for scheme, frozen, predictions, variants, audit, error in [(PEER, *s1), *s2]:
        count = int(frozen.id.map(parents).nunique())
        results[scheme] = save_results(output, scheme, frozen, predictions, variants,
                                       schemes[scheme], references[scheme], count)
        audits[scheme] = audit
        replays[scheme] = {'rows': len(frozen), 'max_abs_prediction_difference_mm': error,
                           'same_keys_labels_and_splits': True}
        print(f'{scheme} replay: {len(frozen)} rows, max difference {error:.3g} mm', flush=True)
    for name, config in setup.items():
        folder = output / 'hypotheses' / name
        write_json(folder / 'config.json', config)
        by_scheme = {key: items[name] for key, items in results.items() if name in items}
        write_json(folder / 'summary.json', {'primary': config['primary'], 'schemes': by_scheme})
        for scheme, variants in by_scheme.items():
            score = variants[config['primary']]['candidate']['pearson_squared']
            print(f'{name} {scheme}: r2={score}', flush=True)
    write_json(output / 'fit_audit.json', json_safe(audits))
    write_json(output / 'stage_results.json', {'results': results, 'replays': replays,
                                              'H12_base_max_abs_difference_mm': reference_error})
    sources = [baseline_protocol, Path(__file__), h12_path, *(p for scheme in results
                                                   for p in (archive / scheme).iterdir() if p.is_file())]
    write_json(output / 'run_manifest.json', {
        'protocol_version': protocol['protocol_version'], 'protocol_sha256': sha256(protocol_path),
        'code_sha256': sha256(Path(__file__)),
        'source_snapshot_sha256': sha256(output / 'source_snapshot.json'),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'data_root': data.root.as_posix(),
        'data_sha256': data.hashes,
        'input_sha256': {path.as_posix(): sha256(path) for path in sources},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - start,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage4 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'exclusions': ['S2 target diagnostic features and labels', 'S1 cutoff-day thickness inputs',
                       'S1 angles and individual defect matching', 'H20 predictions from invalid scenarios'],
        'replays': replays, 'status': 'completed',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)

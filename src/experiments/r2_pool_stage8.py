"""Execute the final PipeTwin hypothesis batch on the frozen benchmarks."""
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
from src.experiments.r2_pool_stage1_strong import freeze_checkout, mapped_labels
from src.experiments.r2_pool_stage2 import PEER, read_baseline
from src.experiments.r2_pool_stage3 import clipped, outer_train, peer_grids
from src.experiments.r2_pool_stage4 import (
    KEY_DTYPES, LOSO, RETRO, assert_replay, keys_match, prior_point_summaries, s1_historical, save_results,
)
from src.experiments.r2_pool_stage6 import read_comparator
from src.experiments.r2_pool_stage7 import blend, replay_dependency
from src.experiments.r2_pool_stage8_models import (
    CALIBRATION, GROUND, REPLAY, ROBUST, SCENARIOS, SOFT, centered_temperature, early_thermal_features,
    fit_shape_calibration, select_by_inner_r_squared, shape_bounds, shape_calibration, thermal_name,
)
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.next_common import json_safe, predict_ridge, versions, write_csv
from src.ml.s1_contract import RIDGE_FEATURES, S1_CUTOFF, S1_ID, S2_ID
from src.ml.s1_model import fit_s1_ridge, predict_s1_ridge
from src.ml.s2_next import fit_spatial_residual
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s2 import (
    SPATIAL_FEATURES, S2_CUTOFF, _apply_candidate, _fit_physics, _predict_physics, prepare_full_grid,
)
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_08.json')
THERMAL = 'selected_thermal_length'
DEFAULT_THERMAL = 'default_thermal_length'
ABLATED = 'selected_feature_ablation'


def checked_inner_baseline(source, outer_column, outer_id, train):
    """Verify archived OOF provenance against the reconstructed outer train."""
    part = source.loc[source[outer_column].astype(str).eq(str(outer_id))].copy()
    keys_match(train, part, KEY)
    truth = train[KEY + ['next_mm']].merge(part[KEY + ['next_mm']], on=KEY, validate='one_to_one')
    np.testing.assert_array_equal(truth.next_mm_x, truth.next_mm_y)
    for pid, held in part.groupby('id', sort=True):
        fit = train.loc[train.id.ne(pid)]
        actual = held.inner_training_ids.drop_duplicates()
        if len(actual) != 1 or set(map(int, actual.iloc[0].split(','))) != set(fit.id):
            raise ValueError('Archived inner baseline has different training section membership')
        if not held.inner_training_rows.eq(len(fit)).all():
            raise ValueError('Archived inner baseline has different training row count')
    if not np.isfinite(part.baseline).all():
        raise ValueError('Missing archived inner baseline prediction')
    return part


def s1_ablations(train, extensions, groups):
    geometry_and_distribution = extensions['geometry_distribution_features']
    family = {'geometry': geometry_and_distribution[:2],
              'distribution': geometry_and_distribution[2:], 'type': extensions['type_features']}
    linear = fit_s1_ridge(train)
    fits = {name: fit_s1_ridge(train, feature_order=RIDGE_FEATURES
                              + [field for group in group_names for field in family[group]])
            for name, group_names in groups.items() if group_names}
    return linear, fits


def s1_ablation_predictions(frame, base, direct, models):
    linear, fits = models
    original = predict_s1_ridge(linear, frame).ridge_delta_mm
    output = {'no_extension': np.asarray(base, dtype=float)}
    for name, model in fits.items():
        correction = predict_s1_ridge(model, frame).ridge_delta_mm - original
        output[name] = clipped(base + np.where(direct, correction, 0.), frame)
    return output


def s1_select_ablations(train, extensions, groups, minimum):
    parts = []
    for pid, held in train.groupby('id', sort=True):
        models = s1_ablations(train.loc[train.id.ne(pid)], extensions, groups)
        variants = s1_ablation_predictions(held, held.proxy, held.has_own_prior_vtd, models)
        parts.append(held[KEY + ['next_mm']].assign(**variants))
    predictions = pd.concat(parts, ignore_index=True)
    name, audit = select_by_inner_r_squared(predictions, groups, minimum, 'no_extension')
    return name, audit, predictions


def raw_joint_prediction(full):
    minimum, percentile = full.joint_min.to_numpy(float), full.joint_p05.to_numpy(float)
    if percentile.std() > 0:
        percentile = minimum.mean() + (percentile - percentile.mean()) * minimum.std() / percentile.std()
    else:
        percentile = minimum
    return (minimum + percentile) / 2.


def replay_s1(data, path, spec, protocol, spec4, oof, output):
    setup, minimum = protocol['hypotheses'], protocol['inner_minimum_rows']
    raw = data.read('data/cleaned/vtd')
    raw = raw.loc[raw.id.isin([S1_ID, *joint.PEERS])]
    grids = data.read('processed/segments_grid')
    support, extension = (spec4['hypotheses'][name] for name in ('H17', 'H18'))
    history = s1_historical(data, raw, grids, support)
    groups = setup['H39']['s1_groups']
    split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
    labels = mapped_labels(data.read('processed/vtd_by_segment'))
    labels = labels.loc[labels.id.isin(joint.PEERS) & labels.prediction_date.eq(S1_CUTOFF)]
    keys_match(split, labels, KEY)
    base = read_baseline(path, spec)
    keys_match(base, split.loc[split.supported], KEY)
    anchor_specs = {f"{row['id']}:{row['fold']}": row
                    for row in json.loads((path / 'anchor_folds.json').read_bytes())}
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    cache, audits, inner_parts, parts = {}, {}, [], []
    for pid, full, held in peer_grids(data, raw, grids, path):
        if pid not in cache:
            train = outer_train(history, pid, S1_CUTOFF)
            valid_oof = checked_inner_baseline(oof, 'outer_held_id', pid, train)
            model, calibration = fit_shape_calibration(valid_oof, setup['H37']['quantile_bins'], minimum)
            selected, selection, inner = s1_select_ablations(train, extension, groups, minimum)
            ablations = s1_ablations(train, extension, groups)
            cache[pid] = model, selected, ablations
            audits[pid] = {'H37': calibration, 'H39': selection, 'train_rows': len(train),
                           'train_profiles': sorted(int(value) for value in train.id.unique())}
            inner_parts.append(inner.assign(outer_held_id=pid))
        model, selected, ablations = cache[pid]
        raw_pred = raw_joint_prediction(full)
        bounded = full.joint_blend.to_numpy(float)
        nominal = full.nominal_thickness_mm.to_numpy(float)
        np.testing.assert_allclose(np.clip(raw_pred, 0., nominal), bounded, atol=1e-10, rtol=0.)
        fold = anchor_specs[full.fold.iloc[0]]
        refs = joint.heldout_anchors(anchors.loc[anchors.id.eq(pid)],
                                    fold['lo_with_buffer_m'], fold['hi_with_buffer_m'])
        stats = prior_point_summaries(raw, full, S1_CUTOFF, joint.PRIOR_DATE, support, refs)
        measured = full.merge(stats, on=joint.KEYS, validate='one_to_one')
        direct = full.aligned_measurement.to_numpy(bool)
        np.testing.assert_array_equal(direct, measured.prior_direct_mm.notna())
        features = s1_ablation_predictions(measured, bounded, direct, ablations)
        values = {
            CALIBRATION: shape_calibration(bounded, full.length_m, nominal, model, setup['H37']['shape_blend']),
            **shape_bounds(raw_pred, bounded, nominal, setup['H38']), **features, ABLATED: features[selected],
        }
        part = full[KEY + ['fold']].assign(prediction=bounded, **values)
        parts.append(part.loc[held])
    pred = pd.concat(parts, ignore_index=True).merge(
        labels, on=KEY, how='left', validate='one_to_one').astype(KEY_DTYPES)
    keys_match(split.loc[split.supported], pred, KEY + ['fold'])
    write_csv(split, output / PEER / 'split_keys.csv')
    write_csv(pd.concat(inner_parts), output / PEER / 'feature_selection_inner_oof.csv', float_format='%.15g')
    method_names = {'H37': [CALIBRATION], 'H38': [ROBUST, SOFT], 'H39': [ABLATED, *groups]}
    return PEER, base, pred, audits, method_names


def prepare_thermal_grids(prepared, setup, output):
    variants = setup['H36']['conductance_w_m2_k']
    target_key = (S2_ID, S2_CUTOFF)
    source = {**prepared['full_frames'], target_key: prepared['target']}
    grids, audits = {}, []
    for key, full in sorted(source.items()):
        if full[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
            raise ValueError('S2 shape features may not use inspection inputs')
        if (full.hydraulic_source_date_max.notna() & full.hydraulic_source_date_max.ge(full.prediction_date)).any():
            raise ValueError('S2 training or inference received future hydraulic rows')
        grid = terrain_features(full)
        temperatures = {thermal_name(value): centered_temperature(grid, prepared['params'], value)
                        for value in variants}
        history, audit = early_thermal_features(grid, prepared['interval_map'][key],
                                                prepared['params'], setup['H40'])
        grids[key] = grid.assign(**temperatures, **history)
        audits.append({'id': int(key[0]), 'prediction_date': str(key[1].date()), 'n': len(grid),
                       **audit, **{f'{name}_std_c': float(np.std(profile))
                                   for name, profile in temperatures.items()}})
    write_csv(pd.DataFrame(audits), output / 'thermal_scenario_input_audit.csv', float_format='%.15g')
    return grids


def s2_models(train, intervals, features, groups, thermal, scenario):
    physics = _fit_physics(train, intervals)
    level = _predict_physics(physics, train, intervals)
    layouts = {name: [field for field in features if field.removesuffix('_rank') not in removed]
               for name, removed in groups.items()}
    layouts.update({name: features + [name] for name in thermal + scenario})
    return physics, {name: fit_spatial_residual(train, level, order) for name, order in layouts.items()}


def raw_s2_predictions(full, intervals, models, optional):
    physics, candidates = models
    level = _predict_physics(physics, full, intervals)
    base = _apply_candidate(level, predict_ridge(candidates['full_spatial_features'], full), full, True)
    values = {}
    for name, model in candidates.items():
        if name in optional and not full[name].notna().all():
            values[name] = base['h_clipped']
        else:
            values[name] = _apply_candidate(level, predict_ridge(model, full), full, True)['h_clipped']
    return base, values


def s2_inner(train, grids, intervals, features, groups, thermal):
    parts = []
    for pid, held in train.groupby('id', sort=True):
        models = s2_models(train.loc[train.id.ne(pid)], intervals, features, groups, thermal, [])
        for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
            full = grids[key]
            _, predictions = raw_s2_predictions(full, intervals, models, thermal)
            part = labels[KEY + ['next_mm']].merge(full[KEY].assign(**predictions), on=KEY,
                                                   how='left', validate='one_to_one')
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def scenario_difference(full, key, a, b):
    affine = np.column_stack([np.ones(len(a)), a])
    projected = affine @ np.linalg.lstsq(affine, b, rcond=None)[0]
    return {'id': int(key[0]), 'prediction_date': str(key[1].date()), 'n_grid': len(full),
            'has_thermal_history': bool(full[GROUND].notna().all()),
            'max_abs_difference_mm': float(np.max(np.abs(a - b))),
            'max_abs_affine_residual_mm': float(np.max(np.abs(projected - b)))}


def replay_s2(prepared, archive, specs, protocol, oof_by_scheme, output):
    setup, minimum = protocol['hypotheses'], protocol['inner_minimum_rows']
    strict, post = prepared['strict'], prepared['post']
    if (not strict.prediction_date.lt(S2_CUTOFF).all() or not post.prediction_date.gt(S2_CUTOFF).all()
            or prepared['measured'].id.eq(S2_ID).any()):
        raise ValueError('S2 historical training boundary or target exclusion failed')
    intervals = prepared['interval_map']
    grids = prepare_thermal_grids(prepared, setup, output)
    thermal = [thermal_name(value) for value in setup['H36']['conductance_w_m2_k']]
    scenarios = setup['H40']['controls']
    ranks = VARIANTS['basin_ranks']
    columns = thermal + scenarios + ranks
    lookup = pd.concat([full[KEY + columns] for full in grids.values()], ignore_index=True)
    strict = strict.merge(lookup, on=KEY, how='left', validate='one_to_one')
    if strict[columns].isna().any().any():
        raise ValueError('Incomplete thermal or terrain feature in frozen training set')
    groups = setup['H39']['s2_drop_groups']
    variants = {'H36': [THERMAL, DEFAULT_THERMAL, *thermal],
                'H37': [CALIBRATION], 'H38': [ROBUST, SOFT],
                'H39': [ABLATED, *groups], 'H40': [SCENARIOS, *scenarios]}
    for scheme in (RETRO, LOSO):
        path, spec = archive / scheme, specs[scheme]
        base = read_baseline(path, spec)
        split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        features = SPATIAL_FEATURES if scheme == RETRO else ranks
        folds = ([(None, strict, post)] if scheme == RETRO else
                 [(int(pid), strict.loc[strict.id.ne(pid)], held) for pid, held in strict.groupby('id', sort=True)])
        parts, splits, audits, inner_parts, differences = [], [], {}, [], []
        for fold, train, held in folds:
            membership = pd.concat([train[KEY].assign(role='train'), held[KEY].assign(role='test')])
            splits.append(membership if fold is None else membership.assign(fold=fold))
            name = str(fold) if fold is not None else 'strict_before_2022_07_01'
            nested = s2_inner(train, grids, intervals, features, groups, thermal)
            method, feature_audit = select_by_inner_r_squared(nested, groups, minimum, 'full_spatial_features')
            conductance, thermal_audit = select_by_inner_r_squared(nested, thermal, minimum, thermal[0])
            calibrated = checked_inner_baseline(oof_by_scheme[scheme], 'outer_fold', name, train)
            model, calibration = fit_shape_calibration(calibrated, setup['H37']['quantile_bins'], minimum)
            fits = s2_models(train, intervals, features, groups, thermal, scenarios)
            audits[name] = {'H36': thermal_audit, 'H37': calibration, 'H39': feature_audit,
                            'fit': {n: value['metadata'] for n, value in fits[1].items()},
                            'train_rows': len(train), 'test_rows': len(held)}
            inner_parts.append(nested.assign(outer_fold=name))
            for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
                full = grids[key]
                control, results = raw_s2_predictions(full, intervals, fits, thermal + scenarios)
                bounded, raw = control['h_clipped'], control['h_raw']
                nominal = full.nominal_thickness_mm.to_numpy(float)
                values = {**results, THERMAL: results[conductance], DEFAULT_THERMAL: results[thermal[0]],
                          CALIBRATION: shape_calibration(bounded, full.length_m, nominal, model,
                                                         setup['H37']['shape_blend']),
                          **shape_bounds(raw, bounded, nominal, setup['H38']), ABLATED: results[method],
                          SCENARIOS: blend(results[GROUND], results[REPLAY], setup['H40']['early_replay_weight'])}
                predictions = full[KEY].assign(prediction=bounded, **values)
                parts.append(labels[KEY + ['next_mm']].merge(predictions, on=KEY, how='left', validate='one_to_one'))
                differences.append(scenario_difference(full, key, results[GROUND], results[REPLAY]))
            if scheme == RETRO:
                key = (S2_ID, S2_CUTOFF)
                full = grids[key]
                _, values = raw_s2_predictions(full, intervals, fits, thermal + scenarios)
                differences.append({**scenario_difference(full, key, values[GROUND], values[REPLAY]),
                                    'qualification': 'target_shape_audit_without_labels'})
        membership = pd.concat(splits, ignore_index=True)
        keys_match(split, membership, KEY + ['role'] + (['fold'] if scheme == LOSO else []))
        write_csv(membership, output / scheme / 'split_keys.csv')
        write_csv(pd.concat(inner_parts), output / scheme / 'selection_inner_oof.csv', float_format='%.15g')
        write_csv(pd.DataFrame(differences), output / scheme / 'scenario_shape_diagnostics.csv', float_format='%.15g')
        yield scheme, base, pd.concat(parts, ignore_index=True).astype(KEY_DTYPES), audits, variants


def run(protocol_path, output):
    tick = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    baseline_protocol = Path(protocol['baseline_protocol'])
    base_contract = json.loads(baseline_protocol.read_bytes())
    specs, tolerance = base_contract['schemes'], base_contract['archive_tolerance_mm']
    deps = {name: Path(path) for name, path in protocol['dependencies'].items()}
    spec4 = json.loads(deps['batch_04_config'].read_bytes())
    archive = Path(protocol['baseline_root'])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = freeze_checkout(output)
    write_json(output / 'source_snapshot.json', snapshot)
    data = RepositoryData(Path(base_contract['build']))
    paths = {scheme: deps['batch_07'] / scheme / 'ensemble_inner_predictions.csv'
             for scheme in (PEER, RETRO, LOSO)}
    oof = {scheme: pd.read_csv(path, parse_dates=['prediction_date'], dtype={'inner_training_ids': str})
           for scheme, path in paths.items()}
    prepared = prepare_full_grid(data)
    runs = [replay_s1(data, archive / PEER, specs[PEER], protocol, spec4, oof[PEER], output),
            *replay_s2(prepared, archive, specs, protocol, oof, output)]
    parents = data.parent_groups()
    results, references, replays, audits, dependency_replays = {}, {}, {}, {}, {}
    files = [baseline_protocol, Path(__file__), Path(__file__).with_name('r2_pool_stage8_models.py'),
             *(value for value in deps.values() if value.is_file()), *paths.values()]
    for scheme, base, pred, audit, variants in runs:
        error = assert_replay(base, pred, tolerance)
        reference = protocol['reference_candidates'][scheme]
        ref_file = Path(reference['predictions'])
        _, r2 = read_comparator(ref_file, base, tolerance)
        references[scheme] = {**reference, 'r2': r2}
        files.extend([ref_file, *((archive / scheme).glob('*'))])
        if scheme == PEER:
            h12_pair, h12_r2 = read_comparator(deps['H12_S1'], base, tolerance)
            assert len(h12_pair) == len(base)
            dependency_replays[scheme] = {'H12_r2': h12_r2,
                                        'H18': replay_dependency(deps['H18_S1'], base, pred, 'full_extension', tolerance)}
        else:
            ref_name = 'H20_S2_retro' if scheme == RETRO else 'H20_S2_loso'
            dependency_replays[scheme] = {'H20': replay_dependency(deps[ref_name], base, pred,
                                                                  DEFAULT_THERMAL, tolerance)}
        results[scheme] = save_results(output, scheme, base, pred, variants, specs[scheme],
                                       references[scheme], int(base.id.map(parents).nunique()))
        if scheme == PEER:
            for h, methods in results[scheme].items():
                for variant, summary in methods.items():
                    value = summary['candidate']['pearson_squared']
                    summary['delta_vs_H12'] = None if value is None else value - h12_r2
                    write_json(output / 'hypotheses' / h / scheme / variant / 'summary.json', summary)
        audits[scheme] = audit
        replays[scheme] = {'rows': len(base), 'max_abs_prediction_difference_mm': error,
                           'same_keys_labels_and_splits': True}
        print(f'{scheme}: {len(base)} frozen keys, max replay difference {error:.3g} mm', flush=True)
        for h, methods in results[scheme].items():
            name = protocol['hypotheses'][h]['primary']
            print(f'{h} {scheme}: {methods[name]["candidate"]["pearson_squared"]}', flush=True)
    for h, config in protocol['hypotheses'].items():
        folder = output / 'hypotheses' / h
        write_json(folder / 'config.json', config)
        write_json(folder / 'summary.json', {'primary': config['primary'],
                   'schemes': {name: values[h] for name, values in results.items() if h in values}})
    write_json(output / 'fit_audit.json', json_safe(audits))
    write_json(output / 'stage_results.json', {'results': results, 'references': references,
                                              'replays': replays, 'dependency_replays': dependency_replays})
    write_json(output / 'run_manifest.json', {
        'protocol_version': protocol['protocol_version'], 'protocol_sha256': sha256(protocol_path),
        'code_sha256': sha256(Path(__file__)), 'source_snapshot_sha256': sha256(output / 'source_snapshot.json'),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'data_root': data.root.as_posix(), 'data_sha256': data.hashes,
        'input_sha256': {path.as_posix(): sha256(path) for path in files},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - tick,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage8 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'linear_algebra_threads': 1, 'status': 'completed', 'replays': replays,
        'heat_loss_assumptions': prepared['params'].assumptions(),
        'candidate_scope': protocol['candidate_scope'],
        'exclusions': ['target S2 diagnostic data', 'outer labels for all fitting and selection',
                       'unobserved gaps other than the declared H40 leading-gap scenarios']
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.protocol, args.output)

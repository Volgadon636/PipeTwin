"""Evaluate independent S1 variants H11–H15 on the frozen stage 1 benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage1 import KEY
from src.experiments.r2_pool_stage1_strong import freeze_checkout
from src.experiments.r2_pool_stage2 import PEER, compare, read_baseline
from src.grid_builder import assign_segments
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.next_common import (
    fit_ridge, json_safe, predict_ridge, s1_fit_rows, s1_training_table, versions, write_csv,
)
from src.ml.s1_alignment import predict_section, section_snapshot
from src.ml.s1_contract import RIDGE_FEATURES, S1_CUTOFF, enrich_s1, production_frame
from src.ml.s1_model import fit_s1_ridge, predict_s1_ridge
from src.ml.windows import make_snapshot
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_03.json')
KS = joint.KEYS


def clipped(values, frame):
    return np.clip(np.asarray(values, dtype=float), 0., frame.nominal_thickness_mm.to_numpy(float))


def axis(frame):
    return ((frame.start_m + frame.end_m) / 2).to_numpy(float)


def mix_profiles(minimum, p05, frame):
    q_std = np.std(p05)
    scaled = (np.mean(minimum) + (p05 - np.mean(p05)) * np.std(minimum) / q_std
              if q_std > 0 else minimum)
    return clipped((minimum + scaled) / 2, frame)


def prior_profile(raw, section, snapshot, cutoff, source_date):
    if not source_date < cutoff:
        raise ValueError('Profile source must precede prediction date')
    prior = raw[raw.id.eq(section.id.iloc[0]) & raw.date.eq(source_date)
                & raw.target_eligible & raw.residual_for_target_mm.notna()
                & raw.distance.between(0, section.end_m.max())]
    grouped = assign_segments(prior, 'distance', section).groupby(KS).residual_for_target_mm
    frame = snapshot.copy()
    direct = grouped.min().reindex(pd.MultiIndex.from_frame(frame[KS])).to_numpy(float)
    quantile = grouped.quantile(0.05).reindex(pd.MultiIndex.from_frame(frame[KS])).to_numpy(float)
    fallback = frame.h0_mm.to_numpy(float)
    return frame.assign(last_mm=np.where(np.isnan(direct), fallback, direct),
                        direct_min=direct,
                        proxy=mix_profiles(np.where(np.isnan(direct), fallback, direct),
                                           np.where(np.isnan(quantile), fallback, quantile), frame))


def historical_training(data, raw, grids, radii):
    fit = s1_fit_rows(s1_training_table(data))
    parts = []
    for (pid, date), labels in fit.groupby(['id', 'prediction_date'], sort=True):
        sources = labels.last_vtd_date.drop_duplicates()
        if len(sources) != 1 or not sources.lt(date).all():
            raise ValueError('Expected one strictly earlier historical VTD source per section')
        section = grids[grids.id.eq(pid)]
        snap, _ = make_snapshot(data, {'id': int(pid), 'prediction_date': str(date.date()),
                                       'case': 'between_vtd'})
        prior = prior_profile(raw, section, snap, date, sources.iloc[0])
        columns = window_features(prior, prior.direct_min, radii)
        prior = prior.assign(**columns)
        part = labels.merge(prior[KS + ['proxy', *columns]], on=KS, how='left', validate='one_to_one')
        if part.proxy.isna().any():
            raise ValueError('Historical baseline is missing an eligible training key')
        parts.append(part.assign(proxy_residual=part.next_mm - part.proxy))
    return pd.concat(parts, ignore_index=True)


def peer_grids(data, raw, grids, archive):
    mapped = data.read('processed/vtd_by_segment')
    terrain = data.read('processed/profile_by_segment')
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    keys = pd.read_csv(archive / 'split_keys.csv', parse_dates=['prediction_date'])
    specs = json.loads((archive / 'anchor_folds.json').read_bytes())
    for pid in joint.PEERS:
        grid = grids[grids.id.eq(pid)].sort_values('segment_id')
        snap = section_snapshot(data, pid)
        fallback, _ = predict_section(snap, raw, grid, S1_CUTOFF, 0)
        enriched = enrich_s1(snap, mapped, terrain, S1_CUTOFF)
        for spec in specs:
            if spec['id'] != pid or not spec['supported']:
                continue
            fold = f"{pid}:{spec['fold']}"
            refs = joint.heldout_anchors(anchors[anchors.id.eq(pid)], spec['lo_with_buffer_m'],
                                         spec['hi_with_buffer_m'])
            if len(refs) != spec['anchors_used']:
                raise ValueError(f'Fold anchors changed: {fold}')
            values, _ = joint.predict_aligned(raw, grid, fallback, refs)
            columns = ['joint_min', 'joint_p05', 'joint_blend', 'aligned_measurement']
            full = enriched.merge(values[KS + columns], on=KS, validate='one_to_one')
            full = full.sort_values('segment_id').reset_index(drop=True).assign(fold=fold)
            held = keys.loc[keys.fold.eq(fold) & keys.supported, 'segment_id']
            yield pid, full, full.segment_id.isin(held)


def outer_train(fit, held_id, cutoff):
    train = fit[fit.id.ne(held_id) & fit.prediction_date.lt(cutoff)]
    if train.id.nunique() < 2 or not train.last_vtd_date.lt(train.prediction_date).all():
        raise ValueError('Expected at least two other sections with strictly earlier train labels')
    return train


def select_lambda(train, candidates):
    folds = {value: [] for value in candidates}
    for pid, held in train.groupby('id', sort=True):
        model = fit_s1_ridge(train[train.id.ne(pid)])
        delta = predict_s1_ridge(model, held).ridge_delta_mm
        for value in candidates:
            pred = clipped(held.s1b_prediction + value * delta, held)
            folds[value].append(metrics(held.next_mm, pred)['pearson_squared'] if len(held) >= 3 else None)
    scores = {value: None if any(v is None for v in values) else float(np.mean(values))
              for value, values in folds.items()}
    valid = [value for value in candidates if scores[value] is not None]
    if not valid:
        raise ValueError('No defined inner score for lambda selection')
    return max(valid, key=lambda value: (scores[value], -value)), scores, folds


def window_features(frame, values, radii):
    if frame.id.nunique() != 1:
        raise ValueError('Neighborhoods require a single section')
    x, values = axis(frame), np.asarray(values, dtype=float)
    distance = np.abs(x[:, None] - x[None, :])
    available = np.isfinite(values)[None, :] & (distance > 0.)
    output = {}
    for radius in radii:
        keep = available & (distance <= radius + 1e-9)
        minimum = np.min(np.where(keep, values, np.inf), axis=1)
        output[f'prior_min_{radius}m'] = np.where(np.isfinite(minimum), minimum, np.nan)
        output[f'prior_count_{radius}m'] = keep.sum(axis=1)
    return output


def impute_at(x, values, index, config):
    """Reconstruct one missing value using only the remaining measured neighbors."""
    if np.isfinite(values[index]):
        raise ValueError('Imputation target must be masked')
    available = np.flatnonzero(np.isfinite(values))
    distances = np.abs(x[available] - x[index])
    if not len(available):
        raise ValueError('Imputation requires at least one remaining donor')
    nearest = values[available[np.argmin(distances)]]
    local = available[distances <= config['local_radius_m'] + 1e-9]
    left, right = local[local < index], local[local > index]
    interpolation = linear = nearest
    if len(left) and len(right):
        bracket = [left[-1], right[0]]
        interpolation = np.interp(x[index], x[bracket], values[bracket])
        if len(local) >= config['minimum_regression_neighbors']:
            linear = np.linalg.lstsq(
                np.column_stack([np.ones(len(local)), x[local] - x[index]]),
                values[local], rcond=None)[0][0]
    return {'nearest': nearest, 'median': np.median(values[local]) if len(local) else nearest,
            'interpolation': interpolation, 'local_linear': linear}


def target_reconstruction(data, config, output):
    target, _ = production_frame(data)
    x, measured = axis(target), target.own_last_thickness_mm.to_numpy(float)
    methods = config['methods']
    masked, actual = [], []
    for idx, value in enumerate(measured):
        sample = measured.copy()
        sample[idx] = np.nan
        if np.isfinite(value):
            if idx == 0 or idx == len(measured) - 1:
                continue
            destination = masked
        else:
            destination = actual
        guesses = impute_at(x, sample, idx, config)
        upper = float(target.nominal_thickness_mm.iloc[idx])
        destination.append({
            'segment_id': int(target.segment_id.iloc[idx]), 'truth_mm': value,
            'production_fallback_mm': float(target.s1a_prediction.iloc[idx]),
            **{method: float(np.clip(guesses[method], 0., upper)) for method in methods}})
    reconstructed = pd.DataFrame(masked)
    write_csv(reconstructed, output / 'masked_past_measurements.csv')
    write_csv(pd.DataFrame(actual), output / 'four_unmeasured_target_segments.csv')
    return {'masked_measurements': len(masked), 'actual_unmeasured_segments': len(actual),
            'reconstruction': {method: metrics(reconstructed.truth_mm, reconstructed[method])
                               for method in methods}}


def masked_forecast(frame, held, config):
    x = axis(frame)
    direct = frame.aligned_measurement.to_numpy(bool)
    rows = []
    for idx in np.flatnonzero(held.to_numpy() & direct):
        minimum = frame.joint_min.where(direct).to_numpy(float).copy()
        p05 = frame.joint_p05.where(direct).to_numpy(float).copy()
        minimum[idx] = p05[idx] = np.nan
        guesses = impute_at(x, minimum, idx, config)
        # Both arms share the same donors and the same fallback for other gaps.
        other = [j for j in np.flatnonzero(~np.isfinite(minimum)) if j != idx]
        common = {j: impute_at(x, minimum, j, config)['nearest'] for j in other}
        for j, value in common.items():
            minimum[j] = p05[j] = value
        record = frame.loc[idx, KEY].to_dict()
        for method in config['methods']:
            minimum[idx] = p05[idx] = np.clip(guesses[method], 0., frame.nominal_thickness_mm.iloc[idx])
            record[method] = mix_profiles(minimum, p05, frame)[idx]
        rows.append(record)
    return rows


def guarded_smooth(frame, radius, weight):
    x = axis(frame)
    z = frame.joint_blend.to_numpy(float)
    direct = frame.aligned_measurement.to_numpy(bool)
    distance = np.abs(x[:, None] - x[None, :])
    neighbor = (distance <= radius + 1e-9) & direct[None, :]
    local = pd.DataFrame(np.where(neighbor, z, np.nan)).median(axis=1).to_numpy(float)
    proposal = np.where(direct, z + weight * (local - z), z)
    immediate = (distance <= 10. + 1e-9) & (distance > 0.) & direct[None, :]
    left = immediate & (x[None, :] < x[:, None])
    right = immediate & (x[None, :] > x[:, None])
    trough = (direct & left.any(axis=1) & right.any(axis=1)
              & (z < np.min(np.where(immediate, z, np.inf), axis=1)))
    return clipped(np.where(trough, np.minimum(z, proposal), proposal), frame), trough


def variants(setup):
    return {
        'H11': 'spatial_residual',
        'H12': 'selected_lambda',
        'H13': 'natural_missingness',
        'H14': f"radius_{setup['H14']['primary_radius_m']}m",
        'H15': setup['H15']['primary'],
    }


def run(protocol_path, output):
    tick = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    baseline_protocol = Path(protocol['baseline_protocol'])
    baseline = json.loads(baseline_protocol.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source_snapshot.json', freeze_checkout(output))
    data = RepositoryData(Path(baseline['build']))
    raw = data.read('data/cleaned/vtd')
    grids = data.read('processed/segments_grid')
    setup = protocol['hypotheses']
    radii = setup['H15']['radii_m']
    archive = Path(protocol['baseline_root'])
    spec = baseline['schemes'][PEER]
    base = read_baseline(archive / PEER, spec)
    historical = historical_training(data, raw, grids, radii)
    pools, masked, metadata, train_keys = [], [], {}, []
    for pid, full, held in peer_grids(data, raw, grids, archive / PEER):
        if pid not in metadata:
            train = outer_train(historical, pid, S1_CUTOFF)
            train_keys.append(train[KEY + ['last_vtd_date']].assign(held_id=pid))
            cfg = setup['H11']
            residual = fit_ridge(train, cfg['features'], 'proxy_residual', np.ones(len(train)),
                                 cfg['ridge_alpha'], model_name='H11')
            s1c = fit_s1_ridge(train)
            weight, inner, folds = select_lambda(train, setup['H12']['lambdas'])
            window_sets = {f'radius_{r}m': [f'prior_min_{r}m', f'prior_count_{r}m'] for r in radii}
            window_sets['multiscale'] = sum(window_sets.values(), [])
            extensions = {name: fit_s1_ridge(train, feature_order=RIDGE_FEATURES + features)
                          for name, features in window_sets.items()}
            metadata[pid] = (residual, s1c, weight, inner, folds, extensions)
        residual, s1c, weight, inner, folds, extensions = metadata[pid]
        frame = full[KEY + ['fold', 'joint_blend', 'aligned_measurement']].copy()
        direct = full.aligned_measurement.to_numpy(bool)
        correction = predict_ridge(residual, full.assign(last_mm=full.joint_min))
        center = correction[direct].mean() if direct.any() else 0.
        shape = correction - center
        frame['H11_spatial_residual'] = clipped(full.joint_blend + np.where(direct, shape, 0.), full)
        delta = predict_s1_ridge(s1c, full).ridge_delta_mm
        frame['H12_selected_lambda'] = clipped(full.joint_blend + weight * delta, full)
        frame['H12_lambda_1'] = clipped(full.joint_blend + delta, full)
        frame['H12_lambda_0'] = full.joint_blend
        minimum = full.joint_min.where(direct).to_numpy(float)
        natural = np.flatnonzero(~direct)
        reconstructed = minimum.copy()
        for idx in natural:
            reconstructed[idx] = impute_at(axis(full), minimum, idx, setup['H13'])[setup['H13']['primary']]
        frame['H13_natural_missingness'] = np.where(
            direct, full.joint_blend, clipped(reconstructed, full))
        masked.extend(masked_forecast(full, held, setup['H13']))
        for radius in setup['H14']['radii_m']:
            smooth, trough = guarded_smooth(full, radius, setup['H14']['smooth_weight'])
            frame[f'H14_radius_{radius}m'] = smooth
            frame[f'trough_{radius}m'] = trough
        enriched = full.assign(**window_features(full, minimum, radii))
        for name, model in extensions.items():
            extra = predict_s1_ridge(model, enriched).ridge_delta_mm - delta
            frame[f'H15_{name}'] = clipped(full.joint_blend + np.where(direct, extra, 0.), full)
        pools.append(frame.loc[held])
    joined = pd.concat(pools, ignore_index=True)
    paired = base.merge(joined, on=KEY, how='outer', validate='one_to_one')
    np.testing.assert_allclose(paired.prediction, paired.joint_blend,
                               atol=baseline['archive_tolerance_mm'], rtol=0., equal_nan=False)
    base = paired[KEY + ['next_mm', 'joint_blend']].rename(columns={'joint_blend': 'prediction'})
    write_csv(pd.concat(train_keys), output / 'training_keys.csv')
    write_csv(joined, output / 'fold_predictions.csv', float_format='%.15g')

    primary = variants(setup)
    results = {key: {} for key in primary}
    for column in joined:
        name, _, variant = column.partition('_')
        if name in primary:
            path = output / 'hypotheses' / name / PEER / variant
            pred = paired[KEY + [column]].rename(columns={column: 'prediction'})
            results[name][variant] = compare(base, pred, spec, path)
    h12 = results['H12']
    h12['delta_vs_lambda_1'] = (h12['selected_lambda']['candidate']['pearson_squared']
                               - h12['lambda_1']['candidate']['pearson_squared'])
    audits = {
        'H11': {str(pid): entry[0]['metadata'] for pid, entry in metadata.items()},
        'H12': {str(pid): {'selected_lambda': entry[2], 'inner_mean_r2': entry[3],
                          'inner_r2_by_profile': entry[4], 'model': entry[1]['metadata']}
                for pid, entry in metadata.items()},
        'H13': target_reconstruction(data, setup['H13'], output / 'hypotheses/H13'),
        'H14': {col: int(joined[col].sum()) for col in joined if col.startswith('trough_')},
        'H15': {str(pid): {name: model['metadata'] for name, model in entry[5].items()}
                for pid, entry in metadata.items()},
    }
    masks = base[KEY + ['next_mm']].merge(pd.DataFrame(masked), on=KEY, validate='one_to_one')
    if len(masks) != int(joined.aligned_measurement.sum()):
        raise ValueError('Missing or duplicate masked control forecasts')
    write_csv(masks, output / 'hypotheses/H13/masked_forecast_predictions.csv', float_format='%.15g')
    mask_spec = {**spec, 'n': len(masks), 'r2': metrics(masks.next_mm, masks.nearest)['pearson_squared']}
    audits['H13'].update(natural_gaps=int((~joined.aligned_measurement).sum()), masked_forecast={})
    for method in setup['H13']['methods'][1:]:
        audits['H13']['masked_forecast'][method] = compare(
            masks[KEY + ['next_mm', 'nearest']].rename(columns={'nearest': 'prediction'}),
            masks[KEY + [method]].rename(columns={method: 'prediction'}), mask_spec,
            output / 'hypotheses/H13/masked_forecast' / method)
    for name, config in setup.items():
        path = output / 'hypotheses' / name
        write_json(path / 'config.json', config)
        write_json(path / 'summary.json', json_safe({'primary': primary[name],
                                                    'results': results[name], 'audit': audits[name]}))
        print(f"{name}: {primary[name]} delta={results[name][primary[name]]['delta_pearson_squared']}",
              flush=True)
    s2 = baseline['schemes']['S2_retro_2023']
    s2_file = archive / 'S2_retro_2023' / 'validation_predictions.csv'
    frozen_s2 = read_baseline(s2_file.parent, s2)
    s2_metrics = metrics(frozen_s2.next_mm, frozen_s2.prediction)
    np.testing.assert_allclose(s2_metrics['pearson_squared'], s2['r2'], atol=1e-8, rtol=0.)
    write_json(output / 'stage_results.json', json_safe({
        'primary': primary, 'results': results, 'audits': audits,
        'S2_frozen': {'n': len(frozen_s2), **s2_metrics},
    }))
    sources = [baseline_protocol, s2_file, *(archive / PEER).glob('*')]
    write_json(output / 'run_manifest.json', {
        'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(Path(__file__)),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'input_sha256': {**data.hashes, **{path.as_posix(): sha256(path) for path in sources}},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - tick,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage3 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'status': 'completed',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)

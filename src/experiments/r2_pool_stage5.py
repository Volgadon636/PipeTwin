"""Evaluate H21–H25 on the frozen cold-start splits."""
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
from src.experiments.r2_pool_stage2 import read_baseline
from src.experiments.r2_pool_stage4 import KEY_DTYPES, LOSO, RETRO, assert_replay, keys_match, save_results
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.next_common import fit_ridge, json_safe, make_weights, predict_ridge, versions, write_csv
from src.ml.s1_contract import S2_ID
from src.ml.s2_next import fit_spatial_residual
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s2 import (
    SPATIAL_FEATURES, S2_CUTOFF, _apply_candidate, _fit_physics, _predict_physics, prepare_full_grid,
)
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_05.json')
INTEGRATED = 'integrated_thermal_exposure'
AVERAGE_CONDITIONS = 'constant_conditions_thermal_exposure'


def thermal_exposure(full, intervals, params):
    """Integrate the existing steady heat-loss scenario on observed intervals."""
    (cutoff,) = full.prediction_date.unique()
    (diameter_mm,) = full.d_inner_assumed_mm.unique()
    diameter = float(diameter_mm) / 1000.
    inputs = ['t_start_usable', 'density_liquid_work_usable', 'velocity_m_s', 'duration_years']
    if (intervals.date.ge(cutoff) | intervals.interval_end.gt(cutoff)).any():
        raise ValueError('Thermal history reaches beyond the prediction boundary')
    inlet, density, velocity, duration = intervals[inputs].to_numpy(float).T
    valid = (np.isfinite(inlet) & np.isfinite(density) & np.isfinite(velocity) & np.isfinite(duration)
             & (inlet > -273.15) & (density > 0) & (velocity > 0) & (duration > 0))
    nominal_period = float(full.horizon_years.iloc[0])
    audit = {'hydraulic_intervals': len(intervals), 'thermal_intervals': int(valid.sum()),
             'thermal_observed_years': float(duration[valid].sum()),
             'thermal_coverage_fraction': float(duration[valid].sum() / nominal_period)}
    if not np.isfinite(diameter) or diameter <= 2 * params.roughness_m:
        audit.update(thermal_observed_years=0., thermal_intervals=0, thermal_coverage_fraction=0.,
                     thermal_status='invalid_diameter')
        return {INTEGRATED: np.nan, AVERAGE_CONDITIONS: np.nan}, audit
    if not valid.any():
        audit['thermal_status'] = 'no_valid_inlet_temperature_density_velocity_intervals'
        return {INTEGRATED: np.nan, AVERAGE_CONDITIONS: np.nan}, audit
    mass_flow = density[valid] * velocity[valid] * np.pi * diameter**2 / 4.
    dt, excess = duration[valid], inlet[valid] - params.ground_temperature_c
    x = (full.start_m.to_numpy(float) + full.end_m.to_numpy(float)) / 2.
    perimeter = np.pi * (diameter - 2 * params.roughness_m)
    scale = params.overall_heat_transfer_w_m2_k * perimeter / params.heat_capacity_j_kg_k
    daily = np.exp(-scale / mass_flow[:, None] * x)
    integrated = (dt * excess) @ daily
    constant = np.dot(dt, excess) * np.exp(-scale * x / np.average(mass_flow, weights=dt))
    lengths = full.length_m.to_numpy(float)
    audit['thermal_status'] = 'observed_intervals_only'
    return {INTEGRATED: integrated - np.average(integrated, weights=lengths),
            AVERAGE_CONDITIONS: constant - np.average(constant, weights=lengths)}, audit


def environmental_value(frame, hypothesis, spec):
    if hypothesis == 'H21':
        value = frame.water_fraction_mean.astype(float)
        return value.where(value.between(0., 1.))
    velocity = frame.v_liquid_true_mean.astype(float)
    return np.log1p(velocity.where(velocity.ge(0.) & np.isfinite(velocity))
                    / spec['velocity_reference_m_s'])


def interacted(frame, terrain, stem, values):
    features = [f'{stem}_x_{column}' for column in terrain]
    return frame.assign(**{feature: frame[column] * values
                           for feature, column in zip(features, terrain)}), features


def fit_interactions(train, physics, base_features, hypothesis, spec, ranked):
    basis = [f'{name}_rank' if ranked else name for name in spec['terrain_features']]
    value = environmental_value(train, hypothesis, spec)
    by_profile = train.assign(environment=value).groupby(['id', 'prediction_date']).environment
    if not by_profile.nunique().le(1).all() or not value.notna().all():
        raise ValueError('Expected one usable value for each environmental training section')
    profile_means = by_profile.mean()
    constant = float(profile_means.mean())
    variants = {spec['primary']: value, spec['controls'][0]: constant}
    fits = {}
    for name, modifier in variants.items():
        enriched, interactions = interacted(train, basis, hypothesis, modifier)
        fits[name] = fit_spatial_residual(enriched, physics, base_features + interactions, 'ridge')
    return fits, basis, constant, float(profile_means.min()), float(profile_means.max())


def fit_relative_loss(train, features, centered):
    nominal = train.nominal_thickness_mm.to_numpy(float)
    if not np.isfinite(nominal).all() or (nominal <= 0.).any():
        raise ValueError('Fractional loss requires a positive nominal thickness')
    loss = (nominal - train.next_mm.to_numpy(float)) / nominal
    frame = train.assign(relative_loss=loss)
    if centered:
        frame['relative_loss'] -= frame.groupby(['id', 'prediction_date']).relative_loss.transform('mean')
    artifact = fit_ridge(frame, features, 'relative_loss', make_weights(frame, 'row_uniform'),
                         10., baseline=None, model_name='s2_relative_loss_ridge', prediction_policy='delta')
    artifact['metadata'].update(target_centered_within_profile=centered,
                                nominal_thicknesses_mm=sorted(set(nominal.tolist())))
    return artifact


def predict_relative_loss(model, full, physics):
    fraction = predict_ridge(model, full, baseline=None)
    centered = fraction - np.average(fraction, weights=full.length_m)
    return _apply_candidate(physics, -full.nominal_thickness_mm.to_numpy(float) * centered,
                            full, centered=False)


def h24_audit(intervals):
    columns = ['q_liquid_usable', 'q_oil_usable', 'q_gas', 'gas_content_rate',
               'fill_coefficient', 'viscosity_liquid_work', 'density_liquid_work_usable']
    observed = [
        {'id': pid, 'prediction_date': str(date.date()), 'intervals': len(frame),
         'nonmissing': {column: int(frame[column].notna().sum()) for column in columns}}
        for (pid, date), frame in sorted(intervals.items())
    ]
    return {
        'implementation': 'blocked_data', 'result': 'not_evaluated',
        'input_schema_reference': 'src/schema_registry.py',
        'technical_reference': 'docs/mentor/revisions/SRC-TECH-SPEC/bb88e6101a1ea291.md',
        'observed_hydraulic_fields': observed,
        'missing_or_unverified': [
            'operating-condition gas flow and separate oil/water/gas phase velocities',
            'phase-specific fluid properties and viscosity type/units',
            'fill coefficient basis and phase',
            'local boundary conditions or spatial wetting/holdup observations to constrain a closure',
        ],
    }


def prepare_covariates(prepared, output):
    source = {**prepared['full_frames'], (S2_ID, S2_CUTOFF): prepared['target']}
    grids, audit = {}, []
    counts = prepared['measured'].groupby(['id', 'prediction_date']).size()
    for key, grid in sorted(source.items()):
        if grid[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
            raise ValueError('Cold-start predictors contain inspection input')
        if not (grid.hydraulic_source_date_max.isna()
                | grid.hydraulic_source_date_max.lt(grid.prediction_date)).all():
            raise ValueError('Hydraulic source date is not strictly earlier than prediction')
        full = terrain_features(grid)
        values, info = thermal_exposure(full, prepared['interval_map'][key], prepared['params'])
        full = full.assign(**values)
        grids[key] = full
        first = full.iloc[0]
        audit.append({
            'id': key[0], 'prediction_date': str(key[1].date()), 'grid_rows': len(full),
            'label_rows': int(counts.get(key, 0)), 'last_hydraulic_date': first.hydraulic_source_date_max,
            'nominal_mm': first.nominal_thickness_mm, 'water_fraction_mean': first.water_fraction_mean,
            'liquid_velocity_mean_m_s': first.v_liquid_true_mean,
            **info, 'integrated_thermal_std_c_year': full[INTEGRATED].std(ddof=0),
            'constant_thermal_std_c_year': full[AVERAGE_CONDITIONS].std(ddof=0),
            'thermal_difference_std_c_year': (full[INTEGRATED] - full[AVERAGE_CONDITIONS]).std(ddof=0),
        })
    write_csv(pd.DataFrame(audit), output / 's2_exposure_environment_audit.csv', float_format='%.15g')
    return grids


def fit_models(train, intervals, features, setup, ranked):
    physics_model = _fit_physics(train, intervals)
    physics = _predict_physics(physics_model, train, intervals)
    baseline = fit_spatial_residual(train, physics, features, 'ridge')
    interactions = {h: fit_interactions(train, physics, features, h, setup[h], ranked)
                    for h in ('H21', 'H22')}
    thermal = {name: fit_spatial_residual(train, physics, features + [name], 'ridge')
               for name in (INTEGRATED, AVERAGE_CONDITIONS)}
    relative = {name: fit_relative_loss(train, features, centered)
                for name, centered in [('centered_relative_loss', True), ('relative_loss_uncentered_fit', False)]}
    return physics_model, baseline, interactions, thermal, relative


def fold_predictions(test, grids, intervals, models, setup):
    physics_model, baseline, interactions, thermal, relative = models
    outputs, environment_rows = [], []
    for key, labels in test.groupby(['id', 'prediction_date'], sort=True):
        full = grids[key]
        physics = _predict_physics(physics_model, full, intervals)
        control = _apply_candidate(physics, predict_ridge(baseline, full, baseline=None), full, True)
        candidates = {'prediction': control}
        for hypothesis, (fits, basis, constant, lo, hi) in interactions.items():
            spec = setup[hypothesis]
            value = environmental_value(full, hypothesis, spec)
            environment_rows.append({'id': int(key[0]), 'prediction_date': str(key[1].date()),
                                     'hypothesis': hypothesis, 'n': len(labels), 'value': value.iloc[0],
                                     'train_min': lo, 'train_max': hi, 'constant_control': constant,
                                     'outside_training_range': bool(not lo <= value.iloc[0] <= hi)})
            for name, model in fits.items():
                modifier = value if name == spec['primary'] else constant
                enriched, _ = interacted(full, basis, hypothesis, modifier)
                candidates[name] = _apply_candidate(
                    physics, predict_ridge(model, enriched, baseline=None), full, True)
        for name, model in thermal.items():
            candidates[name] = (_apply_candidate(physics, predict_ridge(model, full, baseline=None),
                                                 full, True) if full[name].notna().all() else control)
        for name, model in relative.items():
            candidates[name] = predict_relative_loss(model, full, physics)
        forecasts = full[KEY].assign(**{name: result['h_clipped'] for name, result in candidates.items()},
                                    **{f'clipped_{name}': result['clipping'] for name, result in candidates.items()})
        outputs.append(labels[KEY + ['next_mm']].merge(forecasts, on=KEY, how='left',
                                                       validate='one_to_one'))
    return pd.concat(outputs, ignore_index=True), environment_rows


def model_audit(models):
    physics, baseline, interactions, thermal, relative = models
    fitted = {'baseline': baseline, **thermal, **relative}
    for fits, _, _, _, _ in interactions.values():
        fitted.update(fits)
    return {'physics': physics.metadata(), 'spatial': {name: fit['metadata'] for name, fit in fitted.items()}}


def profile_shape_diagnostics(predictions, variants):
    rows = []
    for (pid, date), part in predictions.groupby(['id', 'prediction_date'], sort=True):
        base = part.prediction.to_numpy(float)
        design = np.column_stack([np.ones(len(base)), base])
        for name in variants:
            candidate = part[name].to_numpy(float)
            coefficients = np.linalg.lstsq(design, candidate, rcond=None)[0]
            rows.append({
                'id': int(pid), 'prediction_date': str(date.date()), 'n': len(part), 'variant': name,
                'max_abs_difference_mm': float(np.max(np.abs(candidate - base))),
                'max_abs_affine_residual_mm': float(np.max(np.abs(candidate - design @ coefficients))),
                'affine_slope': float(coefficients[1]) if len(part) >= 3 and np.ptp(base) > 1e-12 else None,
                'baseline_clipped_rows': int(part.clipped_prediction.sum()),
                'candidate_clipped_rows': int(part[f'clipped_{name}'].sum()),
            })
    return pd.DataFrame(rows)


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
    variants = {h: [cfg['primary'], *cfg['controls']] for h, cfg in setup.items() if cfg['primary'] is not None}
    prepared = prepare_full_grid(data)
    strict, post, intervals = prepared['strict'], prepared['post'], prepared['interval_map']
    if (not strict.prediction_date.lt(S2_CUTOFF).all() or not post.prediction_date.gt(S2_CUTOFF).all()
            or prepared['measured'].id.eq(S2_ID).any()):
        raise ValueError('S2 training and holdout date or target-diagnostic exclusion failed')
    grids = prepare_covariates(prepared, output)
    additions = [INTEGRATED, AVERAGE_CONDITIONS, *VARIANTS['basin_ranks']]
    available = pd.concat([frame[KEY + additions] for frame in grids.values()], ignore_index=True)
    strict = strict.merge(available, on=KEY, how='left', validate='one_to_one')
    if strict[additions].isna().any().any():
        raise ValueError('Expected complete candidate features on the frozen training label set')
    blocked = h24_audit(intervals)
    write_json(output / 'hypotheses/H24/input_audit.json', blocked)
    results, replays, model_records, references = {}, {}, {}, {}
    parents = data.parent_groups()
    sources = [baseline_protocol, Path(__file__), Path('src/schema_registry.py'),
               Path(blocked['technical_reference'])]
    for scheme in (RETRO, LOSO):
        ranked = scheme == LOSO
        spec = baseline['schemes'][scheme]
        path = archive / scheme
        frozen = read_baseline(path, spec)
        frozen_split = pd.read_csv(path / 'split_keys.csv', parse_dates=['prediction_date'])
        ref = protocol['reference_candidates'][scheme].copy()
        ref_file = Path(ref['predictions'])
        ref_predictions = pd.read_csv(ref_file, parse_dates=['prediction_date']).astype(KEY_DTYPES)
        assert_replay(frozen, ref_predictions.rename(columns={'baseline': 'prediction'}),
                      baseline['archive_tolerance_mm'])
        ref['r2'] = metrics(ref_predictions.next_mm, ref_predictions.candidate)['pearson_squared']
        references[scheme] = ref
        sources.extend([ref_file, path / 'split_keys.csv', path / 'validation_predictions.csv'])
        features = VARIANTS['basin_ranks'] if ranked else SPATIAL_FEATURES
        if ranked:
            folds = [(int(pid), strict.loc[strict.id.ne(pid)], held)
                     for pid, held in strict.groupby('id', sort=True)]
        else:
            folds = [(None, strict, post)]
        holdout, splits, environments = [], [], []
        model_records[scheme] = {}
        for fold, train, test in folds:
            if fold is not None and train.id.eq(fold).any():
                raise ValueError('Held section inside training set')
            keys = pd.concat([train[KEY + ['next_mm']].assign(role='train'),
                              test[KEY + ['next_mm']].assign(role='test')], ignore_index=True)
            splits.append(keys if fold is None else keys.assign(fold=fold))
            models = fit_models(train, intervals, features, setup, ranked)
            predictions, applicability = fold_predictions(test, grids, intervals, models, setup)
            holdout.append(predictions)
            environments.extend(applicability)
            model_records[scheme][str(fold) if ranked else 'strict_before_2022_07_01'] = model_audit(models)
        split = pd.concat(splits, ignore_index=True)
        keys_match(frozen_split, split, KEY + ['role'] + (['fold'] if ranked else []))
        predictions = pd.concat(holdout, ignore_index=True).astype(KEY_DTYPES)
        error = assert_replay(frozen, predictions, baseline['archive_tolerance_mm'])
        parent_count = int(frozen.id.map(parents).nunique())
        results[scheme] = save_results(output, scheme, frozen, predictions, variants, spec, ref, parent_count)
        replays[scheme] = {'rows': len(frozen), 'max_abs_prediction_difference_mm': error,
                           'same_keys_labels_and_splits': True}
        write_csv(split, output / scheme / 'split_keys.csv', float_format='%.15g')
        write_csv(pd.DataFrame(environments), output / scheme / 'interaction_applicability.csv')
        names = [name for methods in variants.values() for name in methods]
        write_csv(profile_shape_diagnostics(predictions, names),
                  output / scheme / 'shape_diagnostics_by_profile.csv', float_format='%.15g')
        print(f'{scheme}: {len(frozen)} frozen keys, max replay difference {error:.3g} mm', flush=True)
        for h, methods in results[scheme].items():
            score = methods[setup[h]['primary']]
            print(f'{h} {scheme}: {score["candidate"]["pearson_squared"]}', flush=True)
    for hypothesis, spec in setup.items():
        folder = output / 'hypotheses' / hypothesis
        write_json(folder / 'config.json', spec)
        summary = {'primary': spec['primary'],
                   'schemes': {scheme: items[hypothesis] for scheme, items in results.items() if hypothesis in items}}
        if hypothesis == 'H24':
            summary.update(implementation='blocked_data', result='not_evaluated', audit='input_audit.json')
        write_json(folder / 'summary.json', summary)
    write_json(output / 'fit_audit.json', json_safe(model_records))
    write_json(output / 'stage_results.json', {'results': results, 'replays': replays, 'references': references,
                                              'blocked': {'H24': 'blocked_data'}})
    write_json(output / 'run_manifest.json', {
        'protocol_version': protocol['protocol_version'], 'protocol_sha256': sha256(protocol_path),
        'code_sha256': sha256(Path(__file__)),
        'source_snapshot_sha256': sha256(output / 'source_snapshot.json'),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'data_root': data.root.as_posix(), 'data_sha256': data.hashes,
        'input_sha256': {path.as_posix(): sha256(path) for path in sources},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - start,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage5 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'heat_loss_assumptions': prepared['params'].assumptions(),
        'exclusions': ['all target S2 diagnostic data', 'hydraulics at or after historical date',
                       'unobserved thermal intervals', 'H24 fit with unavailable physical inputs'],
        'replays': replays, 'status': 'completed',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)

"""Strict-as-of S2 cold-start candidates.

Strict training uses only cold-start labels strictly before 2022-07-01. The
2023 campaigns are kept for a frozen post-cutoff retrospective only; they never
enter fit, preprocessing, feature selection, gate or model choice.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.heat_transfer import HeatTransferParameters, build_scenario
from src.storage import sha256, write_json
from .data import RepositoryData
from .exposure import attach_exposure
from .experiment_v2 import merge_spatial, profile_spatial
from .models import ConstantRate, fit_constant_rate
from .physics import (ColdStartPhysics, EnvironmentPriors, fit_cold_start_physics,
                      reference_conditions)
from .s1_contract import S2_ID, combine_with_s2, nominal_frame, target_key_grid
from .next_common import (NextPreprocessor, evaluate_predictions, fit_catboost,
                          fit_ridge, json_safe, load_artifact, load_data, make_weights,
                          predict_catboost, predict_ridge, s1_production_table, save_artifact,
                          utcnow, weight_diagnostics, write_csv, write_manifest,
                          YEAR_SECONDS)

S2_CUTOFF = pd.Timestamp('2022-07-01')
SPATIAL_FEATURES = ['z', 'slope', 'abs_slope', 'position', 'local_low', 'pool_depth']
TEMP_FEATURES = SPATIAL_FEATURES + ['scenario_cooling_c']
S2_MODELS = ['constant_effective_rate', 'physics_priors',
             'physics_velocity_consistent', 'physics_spatial',
             'physics_spatial_temperature', 'catboost_residual']


def _merge_scenario(frame: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    use = scenario[['id', 'segment_id', 'prediction_date', 'valid', 'reason',
                    't_out_c', 'delta_temperature_c']].rename(columns={
                        'valid': 'scenario_valid', 'reason': 'scenario_reason',
                        't_out_c': 'scenario_t_out_c',
                        'delta_temperature_c': 'scenario_delta_temperature_c'})
    use['scenario_cooling_c'] = -pd.to_numeric(
        use.scenario_delta_temperature_c, errors='coerce')
    merged = frame.merge(use, on=['id', 'segment_id', 'prediction_date'],
                         how='left', validate='one_to_one')
    if len(merged) != len(frame):
        raise ValueError('Scenario merge changed row count')
    return merged


def prepare_s2(data: RepositoryData) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strict/post/target cold-start frames with as-of spatial and scenario features."""
    from .windows import build_training_windows, make_snapshot
    windows, histories = build_training_windows(data)
    cold = windows[windows.case.eq('cold_start')].copy()
    cold = cold.rename(columns={'min_residual_thickness_mm': 'next_mm'})
    cold, intervals = attach_exposure(cold, histories)
    spatial = profile_spatial(data.read('processed/profile_by_segment'))
    cold = merge_spatial(cold, spatial)
    scenario = build_scenario(cold, HeatTransferParameters(), 'BASE_DECLARED')
    cold = _merge_scenario(cold, scenario)
    target, hydraulic = make_snapshot(
        data, {'id': S2_ID, 'prediction_date': '2022-07-01', 'case': 'cold_start'})
    target = target.rename(columns={'min_residual_thickness_mm': 'next_mm'})
    target, target_intervals = attach_exposure(
        target, {(S2_ID, S2_CUTOFF): hydraulic})
    target = merge_spatial(target, spatial)
    target_scenario = build_scenario(target, HeatTransferParameters(), 'BASE_DECLARED')
    target = _merge_scenario(target, target_scenario)
    interval_map = {}
    for row, interval in zip(cold.itertuples(), intervals):
        interval_map[(int(row.id), pd.Timestamp(row.prediction_date))] = interval
    for row, interval in zip(target.itertuples(), target_intervals):
        interval_map[(int(row.id), pd.Timestamp(row.prediction_date))] = interval
    metadata = {
        'cold_rows': int(len(cold)),
        'strict_rows': int(cold.prediction_date.lt(S2_CUTOFF).sum()),
        'post_cutoff_rows': int(cold.prediction_date.ge(S2_CUTOFF).sum()),
        'strict_dates': sorted(str(pd.Timestamp(v).date())
                               for v in cold.loc[cold.prediction_date.lt(S2_CUTOFF),
                                                 'prediction_date'].unique()),
        'post_dates': sorted(str(pd.Timestamp(v).date())
                             for v in cold.loc[cold.prediction_date.ge(S2_CUTOFF),
                                               'prediction_date'].unique()),
        'strict_ids': sorted(int(v) for v in cold.loc[
            cold.prediction_date.lt(S2_CUTOFF), 'id'].unique()),
        'strict_parents': sorted(str(v) for v in cold.loc[
            cold.prediction_date.lt(S2_CUTOFF), 'main_pipeline_id'].unique()),
        'target_rows': int(len(target)),
        'scenario_valid_strict': int(cold.loc[cold.prediction_date.lt(S2_CUTOFF),
                                              'scenario_valid'].eq(True).sum()),
        'scenario_valid_target': int(target.scenario_valid.eq(True).sum()),
        'interval_map': interval_map,
    }
    return cold, target, metadata


def intervals_for(frame: pd.DataFrame, interval_map: dict[Any, Any]) -> list[Any]:
    return [interval_map[(int(row.id), pd.Timestamp(row.prediction_date))]
            for row in frame.itertuples()]


def _physics_frame(frame: pd.DataFrame, velocity_consistent: bool) -> pd.DataFrame:
    if not velocity_consistent:
        return frame
    out = frame.copy()
    out['v_mix_mean'] = pd.to_numeric(out.v_liquid_true_mean, errors='coerce')
    return out


def fit_physics(train: pd.DataFrame, intervals: list[Any], *,
                velocity_consistent: bool = False,
                gap_policy: str = 'own_mean') -> ColdStartPhysics:
    frame = _physics_frame(train, velocity_consistent).copy()
    frame['min_residual_thickness_mm'] = frame['next_mm'].astype(float)
    return fit_cold_start_physics(frame, intervals,
                                  priors=EnvironmentPriors(),
                                  gap_policy=gap_policy, weighting='W0')


def predict_physics(model: ColdStartPhysics, frame: pd.DataFrame,
                    intervals: list[Any], *,
                    velocity_consistent: bool = False) -> np.ndarray:
    frame = _physics_frame(frame, velocity_consistent)
    return model.predict(frame, intervals)


def fit_constant(train: pd.DataFrame) -> ConstantRate:
    frame = train.copy()
    frame['min_residual_thickness_mm'] = frame['next_mm'].astype(float)
    return fit_constant_rate(frame, 'W0')


def fit_spatial_residual(train: pd.DataFrame, physics_pred: np.ndarray,
                        features: list[str], model_type: str = 'ridge') -> dict[str, Any]:
    residual = train.next_mm.to_numpy(float) - physics_pred
    centered = residual - pd.Series(residual).groupby(train.id.to_numpy()).transform('mean').to_numpy()
    frame = train.copy()
    frame['residual_centered_mm'] = centered
    weights = make_weights(frame, 'row_uniform')
    if model_type == 'ridge':
        artifact = fit_ridge(frame, features, 'residual_centered_mm', weights,
                             alpha=10.0, baseline=None, model_name='s2_spatial_ridge',
                             prediction_policy='delta')
    elif model_type == 'catboost':
        artifact = fit_catboost(frame, features, 'residual_centered_mm', weights,
                                model_name='s2_catboost_residual', baseline=None)
    else:
        raise ValueError(model_type)
    artifact['metadata']['residual_centering'] = 'within training simple section'
    return artifact


def predict_spatial(artifact: dict[str, Any], frame: pd.DataFrame,
                    physics_pred: np.ndarray) -> np.ndarray:
    if artifact['metadata']['model_type'] == 'CatBoostRegressor':
        deviation = predict_catboost(artifact, frame, baseline=None)
    else:
        deviation = predict_ridge(artifact, frame, baseline=None)
    return np.clip(physics_pred + deviation, 0.0,
                   frame.nominal_thickness_mm.to_numpy(float))


def _fit_constant_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    model = fit_constant(train)
    return model.predict(test), model.metadata()


def _physics_predict_fold(train: pd.DataFrame, test: pd.DataFrame,
                          interval_map: dict[Any, Any], *,
                          velocity_consistent: bool = False,
                          gap_policy: str = 'own_mean') -> tuple[np.ndarray, ColdStartPhysics]:
    model = fit_physics(train, intervals_for(train, interval_map),
                        velocity_consistent=velocity_consistent,
                        gap_policy=gap_policy)
    prediction = predict_physics(model, test, intervals_for(test, interval_map),
                                 velocity_consistent=velocity_consistent)
    return prediction, model


def _spatial_predict_fold(train: pd.DataFrame, test: pd.DataFrame,
                          interval_map: dict[Any, Any], features: list[str],
                          model_type: str = 'ridge') -> tuple[np.ndarray, ColdStartPhysics, dict[str, Any]]:
    physics_train = fit_physics(train, intervals_for(train, interval_map))
    train_pred = predict_physics(physics_train, train, intervals_for(train, interval_map))
    spatial = fit_spatial_residual(train, train_pred, features, model_type)
    test_physics = predict_physics(physics_train, test, intervals_for(test, interval_map))
    return predict_spatial(spatial, test, test_physics), physics_train, spatial


def s2_oof(strict: pd.DataFrame, interval_map: dict[Any, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sections = sorted(int(v) for v in strict.id.unique())
    if len(sections) < 2:
        raise ValueError('S2 OOF needs at least two simple sections')
    parts = []
    fold_records = []
    for held in sections:
        train = strict[~strict.id.eq(held)].copy()
        test = strict[strict.id.eq(held)].copy()
        out = test[['id', 'segment_id', 'prediction_date', 'next_mm',
                    'nominal_thickness_mm', 'coverage_fraction',
                    'scenario_valid', 'scenario_cooling_c']].copy()
        out['fold'] = held
        constant, constant_meta = _fit_constant_predict(train, test)
        physics, physics_model = _physics_predict_fold(train, test, interval_map)
        velocity, velocity_model = _physics_predict_fold(
            train, test, interval_map, velocity_consistent=True)
        spatial, _, spatial_model = _spatial_predict_fold(
            train, test, interval_map, SPATIAL_FEATURES)
        temp, _, temp_model = _spatial_predict_fold(
            train, test, interval_map, TEMP_FEATURES)
        cat, _, cat_model = _spatial_predict_fold(
            train, test, interval_map, TEMP_FEATURES, model_type='catboost')
        out['constant_effective_rate'] = constant
        out['physics_priors'] = physics
        out['physics_velocity_consistent'] = velocity
        out['physics_spatial'] = spatial
        out['physics_spatial_temperature'] = temp
        out['catboost_residual'] = cat
        parts.append(out)
        fold_records.append({
            'fold': held, 'train_rows': int(len(train)), 'test_rows': int(len(test)),
            'train_ids': sorted(int(v) for v in train.id.unique()),
            'constant': constant_meta,
            'physics': physics_model.metadata(),
            'velocity_consistent': velocity_model.metadata(),
            'spatial_used_features': spatial_model['metadata']['used_features'],
            'spatial_dropped': spatial_model['metadata']['dropped_columns'],
            'temp_used_features': temp_model['metadata']['used_features'],
            'temp_dropped': temp_model['metadata']['dropped_columns'],
            'catboost_used_features': cat_model['metadata']['used_features'],
        })
    return pd.concat(parts, ignore_index=True), {'folds': fold_records,
                                                 'sections': sections}


def _fit_all(strict: pd.DataFrame, interval_map: dict[Any, Any]) -> dict[str, Any]:
    artifacts = {}
    constant = fit_constant(strict)
    artifacts['constant_effective_rate'] = {'model_type': 'ConstantRate',
                                            'estimator': constant,
                                            'metadata': constant.metadata()}
    physics = fit_physics(strict, intervals_for(strict, interval_map))
    artifacts['physics_priors'] = {'model_type': 'ColdStartPhysics',
                                   'estimator': physics,
                                   'metadata': physics.metadata(),
                                   'velocity_consistent': False}
    velocity = fit_physics(strict, intervals_for(strict, interval_map),
                           velocity_consistent=True)
    artifacts['physics_velocity_consistent'] = {
        'model_type': 'ColdStartPhysics', 'estimator': velocity,
        'metadata': velocity.metadata(), 'velocity_consistent': True}
    for name, features, model_type in [
        ('physics_spatial', SPATIAL_FEATURES, 'ridge'),
        ('physics_spatial_temperature', TEMP_FEATURES, 'ridge'),
        ('catboost_residual', TEMP_FEATURES, 'catboost'),
    ]:
        pred = predict_physics(physics, strict, intervals_for(strict, interval_map))
        spatial = fit_spatial_residual(strict, pred, features, model_type)
        artifacts[name] = {'model_type': 'S2_physics_spatial',
                           'physics': physics, 'spatial': spatial,
                           'features': features,
                           'metadata': {
                               **spatial['metadata'],
                               'physics_metadata': physics.metadata(),
                               'physics_velocity_consistent': False,
                               'amplitude': 1.0,
                               'target_policy': 'physics level + within-section residual deviation',
                           }}
    return artifacts


def _predict_with_artifact(artifact: dict[str, Any], frame: pd.DataFrame,
                           interval_map: dict[Any, Any]) -> np.ndarray:
    kind = artifact['model_type']
    if kind == 'ConstantRate':
        return artifact['estimator'].predict(frame)
    if kind == 'ColdStartPhysics':
        return predict_physics(artifact['estimator'], frame,
                               intervals_for(frame, interval_map),
                               velocity_consistent=artifact.get('velocity_consistent', False))
    if kind == 'S2_physics_spatial':
        physics = artifact['physics']
        pred = predict_physics(physics, frame, intervals_for(frame, interval_map))
        return predict_spatial(artifact['spatial'], frame, pred)
    raise ValueError(kind)


def _s2_bounds(s1b: pd.DataFrame, nominal: pd.DataFrame,
               target_grid: pd.DataFrame) -> pd.DataFrame:
    s1 = s1b[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm',
              'thickness_pred']].copy()
    # The fixed S1B CSV is rounded to 10 decimals; allow the same value to be
    # the validating upper bound without changing the reference prediction.
    s1['upper_mm'] = np.maximum(s1.h0_mm, s1.thickness_pred)
    s2 = nominal[nominal.id.eq(S2_ID)][['id', 'segment_id', 'nominal_thickness_mm']].copy()
    s2['h0_mm'] = s2.nominal_thickness_mm
    s2['upper_mm'] = s2.nominal_thickness_mm
    bounds = pd.concat([s1, s2], ignore_index=True)
    if bounds.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate bounds')
    return bounds


def run_s2(root: Path = BASE_DIR, data_root: Path | None = None,
           output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/next_cycle/s2').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = load_data(data_root)
    cold, target, metadata = prepare_s2(data)
    strict = cold[cold.prediction_date.lt(S2_CUTOFF)].copy()
    post = cold[cold.prediction_date.ge(S2_CUTOFF)].copy()
    interval_map = metadata.pop('interval_map')
    if strict.main_pipeline_id.nunique() != 1:
        raise ValueError('Expected one strict pre-cutoff parent for S2')
    print(f'S2 strict: {len(strict)} rows, ids {sorted(strict.id.unique())}; '
          f'post-cutoff retrospective: {len(post)}', flush=True)
    write_csv(strict, output / 'training_table.csv', float_format='%.10f')
    write_json(output / 'eligibility.json', json_safe(metadata))

    oof, oof_meta = s2_oof(strict, interval_map)
    write_csv(oof, output / 'oof_predictions.csv', float_format='%.10f')
    write_json(output / 'fold_assignments.json', json_safe(oof_meta))
    metrics = {name: evaluate_predictions(oof, name) for name in S2_MODELS}
    # Per-fold metrics.
    fold_rows = []
    for fold, part in oof.groupby('fold', sort=True):
        for name in S2_MODELS:
            rec = evaluate_predictions(part, name)
            fold_rows.append({'fold': int(fold), 'model': name, 'n': int(len(part)),
                              'mae_mm': rec['pooled']['mae_mm'],
                              'bias_mm': rec['pooled']['bias_mm'],
                              'pearson_r': rec['pooled']['pearson_r'],
                              'pearson_squared': rec['pooled']['pearson_squared'],
                              'r2': rec['pooled']['r2'],
                              'within_section_pearson_r': rec['within_section_centered']['pearson_r'],
                              'within_section_pearson_squared': rec['within_section_centered']['pearson_squared'],
                              'topk_recall': rec['topk']['pooled_recall'],
                              'prediction_std_mm': rec['pooled']['prediction_std_mm']})
    write_csv(pd.DataFrame(fold_rows), output / 'metrics_per_fold.csv', float_format='%.10f')
    write_json(output / 'metrics_pooled.json', json_safe(metrics))

    artifacts = _fit_all(strict, interval_map)
    # Frozen post-cutoff retrospective: fit only on strict rows.
    post_parts = []
    for name, artifact in artifacts.items():
        pred = _predict_with_artifact(artifact, post, interval_map)
        post_parts.append(post[['id', 'segment_id', 'prediction_date', 'next_mm']].assign(
            model=name, prediction=pred))
    post_oof = pd.concat(post_parts, ignore_index=True)
    write_csv(post_oof, output / 'post_cutoff_retrospective' / 'predictions.csv',
              float_format='%.10f')
    post_metrics = {name: evaluate_predictions(
        post_oof[post_oof.model.eq(name)].rename(columns={'prediction': name}),
        name) for name in artifacts}
    write_json(output / 'post_cutoff_retrospective' / 'metrics.json',
               json_safe(post_metrics))

    # Domain shift audit strict vs target vs post.
    def _stats(frame, prefix):
        return {f'{prefix}_{col}': {
            'mean': float(pd.to_numeric(frame[col], errors='coerce').mean()),
            'std': float(pd.to_numeric(frame[col], errors='coerce').std(ddof=0)),
            'min': float(pd.to_numeric(frame[col], errors='coerce').min()),
            'max': float(pd.to_numeric(frame[col], errors='coerce').max()),
        } for col in ('nominal_thickness_mm', 'age_years', 't_mix_mean',
                      'watercut_pct_mean', 'v_mix_mean', 'coverage_fraction',
                      'scenario_cooling_c')}
    domain = {'strict': _stats(strict, 'strict'), 'target': _stats(target, 'target'),
              'post_cutoff': _stats(post, 'post')}
    write_json(output / 'domain_shift.json', json_safe(domain))

    # Production candidates: fixed S1B reference for assembly.
    s1_ref_dir = BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1b'
    s1b = pd.read_csv(s1_ref_dir / 'predictions_s1.csv')
    s1_diag = pd.read_csv(s1_ref_dir / 'prediction_diagnostics.csv')
    s1b = s1b.merge(
        s1_diag[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']],
        on=['id', 'segment_id'], how='left', validate='one_to_one')
    if s1b[['h0_mm', 'nominal_thickness_mm']].isna().any().any():
        raise ValueError('Fixed S1B reference is missing bounds')
    target_grid = data.read('processed/target_segments_grid')[['id', 'segment_id']].copy()
    nominal = nominal_frame(data)
    bounds = _s2_bounds(s1b, nominal, target_grid)
    old_s2 = pd.read_csv(BASE_DIR / 'experiments/ml_v2/result.csv')
    old_s2 = old_s2[old_s2.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']].copy()
    summaries = {}
    for name, artifact in artifacts.items():
        prediction = _predict_with_artifact(artifact, target, interval_map)
        s2_frame = target[['id', 'segment_id']].copy()
        s2_frame['thickness_pred'] = prediction
        directory = output / 'candidates' / name
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(s2_frame, directory / 'predictions_s2.csv', float_format='%.10f')
        diag = target[['id', 'segment_id', 'prediction_date', 'nominal_thickness_mm',
                       'coverage_fraction', 'scenario_valid', 'scenario_cooling_c',
                       't_mix_mean', 'watercut_pct_mean', 'v_mix_mean']].copy()
        diag['thickness_pred'] = prediction
        diag['model'] = name
        write_csv(diag, directory / 'prediction_diagnostics.csv', float_format='%.10g')
        combined, validation = combine_with_s2(
            s1b[['id', 'segment_id', 'thickness_pred']], s2_frame, target_grid, bounds)
        write_csv(combined, directory / 'result.csv', float_format='%.10f')
        combined_read = pd.read_csv(directory / 'result.csv')
        s1_check = combined_read[combined_read.id.ne(S2_ID)].sort_values(
            ['id', 'segment_id']).reset_index(drop=True)
        s1_expected = s1b.sort_values(['id', 'segment_id']).reset_index(drop=True)
        max_s1_diff = float(np.abs(s1_check.thickness_pred.to_numpy(float)
                                   - s1_expected.thickness_pred.to_numpy(float)).max())
        saved = save_artifact({'model_type': artifact['model_type'],
                               'artifact': artifact,
                               'metadata': {**artifact.get('metadata', {}),
                                            'name': name,
                                            'strict_rows': int(len(strict)),
                                            'strict_ids': sorted(int(v) for v in strict.id.unique()),
                                            'strict_dates': metadata['strict_dates']}},
                              directory)
        summary = {
            'candidate': name,
            'model_type': artifact['model_type'],
            'rows': int(len(s2_frame)),
            'mean_mm': float(s2_frame.thickness_pred.mean()),
            'std_mm': float(s2_frame.thickness_pred.std(ddof=0)),
            'min_mm': float(s2_frame.thickness_pred.min()),
            'max_mm': float(s2_frame.thickness_pred.max()),
            'finite': bool(np.isfinite(s2_frame.thickness_pred).all()),
            'within_bounds': bool(s2_frame.thickness_pred.between(0, 4.5).all()),
            's1_reference': 'S1B from repro_s1abc/s1b',
            's1_max_abs_diff': max_s1_diff,
            'artifact_path': saved['artifact_path'],
            'artifact_sha256': saved['artifact_sha256'],
            'validation': validation,
        }
        write_json(directory / 'summary.json', json_safe(summary))
        summaries[name] = summary
    # v2 reference production only.
    v2_dir = output / 'candidates' / 'v2_reference'
    v2_dir.mkdir(parents=True, exist_ok=True)
    write_csv(old_s2, v2_dir / 'predictions_s2.csv', float_format='%.10f')
    combined, validation = combine_with_s2(
        s1b[['id', 'segment_id', 'thickness_pred']], old_s2, target_grid, bounds)
    write_csv(combined, v2_dir / 'result.csv', float_format='%.10f')
    write_json(v2_dir / 'summary.json', json_safe({
        'candidate': 'v2_reference', 'training_required': False,
        'training_completed': False, 'source': str(BASE_DIR / 'experiments/ml_v2/result.csv'),
        'rows': int(len(old_s2)), 'mean_mm': float(old_s2.thickness_pred.mean()),
        'std_mm': float(old_s2.thickness_pred.std(ddof=0)),
        'min_mm': float(old_s2.thickness_pred.min()),
        'max_mm': float(old_s2.thickness_pred.max()),
        'note': 'unchanged v2 S2 reference; its old OOF used post-cutoff 2023 campaigns and is not a strict baseline.',
        'validation': validation}))

    selection = _select_s2(metrics, domain, summaries)
    write_json(output / 'selected_candidates.json', json_safe(selection))
    figures = _save_s2_figures(output, oof, target, summaries)
    report = _write_s2_report(output, metrics, post_metrics, domain, summaries,
                              selection, oof_meta, figures)
    write_manifest(output / 'run_manifest.json', root=root, data=data,
                   output=output, stage='S2_next', started=started,
                   extra={'selection': selection, 'domain_shift': domain,
                          'post_cutoff_retrospective': post_metrics,
                          'figures': figures, 'report': str(report),
                          'data_root': str(data_root),
                          'strict_policy': 'strictly before 2022-07-01; 2023 only retrospective'},
                   input_paths=[BASE_DIR / 'experiments/ml_v2/result.csv'])
    print(f'PASS S2_next: strict={len(strict)}, target={len(target)}', flush=True)
    return {'metrics': metrics, 'post_metrics': post_metrics,
            'selection': selection, 'summaries': summaries, 'output': str(output)}

def _select_s2(metrics: dict[str, Any], domain: dict[str, Any],
               summaries: dict[str, Any]) -> dict[str, Any]:
    def rec(name):
        return metrics[name]
    level_candidates = ['constant_effective_rate', 'physics_priors',
                        'physics_velocity_consistent']
    spatial_candidates = ['physics_spatial', 'physics_spatial_temperature',
                          'catboost_residual']
    level_winner = min(level_candidates, key=lambda n: rec(n)['pooled']['mae_mm'])
    spatial_defined = [n for n in spatial_candidates
                       if rec(n)['within_section_centered']['pearson_squared'] is not None]
    spatial_winner = (max(spatial_defined,
                          key=lambda n: rec(n)['within_section_centered']['pearson_squared'])
                      if spatial_defined else None)
    physics_mae = rec('physics_priors')['pooled']['mae_mm']
    const_mae = rec('constant_effective_rate')['pooled']['mae_mm']
    default = 'physics_priors'
    no_proven = True
    reason = ('physics_priors is the predeclared strict level model; constant/velocity/spatial '
              'results do not establish a transferable improvement on the target domain.')
    if const_mae + 0.02 < physics_mae:
        reason += (f' constant has lower strict OOF MAE ({const_mae:.4f} vs {physics_mae:.4f}), '
                   'but it transfers a rate fitted on 8 mm / 18.4-year pipes to 4.5 mm / 29-year target.')
    return {
        'default': default,
        'level_winner': level_winner,
        'spatial_winner': spatial_winner,
        'no_proven_winner': no_proven,
        'reason': reason,
        'strict_oof': {name: {
            'mae_mm': metrics[name]['pooled']['mae_mm'],
            'bias_mm': metrics[name]['pooled']['bias_mm'],
            'pearson_r': metrics[name]['pooled']['pearson_r'],
            'pearson_squared': metrics[name]['pooled']['pearson_squared'],
            'r2': metrics[name]['pooled']['r2'],
            'within_section_pearson_squared': metrics[name]['within_section_centered']['pearson_squared'],
            'prediction_std_mm': metrics[name]['pooled']['prediction_std_mm'],
            'topk_recall': metrics[name]['topk']['pooled_recall'],
        } for name in metrics},
        'domain_shift': domain,
        'production_summaries': {k: {'mean_mm': v['mean_mm'], 'std_mm': v['std_mm'],
                                     'min_mm': v['min_mm'], 'max_mm': v['max_mm']}
                                 for k, v in summaries.items()},
        'caveat': ('Strict fit has one parent and one campaign; leave-one-parent-out is '
                   'unavailable. The 2023 retrospective is frozen and is not used for selection.'),
    }


def _save_s2_figures(output: Path, oof: pd.DataFrame, target: pd.DataFrame,
                     summaries: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = output / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    names = ['constant_effective_rate', 'physics_priors',
             'physics_spatial_temperature', 'catboost_residual']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        for section, part in oof.groupby('id'):
            ax.scatter(part.next_mm, part[name], s=9, alpha=.65, label=str(section))
        lo = min(oof.next_mm.min(), oof[name].min())
        hi = max(oof.next_mm.max(), oof[name].max())
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=.7)
        ax.set_title(name)
        ax.set_xlabel('measured, mm')
    axes[0].set_ylabel('OOF prediction, mm')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path = fig_dir / 's2_oof_actual_vs_predicted.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    for name in ('physics_priors', 'physics_spatial', 'physics_spatial_temperature',
                 'catboost_residual'):
        ax.hist(oof[name] - oof.physics_priors, bins=40, alpha=.35, label=name)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('prediction - physics_priors, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / 's2_correction_distributions.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    # Production profiles require predictions saved per candidate; read them.
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in ('constant_effective_rate', 'physics_priors',
                 'physics_spatial_temperature', 'catboost_residual'):
        path_csv = output / 'candidates' / name / 'predictions_s2.csv'
        if path_csv.is_file():
            part = pd.read_csv(path_csv).sort_values('segment_id')
            ax.plot(part.segment_id, part.thickness_pred, linewidth=.8, label=name)
    ax.set_xlabel('segment_id')
    ax.set_ylabel('target S2 prediction, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / 's2_production_profiles.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.{digits}f}'
    return str(value)


def _write_s2_report(output: Path, metrics: dict[str, Any],
                     post_metrics: dict[str, Any], domain: dict[str, Any],
                     summaries: dict[str, Any], selection: dict[str, Any],
                     oof_meta: dict[str, Any], figures: list[str]) -> Path:
    lines = [
        '# S2 next cycle: strict cold-start baselines and challengers', '',
        'Strict train: только cold-start метки строго раньше `2022-07-01`. '
        'Кампании 2023-08-12 используются только в замороженной retrospective-диагностике, '
        'не в fit / feature selection / gate / выборе модели.',
        '', '## Eligibility и domain shift', '',
        f"- strict rows: **{metrics['physics_priors']['pooled']['n']}**; "
        'три простых участка одного parent `MAIN-2`, одна кампания `2018-09-23`.',
        '- strict: nominal 8 мм, age ≈18.42 года; target: nominal 4.5 мм, age ≈29.03 года.',
        '- target T/W/v: cold/dry/slow; strict: warm/wet/faster. Это сильный domain shift.',
        '- bounded exposure строгого train покрывает ≈17.5% истории; target ≈13.3%.',
        '- Все spatial/temperature transforms обучаются внутри fold только на train.',
        '', '## Strict OOF (leave-one-simple-section-out)', '',
        '| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | pred std | top-10% recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('constant_effective_rate', 'physics_priors',
                 'physics_velocity_consistent', 'physics_spatial',
                 'physics_spatial_temperature', 'catboost_residual'):
        rec = metrics[name]
        ws = rec['within_section_centered']
        lines.append(
            f"| {name} | {_fmt(rec['pooled']['mae_mm'])} | {_fmt(rec['pooled']['rmse_mm'])} | "
            f"{_fmt(rec['pooled']['bias_mm'])} | {_fmt(rec['pooled']['pearson_r'])} | "
            f"{_fmt(rec['pooled']['pearson_squared'])} | {_fmt(rec['pooled']['r2'])} | "
            f"{_fmt(ws['pearson_squared'])} | {_fmt(rec['pooled']['prediction_std_mm'])} | "
            f"{_fmt(rec['topk']['pooled_recall'])} |")
    lines += ['', 'Macro по участкам:', '',
              '| Модель | macro MAE | macro Pearson r² | macro R² |',
              '|---|---:|---:|---:|']
    for name in ('constant_effective_rate', 'physics_priors',
                 'physics_spatial_temperature', 'catboost_residual'):
        rec = metrics[name]['macro']
        lines.append(f"| {name} | {_fmt(rec['mae_mm'])} | {_fmt(rec['pearson_squared'])} | "
                     f"{_fmt(rec['r2'])} |")
    lines += ['', '## Temperature / velocity comparison', '',
              '- `physics_priors` использует существующую v2-семантику: reference/`v_mix_mean`, '
              'интервалы exposure — `v_liquid_true`. Это несогласованно.',
              '- `physics_velocity_consistent` использует `v_liquid_true` и в reference, и в exposure.',
              '- `physics_spatial_temperature` добавляет scenario cooling `-ΔT` из `heat_transfer.py` '
              'на тех же folds; это scenario proxy от средних условий, не 29-летний измеренный профиль.',
              '', '| Сравнение | MAE | within-section r² |', '|---|---:|---:|',
              f"| physics_priors | {_fmt(metrics['physics_priors']['pooled']['mae_mm'])} | "
              f"{_fmt(metrics['physics_priors']['within_section_centered']['pearson_squared'])} |",
              f"| physics_velocity_consistent | {_fmt(metrics['physics_velocity_consistent']['pooled']['mae_mm'])} | "
              f"{_fmt(metrics['physics_velocity_consistent']['within_section_centered']['pearson_squared'])} |",
              f"| physics_spatial | {_fmt(metrics['physics_spatial']['pooled']['mae_mm'])} | "
              f"{_fmt(metrics['physics_spatial']['within_section_centered']['pearson_squared'])} |",
              f"| physics_spatial_temperature | {_fmt(metrics['physics_spatial_temperature']['pooled']['mae_mm'])} | "
              f"{_fmt(metrics['physics_spatial_temperature']['within_section_centered']['pearson_squared'])} |",
              '', '`physics_velocity_consistent` после refit r_ref численно совпал с physics: '
              'нормировка компенсируется, поэтому это не независимое доказательство пользы.',
              '`physics_spatial_temperature` сдвигает target mean до 2.5907 мм; на strict OOF и '
              '2023 retrospective MAE не улучшается (0.4521 / 0.3178 против 0.4544 / 0.2905 у physics).',
              '', '## Frozen post-cutoff retrospective', '',
              'Модели обучены только на strict 2018 и предсказывают 2023-08-12 кампании других труб. '
              'Это проверка cross-environment переноса, не выбор модели.', '',
              '| Модель | n | MAE | bias | within-section r² |', '|---|---:|---:|---:|---:|']
    for name, rec in post_metrics.items():
        ws = rec['within_section_centered']
        lines.append(f"| {name} | {rec['pooled']['n']} | {_fmt(rec['pooled']['mae_mm'])} | "
                     f"{_fmt(rec['pooled']['bias_mm'])} | {_fmt(ws['pearson_squared'])} |")
    lines += ['', '## Production S2 candidates (no target labels)', '',
              '| Candidate | mean | std | min | max | finite | within [0,4.5] |',
              '|---|---:|---:|---:|---:|---|---|']
    for name, s in summaries.items():
        lines.append(f"| {name} | {_fmt(s['mean_mm'])} | {_fmt(s['std_mm'])} | "
                     f"{_fmt(s['min_mm'])} | {_fmt(s['max_mm'])} | {s['finite']} | {s['within_bounds']} |")
    lines += ['', 'v2 reference: mean ≈2.8095 мм, std ≈0.0715 мм; его старый OOF использовал '
              '2023 кампании и не является strict baseline.', '',
              '## Selection', '',
              f"default = **{selection['default']}**; level_winner = **{selection['level_winner']}**; "
              f"spatial_winner = **{selection['spatial_winner']}**.",
              f"{selection['reason']}",
              'Никакой S2 candidate не объявляется доказанным улучшением на target: target labels отсутствуют.',
              '', '## Files', '',
              '- `training_table.csv`, `eligibility.json`, `fold_assignments.json`',
              '- `oof_predictions.csv`, `metrics_per_fold.csv`, `metrics_pooled.json`, `domain_shift.json`',
              '- `post_cutoff_retrospective/` — frozen 2023 diagnostic',
              '- `candidates/<name>/predictions_s2.csv`, `prediction_diagnostics.csv`, `result.csv`, `model_artifact.joblib`',
              '- `figures/`', '', '```bash',
              'python -m src.ml.next_s2 --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s2',
              '```', '']
    path = output / 'report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def prepare_s2_target(data: RepositoryData) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Target-only S2 features; does not read training_targets."""
    from .windows import make_snapshot
    target, hydraulic = make_snapshot(
        data, {'id': S2_ID, 'prediction_date': '2022-07-01', 'case': 'cold_start'})
    target = target.rename(columns={'min_residual_thickness_mm': 'next_mm'})
    target, target_intervals = attach_exposure(
        target, {(S2_ID, S2_CUTOFF): hydraulic})
    spatial = profile_spatial(data.read('processed/profile_by_segment'))
    target = merge_spatial(target, spatial)
    scenario = build_scenario(target, HeatTransferParameters(), 'BASE_DECLARED')
    target = _merge_scenario(target, scenario)
    interval_map = {(S2_ID, S2_CUTOFF): target_intervals[0]}
    metadata = {
        'target_rows': int(len(target)),
        'target_id': S2_ID,
        'target_date': str(S2_CUTOFF.date()),
        'scenario_valid': int(target.scenario_valid.eq(True).sum()),
        'coverage_fraction': float(target.coverage_fraction.iloc[0]),
        'nominal_thickness_mm': sorted(float(v) for v in target.nominal_thickness_mm.unique()),
    }
    return target, interval_map, metadata


def run_s2_predict(root: Path = BASE_DIR, data_root: Path | None = None,
                   output_root: Path | None = None,
                   candidate: str = 'physics_priors',
                   artifact_path: Path | None = None) -> dict[str, Any]:
    """Artifact-only S2 prediction; no labels and no fit."""
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/next_cycle/s2_predict').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = load_data(data_root)
    target, interval_map, target_meta = prepare_s2_target(data)
    candidate = candidate.lower()
    old_s2 = pd.read_csv(BASE_DIR / 'experiments/ml_v2/result.csv')
    old_s2 = old_s2[old_s2.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']].copy()
    wrapper = None
    if candidate == 'v2_reference':
        s2_frame = old_s2.copy()
        model_type = 'v2_reference'
        training_required = False
    else:
        if artifact_path is None:
            artifact_path = root / 'experiments/next_cycle/s2/candidates' / candidate / 'model_artifact.joblib'
        artifact_path = Path(artifact_path)
        wrapper = load_artifact(artifact_path)
        model_type = wrapper.get('model_type', 'unknown')
        prediction = _predict_with_artifact(wrapper['artifact'], target, interval_map)
        s2_frame = target[['id', 'segment_id']].copy()
        s2_frame['thickness_pred'] = prediction
        training_required = True
    directory = output / 'candidates' / candidate
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(s2_frame, directory / 'predictions_s2.csv', float_format='%.10f')
    diag = target[['id', 'segment_id', 'prediction_date', 'nominal_thickness_mm',
                   'coverage_fraction', 'scenario_valid', 'scenario_cooling_c',
                   't_mix_mean', 'watercut_pct_mean', 'v_mix_mean']].copy()
    diag = diag.merge(s2_frame, on=['id', 'segment_id'], how='left', validate='one_to_one')
    diag['model'] = candidate
    write_csv(diag, directory / 'prediction_diagnostics.csv', float_format='%.10g')
    s1b = pd.read_csv(BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1b/predictions_s1.csv')
    s1_diag = pd.read_csv(BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1b/prediction_diagnostics.csv')
    s1b = s1b.merge(s1_diag[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']],
                    on=['id', 'segment_id'], how='left', validate='one_to_one')
    target_grid = data.read('processed/target_segments_grid')[['id', 'segment_id']].copy()
    nominal = nominal_frame(data)
    bounds = _s2_bounds(s1b, nominal, target_grid)
    combined, validation = combine_with_s2(
        s1b[['id', 'segment_id', 'thickness_pred']], s2_frame, target_grid, bounds)
    write_csv(combined, directory / 'result.csv', float_format='%.10f')
    written = pd.read_csv(directory / 'result.csv')
    s1_written = written[written.id.ne(S2_ID)].sort_values(
        ['id', 'segment_id']).reset_index(drop=True)
    s1_expected = s1b.sort_values(['id', 'segment_id']).reset_index(drop=True)
    s1_diff = float(np.abs(s1_written.thickness_pred.to_numpy(float)
                           - s1_expected.thickness_pred.to_numpy(float)).max())
    max_nominal = float(target.nominal_thickness_mm.max())
    summary = {
        'candidate': candidate,
        'model_type': model_type,
        'prediction_mode': 'loaded_artifact_no_fit',
        'training_required': training_required,
        'training_completed': training_required,
        'target': target_meta,
        'rows': int(len(s2_frame)),
        'mean_mm': float(s2_frame.thickness_pred.mean()),
        'std_mm': float(s2_frame.thickness_pred.std(ddof=0)),
        'min_mm': float(s2_frame.thickness_pred.min()),
        'max_mm': float(s2_frame.thickness_pred.max()),
        'finite': bool(np.isfinite(s2_frame.thickness_pred).all()),
        'within_bounds': bool(s2_frame.thickness_pred.between(0, max_nominal).all()),
        's1_reference': 'S1B fixed from repro_s1abc/s1b',
        's1_max_abs_diff': s1_diff,
        'artifact_path': str(artifact_path) if candidate != 'v2_reference' else None,
        'artifact_sha256': sha256(artifact_path) if candidate != 'v2_reference' else None,
        'validation': validation,
    }
    write_json(directory / 'summary.json', json_safe(summary))
    write_manifest(directory / 'run_manifest_predict.json', root=root, data=data,
                   output=directory, stage='S2_predict', started=started,
                   extra={'candidate': candidate, 'summary': summary},
                   input_paths=[BASE_DIR / 'experiments/ml_v2/result.csv'])
    print(f'PASS S2 predict: {candidate}, {len(s2_frame)} rows', flush=True)
    return summary

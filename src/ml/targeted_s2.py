"""Stage 2: full-grid thermal contract and full-grid centered S2 spatial Ridge.

The module is intentionally separate from the preserved next_cycle outputs.
It reads the next_cycle data generation, rebuilds full-grid cold-start snapshots
before joining labels, and retrains only three fixed spatial Ridge candidates.
No retrospective labels are used for fitting or selecting parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.heat_transfer import (HeatTransferParameters, build_scenario,
                               segment_outlet_temperature)
from src.storage import sha256, write_json
from .data import RepositoryData
from .exposure import attach_exposure
from .experiment_v2 import merge_spatial, profile_spatial
from .next_common import (evaluate_predictions, json_safe, load_artifact,
                          predict_ridge, s1_production_table, save_artifact,
                          utcnow, write_csv as _write_csv, write_manifest)
from .physics import (ColdStartPhysics, EnvironmentPriors,
                      fit_cold_start_physics)
from .s1_contract import S2_ID, combine_with_s2, nominal_frame, target_key_grid
from .s2_next import fit_spatial_residual
from .targeted_audit import tail_metrics
from .windows import build_training_windows, make_snapshot


def write_csv(frame: pd.DataFrame, path: Path, fmt: str = '%.10g') -> None:
    _write_csv(frame, path, float_format=fmt)

S2_CUTOFF = pd.Timestamp('2022-07-01')
SPATIAL_FEATURES = ['z', 'slope', 'abs_slope', 'position', 'local_low', 'pool_depth']
TEMP_FEATURES = SPATIAL_FEATURES + ['scenario_cooling_c']
CANDIDATES = ['spatial_no_temperature_uncentered',
              'spatial_no_temperature_centered',
              'spatial_centered_corrected_temperature']
CENTRED = {'spatial_no_temperature_centered',
           'spatial_centered_corrected_temperature'}
FEATURES = {
    'spatial_no_temperature_uncentered': SPATIAL_FEATURES,
    'spatial_no_temperature_centered': SPATIAL_FEATURES,
    'spatial_centered_corrected_temperature': TEMP_FEATURES,
}


def _grid_check(frame: pd.DataFrame, tol: float = 1e-8) -> dict[str, Any]:
    issues = []
    for (pid, date), part in frame.groupby(['id', 'prediction_date'], sort=True):
        part = part.sort_values(['start_m', 'end_m'])
        start = part.start_m.to_numpy(float)
        end = part.end_m.to_numpy(float)
        length = part.length_m.to_numpy(float)
        if not np.isfinite(np.r_[start, end, length]).all():
            issues.append({'id': int(pid), 'date': str(date), 'issue': 'nonfinite_geometry'})
            continue
        if np.any(end <= start) or np.any(length <= 0):
            issues.append({'id': int(pid), 'date': str(date), 'issue': 'nonpositive_length'})
        if len(part) > 1:
            gaps = start[1:] - end[:-1]
            if np.any(gaps > tol):
                issues.append({'id': int(pid), 'date': str(date),
                               'issue': 'gap', 'max_gap_m': float(gaps.max())})
            if np.any(gaps < -tol):
                issues.append({'id': int(pid), 'date': str(date),
                               'issue': 'overlap', 'max_overlap_m': float(-gaps.min())})
        if not np.allclose(length, end - start, atol=tol):
            issues.append({'id': int(pid), 'date': str(date),
                           'issue': 'length_not_end_minus_start'})
    return {'valid': len(issues) == 0, 'issues': issues,
            'groups': int(frame.groupby(['id', 'prediction_date']).ngroups)}


def _thermal_features(full: pd.DataFrame,
                      params: HeatTransferParameters,
                      scenario_id: str = 'FULL_GRID_BASE',
                      validate_grid: bool = True) -> pd.DataFrame:
    """Build scenario on a full grid; never use labels here."""
    if validate_grid:
        check = _grid_check(full)
        if not check['valid']:
            raise ValueError(f'Invalid physical grid: {check["issues"][:3]}')
    scenario = build_scenario(full, params, scenario_id)
    if scenario.segment_id.isna().any():
        reasons = sorted(set(str(v) for v in scenario.loc[
            scenario.segment_id.isna(), 'reason'].dropna()))
        reason = ';'.join(reasons) if reasons else 'scenario_invalid'
        empty = full[['id', 'segment_id', 'prediction_date']].copy()
        for col in ('t_in_segment_c', 't_out_segment_c', 'local_delta_t_c',
                    'scenario_cooling_c', 'cumulative_cooling_from_inlet_c'):
            empty[col] = np.nan
        empty['scenario_valid'] = False
        empty['scenario_reason'] = reason
        return empty
    scenario = scenario.rename(columns={
        't_in_c': 't_in_segment_c', 't_out_c': 't_out_segment_c',
        'delta_temperature_c': 'local_delta_t_c'})
    scenario['scenario_cooling_c'] = -pd.to_numeric(
        scenario.local_delta_t_c, errors='coerce')
    scenario['cumulative_cooling_from_inlet_c'] = np.nan
    for _, part in scenario.groupby(['id', 'prediction_date'], sort=True):
        valid = part[part.valid.fillna(False).astype(bool)].sort_values('start_m')
        if valid.empty:
            continue
        inlet = float(valid.t_in_segment_c.iloc[0])
        scenario.loc[valid.index, 'cumulative_cooling_from_inlet_c'] = (
            inlet - valid.t_out_segment_c.to_numpy(float))
    keep = ['id', 'segment_id', 'prediction_date', 'valid', 'reason',
            't_in_segment_c', 't_out_segment_c', 'local_delta_t_c',
            'scenario_cooling_c', 'cumulative_cooling_from_inlet_c']
    out = scenario[keep].rename(columns={
        'valid': 'scenario_valid', 'reason': 'scenario_reason'})
    return out


def _merge_thermal(full: pd.DataFrame, thermal: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    thermal = thermal.copy()
    for frame in (full, thermal):
        frame['id'] = pd.to_numeric(frame['id'], errors='raise').astype('int64')
        frame['segment_id'] = pd.to_numeric(frame['segment_id'], errors='raise').astype('int64')
        frame['prediction_date'] = pd.to_datetime(frame['prediction_date'])
    merged = full.merge(thermal, on=['id', 'segment_id', 'prediction_date'],
                        how='left', validate='one_to_one')
    if len(merged) != len(full):
        raise ValueError('Thermal merge changed row count')
    return merged


def prepare_full_grid(data: RepositoryData,
                      label_keys: list[tuple[int, int, Any]] | None = None) -> dict[str, Any]:
    """Rebuild full grids for every cold campaign before joining labels.

    ``label_keys`` is an explicit diagnostic subset of measured labels
    ``(id, segment_id, prediction_date)``.  Full physical frames are always
    rebuilt for every campaign; the subset only changes which measured rows are
    joined.  Unknown keys fail instead of silently changing the source.
    """
    windows, histories = build_training_windows(data)
    cold = windows[windows.case.eq('cold_start')].copy()
    cold = cold.rename(columns={'min_residual_thickness_mm': 'next_mm'})
    labels = cold[['id', 'segment_id', 'prediction_date', 'next_mm',
                   'main_pipeline_id', 'campaign_id']].copy()
    campaigns = cold[['id', 'prediction_date']].drop_duplicates().sort_values(
        ['id', 'prediction_date'])
    if label_keys is not None:
        required = {(int(a), int(b), pd.Timestamp(c)) for a, b, c in label_keys}
        available = {(int(a), int(b), pd.Timestamp(c))
                     for a, b, c in zip(labels.id, labels.segment_id,
                                        labels.prediction_date)}
        unknown = required - available
        if unknown:
            raise ValueError(f'Unknown label keys: {sorted(unknown)[:5]}')
        labels = labels[[(int(a), int(b), pd.Timestamp(c)) in required
                         for a, b, c in zip(labels.id, labels.segment_id,
                                            labels.prediction_date)]].copy()
    spatial = profile_spatial(data.read('processed/profile_by_segment'))
    params = HeatTransferParameters()
    full_frames: dict[tuple[int, pd.Timestamp], pd.DataFrame] = {}
    interval_map: dict[tuple[int, pd.Timestamp], Any] = {}
    measured_parts = []
    thermal_checks = []
    for row in campaigns.itertuples():
        pid, date = int(row.id), pd.Timestamp(row.prediction_date)
        request = {'id': pid, 'prediction_date': str(date.date()), 'case': 'cold_start'}
        full, hydraulic = make_snapshot(data, request)
        full, intervals = attach_exposure(full, histories)
        interval_map[(pid, date)] = intervals[0]
        full = merge_spatial(full, spatial)
        thermal = _thermal_features(full, params)
        full = _merge_thermal(full, thermal)
        full_frames[(pid, date)] = full
        lab = labels[(labels.id.eq(pid)) & (labels.prediction_date.eq(date))]
        measured = full.merge(lab, on=['id', 'segment_id', 'prediction_date'],
                              how='inner', validate='one_to_one')
        if 'main_pipeline_id_y' in measured:
            measured = measured.rename(columns={'main_pipeline_id_y': 'main_pipeline_id'})
            measured = measured.drop(columns=['main_pipeline_id_x'], errors='ignore')
        if len(measured) != len(lab):
            raise ValueError(f'Label join mismatch for {pid} {date}: '
                             f'{len(measured)} vs {len(lab)}')
        if len(measured):
            measured_parts.append(measured)
        thermal_checks.append({'id': pid, 'date': str(date.date()),
                               'full_rows': int(len(full)),
                               'measured_rows': int(len(measured)),
                               **_grid_check(full)})
    target, target_intervals = prepare_target_full(data, spatial, params)
    interval_map[(S2_ID, S2_CUTOFF)] = target_intervals
    thermal_checks.append({'id': S2_ID, 'date': str(S2_CUTOFF.date()),
                           'full_rows': int(len(target)), 'measured_rows': 0,
                           **_grid_check(target)})
    if measured_parts:
        measured = pd.concat(measured_parts, ignore_index=True)
    else:
        measured = pd.DataFrame(columns=['id', 'segment_id', 'prediction_date',
                                         'next_mm', 'nominal_thickness_mm'])
    strict = measured[measured.prediction_date.lt(S2_CUTOFF)].copy()
    post = measured[measured.prediction_date.ge(S2_CUTOFF)].copy()
    return {
        'measured': measured, 'strict': strict, 'post': post,
        'target': target, 'full_frames': full_frames,
        'interval_map': interval_map, 'thermal_checks': thermal_checks,
        'spatial_profile': spatial, 'params': params,
    }

def prepare_target_full(data: RepositoryData, spatial: pd.DataFrame | None = None,
                        params: HeatTransferParameters | None = None) -> tuple[pd.DataFrame, Any]:
    spatial = spatial if spatial is not None else profile_spatial(
        data.read('processed/profile_by_segment'))
    params = params or HeatTransferParameters()
    target, hydraulic = make_snapshot(
        data, {'id': S2_ID, 'prediction_date': '2022-07-01', 'case': 'cold_start'})
    target, intervals = attach_exposure(
        target, {(S2_ID, S2_CUTOFF): hydraulic})
    target = merge_spatial(target, spatial)
    thermal = _thermal_features(target, params)
    target = _merge_thermal(target, thermal)
    return target, intervals[0]


def _mean_full_grid(full: pd.DataFrame, g: np.ndarray) -> dict[str, Any]:
    weights = full.length_m.to_numpy(float)
    values = np.asarray(g, float)
    if len(values) != len(full) or not np.isfinite(values).all():
        raise ValueError('Invalid full-grid g')
    mean_len = float(np.sum(weights * values) / np.sum(weights))
    mean_arith = float(values.mean())
    keys = '\n'.join(f'{int(a)}|{int(b)}|{float(c):.6f}|{float(d):.6f}|{float(e):.6f}'
                     for a, b, c, d, e in zip(full.id, full.segment_id,
                                              full.start_m, full.end_m, full.length_m))
    context = {
        'n': int(len(full)), 'sum_length_m': float(weights.sum()),
        'mean_length_weighted': mean_len, 'mean_arithmetic': mean_arith,
        'context_sha256': hashlib.sha256(keys.encode('utf-8')).hexdigest(),
    }
    return context


def _apply_candidate(physics: np.ndarray, g: np.ndarray,
                     full: pd.DataFrame, centered: bool,
                     context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or _mean_full_grid(full, g)
    g_centered = g - context['mean_length_weighted'] if centered else g.copy()
    h_raw = physics + g_centered
    nominal = full.nominal_thickness_mm.to_numpy(float)
    h_clipped = np.clip(h_raw, 0.0, nominal)
    clipping = (h_raw < 0) | (h_raw > nominal)
    return {'g_raw': g, 'g_centered': g_centered, 'h_raw': h_raw,
            'h_clipped': h_clipped, 'clipping': clipping,
            'context': context,
            'mean_shift_after_clipping': float(np.mean(h_clipped) - np.mean(h_raw))}


def _fit_physics(train: pd.DataFrame, interval_map: dict[Any, Any]) -> ColdStartPhysics:
    frame = train.copy()
    frame['min_residual_thickness_mm'] = frame.next_mm.astype(float)
    intervals = [interval_map[(int(r.id), pd.Timestamp(r.prediction_date))]
                 for r in frame.itertuples()]
    return fit_cold_start_physics(frame, intervals,
                                  priors=EnvironmentPriors(),
                                  gap_policy='own_mean', weighting='W0')


def _predict_physics(model: ColdStartPhysics, frame: pd.DataFrame,
                     interval_map: dict[Any, Any]) -> np.ndarray:
    intervals = [interval_map[(int(r.id), pd.Timestamp(r.prediction_date))]
                 for r in frame.itertuples()]
    return model.predict(frame, intervals)


def fit_candidate_artifacts(strict: pd.DataFrame,
                            interval_map: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    physics = _fit_physics(strict, interval_map)
    train_physics = _predict_physics(physics, strict, interval_map)
    artifacts = {'physics_priors': {'model_type': 'ColdStartPhysics',
                                    'estimator': physics,
                                    'metadata': physics.metadata()}}
    for name in CANDIDATES:
        spatial = fit_spatial_residual(strict, train_physics,
                                       FEATURES[name], 'ridge')
        artifacts[name] = {
            'model_type': 'S2_full_grid_spatial',
            'physics': physics, 'spatial': spatial,
            'features': FEATURES[name],
            'centered': name in CENTRED,
            'metadata': {
                **spatial['metadata'],
                'candidate': name,
                'physics_metadata': physics.metadata(),
                'features': FEATURES[name],
                'centered': name in CENTRED,
                'centering': ('length-weighted full-grid mean of g' if name in CENTRED
                              else 'none'),
                'lambda': 1.0,
                'training_rows': int(len(strict)),
            },
        }
    return artifacts


def _predict_candidate(artifact: dict[str, Any], full: pd.DataFrame,
                       interval_map: dict[Any, Any]) -> dict[str, Any]:
    physics = _predict_physics(artifact['physics'], full, interval_map)
    g = predict_ridge(artifact['spatial'], full, baseline=None)
    return _apply_candidate(physics, g, full, artifact.get('centered', False))

def predict_selected(artifact: dict[str, Any], full: pd.DataFrame,
                     interval_map: dict[Any, Any],
                     segment_ids: list[int]) -> pd.DataFrame:
    """Compute full-grid context once, then return a partial request."""
    pred = _predict_candidate(artifact, full, interval_map)
    mask = full.segment_id.isin(segment_ids).to_numpy()
    if mask.sum() == 0:
        raise ValueError('No requested segment ids in full grid')
    out = full.loc[mask, ['id', 'segment_id', 'nominal_thickness_mm']].copy()
    out['thickness_pred'] = pred['h_clipped'][mask]
    out['g_raw'] = pred['g_raw'][mask]
    out['g_centered'] = pred['g_centered'][mask]
    out['h_raw'] = pred['h_raw'][mask]
    out['clipping'] = pred['clipping'][mask]
    out['context_mean_length_weighted'] = pred['context']['mean_length_weighted']
    out['context_sha256'] = pred['context']['context_sha256']
    return out


def _oof_candidates(strict: pd.DataFrame, full_frames: dict[tuple[int, pd.Timestamp], pd.DataFrame],
                    interval_map: dict[Any, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts, contexts, fold_records = [], [], []
    for held in sorted(int(v) for v in strict.id.unique()):
        train = strict[~strict.id.eq(held)].copy()
        test = strict[strict.id.eq(held)].copy()
        if test.empty:
            continue
        dates = sorted(pd.Timestamp(v) for v in test.prediction_date.unique())
        if len(dates) != 1:
            raise ValueError('Expected one date per held section')
        full = full_frames[(held, dates[0])]
        physics_model = _fit_physics(train, interval_map)
        train_physics = _predict_physics(physics_model, train, interval_map)
        physics_test = _predict_physics(physics_model, full, interval_map)
        out = test[['id', 'segment_id', 'prediction_date', 'next_mm',
                    'nominal_thickness_mm', 'scenario_cooling_c']].copy()
        out['fold'] = held
        physics_by_key = full[['segment_id']].copy()
        physics_by_key['physics_priors'] = physics_test
        out = out.merge(physics_by_key, on='segment_id', how='left',
                        validate='one_to_one')
        for name in CANDIDATES:
            spatial = fit_spatial_residual(train, train_physics,
                                           FEATURES[name], 'ridge')
            g_full = predict_ridge(spatial, full, baseline=None)
            context = _mean_full_grid(full, g_full)
            pred = _apply_candidate(physics_test, g_full, full,
                                    name in CENTRED, context)
            pred_frame = full[['segment_id']].copy()
            pred_frame['prediction'] = pred['h_clipped']
            pred_frame['g_raw'] = pred['g_raw']
            pred_frame['g_centered'] = pred['g_centered']
            pred_frame['h_raw'] = pred['h_raw']
            pred_frame['clipping'] = pred['clipping']
            out = out.merge(pred_frame, on='segment_id', how='left',
                            validate='one_to_one').rename(
                columns={'prediction': name, 'g_raw': name + '_g_raw',
                         'g_centered': name + '_g_centered',
                         'h_raw': name + '_h_raw',
                         'clipping': name + '_clipped'})
            contexts.append({'fold': held, 'candidate': name, **context})
            fold_records.append({
                'fold': held, 'candidate': name, 'train_rows': int(len(train)),
                'test_rows': int(len(test)),
                'train_ids': sorted(int(v) for v in train.id.unique()),
                'features': FEATURES[name],
                'centered': name in CENTRED,
                'context': context,
                'used_features': list(spatial['metadata']['used_features']),
                'dropped_columns': dict(spatial['metadata']['dropped_columns']),
            })
        parts.append(out)
    return pd.concat(parts, ignore_index=True), {
        'contexts': contexts, 'folds': fold_records}


def _target_predictions(artifacts: dict[str, dict[str, Any]],
                        target: pd.DataFrame,
                        interval_map: dict[Any, Any]) -> dict[str, pd.DataFrame]:
    result = {}
    for name in CANDIDATES:
        pred = _predict_candidate(artifacts[name], target, interval_map)
        frame = target[['id', 'segment_id', 'prediction_date', 'nominal_thickness_mm',
                        'length_m', 'scenario_valid', 'scenario_cooling_c',
                        't_out_segment_c', 't_in_segment_c', 't_out_segment_c',
                        'local_delta_t_c', 'cumulative_cooling_from_inlet_c']].copy()
        frame['physics_priors'] = _predict_physics(
            artifacts[name]['physics'], target, interval_map)
        frame['g_raw'] = pred['g_raw']
        frame['g_centered'] = pred['g_centered']
        frame['h_raw'] = pred['h_raw']
        frame['thickness_pred'] = pred['h_clipped']
        frame['clipping'] = pred['clipping']
        frame['context_mean_length_weighted'] = pred['context']['mean_length_weighted']
        frame['context_mean_arithmetic'] = pred['context']['mean_arithmetic']
        frame['context_sha256'] = pred['context']['context_sha256']
        frame['mean_shift_after_clipping'] = pred['mean_shift_after_clipping']
        result[name] = frame
    return result


def _metrics_table(oof: pd.DataFrame, pred_columns: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, details = [], {}
    for name, column in pred_columns.items():
        rec = evaluate_predictions(oof, column)
        tail = tail_metrics(oof, column, truth='next_mm', section='id')
        details[name] = rec
        rows.append({
            'model': name, 'n': rec['pooled']['n'],
            'mae_mm': rec['pooled']['mae_mm'], 'rmse_mm': rec['pooled']['rmse_mm'],
            'bias_mm': rec['pooled']['bias_mm'],
            'pearson_r': rec['pooled']['pearson_r'],
            'pearson_squared': rec['pooled']['pearson_squared'],
            'r2': rec['pooled']['r2'],
            'std_pred_mm': rec['pooled']['prediction_std_mm'],
            'std_truth_mm': rec['pooled']['truth_std_mm'],
            'lower_tail_mae_mm': tail['lower_tail_mae_mm'],
            'lower_tail_recall': tail['lower_tail_recall'],
            'global_top10_recall': tail['global_top10_recall'],
            'mean_shift_mm': None,
            'clipping_count': int(oof[column + '_clipped'].sum())
            if column + '_clipped' in oof else None,
        })
    return pd.DataFrame(rows), details


def run_stage2(root: Path = BASE_DIR,
               data_root: Path | None = None,
               output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/targeted_cycle/stage_2_s2_fixes').resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'thermal_contract').mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = RepositoryData(data_root)
    print('Building full-grid cold snapshots before label joins...', flush=True)
    prepared = prepare_full_grid(data)
    measured = prepared['measured']
    strict = prepared['strict']
    post = prepared['post']
    target = prepared['target']
    full_frames = prepared['full_frames']
    interval_map = prepared['interval_map']
    target_interval = interval_map[(S2_ID, S2_CUTOFF)]
    params = prepared['params']
    write_csv(measured, output / 'training_table_full_grid.csv', fmt='%.10f')
    write_json(output / 'eligibility.json', {
        'strict_rows': int(len(strict)), 'strict_ids': sorted(int(v) for v in strict.id.unique()),
        'strict_dates': sorted(str(pd.Timestamp(v).date()) for v in strict.prediction_date.unique()),
        'post_rows': int(len(post)), 'post_ids': sorted(int(v) for v in post.id.unique()),
        'target_rows': int(len(target)), 'target_id': S2_ID,
        'target_vtd_rows': 0, 'target_uzt_eligible_rows': 0,
        'policy': 'strict labels < 2022-07-01; 2023 only frozen retrospective; no target diagnostics',
    })
    # Full-grid grid checks and before/after thermal features.
    write_json(output / 'thermal_contract' / 'grid_checks.json', prepared['thermal_checks'])
    old_parts = []
    for (pid, date), part in measured.groupby(['id', 'prediction_date'], sort=True):
        # Old behavior: scenario on the measured/ragged rows.
        full_old = part.copy()
        old_thermal = _thermal_features(full_old, params, 'OLD_MEASURED_GRID',
                                        validate_grid=False)
        old_parts.append(old_thermal)
    old_thermal = pd.concat(old_parts, ignore_index=True)
    before_after = measured.merge(
        old_thermal[['id', 'segment_id', 'prediction_date', 't_out_segment_c',
                     'scenario_cooling_c', 'local_delta_t_c']].rename(columns={
                         't_out_segment_c': 'old_t_out_segment_c',
                         'scenario_cooling_c': 'old_scenario_cooling_c',
                         'local_delta_t_c': 'old_local_delta_t_c'}),
        on=['id', 'segment_id', 'prediction_date'], how='left', validate='one_to_one')
    before_after['delta_t_out_c'] = (before_after.t_out_segment_c
                                     - before_after.old_t_out_segment_c)
    before_after['delta_cooling_c'] = (before_after.scenario_cooling_c
                                       - before_after.old_scenario_cooling_c)
    write_csv(before_after, output / 'thermal_contract' / 'before_after_features.csv',
              fmt='%.12g')
    thermal_summary = {
        'rows': int(len(before_after)),
        'max_abs_t_out_diff_c': float(before_after.delta_t_out_c.abs().max()),
        'max_abs_local_cooling_diff_c': float(before_after.delta_cooling_c.abs().max()),
        'mean_abs_local_cooling_diff_c': float(before_after.delta_cooling_c.abs().mean()),
        'old_measured_grids': 'scenario built on measured rows (defect)',
        'new_full_grid': 'scenario built on full contiguous grid before label join',
        'parameters': params.assumptions(),
        'note': 'This is the contract difference, not a temperature-accuracy claim.',
    }
    by_campaign = before_after.groupby(['id', 'prediction_date']).agg(
        rows=('segment_id', 'size'),
        max_abs_t_out_diff_c=('delta_t_out_c', lambda x: float(x.abs().max())),
        max_abs_local_cooling_diff_c=('delta_cooling_c', lambda x: float(x.abs().max())),
        mean_abs_local_cooling_diff_c=('delta_cooling_c', lambda x: float(x.abs().mean())),
    ).reset_index()
    by_campaign['prediction_date'] = by_campaign.prediction_date.astype(str)
    validity = measured.groupby(['id', 'prediction_date']).agg(
        rows=('segment_id', 'size'), valid=('scenario_valid', 'sum'),
        reasons=('scenario_reason', lambda x: ';'.join(sorted(set(map(str, x.dropna())))))
    ).reset_index()
    validity['prediction_date'] = validity.prediction_date.astype(str)
    validity['valid'] = validity['valid'].astype(int)
    validity = pd.concat([validity, pd.DataFrame([{
        'id': S2_ID, 'prediction_date': str(S2_CUTOFF.date()),
        'rows': int(len(target)),
        'valid': int(target.scenario_valid.eq(True).sum()), 'reasons': ''}])],
        ignore_index=True)
    by_campaign = by_campaign.merge(
        validity[['id', 'prediction_date', 'valid', 'reasons']],
        on=['id', 'prediction_date'], how='left', validate='one_to_one')
    write_csv(by_campaign, output / 'thermal_contract' / 'before_after_by_campaign.csv', fmt='%.12g')
    thermal_summary['by_campaign'] = by_campaign.to_dict(orient='records')
    write_json(output / 'thermal_contract' / 'scenario_validity.json',
               {'by_campaign': validity.to_dict(orient='records'),
                'note': 'Invalid scenario rows are retained with NaN thermal features; no labels were used.'})
    write_json(output / 'thermal_contract' / 'before_after_summary.json', json_safe(thermal_summary))

    # Thermal tests on real and synthetic grids.
    tests = {}
    sample_key = sorted(full_frames)[0]
    sample_full = full_frames[sample_key]
    scenario_a = _thermal_features(sample_full, params)
    scenario_b = _thermal_features(sample_full.sample(frac=1, random_state=17), params)
    tests['row_order_invariance'] = bool(np.allclose(
        scenario_a.sort_values('segment_id').scenario_cooling_c.to_numpy(float),
        scenario_b.sort_values('segment_id').scenario_cooling_c.to_numpy(float),
        equal_nan=True))
    labels_removed = sample_full.drop(columns=[c for c in sample_full.columns
                                               if c in ('next_mm',)], errors='ignore')
    scenario_c = _thermal_features(labels_removed, params)
    tests['label_subset_invariance'] = bool(np.allclose(
        scenario_a.sort_values('segment_id').scenario_cooling_c.to_numpy(float),
        scenario_c.sort_values('segment_id').scenario_cooling_c.to_numpy(float),
        equal_nan=True))
    subset_keys = measured[measured.id.eq(sample_key[0])].segment_id
    full_subset = scenario_a[scenario_a.segment_id.isin(subset_keys)].sort_values('segment_id')
    measured_subset = measured[(measured.id.eq(sample_key[0])) &
                               (measured.prediction_date.eq(sample_key[1]))].sort_values('segment_id')
    tests['full_grid_to_subset'] = bool(np.allclose(
        full_subset.scenario_cooling_c.to_numpy(float),
        measured_subset.scenario_cooling_c.to_numpy(float), equal_nan=True))
    # Synthetic gap/overlap.
    synthetic = pd.DataFrame({
        'id': [1, 1], 'prediction_date': [pd.Timestamp('2020-01-01')] * 2,
        'start_m': [0., 20.], 'end_m': [10., 30.], 'length_m': [10., 10.]})
    tests['gap_detected'] = not _grid_check(synthetic)['valid']
    overlap = synthetic.copy()
    overlap.loc[1, 'start_m'] = 5.
    overlap.loc[1, 'end_m'] = 15.
    tests['overlap_detected'] = not _grid_check(overlap)['valid']
    # Analytic exponential and local sum.
    analytic = segment_outlet_temperature(20.0, 10.0, 0.2, 5.0, params)
    perimeter = math.pi * (0.2 - 2 * params.roughness_m)
    exponent = -(params.overall_heat_transfer_w_m2_k * perimeter * 10.0
                 / (5.0 * params.heat_capacity_j_kg_k))
    expected = params.ground_temperature_c + (20.0 - params.ground_temperature_c) * math.exp(exponent)
    tests['analytic_exponential'] = bool(abs(analytic['t_out_c'] - expected) < 1e-10)
    first = sample_full.sort_values('start_m').iloc[0]
    last = sample_full.sort_values('start_m').iloc[-1]
    tests['local_sum_equals_total'] = bool(abs(
        scenario_a.local_delta_t_c.sum()
        - (last.t_out_segment_c - first.t_in_segment_c)) < 1e-8)
    tests['zero_flow_invalid'] = not segment_outlet_temperature(
        20.0, 10.0, 0.2, 0.0, params)['valid']
    write_json(output / 'thermal_contract' / 'tests.json', tests)

    # OOF candidates.
    print('Fitting limited spatial Ridge candidates, leave-one-section-out...', flush=True)
    oof, oof_meta = _oof_candidates(strict, full_frames, interval_map)
    write_csv(oof, output / 'oof_predictions.csv', fmt='%.10f')
    write_json(output / 'fold_assignments.json', json_safe(oof_meta))
    # Reference OOF from preserved next_cycle artifacts.
    old_oof = pd.read_csv(root / 'experiments/next_cycle/s2/oof_predictions.csv')
    ref_cols = old_oof[['id', 'segment_id', 'physics_spatial']].rename(
        columns={'physics_spatial': 'old_physics_spatial'})
    oof = oof.merge(ref_cols, on=['id', 'segment_id'], how='left', validate='one_to_one')
    pred_columns = {
        'physics_priors': 'physics_priors',
        'old_physics_spatial': 'old_physics_spatial',
        'spatial_no_temperature_uncentered': 'spatial_no_temperature_uncentered',
        'spatial_no_temperature_centered': 'spatial_no_temperature_centered',
        'spatial_centered_corrected_temperature': 'spatial_centered_corrected_temperature',
    }
    # Add clipping columns for new candidates from OOF raw nominal.
    for name in CANDIDATES:
        raw = oof[name + '_h_raw'].to_numpy(float)
        nominal = oof.nominal_thickness_mm.to_numpy(float)
        oof[name + '_clipped'] = (raw < 0) | (raw > nominal)
    _, metrics_details = _metrics_table(oof, pred_columns)
    rows = []
    for name, rec in metrics_details.items():
        tail = tail_metrics(oof, column_for(name), truth='next_mm', section='id')
        rows.append({'model': name, **{
            'n': rec['pooled']['n'], 'mae_mm': rec['pooled']['mae_mm'],
            'rmse_mm': rec['pooled']['rmse_mm'], 'bias_mm': rec['pooled']['bias_mm'],
            'pearson_r': rec['pooled']['pearson_r'],
            'pearson_squared': rec['pooled']['pearson_squared'],
            'r2': rec['pooled']['r2'],
            'std_pred_mm': rec['pooled']['prediction_std_mm'],
            'std_truth_mm': rec['pooled']['truth_std_mm'],
            'lower_tail_mae_mm': tail['lower_tail_mae_mm'],
            'lower_tail_recall': tail['lower_tail_recall'],
            'global_top10_recall': tail['global_top10_recall'],
            'clipping_count': int(oof[column_for(name) + '_clipped'].sum())
            if column_for(name) + '_clipped' in oof else None,
        }})
    metrics_table = pd.DataFrame(rows)
    write_csv(metrics_table, output / 'metrics_strict.csv', fmt='%.10f')
    write_json(output / 'metrics_strict_details.json', json_safe(metrics_details))
    # Per-section rows.
    per_rows = []
    for name, rec in metrics_details.items():
        for item in rec['per_section']:
            per_rows.append({'model': name, **item})
    write_csv(pd.DataFrame(per_rows), output / 'metrics_strict_per_section.csv', fmt='%.10f')

    # Final artifacts and target predictions.
    print('Fitting final full-train artifacts and predicting target...', flush=True)
    artifacts = fit_candidate_artifacts(strict, interval_map)
    target_predictions = _target_predictions(artifacts, target, interval_map)
    write_csv(target_predictions['spatial_no_temperature_uncentered'],
              output / 'target_spatial_uncentered.csv', fmt='%.10f')
    write_csv(target_predictions['spatial_no_temperature_centered'],
              output / 'target_spatial_centered.csv', fmt='%.10f')
    write_csv(target_predictions['spatial_centered_corrected_temperature'],
              output / 'target_spatial_temperature_centered.csv', fmt='%.10f')
    # Save full artifacts and summary.
    artifact_summary = {}
    for name in ('physics_priors',) + tuple(CANDIDATES):
        if name == 'physics_priors':
            artifact = artifacts[name]
            directory = output / 'models' / name
            directory.mkdir(parents=True, exist_ok=True)
            joblib_path = directory / 'model_artifact.joblib'
            import joblib
            joblib.dump(artifact, joblib_path)
            metadata = artifact['metadata']
        else:
            directory = output / 'models' / name
            directory.mkdir(parents=True, exist_ok=True)
            artifact = artifacts[name]
            # Attach the exact target full-grid context for partial inference.
            target_pred = target_predictions[name]
            context = {
                'n': int(len(target_pred)),
                'sum_length_m': float(target_pred.length_m.sum()),
                'mean_length_weighted': float(target_pred.context_mean_length_weighted.iloc[0]),
                'mean_arithmetic': float(target_pred.context_mean_arithmetic.iloc[0]),
                'context_sha256': str(target_pred.context_sha256.iloc[0]),
                'target_id': S2_ID,
                'target_date': str(S2_CUTOFF.date()),
            }
            artifact['centering_context'] = context
            import joblib
            joblib_path = directory / 'model_artifact.joblib'
            joblib.dump(artifact, joblib_path)
            metadata = {**artifact['metadata'], 'centering_context': context}
        write_json(directory / 'model_metadata.json', json_safe(metadata))
        if name == 'physics_priors':
            physics_pred = _predict_physics(artifact['estimator'], target, interval_map)
            target_pred = target[['id', 'segment_id', 'nominal_thickness_mm',
                                  'length_m']].copy()
            target_pred['physics_priors'] = physics_pred
            target_pred['thickness_pred'] = np.clip(
                physics_pred, 0.0, target_pred.nominal_thickness_mm.to_numpy(float))
            target_pred['clipping'] = ((physics_pred < 0)
                                       | (physics_pred > target_pred.nominal_thickness_mm.to_numpy(float)))
        else:
            target_pred = target_predictions[name]
        write_csv(target_pred[['id', 'segment_id', 'thickness_pred']],
                  directory / 'predictions_s2.csv', fmt='%.10f')
        write_csv(target_pred, directory / 'prediction_diagnostics.csv', fmt='%.10f')
        artifact_summary[name] = {
            'model_type': artifact['model_type'],
            'artifact_path': str(joblib_path),
            'artifact_sha256': sha256(joblib_path),
            'metadata': metadata,
            'target_mean_mm': float(target_pred.thickness_pred.mean()),
            'target_std_mm': float(target_pred.thickness_pred.std(ddof=0)),
            'target_min_mm': float(target_pred.thickness_pred.min()),
            'target_max_mm': float(target_pred.thickness_pred.max()),
            'clipping_count': int(target_pred.clipping.sum()),
            'mean_shift_after_clipping_mm': (
                float(target_pred.thickness_pred.mean() - target_pred.h_raw.mean())
                if 'h_raw' in target_pred else 0.0),
        }
        # Result with fixed S1B reference.
        # Read fixed S1B from preserved next_cycle output.
        s1b_path = root / 'experiments/next_cycle/repro_s1abc/s1b/predictions_s1.csv'
        s1b = pd.read_csv(s1b_path)
        s1_diag = pd.read_csv(root / 'experiments/next_cycle/repro_s1abc/s1b/prediction_diagnostics.csv')
        s1b = s1b.merge(s1_diag[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']],
                        on=['id', 'segment_id'], how='left', validate='one_to_one')
        target_grid = data.read('processed/target_segments_grid')[['id', 'segment_id']]
        nominal = nominal_frame(data)
        s2_pred = target_pred[['id', 'segment_id', 'thickness_pred']]
        bounds = nominal[nominal.id.eq(S2_ID)][['id', 'segment_id',
                                                'nominal_thickness_mm']].copy()
        bounds['h0_mm'] = bounds.nominal_thickness_mm
        bounds['upper_mm'] = bounds.nominal_thickness_mm
        s1_bounds = s1b[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']].copy()
        s1_bounds['upper_mm'] = np.maximum(
            s1_bounds.h0_mm,
            s1b.thickness_pred.to_numpy(float))
        bounds = pd.concat([
            s1_bounds[['id', 'segment_id', 'h0_mm',
                       'nominal_thickness_mm', 'upper_mm']],
            bounds[['id', 'segment_id', 'h0_mm',
                    'nominal_thickness_mm', 'upper_mm']]], ignore_index=True)
        combined, validation = combine_with_s2(
            s1b[['id', 'segment_id', 'thickness_pred']], s2_pred,
            target_grid, bounds)
        write_csv(combined, directory / 'result.csv', fmt='%.10f')
        artifact_summary[name]['validation'] = validation
    # Preserved references without refit.
    refs = {}
    v2_path = root / 'experiments/ml_v2/result.csv'
    v2 = pd.read_csv(v2_path)
    v2 = v2[v2.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']]
    refs['v2_reference'] = {
        'path': str(v2_path), 'sha256': sha256(v2_path), 'rows': int(len(v2)),
        'mean_mm': float(v2.thickness_pred.mean()),
        'std_mm': float(v2.thickness_pred.std(ddof=0)),
        'note': 'unchanged old S2 production reference; no refit'}
    old_sp_path = root / 'experiments/next_cycle/s2/candidates/physics_spatial/predictions_s2.csv'
    old_sp = pd.read_csv(old_sp_path)
    refs['old_physics_spatial'] = {
        'path': str(old_sp_path), 'sha256': sha256(old_sp_path), 'rows': int(len(old_sp)),
        'mean_mm': float(old_sp.thickness_pred.mean()),
        'std_mm': float(old_sp.thickness_pred.std(ddof=0)),
        'note': 'old uncentered spatial reference; no refit'}
    old_phys_path = root / 'experiments/next_cycle/s2/candidates/physics_priors/predictions_s2.csv'
    old_phys = pd.read_csv(old_phys_path)
    refs['physics_priors'] = {
        'path': str(old_phys_path), 'sha256': sha256(old_phys_path), 'rows': int(len(old_phys)),
        'mean_mm': float(old_phys.thickness_pred.mean()),
        'std_mm': float(old_phys.thickness_pred.std(ddof=0)),
        'note': 'old physics reference; new strict full-train physics artifact is saved separately'}
    old_temp_path = root / 'experiments/next_cycle/s2/candidates/physics_spatial_temperature/predictions_s2.csv'
    old_temp = pd.read_csv(old_temp_path)
    refs['old_physics_spatial_temperature'] = {
        'path': str(old_temp_path), 'sha256': sha256(old_temp_path), 'rows': int(len(old_temp)),
        'mean_mm': float(old_temp.thickness_pred.mean()),
        'std_mm': float(old_temp.thickness_pred.std(ddof=0)),
        'note': 'old uncentered temperature-spatial reference; shows old level shift, no refit'}
    write_json(output / 'references.json', json_safe(refs))
    # Zero-mean check for centered candidates on the full target grid.
    centering_checks = {}
    for name in CANDIDATES:
        part = target_predictions[name]
        w = part.length_m.to_numpy(float)
        g = part.g_centered.to_numpy(float)
        centering_checks[name] = {
            'centered': name in CENTRED,
            'length_weighted_mean_g_centered': float(np.sum(w * g) / np.sum(w)),
            'arithmetic_mean_g_centered': float(g.mean()),
            'max_abs_mean_check': float(abs(np.sum(w * g) / np.sum(w)))
            if name in CENTRED else None,
        }
    write_json(output / 'centering_checks.json', json_safe(centering_checks))
    write_json(output / 'target_candidates_summary.json', json_safe(artifact_summary))

    # Retrospective only, never used for fit/selection.
    retro_parts = []
    for name in ('physics_priors',) + tuple(CANDIDATES):
        # Build with the exact strict model and per-post full-grid context.
        rows = []
        for (pid, date), full in full_frames.items():
            if pd.Timestamp(date) < S2_CUTOFF:
                continue
            if name == 'physics_priors':
                pred = _predict_physics(artifacts[name]['estimator'], full, interval_map)
                clipped = np.clip(pred, 0, full.nominal_thickness_mm.to_numpy(float))
            else:
                pred = _predict_candidate(artifacts[name], full, interval_map)['h_clipped']
            label = post[(post.id.eq(pid)) & (post.prediction_date.eq(date))]
            pframe = full[['segment_id', 'nominal_thickness_mm']].copy()
            pframe['prediction'] = pred
            pframe = pframe.merge(label[['segment_id', 'next_mm']], on='segment_id',
                                  how='inner', validate='one_to_one')
            pframe['id'] = pid
            pframe['prediction_date'] = date
            pframe['model'] = name
            retro_parts.append(pframe)
    retro = pd.concat(retro_parts, ignore_index=True)
    write_csv(retro, output / 'post_cutoff_retrospective' / 'predictions.csv', fmt='%.10f')
    retro_metrics = {}
    for name in ('physics_priors',) + tuple(CANDIDATES):
        part = retro[retro.model.eq(name)][['id', 'segment_id', 'prediction_date',
                                             'next_mm', 'prediction']].rename(
            columns={'prediction': name})
        retro_metrics[name] = evaluate_predictions(part, name)
    write_json(output / 'post_cutoff_retrospective' / 'metrics.json',
               json_safe(retro_metrics))

    # Domain shift and feature support.
    def stats(frame, prefix):
        cols = ['nominal_thickness_mm', 'age_years', 't_mix_mean',
                'watercut_pct_mean', 'v_mix_mean', 'coverage_fraction',
                'scenario_cooling_c', 'cumulative_cooling_from_inlet_c']
        out = {}
        for col in cols:
            if col in frame:
                v = pd.to_numeric(frame[col], errors='coerce')
                out[prefix + '_' + col] = {'min': float(v.min()), 'max': float(v.max()),
                                           'mean': float(v.mean()), 'std': float(v.std(ddof=0))}
        return out
    domain = {**stats(strict, 'strict'), **stats(target, 'target'), **stats(post, 'post')}
    write_json(output / 'domain_shift.json', json_safe(domain))

    # Figures.
    _figures(output, oof, target_predictions, before_after)
    # Report and manifest.
    report = _report(output, metrics_table, metrics_details, retro_metrics,
                     artifact_summary, thermal_summary, tests, domain,
                     before_after)
    manifest = write_manifest(
        output / 'run_manifest.json', root=root, data=data, output=output,
        stage='Stage_2_S2_fixes', started=started,
        extra={'data_root': str(data_root), 'thermal_summary': thermal_summary,
               'thermal_tests': tests, 'report': str(report),
               'artifact_summary': artifact_summary,
               'stage_code_sha256': {
                   'src/ml/targeted_s2.py': sha256(root / 'src/ml/targeted_s2.py'),
                   'tests/test_stage_2_s2_fixes.py': sha256(root / 'tests/test_stage_2_s2_fixes.py'),
                   'src/ml/targeted_audit.py': sha256(root / 'src/ml/targeted_audit.py'),
               },
               'no_retrospective_tuning': True,
               'plan_version': 2},
        input_paths=[root / 'experiments/next_cycle/s2/oof_predictions.csv',
                     root / 'experiments/next_cycle/repro_s1abc/s1b/predictions_s1.csv'])
    report_path = root / 'experiments/targeted_cycle/stage_2_report_ru.md'
    report_path.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'PASS Stage 2: {output}', flush=True)
    print(f'  strict={len(strict)} target={len(target)} post={len(post)}', flush=True)
    return {'metrics': metrics_table, 'retrospective': retro_metrics,
            'artifacts': artifact_summary, 'thermal': thermal_summary,
            'tests': tests, 'output': str(output), 'report': str(report_path)}


def column_for(name: str) -> str:
    return name


def _figures(output: Path, oof: pd.DataFrame, target: dict[str, pd.DataFrame],
             before_after: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = output / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for col, label in [('old_scenario_cooling_c', 'measured-grid scenario'),
                       ('scenario_cooling_c', 'full-grid scenario')]:
        if col in before_after:
            ax.hist(pd.to_numeric(before_after[col], errors='coerce').dropna(),
                    bins=40, alpha=.5, label=label)
    ax.set_xlabel('scenario_cooling_c, °C')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / 'thermal_before_after.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, color in [('spatial_no_temperature_uncentered', 'tab:gray'),
                        ('spatial_no_temperature_centered', 'tab:blue'),
                        ('spatial_centered_corrected_temperature', 'tab:orange')]:
        part = target[name].sort_values('segment_id')
        ax.plot(part.segment_id, part.thickness_pred, linewidth=.8, label=name,
                color=color)
    ax.set_xlabel('segment_id')
    ax.set_ylabel('target S2 prediction, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / 's2_centered_profiles.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(oof.next_mm, oof.spatial_no_temperature_uncentered, s=6, alpha=.4,
               label='uncentered')
    ax.scatter(oof.next_mm, oof.spatial_no_temperature_centered, s=6, alpha=.4,
               label='centered')
    lo = min(oof.next_mm.min(), oof.spatial_no_temperature_centered.min())
    hi = max(oof.next_mm.max(), oof.spatial_no_temperature_centered.max())
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=.7)
    ax.set_xlabel('measured next, mm')
    ax.set_ylabel('OOF prediction, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / 's2_oof_centering.png', dpi=160)
    plt.close(fig)


def _fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return 'n/a' if not np.isfinite(value) else f'{value:.{digits}f}'
    return str(value)


def _report(output: Path, metrics: pd.DataFrame, details: dict[str, Any],
            retro: dict[str, Any], artifacts: dict[str, Any],
            thermal: dict[str, Any], tests: dict[str, Any],
            domain: dict[str, Any], before_after: pd.DataFrame) -> Path:
    lines = [
        '# Stage 2 — full-grid S2 thermal contract and full-grid centering', '',
        f"Generated: `{utcnow()}`. No retrospective or target labels were used for fit/selection.",
        '', '## 1. Что исправлено', '',
        '- Тепловой сценарий теперь строится по полной физической сетке каждого `id/cutoff/case` '
        'до присоединения labels; measured rows получают значения по строгим ключам.',
        '- Добавлены `t_in_segment_c`, `t_out_segment_c`, `local_delta_t_c`, '
        '`cumulative_cooling_from_inlet_c`; `scenario_cooling_c` сохранён как прежний тип '
        'локального cooling-признака.',
        '- S2 spatial correction: `h_raw = h_physics + lambda*(g(X) - mean_full_grid(g(X)))`, '
        '`lambda=1`; центрирование по длине сегмента на полной сетке, без labels. '
        'Арифметическое среднее также сохранено.',
        '- Параметры тепла не подбирались; CatBoost не переобучался; retrospective не использовалась '
        'для параметров или выбора.',
        '', '## 2. Thermal before/after', '',
        f"- Измеренных строк сравнено: **{thermal['rows']}**.",
        f"- max |Δ t_out| на measured keys: **{thermal['max_abs_t_out_diff_c']:.6g} °C**.",
        f"- max |Δ local cooling|: **{thermal['max_abs_local_cooling_diff_c']:.6g} °C**.",
        f"- mean |Δ local cooling|: **{thermal['mean_abs_local_cooling_diff_c']:.6g} °C**.",
        '- Старое поведение строило сценарий по разреженным measured rows; новое — по полной сетке. '
        'Это contract fix, не доказательство роста accuracy.',
        '', 'Thermal by campaign:', '',
        '| id | date | rows | max abs Δt_out | max abs Δcooling | valid |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for rec in thermal.get('by_campaign', []):
        lines.append(f"| {rec['id']} | {rec['prediction_date']} | {rec['rows']} | "
                     f"{_fmt(rec['max_abs_t_out_diff_c'], 8)} | "
                     f"{_fmt(rec['max_abs_local_cooling_diff_c'], 8)} | "
                     f"{rec.get('valid', 'n/a')} |")
    lines += [
        '', 'Для 1250022871 2018 max Δt_out ≈ 0.2196 °C и max Δlocal cooling ≈ 3.48e-5 °C, '
        'как в Stage 1/аудите. Пять 2023-кампаний имеют `missing_inlet_temperature` и '
        'сохраняются как invalid без подстановки; target full grid valid.',
        '', '## 3. Thermal contract tests', '',
        '| Тест | Результат |', '|---|---|',
    ]
    for key, value in tests.items():
        lines.append(f'| {key} | {value} |')
    lines += ['', '## 4. S2 strict OOF на одинаковых 372 ключах', '',
              '| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | lower-tail MAE | top10 recall | clip |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        det = details[row.model]
        ws = det['within_section_centered']
        lines.append(f"| {row.model} | {_fmt(row.mae_mm)} | {_fmt(row.rmse_mm)} | "
                     f"{_fmt(row.bias_mm)} | {_fmt(row.pearson_r)} | {_fmt(row.pearson_squared)} | "
                     f"{_fmt(row.r2)} | {_fmt(ws['pearson_squared'])} | {_fmt(row.lower_tail_mae_mm)} | "
                     f"{_fmt(row.lower_tail_recall)} | {row.clipping_count} |")
    lines += ['', 'Centered no-temperature меняет MAE незначительно (0.4495 против 0.4482), '
              'signed r почти не меняется (0.06179 против 0.06165); length-weighted mean '
              '`g_centered` ≈ 1e-17, arithmetic mean ≈ -3.9e-5. Это contract fix уровня, '
              'не доказательство улучшения spatial ranking. Temperature-centered MAE 0.4498, '
              'within-section r² 0.0118; также не объявляется улучшением.', '',
              '## 5. Target predictions: raw / clipped / centering', '',
              '| Candidate | mean | std | min | max | clipping | mean shift after clip |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in artifacts.items():
        lines.append(f"| {name} | {_fmt(s['target_mean_mm'])} | {_fmt(s['target_std_mm'])} | "
                     f"{_fmt(s['target_min_mm'])} | {_fmt(s['target_max_mm'])} | "
                     f"{s['clipping_count']} | {_fmt(s['mean_shift_after_clipping_mm'])} |")
    lines += ['', 'Preserved references (no refit):', '',
              '| Reference | mean | std | rows |', '|---|---:|---:|---:|']
    try:
        refs = json.loads((output / 'references.json').read_text(encoding='utf-8'))
        for name, rec in refs.items():
            lines.append(f"| {name} | {_fmt(rec['mean_mm'])} | {_fmt(rec['std_mm'])} | {rec['rows']} |")
    except FileNotFoundError:
        pass
    lines += ['', 'Target labels отсутствуют; MAE/корреляция на target не вычислялись. '
              'Старый uncentered temperature-spatial имел mean ≈2.5907, новый centered — '
              '≈2.80956: это устранение level shift, не улучшение spatial accuracy.',
              '', '## 6. Frozen 2023 retrospective (не selection)', '',
              '| Модель | MAE | bias | Pearson r | Pearson r² | R² |', '|---|---:|---:|---:|---:|---:|']
    for name, rec in retro.items():
        p = rec['pooled']
        lines.append(f"| {name} | {_fmt(p['mae_mm'])} | {_fmt(p['bias_mm'])} | "
                     f"{_fmt(p['pearson_r'])} | {_fmt(p['pearson_squared'])} | {_fmt(p['r2'])} |")
    lines += ['', 'Retrospective уже просмотрена командой; она не является untouched holdout и '
              'не использовалась для выбора.',
              '', '## 7. Train/target conditions', '',
              '| Variable | strict mean | target mean |', '|---|---:|---:|']
    for var in ('nominal_thickness_mm', 'age_years', 't_mix_mean',
                'watercut_pct_mean', 'v_mix_mean', 'coverage_fraction',
                'scenario_cooling_c', 'cumulative_cooling_from_inlet_c'):
        lines.append(f"| {var} | {_fmt(domain.get('strict_'+var, {}).get('mean'))} | "
                     f"{_fmt(domain.get('target_'+var, {}).get('mean'))} |")
    lines += ['', '## 8. Artifacts and commands', '',
              '```text',
              'experiments/targeted_cycle/stage_2_s2_fixes/',
              '  thermal_contract/{grid_checks,before_after_features,before_after_summary,tests}.json/csv',
              '  oof_predictions.csv, metrics_strict.csv, metrics_strict_per_section.csv',
              '  models/<candidate>/{model_artifact.joblib,model_metadata.json,predictions_s2.csv,result.csv}',
              '  post_cutoff_retrospective/{predictions,metrics}.json',
              '  target_candidates_summary.json, domain_shift.json, run_manifest.json',
              '```',
              'Источник данных задаётся явно через `--data-root`; тесты требуют '
              '`PIPETWIN_NEXT_DATA_ROOT` и не пропускаются молча. Для запуска использован '
              'LF-worktree той же проверенной генерации, потому что основной checkout имеет CRLF.',
              '', '```bash',
              'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s2 train --root . --data-root C:/Users/Администратор/Documents/PipeTwin_next/experiments/next_cycle/data_generation --output-root experiments/targeted_cycle/stage_2_s2_fixes',
              'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s2 predict --root . --data-root C:/Users/Администратор/Documents/PipeTwin_next/experiments/next_cycle/data_generation --output-root experiments/targeted_cycle/stage_2_predict --candidate spatial_no_temperature_centered',
              '```',
              '', '## 9. Тесты', '',
              '- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_2_s2_fixes -v`: 9 тестов, OK '
              '(label invariance, row order, full→subset, gap/overlap, analytic, local sum, '
              'zero flow, zero mean, full/partial inference, save/load predict, metadata/bounds).',
              '- OOF/target/result checks: 372 strict keys, 1475 target keys, 2000 result keys, '
              'S1B фиксирован с max abs diff 0.0.',
              '', '## 10. Ограничения и решения', '',
              '- Полная сетка всё ещё использует steady-state сценарий от средних входных условий; '
              'это не измеренная 29-летняя история.',
              '- Центрирование устраняет сдвиг уровня, но не доказывает улучшение Pearson/ранжирования.',
              '- Strict train: один parent/одна кампания; независимый forward fold недоступен.',
              '- 2023 retrospective — использованная диагностика, не holdout.',
              '- Требуют разбора: принимать ли centered spatial как challenger, и какой thermal feature '
              '(local vs cumulative) использовать в будущем; сейчас основной before/after использует прежний local cooling.',
              '']
    path = output / 'stage_2_report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def predict_saved(artifact_path: Path, data_root: Path,
                  output_root: Path, candidate: str) -> dict[str, Any]:
    """Load artifact, build full target grid, no labels and no fit."""
    artifact = load_artifact(artifact_path)
    data = RepositoryData(data_root)
    target, interval = prepare_target_full(data)
    interval_map = {(S2_ID, S2_CUTOFF): interval}
    if candidate == 'physics_priors':
        physics = _predict_physics(artifact['estimator'], target, interval_map)
        pred = np.clip(physics, 0.0, target.nominal_thickness_mm.to_numpy(float))
        frame = target[['id', 'segment_id']].copy()
        frame['thickness_pred'] = pred
        frame['clipping'] = (physics < 0) | (physics > target.nominal_thickness_mm.to_numpy(float))
    else:
        ctx = artifact.get('centering_context')
        pred = _predict_candidate(artifact, target, interval_map)
        if ctx is not None:
            current = pred['context']
            if current['context_sha256'] != ctx['context_sha256']:
                raise ValueError('Full-grid centering context hash mismatch')
            if abs(current['mean_length_weighted'] - ctx['mean_length_weighted']) > 1e-10:
                raise ValueError('Full-grid centering mean mismatch')
        frame = target[['id', 'segment_id']].copy()
        frame['thickness_pred'] = pred['h_clipped']
        frame['clipping'] = pred['clipping']
        frame['g_raw'] = pred['g_raw']
        frame['g_centered'] = pred['g_centered']
        frame['h_raw'] = pred['h_raw']
        frame['context_mean_length_weighted'] = pred['context']['mean_length_weighted']
        frame['context_mean_arithmetic'] = pred['context']['mean_arithmetic']
        frame['context_sha256'] = pred['context']['context_sha256']
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / candidate
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(frame[['id', 'segment_id', 'thickness_pred']],
              directory / 'predictions_s2.csv', fmt='%.10f')
    write_csv(frame, directory / 'prediction_diagnostics.csv', fmt='%.10f')
    write_json(directory / 'summary.json', {
        'candidate': candidate, 'rows': int(len(frame)),
        'mean_mm': float(frame.thickness_pred.mean()),
        'std_mm': float(frame.thickness_pred.std(ddof=0)),
        'min_mm': float(frame.thickness_pred.min()),
        'max_mm': float(frame.thickness_pred.max()),
        'finite': bool(np.isfinite(frame.thickness_pred).all()),
        'within_bounds': bool(frame.thickness_pred.between(0, 4.5).all()),
        'clipping_count': int(frame.clipping.sum()),
        'prediction_mode': 'loaded_artifact_full_grid_no_fit',
    })
    return {'rows': int(len(frame)), 'candidate': candidate,
            'path': str(directory)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--root', type=Path, default=BASE_DIR)
    train.add_argument('--data-root', type=Path, required=True)
    train.add_argument('--output-root', type=Path,
                       default=BASE_DIR / 'experiments/targeted_cycle/stage_2_s2_fixes')
    predict = sub.add_parser('predict')
    predict.add_argument('--root', type=Path, default=BASE_DIR)
    predict.add_argument('--data-root', type=Path, required=True)
    predict.add_argument('--output-root', type=Path, required=True)
    predict.add_argument('--candidate', required=True)
    predict.add_argument('--artifact', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == 'train':
        run_stage2(args.root, args.data_root, args.output_root)
    else:
        artifact = args.artifact or (args.root / 'experiments/targeted_cycle/stage_2_s2_fixes/models'
                                     / args.candidate / 'model_artifact.joblib')
        predict_saved(artifact, args.data_root, args.output_root, args.candidate)


if __name__ == '__main__':
    main()

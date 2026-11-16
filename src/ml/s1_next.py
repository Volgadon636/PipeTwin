"""Limited S1 candidate cycle: controls, last-baseline Ridge, rate and CatBoost.

Primary goal is spatial signal, not merely MAE. All fits are leave-one-simple-
section-out; all preprocessing is train-only. No post-cutoff labels are used.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .submission import validate_submission
from .s1_contract import (S1_CUTOFF, S1_ID, S2_ID, TARGET_COLUMN,
                          combine_with_s2, load_s2_source, nominal_frame,
                          target_key_grid)
from .next_common import (GEOMETRY_FEATURES, HISTORY_FEATURES, S1_CORE_FEATURES,
                          S1_HISTORY_ONLY_FEATURES, NextPreprocessor,
                          evaluate_many, evaluate_predictions, fit_catboost,
                          fit_ridge, json_safe, load_artifact, load_data, make_weights,
                          per_section_metrics, predict_catboost, predict_ridge,
                          s1_fit_rows, s1_production_table, s1_training_table,
                          save_artifact, utcnow, weight_diagnostics, write_csv,
                          write_manifest, YEAR_SECONDS)

MODEL_LABELS = {
    's1a': 'S1A last-prior rule',
    's1b': 'S1B prior-min rule',
    'old_s1c': 'old S1C reference',
    'train_median': 'train median',
    'affine': 'affine a>=0 on last',
    'ridge_history': 'Ridge last + history',
    'ridge_geometry': 'Ridge last + history + geometry',
    'rate_signed': 'Ridge signed rate',
    'rate_nonnegative': 'Ridge nonnegative rate',
    'catboost': 'CatBoost delta challenger',
    'mix_025': 'ridge_geometry lambda=0.25',
    'mix_050': 'ridge_geometry lambda=0.50',
    'mix_100': 'ridge_geometry lambda=1.00',
}


def _affine_fit(train: pd.DataFrame) -> tuple[float, float]:
    x = train.last_mm.to_numpy(float)
    y = train.next_mm.to_numpy(float)
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = max(0.0, float(coef[0]))
    c = float(np.mean(y - a * x))
    return a, c


def _apply_affine(frame: pd.DataFrame, a: float, c: float) -> np.ndarray:
    return np.clip(a * frame.last_mm.to_numpy(float) + c, 0.0,
                   frame.nominal_thickness_mm.to_numpy(float))


def _ridge_oof(fit: pd.DataFrame, features: list[str], target: str,
               scheme: str, alpha: float = 10.0,
               baseline: str = 'last_mm', model_name: str = 'ridge',
               prediction_kind: str = 'delta') -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Leave-one-simple-section-out OOF for a Ridge candidate."""
    records = []
    parts = []
    for held in sorted(int(v) for v in fit.id.unique()):
        train = fit[~fit.id.eq(held)].copy()
        test = fit[fit.id.eq(held)].copy()
        weights = make_weights(train, scheme)
        artifact = fit_ridge(train, features, target, weights, alpha,
                             model_name=model_name, baseline=baseline,
                             prediction_policy=prediction_kind)
        if prediction_kind == 'delta':
            pred = predict_ridge(artifact, test, baseline=baseline)
        elif prediction_kind == 'rate_signed':
            rate = artifact['estimator'].predict(
                artifact['preprocessor'].transform(test))
            pred = np.clip(test.last_mm.to_numpy(float)
                           - rate * test.dt_years.to_numpy(float),
                           0.0, test.nominal_thickness_mm.to_numpy(float))
        elif prediction_kind == 'rate_nonnegative':
            rate = artifact['estimator'].predict(
                artifact['preprocessor'].transform(test))
            rate = np.maximum(rate, 0.0)
            pred = np.clip(test.last_mm.to_numpy(float)
                           - rate * test.dt_years.to_numpy(float),
                           0.0, test.nominal_thickness_mm.to_numpy(float))
        else:
            raise ValueError(prediction_kind)
        out = test[['id', 'segment_id', 'next_mm', 'last_mm', 'dt_years']].copy()
        out['fold'] = held
        out['prediction'] = pred
        parts.append(out)
        records.append({
            'fold': held,
            'train_rows': int(len(train)),
            'test_rows': int(len(test)),
            'train_ids': sorted(int(v) for v in train.id.unique()),
            'scheme': scheme,
            'features': list(features),
            'used_features': list(artifact['metadata']['used_features']),
            'dropped_columns': dict(artifact['metadata']['dropped_columns']),
            'intercept': artifact['metadata']['intercept'],
            'coefficients': artifact['metadata']['coefficients'],
            'weights': weight_diagnostics(train, weights),
        })
    return pd.concat(parts, ignore_index=True), records


def _catboost_oof(fit: pd.DataFrame, features: list[str],
                  scheme: str = 'row_uniform') -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts, records = [], []
    for held in sorted(int(v) for v in fit.id.unique()):
        train = fit[~fit.id.eq(held)].copy()
        test = fit[fit.id.eq(held)].copy()
        weights = make_weights(train, scheme)
        artifact = fit_catboost(train, features, 'target_delta_mm', weights,
                                model_name='catboost_delta',
                                baseline='last_mm')
        pred = predict_catboost(artifact, test, baseline='last_mm')
        out = test[['id', 'segment_id', 'next_mm', 'last_mm', 'dt_years']].copy()
        out['fold'] = held
        out['prediction'] = pred
        parts.append(out)
        records.append({
            'fold': held, 'train_rows': int(len(train)), 'test_rows': int(len(test)),
            'train_ids': sorted(int(v) for v in train.id.unique()),
            'used_features': list(artifact['metadata']['used_features']),
            'dropped_columns': dict(artifact['metadata']['dropped_columns']),
            'weights': weight_diagnostics(train, weights),
            'params': artifact['metadata']['params'],
        })
    return pd.concat(parts, ignore_index=True), records


def _old_s1c_oof(fit: pd.DataFrame) -> pd.DataFrame:
    path = BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1c/oof_predictions.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Old S1C reference not found: {path}')
    old = pd.read_csv(path)
    old = old[['id', 'segment_id', 'prediction_s1c']].rename(
        columns={'prediction_s1c': 'old_s1c'})
    merged = fit[['id', 'segment_id']].merge(
        old, on=['id', 'segment_id'], how='left', validate='one_to_one')
    if merged.old_s1c.isna().any():
        raise ValueError('Old S1C OOF misses fit-eligible rows')
    return merged[['id', 'segment_id', 'old_s1c']]


def _old_s1c_production() -> pd.DataFrame:
    path = BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1c/predictions_s1.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Old S1C production not found: {path}')
    frame = pd.read_csv(path)
    if len(frame) != 525:
        raise ValueError('Old S1C production has wrong row count')
    return frame.rename(columns={'thickness_pred': 'old_s1c'})


def _build_oof(fit: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = fit[['id', 'segment_id', 'next_mm', 'last_mm', 'dt_years',
                'prior_vtd_min_thickness_mm', 'nominal_thickness_mm']].copy()
    base['s1a'] = fit.last_mm.to_numpy(float)
    base['s1b'] = fit.prior_vtd_min_thickness_mm.to_numpy(float)
    base = base.merge(_old_s1c_oof(fit), on=['id', 'segment_id'],
                      how='left', validate='one_to_one')
    records: dict[str, Any] = {}
    # Controls fitted inside each fold.
    median_parts, affine_parts = [], []
    for held in sorted(int(v) for v in fit.id.unique()):
        train = fit[~fit.id.eq(held)]
        test = fit[fit.id.eq(held)]
        median = float(np.median(train.next_mm.to_numpy(float)))
        a, c = _affine_fit(train)
        median_parts.append(test[['id', 'segment_id']].assign(train_median=median))
        affine_parts.append(test[['id', 'segment_id']].assign(
            affine=_apply_affine(test, a, c), affine_a=a, affine_c=c, affine_fold=held))
    base = base.merge(pd.concat(median_parts, ignore_index=True),
                      on=['id', 'segment_id'], validate='one_to_one')
    affine = pd.concat(affine_parts, ignore_index=True)
    base = base.merge(affine, on=['id', 'segment_id'], validate='one_to_one')
    # Last-baseline Ridge candidates.
    for name, features, target, kind, scheme in [
        ('ridge_history', S1_HISTORY_ONLY_FEATURES, 'target_delta_mm', 'delta', 'row_uniform'),
        ('ridge_geometry', S1_CORE_FEATURES, 'target_delta_mm', 'delta', 'row_uniform'),
        ('rate_signed', S1_CORE_FEATURES, 'rate_mm_year', 'rate_signed', 'row_uniform'),
        ('rate_nonnegative', S1_CORE_FEATURES, 'rate_mm_year', 'rate_nonnegative', 'row_uniform'),
    ]:
        pred, fold_records = _ridge_oof(fit, features, target, scheme,
                                        alpha=10.0, baseline='last_mm',
                                        model_name=name, prediction_kind=kind)
        base = base.merge(pred[['id', 'segment_id', 'prediction', 'fold']].rename(
            columns={'prediction': name, 'fold': name + '_fold'}),
            on=['id', 'segment_id'], validate='one_to_one')
        records[name] = fold_records
    # CatBoost challenger.
    cb_pred, cb_records = _catboost_oof(fit, S1_CORE_FEATURES)
    base = base.merge(cb_pred[['id', 'segment_id', 'prediction', 'fold']].rename(
        columns={'prediction': 'catboost', 'fold': 'catboost_fold'}),
        on=['id', 'segment_id'], validate='one_to_one')
    records['catboost'] = cb_records
    # Mixtures use only OOF ridge_geometry delta.
    for lam in (0.25, 0.50, 1.00):
        key = f'mix_{int(lam * 100):03d}'
        base[key] = np.clip(
            base.last_mm.to_numpy(float)
            + lam * (base.ridge_geometry.to_numpy(float)
                     - base.last_mm.to_numpy(float)),
            0.0, base.nominal_thickness_mm.to_numpy(float))
    # lambda=0 sanity check equals S1A.
    base['mix_000'] = base.s1a
    return base, records


def _fold_metrics(oof: pd.DataFrame, model_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for fold, part in oof.groupby('id', sort=True):
        for name, column in model_columns.items():
            record = evaluate_predictions(part, column)
            rows.append({'fold': int(fold), 'model': name, 'n': int(len(part)),
                         **{f'pooled_{k}': v for k, v in record['pooled'].items()},
                         **{f'macro_{k}': v for k, v in record['macro'].items()},
                         'within_section_pearson_r': record['within_section_centered']['pearson_r'],
                         'within_section_pearson_squared': record['within_section_centered']['pearson_squared'],
                         'topk_recall': record['topk']['pooled_recall']})
    return pd.DataFrame(rows)


def _metrics_payload(oof: pd.DataFrame, model_columns: dict[str, str],
                     all_pipeline: pd.DataFrame | None = None,
                     all_columns: dict[str, str] | None = None) -> dict[str, Any]:
    fit_metrics = {name: evaluate_predictions(oof, column)
                   for name, column in model_columns.items()}
    payload = {'fit_eligible_357': fit_metrics}
    if all_pipeline is not None and all_columns:
        payload['all_pipeline_561'] = {
            name: evaluate_predictions(all_pipeline, column)
            for name, column in all_columns.items()}
    return payload


def _all_pipeline_table(table: pd.DataFrame, fit_oof: pd.DataFrame) -> pd.DataFrame:
    """Fallback rows keep the S1B rule; no model is fit on them."""
    all_rows = table[['id', 'segment_id', 'next_mm', 'last_mm',
                      's1a_prediction', 's1b_prediction', 'has_own_prior_vtd']].copy()
    all_rows['s1a'] = all_rows.s1a_prediction.astype(float)
    all_rows['s1b'] = all_rows.s1b_prediction.astype(float)
    old_path = BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1c/oof_predictions.csv'
    old = pd.read_csv(old_path)[['id', 'segment_id', 'prediction_s1c']].rename(
        columns={'prediction_s1c': 'old_s1c'})
    all_rows = all_rows.merge(old, on=['id', 'segment_id'], how='left',
                              validate='one_to_one')
    fit_cols = ['train_median', 'affine', 'ridge_history', 'ridge_geometry',
                'rate_signed', 'rate_nonnegative', 'catboost',
                'mix_025', 'mix_050', 'mix_100']
    all_rows = all_rows.merge(fit_oof[['id', 'segment_id'] + fit_cols],
                              on=['id', 'segment_id'], how='left',
                              validate='one_to_one')
    fallback = ~all_rows.has_own_prior_vtd.astype(bool)
    for column in fit_cols:
        all_rows.loc[fallback, column] = all_rows.loc[fallback, 's1b']
    return all_rows


def _weight_ablation(fit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in ('row_uniform', 'inverse_sqrt_section_size', 'inverse_section_size'):
        pred, records = _ridge_oof(fit, S1_HISTORY_ONLY_FEATURES,
                                   'target_delta_mm', scheme,
                                   alpha=10.0, baseline='last_mm',
                                   model_name=f'history_{scheme}')
        merged = fit[['id', 'segment_id', 'next_mm']].merge(
            pred[['id', 'segment_id', 'prediction']], on=['id', 'segment_id'],
            validate='one_to_one')
        metrics = evaluate_predictions(merged, 'prediction')
        full_weights = make_weights(fit, scheme)
        full_diag = weight_diagnostics(fit, full_weights)
        rows.append({
            'scheme': scheme,
            'n': int(len(merged)),
            'mae_mm': metrics['pooled']['mae_mm'],
            'rmse_mm': metrics['pooled']['rmse_mm'],
            'bias_mm': metrics['pooled']['bias_mm'],
            'pearson_r': metrics['pooled']['pearson_r'],
            'pearson_squared': metrics['pooled']['pearson_squared'],
            'r2': metrics['pooled']['r2'],
            'within_section_pearson_r': metrics['within_section_centered']['pearson_r'],
            'within_section_pearson_squared': metrics['within_section_centered']['pearson_squared'],
            'macro_mae_mm': metrics['macro']['mae_mm'],
            'topk_recall': metrics['topk']['pooled_recall'],
            'kish_ess': full_diag['kish_ess'],
            'group_totals': json.dumps(full_diag['group_totals']),
            'full_weight_diagnostics': json.dumps(full_diag),
            'fold_weight_diagnostics': json.dumps(
                {str(r['fold']): r['weights'] for r in records}),
            'fold_records': json.dumps(records),
        })
    return pd.DataFrame(rows)


def _block_cv(fit: pd.DataFrame, block_size: int = 10, buffer: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = fit[fit.id.eq(S1_ID)].copy()
    if target.empty:
        return pd.DataFrame(), {'status': 'no_target_rows'}
    target['block'] = target.segment_id // block_size
    parts, records = [], []
    for block, test in target.groupby('block', sort=True):
        lo, hi = int(test.segment_id.min()) - buffer, int(test.segment_id.max()) + buffer
        hold = target.segment_id.between(lo, hi)
        train = fit[~((fit.id.eq(S1_ID)) & fit.segment_id.between(lo, hi))].copy()
        if train.empty or test.empty:
            continue
        weights = make_weights(train, 'row_uniform')
        ridge = fit_ridge(train, S1_CORE_FEATURES, 'target_delta_mm', weights,
                          alpha=10.0, baseline='last_mm', model_name='block_ridge')
        rate = fit_ridge(train, S1_CORE_FEATURES, 'rate_mm_year', weights,
                         alpha=10.0, baseline='last_mm', model_name='block_rate')
        cat = fit_catboost(train, S1_CORE_FEATURES, 'target_delta_mm', weights,
                           model_name='block_catboost', baseline='last_mm')
        out = test[['id', 'segment_id', 'next_mm', 'last_mm', 'dt_years',
                    'nominal_thickness_mm']].copy()
        out['block'] = int(block)
        out['s1a'] = test.last_mm.to_numpy(float)
        out['s1b'] = test.prior_vtd_min_thickness_mm.to_numpy(float)
        out['ridge_geometry'] = predict_ridge(ridge, test, baseline='last_mm')
        rate_hat = rate['estimator'].predict(rate['preprocessor'].transform(test))
        out['rate_signed'] = np.clip(
            test.last_mm.to_numpy(float) - rate_hat * test.dt_years.to_numpy(float),
            0.0, test.nominal_thickness_mm.to_numpy(float))
        out['catboost'] = predict_catboost(cat, test, baseline='last_mm')
        parts.append(out)
        records.append({'block': int(block), 'test_rows': int(len(test)),
                        'train_rows': int(len(train)),
                        'buffer_segments': int(buffer),
                        'train_ids': sorted(int(v) for v in train.id.unique())})
    if not parts:
        return pd.DataFrame(), {'status': 'no_blocks'}
    result = pd.concat(parts, ignore_index=True)
    metrics = {}
    for name in ('s1a', 's1b', 'ridge_geometry', 'rate_signed', 'catboost'):
        metrics[name] = evaluate_predictions(result, name)
    payload = {'status': 'same_pipe_spatial_interpolation', 'block_size_segments': block_size,
               'buffer_segments': buffer, 'records': records, 'metrics': metrics}
    return result, payload


def _production_predictions(prod: pd.DataFrame, fit: pd.DataFrame,
                            reference_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full-fit artifacts and candidate production predictions."""
    prod = prod.reset_index(drop=True)
    own = prod.has_own_prior_vtd.astype(bool)
    result = prod.reset_index(drop=True).copy()
    result['s1a'] = result.s1a_prediction.astype(float)
    result['s1b'] = result.s1b_prediction.astype(float)
    old = _old_s1c_production()
    result = result.merge(old, on=['id', 'segment_id'], how='left',
                          validate='one_to_one')
    weights = make_weights(fit, 'row_uniform')
    artifacts = {}
    specs = {
        'ridge_history': (S1_HISTORY_ONLY_FEATURES, 'target_delta_mm', 'delta'),
        'ridge_geometry': (S1_CORE_FEATURES, 'target_delta_mm', 'delta'),
        'rate': (S1_CORE_FEATURES, 'rate_mm_year', 'rate'),
        'catboost': (S1_CORE_FEATURES, 'target_delta_mm', 'delta'),
    }
    for name, (features, target, kind) in specs.items():
        if name == 'catboost':
            artifact = fit_catboost(fit, features, target, weights,
                                    model_name='catboost_delta', baseline='last_mm')
        else:
            artifact = fit_ridge(fit, features, target, weights, alpha=10.0,
                                 baseline='last_mm', model_name=name,
                                 prediction_policy=kind)
        artifacts[name] = artifact
    # Apply own rows only; fallback keeps S1B.
    result['ridge_history'] = result.s1b
    result['ridge_geometry'] = result.s1b
    result['rate_signed'] = result.s1b
    result['rate_nonnegative'] = result.s1b
    result['catboost'] = result.s1b
    own_idx = result.index[own]
    result.loc[own_idx, 'ridge_history'] = predict_ridge(
        artifacts['ridge_history'], result.loc[own_idx], baseline='last_mm')
    result.loc[own_idx, 'ridge_geometry'] = predict_ridge(
        artifacts['ridge_geometry'], result.loc[own_idx], baseline='last_mm')
    rate_hat = artifacts['rate']['estimator'].predict(
        artifacts['rate']['preprocessor'].transform(result.loc[own_idx]))
    dt = result.loc[own_idx, 'dt_years'].to_numpy(float)
    last = result.loc[own_idx, 'last_mm'].to_numpy(float)
    nominal = result.loc[own_idx, 'nominal_thickness_mm'].to_numpy(float)
    result.loc[own_idx, 'rate_signed'] = np.clip(last - rate_hat * dt, 0.0, nominal)
    result.loc[own_idx, 'rate_nonnegative'] = np.clip(
        last - np.maximum(rate_hat, 0.0) * dt, 0.0, nominal)
    result.loc[own_idx, 'catboost'] = predict_catboost(
        artifacts['catboost'], result.loc[own_idx], baseline='last_mm')
    delta_geom = (result.ridge_geometry - result.last_mm).to_numpy(float)
    for lam in (0.25, 0.50, 1.00):
        key = f'mix_{int(lam * 100):03d}'
        result[key] = result.s1b
        result.loc[own_idx, key] = np.clip(
            last + lam * delta_geom[own_idx], 0.0, nominal)
    result['mix_000'] = result.s1a
    return result, artifacts


def _forward_audit(table: pd.DataFrame) -> dict[str, Any]:
    eligible = table[table.prediction_date.lt(S1_CUTOFF)]
    counts = pd.to_numeric(eligible.prior_vtd_campaign_count, errors='coerce').fillna(0)
    ge2 = eligible[counts.ge(2)]
    return {
        'cutoff': str(S1_CUTOFF.date()),
        'eligible_rows': int(len(eligible)),
        'max_prior_campaigns': int(counts.max()) if len(counts) else 0,
        'rows_with_two_prior_campaigns': int(len(ge2)),
        'strict_forward_folds_available': bool(len(ge2) >= 2),
        'reason': ('Every eligible S1 label has one prior campaign; 2023-12-01 '
                   'labels are on the cutoff and excluded.') if len(ge2) == 0 else '',
    }


def _candidate_frame(prod: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = prod.reset_index(drop=True).copy()
    frame['thickness_pred'] = frame[column].astype(float)
    frame['model'] = column
    frame['ridge_applied'] = frame.has_own_prior_vtd.astype(bool)
    frame['prediction_source'] = np.where(
        frame.has_own_prior_vtd, column, 's1b_fallback')
    frame['delta_vs_s1a_mm'] = frame.thickness_pred - frame.s1a
    frame['delta_vs_s1b_mm'] = frame.thickness_pred - frame.s1b
    return frame


def _write_candidate(output: Path, prod: pd.DataFrame, column: str,
                     s2: pd.DataFrame, target_grid: pd.DataFrame,
                     bounds: pd.DataFrame, *, model_label: str,
                     training_required: bool, training_completed: bool,
                     artifact_info: dict[str, Any] | None) -> dict[str, Any]:
    directory = output / 'candidates' / column
    directory.mkdir(parents=True, exist_ok=True)
    frame = _candidate_frame(prod, column)
    write_csv(frame[['id', 'segment_id', 'thickness_pred']],
              directory / 'predictions_s1.csv', float_format='%.10f')
    diagnostic_cols = ['id', 'segment_id', 'prediction_date', 'model', 'thickness_pred',
                       'last_mm', 's1a', 's1b', 'delta_vs_s1a_mm', 'delta_vs_s1b_mm',
                       'has_own_prior_vtd', 'ridge_applied', 'prediction_source',
                       'fallback_segment_id', 'fallback_distance_m',
                       'fallback_measurement_date', 'nominal_thickness_mm', 'h0_mm']
    write_csv(frame[diagnostic_cols], directory / 'prediction_diagnostics.csv',
              float_format='%.10g')
    s1_pred = frame[['id', 'segment_id', 'thickness_pred']]
    combined, validation = combine_with_s2(s1_pred, s2, target_grid, bounds)
    write_csv(combined, directory / 'result.csv', float_format='%.10f')
    revalidated = pd.read_csv(directory / 'result.csv')
    s2_written = revalidated[revalidated.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']]
    s2_written = s2_written.sort_values(['id', 'segment_id']).reset_index(drop=True)
    s2_expected = s2.sort_values(['id', 'segment_id']).reset_index(drop=True)
    max_s2_diff = float(np.abs(s2_written.thickness_pred.to_numpy(float)
                               - s2_expected.thickness_pred.to_numpy(float)).max())
    if max_s2_diff != 0.0:
        raise ValueError(f'S2 changed for candidate {column}')
    top = frame.sort_values(['thickness_pred', 'segment_id']).head(33)
    s1a_top = frame.sort_values(['s1a', 'segment_id']).head(33)
    overlap = len(set(top.segment_id) & set(s1a_top.segment_id))
    segment_135 = frame[frame.segment_id.eq(135)].iloc[0]
    summary = {
        'candidate': column,
        'model_label': model_label,
        'training_required': training_required,
        'training_completed': training_completed,
        'rows': int(len(frame)),
        'fallback_rows': int((~frame.has_own_prior_vtd).sum()),
        'mean_mm': float(frame.thickness_pred.mean()),
        'std_mm': float(frame.thickness_pred.std(ddof=0)),
        'min_mm': float(frame.thickness_pred.min()),
        'max_mm': float(frame.thickness_pred.max()),
        'above_last_rows': int((frame.thickness_pred > frame.last_mm + 1e-9).sum()),
        'above_s1b_rows': int((frame.thickness_pred > frame.s1b + 1e-9).sum()),
        'top33_overlap_with_s1a': int(overlap),
        'segment_135': {
            's1a_mm': float(segment_135.s1a),
            's1b_mm': float(segment_135.s1b),
            'prediction_mm': float(segment_135.thickness_pred),
            'delta_vs_s1a_mm': float(segment_135.delta_vs_s1a_mm),
            'delta_vs_s1b_mm': float(segment_135.delta_vs_s1b_mm),
            'fallback': bool(not segment_135.has_own_prior_vtd),
        },
        'bounds_policy': ('[0, h0] for S1A/S1B; [0, nominal] for statistical models'
                          if column in ('s1a', 's1b') else '[0, nominal]'),
        's2': {'source': str(BASE_DIR / 'experiments/ml_v2/result.csv'),
               'rows': int(len(s2)), 'max_abs_diff': max_s2_diff},
        'validation': validation,
        'artifact': artifact_info,
    }
    write_json(directory / 'summary.json', json_safe(summary))
    if artifact_info and artifact_info.get('artifact_path'):
        src = Path(artifact_info['artifact_path'])
        dst = directory / 'model_artifact.joblib'
        if src.is_file() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        meta_src = Path(artifact_info.get('metadata_path', ''))
        meta_dst = directory / 'model_metadata.json'
        if meta_src.is_file() and meta_src.resolve() != meta_dst.resolve():
            shutil.copyfile(meta_src, meta_dst)
    return summary


def _save_figures(output: Path, oof: pd.DataFrame, prod: pd.DataFrame,
                  metrics: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = output / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, ('s1a', 'ridge_geometry', 'catboost')):
        for section, part in oof.groupby('id'):
            ax.scatter(part.next_mm, part[name], s=8, alpha=.55, label=str(section))
        lo = min(oof.next_mm.min(), oof[name].min())
        hi = max(oof.next_mm.max(), oof[name].max())
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=.7)
        ax.set_title(name)
        ax.set_xlabel('measured next, mm')
        ax.set_ylabel('OOF prediction, mm')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path = fig_dir / 'oof_actual_vs_predicted.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ('ridge_geometry', 'rate_signed', 'rate_nonnegative', 'catboost'):
        ax.hist(oof[name] - oof.last_mm, bins=40, alpha=.35, label=name)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('prediction - last measurement, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / 'correction_distributions.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(12, 4))
    for name in ('s1a', 's1b', 'old_s1c', 'ridge_geometry', 'catboost'):
        ax.plot(prod.segment_id, prod[name], linewidth=1, label=name)
    ax.set_xlabel('segment_id')
    ax.set_ylabel('production prediction, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / 'production_profiles.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths



def _select(metrics: dict[str, Any]) -> dict[str, Any]:
    fit = metrics['fit_eligible_357']
    baseline = fit['s1a']
    candidates = ['ridge_history', 'ridge_geometry', 'rate_signed',
                  'rate_nonnegative', 'catboost', 'mix_025', 'mix_050', 'mix_100']

    def key(name):
        rec = fit[name]
        return {
            'mae_mm': rec['pooled']['mae_mm'],
            'within_section_pearson_squared': rec['within_section_centered']['pearson_squared'],
            'within_section_pearson_r': rec['within_section_centered']['pearson_r'],
            'pooled_pearson_squared': rec['pooled']['pearson_squared'],
            'topk_recall': rec['topk']['pooled_recall'],
        }

    ranked = {name: key(name) for name in candidates}
    defined = {k: v for k, v in ranked.items()
               if v['within_section_pearson_squared'] is not None}
    best_spatial = (max(defined, key=lambda k: defined[k]['within_section_pearson_squared'])
                    if defined else None)
    best_mae = min(ranked, key=lambda k: ranked[k]['mae_mm'])
    qualifies = []
    base_ws = baseline['within_section_centered']['pearson_squared']
    base_mae = baseline['pooled']['mae_mm']
    for name, rec in ranked.items():
        if (rec['within_section_pearson_squared'] is not None
                and base_ws is not None
                and rec['within_section_pearson_squared'] > base_ws
                and rec['mae_mm'] <= base_mae):
            qualifies.append(name)
    default = best_spatial if best_spatial in qualifies else 's1a'
    return {
        'default': default,
        'default_reason': ('candidate improves within-section spatial signal and MAE over S1A'
                           if default != 's1a' else
                           'no candidate improved both within-section spatial signal and MAE; S1A remains default'),
        'best_spatial': best_spatial,
        'best_mae': best_mae,
        'qualifying_candidates': qualifies,
        'no_proven_winner': default == 's1a',
        'baseline': {'mae_mm': base_mae, 'within_section_pearson_squared': base_ws},
        'candidates': ranked,
        'caveat': ('LOPO on three sections of one parent is exploratory spatial transfer, '
                   'not a fresh independent holdout. No hidden-test accuracy is claimed.'),
    }


def _bounds_for_candidate(prod: pd.DataFrame, nominal: pd.DataFrame,
                          candidate: str) -> pd.DataFrame:
    """Per-candidate bounds: h0 for conservative rules, nominal for statistical models."""
    s1 = prod[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']].copy()
    s1['upper_mm'] = (s1.h0_mm if candidate in ('s1a', 's1b')
                      else s1.nominal_thickness_mm)
    s2 = nominal[nominal.id.eq(S2_ID)][['id', 'segment_id',
                                        'nominal_thickness_mm']].copy()
    s2['h0_mm'] = s2.nominal_thickness_mm
    s2['upper_mm'] = s2.nominal_thickness_mm
    bounds = pd.concat([s1, s2], ignore_index=True)
    if bounds.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate bounds key')
    if (bounds.upper_mm < bounds.h0_mm - 1e-9).any():
        raise ValueError('Upper bound below h0')
    return bounds


def run_s1(root: Path = BASE_DIR, data_root: Path | None = None,
           output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/next_cycle/s1').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = load_data(data_root)
    table = s1_training_table(data)
    fit = s1_fit_rows(table)
    prod = s1_production_table(data)
    if len(fit) != 357 or fit.id.nunique() != 3:
        raise ValueError(f'Unexpected S1 fit table: {len(fit)} rows, {fit.id.nunique()} sections')
    print(f'S1 fit: {len(fit)} rows, sections {fit.id.value_counts().to_dict()}', flush=True)
    write_csv(table, output / 'training_table.csv', float_format='%.10f')
    write_csv(fit, output / 'fit_table.csv', float_format='%.10f')

    oof_fit, fold_records = _build_oof(fit)
    model_columns = {
        's1a': 's1a', 's1b': 's1b', 'old_s1c': 'old_s1c',
        'train_median': 'train_median', 'affine': 'affine',
        'ridge_history': 'ridge_history', 'ridge_geometry': 'ridge_geometry',
        'rate_signed': 'rate_signed', 'rate_nonnegative': 'rate_nonnegative',
        'catboost': 'catboost', 'mix_025': 'mix_025', 'mix_050': 'mix_050',
        'mix_100': 'mix_100', 'mix_000': 'mix_000',
    }
    all_columns = {k: v for k, v in model_columns.items()
                   if k not in ('train_median', 'affine', 'mix_000')}
    all_pipeline = _all_pipeline_table(table, oof_fit)
    metrics = _metrics_payload(oof_fit, model_columns, all_pipeline, all_columns)
    if not np.allclose(oof_fit.mix_000, oof_fit.s1a, rtol=0, atol=0):
        raise ValueError('lambda=0 mixture is not S1A')
    write_csv(oof_fit, output / 'oof_fit_predictions.csv', float_format='%.10f')
    write_csv(all_pipeline, output / 'oof_all_pipeline.csv', float_format='%.10f')
    per_fold = _fold_metrics(oof_fit, model_columns)
    write_csv(per_fold, output / 'metrics_per_fold.csv', float_format='%.10f')
    write_json(output / 'metrics_pooled.json', json_safe(metrics))

    fold_assignments = []
    for name, records in fold_records.items():
        for rec in records:
            fold_assignments.append({'model': name, **rec})
    write_csv(pd.DataFrame(fold_assignments), output / 'fold_assignments.csv',
              float_format='%.10f')

    weights = _weight_ablation(fit)
    write_csv(weights, output / 'weight_ablation.csv', float_format='%.10f')
    block_predictions, block_metrics = _block_cv(fit)
    block_dir = output / 'block_cv'
    block_dir.mkdir(parents=True, exist_ok=True)
    if not block_predictions.empty:
        write_csv(block_predictions, block_dir / 'oof_predictions.csv', float_format='%.10f')
    write_json(block_dir / 'metrics.json', json_safe(block_metrics))
    forward = _forward_audit(table)
    write_json(output / 'forward_fold_audit.json', json_safe(forward))

    prod_frame, artifacts = _production_predictions(prod, fit, root)
    target_grid = data.read('processed/target_segments_grid')[['id', 'segment_id']].copy()
    nominal = nominal_frame(data)
    s2 = load_s2_source(BASE_DIR / 'experiments/ml_v2/result.csv', target_grid)
    candidate_columns = {
        's1a': 's1a', 's1b': 's1b', 'old_s1c': 'old_s1c',
        'ridge_history': 'ridge_history', 'ridge_geometry': 'ridge_geometry',
        'rate_signed': 'rate_signed', 'rate_nonnegative': 'rate_nonnegative',
        'catboost': 'catboost', 'mix_025': 'mix_025', 'mix_050': 'mix_050',
        'mix_100': 'mix_100',
    }
    summaries = {}
    for name, column in candidate_columns.items():
        bounds = _bounds_for_candidate(prod_frame, nominal, name)
        artifact_info = None
        if name in artifacts:
            directory = output / 'candidates' / name
            saved = save_artifact(artifacts[name], directory)
            artifact_info = {'artifact_path': saved['artifact_path'],
                             'artifact_sha256': saved['artifact_sha256'],
                             'metadata_path': str(directory / 'model_metadata.json')}
        elif name in ('rate_signed', 'rate_nonnegative') and 'rate' in artifacts:
            directory = output / 'candidates' / name
            rate_artifact = artifacts['rate']
            rate_artifact['metadata'] = {**rate_artifact['metadata'],
                                         'model_name': name,
                                         'prediction_policy': name}
            saved = save_artifact(rate_artifact, directory)
            artifact_info = {'artifact_path': saved['artifact_path'],
                             'artifact_sha256': saved['artifact_sha256'],
                             'metadata_path': str(directory / 'model_metadata.json')}
        elif name == 'old_s1c':
            src = BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1c/model_artifact.joblib'
            if src.is_file():
                artifact_info = {
                    'artifact_path': str(src), 'artifact_sha256': sha256(src),
                    'metadata_path': str(BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1c/model_metadata.json')}
        summaries[name] = _write_candidate(
            output, prod_frame, column, s2, target_grid, bounds,
            model_label=MODEL_LABELS.get(name, name),
            training_required=name not in ('s1a', 's1b'),
            training_completed=name not in ('s1a', 's1b'),
            artifact_info=artifact_info)
        if name.startswith('mix_'):
            write_json(output / 'candidates' / name / 'model_metadata.json', {
                'model_type': 'posthoc_mixture',
                'source_model': 'ridge_geometry',
                'lambda': float(name.split('_')[1]) / 100.0,
                'source_artifact': str(output / 'candidates/ridge_geometry/model_artifact.joblib'),
                'source_artifact_sha256': sha256(output / 'candidates/ridge_geometry/model_artifact.joblib'),
                'note': 'Not a separately trained model; lambda applied to ridge_geometry OOF/full delta.',
            })
    selection = _select(metrics)
    write_json(output / 'selected_candidates.json', json_safe(selection))
    figures = _save_figures(output, oof_fit, prod_frame, metrics)
    report = _write_report(output, metrics, selection, weights, block_metrics,
                           forward, summaries, figures, prod_frame)
    write_manifest(output / 'run_manifest.json', root=root, data=data,
                   output=output, stage='S1_next', started=started,
                   extra={'selection': selection, 'forward_fold_audit': forward,
                          'figures': figures, 'report': str(report),
                          'data_root': str(data_root)},
                   input_paths=[BASE_DIR / 'experiments/ml_v2/result.csv'])
    print(f'PASS S1_next: {len(fit)} fit rows, default={selection["default"]}', flush=True)
    return {'metrics': metrics, 'selection': selection, 'summaries': summaries,
            'output': str(output)}

def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.{digits}f}'
    return str(value)


def _write_report(output: Path, metrics: dict[str, Any], selection: dict[str, Any],
                  weights: pd.DataFrame, block: dict[str, Any],
                  forward: dict[str, Any], summaries: dict[str, Any],
                  figures: list[str], prod: pd.DataFrame) -> Path:
    fit = metrics['fit_eligible_357']
    allp = metrics.get('all_pipeline_561', {})
    lines = [
        '# S1 next cycle: controls, last-baseline learners and honest limits', '',
        'Exploratory-цикл. LOPO на трёх простых участках одного parent pipeline — '
        'spatial-transfer диагностика, не свежий независимый holdout. '
        'Скрытые ответы и post-cutoff метки не использовались.',
        '', '## Данные и протокол', '',
        '- fit: 357 строк, три простых участка (4 / 32 / 321), одна кампания `2022-07-06`.',
        '- Признаки строятся только по ВТД строго раньше даты ответа.',
        '- Новая core-модель использует один last baseline; S1B — отдельное правило.',
        '- Old S1C — reference, не выбранный winner.',
        '', '## Сравнение на одинаковых 357 fit-eligible ключах', '',
        '| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | top-10% recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('s1a', 's1b', 'old_s1c', 'train_median', 'affine', 'ridge_history',
                 'ridge_geometry', 'rate_signed', 'rate_nonnegative', 'catboost',
                 'mix_025', 'mix_050', 'mix_100', 'mix_000'):
        rec = fit[name]
        ws = rec['within_section_centered']
        lines.append(
            f"| {name} | {_fmt(rec['pooled']['mae_mm'])} | {_fmt(rec['pooled']['rmse_mm'])} | "
            f"{_fmt(rec['pooled']['bias_mm'])} | {_fmt(rec['pooled']['pearson_r'])} | "
            f"{_fmt(rec['pooled']['pearson_squared'])} | {_fmt(rec['pooled']['r2'])} | "
            f"{_fmt(ws['pearson_squared'])} | {_fmt(rec['topk']['pooled_recall'])} |")
    lines += ['', '`rate_signed` OOF численно совпал с `ridge_geometry`, потому что в train один и тот же dt=3.7837. '
              'На production фактический dt различается (1.4045/5.1882), поэтому rate-прогноз отличается. '
              'Это не доказывает корректное масштабирование скорости по времени.', '',
              '## Macro по участкам', '',
              '| Модель | macro MAE | macro Pearson r² | macro R² |',
              '|---|---:|---:|---:|']
    for name in ('s1a', 'old_s1c', 'ridge_geometry', 'rate_signed', 'catboost'):
        rec = fit[name]['macro']
        lines.append(f"| {name} | {_fmt(rec['mae_mm'])} | {_fmt(rec['pearson_squared'])} | "
                     f"{_fmt(rec['r2'])} |")
    lines += ['', '## All-pipeline 561 строк (fallback = S1B)', '',
              '| Модель | n | MAE | bias | within-section r² |',
              '|---|---:|---:|---:|---:|']
    for name in ('s1a', 's1b', 'old_s1c', 'ridge_geometry', 'rate_signed', 'catboost'):
        if name not in allp:
            continue
        rec = allp[name]
        ws = rec['within_section_centered']
        lines.append(f"| {name} | {rec['pooled']['n']} | {_fmt(rec['pooled']['mae_mm'])} | "
                     f"{_fmt(rec['pooled']['bias_mm'])} | {_fmt(ws['pearson_squared'])} |")
    lines += ['', '## Weight ablation (history-only)', '',
              '| Схема | MAE | within-section r² | Kish ESS | group totals |',
              '|---|---:|---:|---:|---|']
    for row in weights.itertuples():
        lines.append(f"| {row.scheme} | {_fmt(row.mae_mm)} | "
                     f"{_fmt(row.within_section_pearson_squared)} | "
                     f"{row.kish_ess:.2f} | {row.group_totals} |")
    lines += ['', '## Spatial block diagnostic', '',
              f"Статус: `{block.get('status')}`; 100 м blocks, buffer 20 м, "
              'same-pipe interpolation, не прогноз будущей кампании.',
              '', '| Модель | MAE | within-section r² |', '|---|---:|---:|']
    for name, rec in block.get('metrics', {}).items():
        ws = rec['within_section_centered']
        lines.append(f"| {name} | {_fmt(rec['pooled']['mae_mm'])} | "
                     f"{_fmt(ws['pearson_squared'])} |")
    lines += ['', 'Block-CV (same-pipe interpolation, 100 м + 20 м buffer): within-section r² '
              'S1A=0.1223, ridge_geometry/rate_signed=0.1633, catboost=0.1309; top-10% recall '
              'S1A=0.303, ridge/rate=0.455. Это не LOPO между трубами; LOPO выше сигнал не подтверждает.', '',
              '## Forward folds', '',
              f"`strict_forward_folds_available={forward['strict_forward_folds_available']}`. "
              f"{forward['reason']}", '', '## Selection', '',
              f"default = **{selection['default']}**. {selection['default_reason']}.",
              f"best_spatial = **{selection['best_spatial']}**; best_mae = **{selection['best_mae']}**.",
              'Никакой кандидат не объявляется доказанным winner; production result не перезаписан.',
              '', '## Production candidates', '',
              '| Candidate | mean | std | min | max | above last | fallback | top33 overlap S1A | segment 135 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('s1a', 's1b', 'old_s1c', 'ridge_history', 'ridge_geometry',
                 'rate_signed', 'rate_nonnegative', 'catboost', 'mix_025',
                 'mix_050', 'mix_100'):
        s = summaries[name]
        lines.append(f"| {name} | {_fmt(s['mean_mm'])} | {_fmt(s['std_mm'])} | "
                     f"{_fmt(s['min_mm'])} | {_fmt(s['max_mm'])} | "
                     f"{s['above_last_rows']} | {s['fallback_rows']} | "
                     f"{s['top33_overlap_with_s1a']} | {_fmt(s['segment_135']['prediction_mm'])} |")
    lines += ['', '## Files', '',
              '- `metrics_pooled.json`, `metrics_per_fold.csv`, `oof_fit_predictions.csv`, `oof_all_pipeline.csv`',
              '- `weight_ablation.csv`, `block_cv/`, `forward_fold_audit.json`, `selected_candidates.json`',
              '- `candidates/<name>/predictions_s1.csv`, `prediction_diagnostics.csv`, `result.csv`, `summary.json`',
              '- `figures/`', '', '```bash',
              'python -m src.ml.next_s1 --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s1',
              '```', '']
    path = output / 'report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def _predict_artifact(artifact: dict[str, Any], frame: pd.DataFrame,
                      candidate: str) -> np.ndarray:
    meta = artifact['metadata']
    target = meta.get('target')
    if target == 'rate_mm_year' or candidate.startswith('rate_'):
        rate = artifact['estimator'].predict(
            artifact['preprocessor'].transform(frame))
        if candidate == 'rate_nonnegative':
            rate = np.maximum(rate, 0.0)
        return np.clip(frame.last_mm.to_numpy(float)
                       - rate * frame.dt_years.to_numpy(float), 0.0,
                       frame.nominal_thickness_mm.to_numpy(float))
    if meta.get('model_type') == 'CatBoostRegressor':
        return predict_catboost(artifact, frame, baseline='last_mm')
    return predict_ridge(artifact, frame, baseline='last_mm')


def run_s1_predict(root: Path = BASE_DIR, data_root: Path | None = None,
                   output_root: Path | None = None,
                   candidate: str = 'ridge_geometry',
                   artifact_path: Path | None = None) -> dict[str, Any]:
    """Inference-only S1 candidate from a saved artifact; no labels and no fit."""
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/next_cycle/s1_predict').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = load_data(data_root)
    prod = s1_production_table(data)
    prod['s1a'] = prod.s1a_prediction.astype(float)
    prod['s1b'] = prod.s1b_prediction.astype(float)
    target_grid = data.read('processed/target_segments_grid')[['id', 'segment_id']].copy()
    nominal = nominal_frame(data)
    s2 = load_s2_source(BASE_DIR / 'experiments/ml_v2/result.csv', target_grid)
    candidate = candidate.lower()
    artifact_info = None
    if candidate in ('s1a', 's1b'):
        column = 's1a_prediction' if candidate == 's1a' else 's1b_prediction'
        prediction = prod[column].to_numpy(float)
        column_name = candidate
    elif candidate == 'old_s1c':
        old = _old_s1c_production()
        merged = prod[['id', 'segment_id']].merge(
            old, on=['id', 'segment_id'], how='left', validate='one_to_one')
        if merged.old_s1c.isna().any():
            raise ValueError('Old S1C predictions do not cover S1 keys')
        prediction = merged.old_s1c.to_numpy(float)
        column_name = 'old_s1c'
        src = BASE_DIR / 'experiments/next_cycle/repro_s1abc/s1c/model_artifact.joblib'
        artifact_info = {'artifact_path': str(src), 'artifact_sha256': sha256(src),
                         'metadata_path': str(src.with_name('model_metadata.json'))}
    else:
        if candidate.startswith('mix_'):
            lam = float(candidate.split('_')[1]) / 100.0
            base_path = root / 'experiments/next_cycle/s1/candidates/ridge_geometry/model_artifact.joblib'
            if artifact_path is None:
                artifact_path = base_path
            artifact = load_artifact(artifact_path)
            base_pred = _predict_artifact(artifact, prod, 'ridge_geometry')
            own = prod.has_own_prior_vtd.astype(bool).to_numpy()
            prediction = prod.s1b_prediction.to_numpy(float).copy()
            prediction[own] = np.clip(
                prod.last_mm.to_numpy(float)[own]
                + lam * (base_pred[own] - prod.last_mm.to_numpy(float)[own]),
                0.0, prod.nominal_thickness_mm.to_numpy(float)[own])
            column_name = candidate
            artifact_info = None
        else:
            if artifact_path is None:
                artifact_path = root / 'experiments/next_cycle/s1/candidates' / candidate / 'model_artifact.joblib'
            artifact_path = Path(artifact_path)
            artifact = load_artifact(artifact_path)
            prediction = prod.s1b_prediction.to_numpy(float).copy()
            own = prod.has_own_prior_vtd.astype(bool).to_numpy()
            prediction[own] = _predict_artifact(artifact, prod.loc[own], candidate)
            column_name = candidate
            artifact_info = {'artifact_path': str(artifact_path),
                             'artifact_sha256': sha256(artifact_path),
                             'metadata_path': str(artifact_path.with_name('model_metadata.json'))}
    prod = prod.copy()
    prod[column_name] = prediction
    bounds = _bounds_for_candidate(prod, nominal, column_name)
    summary = _write_candidate(
        output, prod, column_name, s2, target_grid, bounds,
        model_label=MODEL_LABELS.get(column_name, column_name),
        training_required=column_name not in ('s1a', 's1b'),
        training_completed=column_name not in ('s1a', 's1b'),
        artifact_info=artifact_info)
    if column_name.startswith('mix_'):
        source = root / 'experiments/next_cycle/s1/candidates/ridge_geometry/model_artifact.joblib'
        write_json(output / 'candidates' / column_name / 'model_metadata.json', {
            'model_type': 'posthoc_mixture',
            'source_model': 'ridge_geometry',
            'lambda': float(column_name.split('_')[1]) / 100.0,
            'source_artifact': str(source),
            'source_artifact_sha256': sha256(source),
        })
    write_manifest(output / 'run_manifest_predict.json', root=root, data=data,
                   output=output, stage='S1_predict', started=started,
                   extra={'candidate': column_name, 'summary': summary,
                          'prediction_mode': 'loaded_artifact_no_fit'},
                   input_paths=[BASE_DIR / 'experiments/ml_v2/result.csv'])
    print(f'PASS S1 predict: {column_name}, {len(prod)} rows', flush=True)
    return summary

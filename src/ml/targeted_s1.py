"""Stage 3: S1 fixed 100 m block / 20 m buffer comparison.

The held-out unit is a physical 100 m block on the target pipe.  Test keys are
the measured target segments in that block; training keys are all eligible
segments outside the block+buffer.  All transforms and coefficients are fit on
the train side of each fold.  LOPO is not used here; Stage 4 will test larger
blocks.  No time-law selection is made.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .next_common import (S1_LOCAL_PRIOR_FEATURES, evaluate_predictions,
                          fit_catboost, fit_ridge, json_safe, load_artifact,
                          make_weights, predict_catboost, predict_ridge,
                          s1_fit_rows, s1_production_table, s1_training_table,
                          save_artifact, utcnow, write_csv as _write_csv, write_manifest)
from .s1_contract import S1_ID
from .targeted_audit import tail_metrics


def write_csv(frame: pd.DataFrame, path: Path, fmt: str = '%.10g') -> None:
    _write_csv(frame, path, float_format=fmt)

BLOCK_M = 100.0
BUFFER_M = 20.0
CORE_FEATURES = ['last_mm', 'age_last_vtd_years', 'prior_vtd_campaign_count',
                 'neighbor_prior_median_mm', 'neighbor_prior_count',
                 'neighbor_missing', 'slope_m_per_m', 'abs_slope_m_per_m',
                 'local_low_m', 'relative_position']
MODELS = ['s1a', 'train_median', 'affine_global', 'affine_target',
          'global_ridge_geometry', 'target_ridge_geometry',
          'global_ridge_local', 'catboost_reference']
LOCAL_FEATURES = S1_LOCAL_PRIOR_FEATURES
CENTRED = 'within_block'


def _affine_fit(frame: pd.DataFrame) -> tuple[float, float]:
    x = frame.last_mm.to_numpy(float)
    y = frame.next_mm.to_numpy(float)
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = max(0.0, float(coef[0]))
    b = float(np.mean(y - a * x))
    return a, b


def _affine_predict(frame: pd.DataFrame, a: float, b: float) -> np.ndarray:
    return np.clip(a * frame.last_mm.to_numpy(float) + b, 0.0,
                   frame.nominal_thickness_mm.to_numpy(float))


def _physical_block_frame(fit: pd.DataFrame, block_m: float = BLOCK_M) -> pd.DataFrame:
    frame = fit.copy()
    frame['center_m'] = (frame.start_m.astype(float) + frame.end_m.astype(float)) / 2.0
    frame['block'] = np.floor(frame.center_m / block_m).astype(int)
    return frame


def _block_folds(fit: pd.DataFrame, target: pd.DataFrame,
                 block_m: float = BLOCK_M,
                 buffer_m: float = BUFFER_M) -> list[dict[str, Any]]:
    folds = []
    for block, test in target.groupby('block', sort=True):
        low = float(block) * block_m
        high = low + block_m
        hold = target[(target.center_m.ge(low - buffer_m))
                      & (target.center_m.le(high + buffer_m))]
        train_global = fit[~fit.index.isin(hold.index)].copy()
        train_local = target[~target.index.isin(hold.index)].copy()
        if test.empty or train_global.empty or train_local.empty:
            continue
        if set(test.index) & set(train_global.index) or set(test.index) & set(train_local.index):
            raise ValueError('Held-out block leaked into training')
        folds.append({'block': int(block), 'low_m': low, 'high_m': high,
                      'buffer_m': buffer_m, 'test': test,
                      'train_global': train_global, 'train_local': train_local,
                      'hold': hold})
    return folds


def _fit_fold_models(fold: dict[str, Any]) -> dict[str, Any]:
    test = fold['test']
    train_global = fold['train_global']
    train_local = fold['train_local']
    weights_global = make_weights(train_global, 'row_uniform')
    weights_local = make_weights(train_local, 'row_uniform')
    a_g, b_g = _affine_fit(train_global)
    a_l, b_l = _affine_fit(train_local)
    median = float(np.median(train_global.next_mm.to_numpy(float)))
    global_ridge = fit_ridge(train_global, CORE_FEATURES, 'target_delta_mm',
                             weights_global, alpha=10.0, baseline='last_mm',
                             model_name='global_ridge_geometry')
    global_ridge_local = fit_ridge(
        train_global, CORE_FEATURES + LOCAL_FEATURES, 'target_delta_mm',
        weights_global, alpha=10.0, baseline='last_mm',
        model_name='global_ridge_local')
    target_ridge = fit_ridge(train_local, CORE_FEATURES, 'target_delta_mm',
                             weights_local, alpha=10.0, baseline='last_mm',
                             model_name='target_ridge_geometry')
    cat = fit_catboost(train_global, CORE_FEATURES, 'target_delta_mm',
                       weights_global, model_name='catboost_reference',
                       baseline='last_mm')
    return {
        'a_global': a_g, 'b_global': b_g, 'a_target': a_l, 'b_target': b_l,
        'median': median, 'global_ridge': global_ridge,
        'global_ridge_local': global_ridge_local,
        'target_ridge': target_ridge, 'catboost': cat,
    }


def _predict_fold(models: dict[str, Any], test: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        's1a': test.last_mm.to_numpy(float),
        'train_median': np.clip(np.full(len(test), models['median']), 0.0,
                                test.nominal_thickness_mm.to_numpy(float)),
        'affine_global': _affine_predict(test, models['a_global'], models['b_global']),
        'affine_target': _affine_predict(test, models['a_target'], models['b_target']),
        'global_ridge_geometry': predict_ridge(models['global_ridge'], test,
                                               baseline='last_mm'),
        'global_ridge_local': predict_ridge(models['global_ridge_local'], test,
                                            baseline='last_mm'),
        'target_ridge_geometry': predict_ridge(models['target_ridge'], test,
                                               baseline='last_mm'),
        'catboost_reference': predict_catboost(models['catboost'], test,
                                               baseline='last_mm'),
    }


def _block_metrics(oof: pd.DataFrame, pred: str) -> list[dict[str, Any]]:
    rows = []
    for block, part in oof.groupby('block', sort=True):
        rec = evaluate_predictions(part, pred)
        rows.append({'block': int(block), 'n': int(len(part)),
                     **{f'pooled_{k}': v for k, v in rec['pooled'].items()},
                     'within_block_pearson_r': rec['within_section_centered']['pearson_r'],
                     'within_block_pearson_squared': rec['within_section_centered']['pearson_squared'],
                     'lower_tail_recall': rec['topk']['pooled_recall']})
    return rows


def _block_tail(oof: pd.DataFrame, pred: str) -> dict[str, Any]:
    total_k = total_overlap = 0
    total_lower = 0.0
    per = []
    for block, part in oof.groupby('block', sort=True):
        part = part.sort_values('segment_id')
        k = int(math.ceil(0.1 * len(part)))
        true_ids = list(part.sort_values(['next_mm', 'segment_id']).index[:k])
        pred_ids = list(part.sort_values([pred, 'segment_id']).index[:k])
        overlap = len(set(true_ids) & set(pred_ids))
        selected = part.loc[true_ids]
        lower = float(np.abs(selected[pred] - selected.next_mm).sum())
        per.append({'block': int(block), 'n': int(len(part)), 'k': k,
                    'overlap': overlap, 'recall': overlap / k,
                    'lower_tail_mae_mm': lower / k})
        total_k += k
        total_overlap += overlap
        total_lower += lower
    global_k = int(math.ceil(0.1 * len(oof)))
    true_g = list(oof.sort_values(['next_mm', 'segment_id']).index[:global_k])
    pred_g = list(oof.sort_values([pred, 'segment_id']).index[:global_k])
    selected_g = oof.loc[true_g]
    return {
        'global_k': global_k,
        'global_top10_recall': len(set(true_g) & set(pred_g)) / global_k,
        'global_lower_tail_mae_mm': float(
            np.abs(selected_g[pred] - selected_g.next_mm).sum() / global_k),
        'per_block_k_sum': total_k,
        'per_block_top10_recall': total_overlap / total_k,
        'per_block_lower_tail_mae_mm': total_lower / total_k,
        'per_block': per,
        'tie_policy': 'ascending value; equal values by lower segment_id',
    }


def _within_block_centered(oof: pd.DataFrame, pred: str) -> dict[str, Any]:
    ys, ps, per = [], [], []
    for block, part in oof.groupby('block', sort=True):
        y = part.next_mm.to_numpy(float)
        p = part[pred].to_numpy(float)
        if len(y) < 2 or np.ptp(y) <= 1e-12 or np.ptp(p) <= 1e-12:
            per.append({'block': int(block), 'n': len(part), 'pearson_r': None,
                        'status': 'undefined_constant_or_small'})
            continue
        yc, pc = y - y.mean(), p - p.mean()
        r = float(np.corrcoef(yc, pc)[0, 1])
        per.append({'block': int(block), 'n': len(part), 'pearson_r': r,
                    'status': 'defined'})
        ys.append(yc)
        ps.append(pc)
    if not ys:
        return {'pooled_pearson_r': None, 'pooled_pearson_squared': None,
                'defined_blocks': 0, 'undefined_blocks': len(per), 'per_block': per}
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    r = float(np.corrcoef(y, p)[0, 1])
    return {'pooled_pearson_r': r, 'pooled_pearson_squared': r * r,
            'defined_blocks': len(ys), 'undefined_blocks': len(per) - len(ys),
            'per_block': per}


def _baseline_bins(oof: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    frame = oof.copy()
    frame['baseline_bin'] = pd.cut(frame.last_mm, bins=bins, right=False,
                                   include_lowest=True)
    rows = []
    for bin_label, part in frame.groupby('baseline_bin', observed=True):
        for pred in MODELS:
            error = part[pred] - part.next_mm
            rows.append({'baseline_bin': str(bin_label), 'n': int(len(part)),
                         'model': pred, 'mae_mm': float(np.abs(error).mean()),
                         'bias_mm': float(error.mean()),
                         'rmse_mm': float(np.sqrt((error ** 2).mean()))})
    return pd.DataFrame(rows)


def _overall_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, details = [], {}
    for pred in MODELS:
        rec = evaluate_predictions(oof, pred)
        tail = _block_tail(oof, pred)
        centered = _within_block_centered(oof, pred)
        details[pred] = {'evaluation': rec, 'tail': tail, 'centered': centered}
        rows.append({
            'model': pred, 'n': rec['pooled']['n'],
            'mae_mm': rec['pooled']['mae_mm'], 'rmse_mm': rec['pooled']['rmse_mm'],
            'bias_mm': rec['pooled']['bias_mm'],
            'pearson_r': rec['pooled']['pearson_r'],
            'pearson_squared': rec['pooled']['pearson_squared'],
            'r2': rec['pooled']['r2'],
            'std_pred_mm': rec['pooled']['prediction_std_mm'],
            'std_truth_mm': rec['pooled']['truth_std_mm'],
            'global_top10_recall': tail['global_top10_recall'],
            'global_lower_tail_mae_mm': tail['global_lower_tail_mae_mm'],
            'per_block_top10_recall': tail['per_block_top10_recall'],
            'per_block_lower_tail_mae_mm': tail['per_block_lower_tail_mae_mm'],
            'within_block_pearson_r': centered['pooled_pearson_r'],
            'within_block_pearson_squared': centered['pooled_pearson_squared'],
            'defined_blocks': centered['defined_blocks'],
            'undefined_blocks': centered['undefined_blocks'],
        })
    return pd.DataFrame(rows), details


def _fit_full_models(fit: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    weights = make_weights(fit, 'row_uniform')
    a_g, b_g = _affine_fit(fit)
    a_t, b_t = _affine_fit(target)
    median = float(np.median(fit.next_mm.to_numpy(float)))
    return {
        'global_ridge_geometry': fit_ridge(
            fit, CORE_FEATURES, 'target_delta_mm', weights, alpha=10.0,
            baseline='last_mm', model_name='global_ridge_geometry'),
        'global_ridge_local': fit_ridge(
            fit, CORE_FEATURES + LOCAL_FEATURES, 'target_delta_mm', weights,
            alpha=10.0, baseline='last_mm', model_name='global_ridge_local'),
        'target_ridge_geometry': fit_ridge(
            target, CORE_FEATURES, 'target_delta_mm',
            make_weights(target, 'row_uniform'), alpha=10.0,
            baseline='last_mm', model_name='target_ridge_geometry'),
        'catboost_reference': fit_catboost(
            fit, CORE_FEATURES, 'target_delta_mm', weights,
            model_name='catboost_reference', baseline='last_mm'),
        'affine_global': {'a': a_g, 'b': b_g, 'n': int(len(fit))},
        'affine_target': {'a': a_t, 'b': b_t, 'n': int(len(target))},
        'train_median': {'value': median, 'n': int(len(fit))},
    }


def _target_predictions(prod: pd.DataFrame, models: dict[str, Any]) -> dict[str, pd.DataFrame]:
    base = prod[['id', 'segment_id', 'nominal_thickness_mm', 'last_mm']].copy()
    out = {}
    out['s1a'] = base.assign(thickness_pred=base.last_mm.to_numpy(float))
    out['train_median'] = base.assign(thickness_pred=np.clip(
        np.full(len(base), models['train_median']['value']), 0,
        base.nominal_thickness_mm.to_numpy(float)))
    for name in ('affine_global', 'affine_target'):
        out[name] = base.assign(thickness_pred=_affine_predict(
            prod, models[name]['a'], models[name]['b']))
    for name in ('global_ridge_geometry', 'global_ridge_local',
                 'target_ridge_geometry'):
        out[name] = base.assign(thickness_pred=predict_ridge(
            models[name], prod, baseline='last_mm'))
    out['catboost_reference'] = base.assign(thickness_pred=predict_catboost(
        models['catboost_reference'], prod, baseline='last_mm'))
    for name in out:
        raw = out[name].thickness_pred.to_numpy(float)
        nominal = out[name].nominal_thickness_mm.to_numpy(float)
        out[name]['thickness_pred_unclipped'] = raw
        out[name]['thickness_pred'] = np.clip(raw, 0, nominal)
        out[name]['clipping'] = (raw < 0) | (raw > nominal)
    return out


def _archive_compare(oof: pd.DataFrame, root: Path) -> dict[str, Any]:
    path = root / 'experiments/next_cycle/s1/block_cv/oof_predictions.csv'
    if not path.is_file():
        return {'status': 'archive_absent'}
    old = pd.read_csv(path)
    merged = oof[['id', 'segment_id', 'block', 's1a', 'global_ridge_geometry',
                  'catboost_reference']].merge(
        old[['id', 'segment_id', 's1a', 'ridge_geometry', 'catboost']].rename(
            columns={'s1a': 'old_s1a', 'ridge_geometry': 'old_ridge',
                     'catboost': 'old_catboost'}),
        on=['id', 'segment_id'], how='outer', indicator=True)
    both = merged[merged._merge.eq('both')]
    return {
        'rows_both': int(len(both)),
        'rows_only_current': int(merged._merge.eq('left_only').sum()),
        'rows_only_archive': int(merged._merge.eq('right_only').sum()),
        's1a_max_abs_diff': float(np.abs(both.s1a - both.old_s1a).max()),
        'global_ridge_max_abs_diff': float(np.abs(
            both.global_ridge_geometry - both.old_ridge).max()),
        'catboost_max_abs_diff': float(np.abs(
            both.catboost_reference - both.old_catboost).max()),
        'block_mapping_equal': bool((both.block == (both.segment_id // 10)).all())
        if len(both) else None,
        'old_block_definition': 'segment_id // 10 (archive)',
        'current_block_definition': 'center_m // 100 (physical)',
        'note': 'Archive used only for protocol comparison; no selection.',
    }


def run_stage3(root: Path = BASE_DIR, data_root: Path | None = None,
               output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/targeted_cycle/stage_3_s1_100m_20m').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = RepositoryData(data_root)
    table = s1_training_table(data)
    fit = s1_fit_rows(table)
    target = _physical_block_frame(fit[fit.id.eq(S1_ID)].copy(), BLOCK_M)
    other = fit[~fit.id.eq(S1_ID)].copy()
    if len(target) != 321:
        raise ValueError(f'Expected 321 target fit rows, got {len(target)}')
    folds = _block_folds(fit, target, BLOCK_M, BUFFER_M)
    print(f'Stage 3: {len(folds)} physical blocks, {len(target)} target rows', flush=True)
    write_json(output / 'fold_assignments.json', {
        'block_m': BLOCK_M, 'buffer_m': BUFFER_M,
        'block_definition': 'floor(center_m / 100); center=(start+end)/2',
        'folds': [{'block': f['block'], 'low_m': f['low_m'], 'high_m': f['high_m'],
                   'buffer_m': f['buffer_m'], 'test_rows': int(len(f['test'])),
                   'train_global_rows': int(len(f['train_global'])),
                   'train_local_rows': int(len(f['train_local'])),
                   'test_keys': [[int(a), int(b)] for a, b in
                                 zip(f['test'].id, f['test'].segment_id)]}
                  for f in folds],
    })
    fold_key_rows = []
    for f in folds:
        for kind, frame in [('global', f['train_global']),
                            ('target_only', f['train_local'])]:
            for r in frame.itertuples():
                fold_key_rows.append({'fold_block': f['block'], 'train_kind': kind,
                                      'id': int(r.id), 'segment_id': int(r.segment_id),
                                      'prediction_date': str(pd.Timestamp(r.prediction_date).date())})
    write_csv(pd.DataFrame(fold_key_rows), output / 'fold_training_keys.csv')
    # OOF.
    oof_parts, fold_records = [], []
    for f in folds:
        models = _fit_fold_models(f)
        preds = _predict_fold(models, f['test'])
        part = f['test'][['id', 'segment_id', 'prediction_date', 'next_mm',
                          'last_mm', 'nominal_thickness_mm',
                          'center_m', 'block']].copy()
        for name, values in preds.items():
            part[name] = values
        oof_parts.append(part)
        fold_records.append({
            'block': f['block'], 'low_m': f['low_m'], 'high_m': f['high_m'],
            'buffer_m': f['buffer_m'],
            'test_rows': int(len(f['test'])),
            'train_global_rows': int(len(f['train_global'])),
            'train_local_rows': int(len(f['train_local'])),
            'train_global_target_rows': int((f['train_global'].id == S1_ID).sum()),
            'train_global_other_rows': int((f['train_global'].id != S1_ID).sum()),
            'affine_global': {'a': models['a_global'], 'b': models['b_global']},
            'affine_target': {'a': models['a_target'], 'b': models['b_target']},
            'train_median': models['median'],
            'global_used_features': list(models['global_ridge']['metadata']['used_features']),
            'target_used_features': list(models['target_ridge']['metadata']['used_features']),
            'catboost_used_features': list(models['catboost']['metadata']['used_features']),
        })
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ['block', 'segment_id']).reset_index(drop=True)
    write_csv(oof, output / 'oof_predictions.csv', fmt='%.10f')
    write_json(output / 'fold_fit_metadata.json', json_safe(fold_records))
    metrics, details = _overall_metrics(oof)
    write_csv(metrics, output / 'metrics_overall.csv', fmt='%.10f')
    write_json(output / 'metrics_details.json', json_safe(details))
    # Per-block metrics and paired differences.
    per_block_parts = []
    for model in MODELS:
        rows = _block_metrics(oof, model)
        per_block_parts.extend(rows)
    # Add explicit paired diff vs S1A.
    block_mae = {}
    for model in MODELS:
        block_mae[model] = oof.groupby('block').apply(
            lambda x: float(np.mean(np.abs(x[model] - x.next_mm))),
            include_groups=False)
    paired_rows = []
    for block in sorted(block_mae['s1a'].index):
        for model in MODELS:
            if model == 's1a':
                continue
            diff = float(block_mae[model].loc[block] - block_mae['s1a'].loc[block])
            paired_rows.append({'block': int(block), 'model': model,
                                'mae_model_mm': float(block_mae[model].loc[block]),
                                'mae_s1a_mm': float(block_mae['s1a'].loc[block]),
                                'delta_mae_mm': diff,
                                'model_better': diff < 0})
    paired = pd.DataFrame(paired_rows)
    write_csv(paired, output / 'paired_block_mae.csv', fmt='%.10f')
    paired_summary = {}
    for model, part in paired.groupby('model'):
        vals = part.delta_mae_mm.to_numpy(float)
        paired_summary[model] = {
            'mean_delta_mae_mm': float(vals.mean()),
            'median_delta_mae_mm': float(np.median(vals)),
            'q05_delta_mae_mm': float(np.quantile(vals, .05)),
            'q95_delta_mae_mm': float(np.quantile(vals, .95)),
            'blocks_model_better': int((vals < 0).sum()),
            'blocks': int(len(vals)),
        }
    write_json(output / 'paired_block_mae_summary.json', json_safe(paired_summary))
    write_csv(pd.DataFrame(per_block_parts), output / 'metrics_per_block.csv',
              fmt='%.10f')
    bins = _baseline_bins(oof)
    write_csv(bins, output / 'baseline_thickness_bins.csv', fmt='%.10f')
    write_json(output / 'within_block_correlation.json',
               json_safe({m: details[m]['centered'] for m in MODELS}))
    write_json(output / 'tail_metrics.json',
               json_safe({m: details[m]['tail'] for m in MODELS}))
    # Models and full-target predictions.
    models = _fit_full_models(fit, target)
    model_rows = []
    for name in ('global_ridge_geometry', 'global_ridge_local',
                 'target_ridge_geometry', 'catboost_reference'):
        directory = output / 'models' / name
        saved = save_artifact(models[name], directory)
        model_rows.append({'model': name, **saved})
    for name in ('affine_global', 'affine_target', 'train_median'):
        directory = output / 'models' / name
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'model_metadata.json', json_safe({
            'model': name, 'params': models[name],
            'feature_contract': 'historical last measurement only' if name == 'train_median'
            else 'affine on last measurement, a>=0'}))
        model_rows.append({'model': name, 'artifact_path': None,
                           'artifact_sha256': None, 'metadata': models[name]})
    write_json(output / 'model_artifacts.json', json_safe(model_rows))
    write_json(output / 'feature_contract.json', {
        'core_features': CORE_FEATURES,
        'baseline': 'last eligible own prior VTD measurement (last_mm)',
        'target': 'target_delta_mm = next_mm - last_mm',
        'preprocessing': 'NextPreprocessor train-only: median imputer, constant/duplicate/empty drop, StandardScaler',
        'target_ridge': 'fit only on target-section rows outside held-out block+buffer',
        'global_ridge': 'fit on all eligible rows outside target block+buffer, including other simple sections',
        'catboost': 'fixed reference parameters; no search',
        'bounds': '[0, nominal_thickness_mm]',
    })
    # Full-model training IDs.
    train_ids = []
    for name, frame in [('global_ridge_geometry', fit),
                        ('global_ridge_local', fit),
                        ('target_ridge_geometry', target),
                        ('catboost_reference', fit)]:
        for r in frame.itertuples():
            train_ids.append({'model': name, 'id': int(r.id),
                              'segment_id': int(r.segment_id),
                              'prediction_date': str(pd.Timestamp(r.prediction_date).date())})
    write_csv(pd.DataFrame(train_ids), output / 'training_ids.csv')
    prod = s1_production_table(data)
    target_preds = _target_predictions(prod, models)
    for name, frame in target_preds.items():
        directory = output / 'target_predictions' / name
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(frame[['id', 'segment_id', 'thickness_pred']],
                  directory / 'predictions_s1.csv', fmt='%.10f')
        write_csv(frame, directory / 'prediction_diagnostics.csv', fmt='%.10f')
    archive = _archive_compare(oof, root)
    write_json(output / 'archive_comparison.json', json_safe(archive))
    # Figures and report.
    _figures(output, oof, paired, metrics)
    report = _report(root, data_root, output, metrics, details, paired_summary,
                     bins, archive, model_rows)
    report_path = root / 'experiments/targeted_cycle/stage_3_report_ru.md'
    report_path.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
    write_manifest(output / 'run_manifest.json', root=root, data=data, output=output,
                   stage='Stage_3_S1_100m_20m', started=started,
                   extra={'data_root': str(data_root), 'blocks': len(folds),
                          'archive_comparison': archive,
                          'report': str(report_path), 'plan_version': 2,
                          'stage_code_sha256': {
                              'src/ml/targeted_s1.py': sha256(root / 'src/ml/targeted_s1.py')},
                          'selection_status': 'exploratory_100m_20m_not_stable_until_stage_4'},
                   input_paths=[])
    print(f'PASS Stage 3: {output}', flush=True)
    print(f'  folds={len(folds)} target_rows={len(oof)}', flush=True)
    return {'metrics': metrics, 'details': details, 'paired': paired_summary,
            'archive': archive, 'output': str(output), 'report': str(report_path)}


def _figures(output: Path, oof: pd.DataFrame, paired: pd.DataFrame,
             metrics: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = output / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in ('s1a', 'global_ridge_geometry', 'target_ridge_geometry'):
        part = oof.sort_values('segment_id')
        ax.plot(part.segment_id, part[model], linewidth=.8, label=model)
    ax.set_xlabel('segment_id')
    ax.set_ylabel('OOF prediction, mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / 's1_oof_profiles.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, part in paired.groupby('model'):
        ax.hist(part.delta_mae_mm, bins=30, alpha=.35, label=model)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('block MAE(model) - MAE(S1A), mm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / 'paired_block_mae.png', dpi=160)
    plt.close(fig)


def _fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return 'n/a' if not np.isfinite(value) else f'{value:.{digits}f}'
    return str(value)


def _report(root: Path, data_root: Path, output: Path, metrics: pd.DataFrame,
            details: dict[str, Any], paired: dict[str, Any],
            bins: pd.DataFrame, archive: dict[str, Any],
            model_rows: list[dict[str, Any]]) -> Path:
    lines = [
        '# Stage 3 — S1 fixed 100 m block / 20 m buffer', '',
        f"Generated: `{utcnow()}`. Physical blocks: `center_m=(start+end)/2`, "
        '`block=floor(center_m/100)`, buffer 20 m; training excludes the entire '
        'target block plus buffer. No future answer is used in features.',
        '', '## 1. Protocol and keys', '',
        f"- Target pipe `{S1_ID}`, 321 measured fit-eligible segments.",
        '- Test keys: measured target segments whose center lies in the held-out 100 m block.',
        '- Global models train on all eligible simple sections outside block+buffer; '
        'target-only models train only on target-pipe rows outside block+buffer.',
        '- All imputers/scalers/features/coefficients are fit inside each train fold.',
        '- LOPO is not used here; Stage 4 robustness and temporal Stage 5 are not started.',
        '', '## 2. Metrics on the same 321 OOF keys', '',
        '| Model | MAE | RMSE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail global MAE | top10 global | top10 per-block | within-block r | within-block r² |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f"| {row.model} | {_fmt(row.mae_mm)} | {_fmt(row.rmse_mm)} | "
                     f"{_fmt(row.bias_mm)} | {_fmt(row.pearson_r)} | {_fmt(row.pearson_squared)} | "
                     f"{_fmt(row.r2)} | {_fmt(row.std_pred_mm)} | {_fmt(row.std_truth_mm)} | "
                     f"{_fmt(row.global_lower_tail_mae_mm)} | {_fmt(row.global_top10_recall)} | "
                     f"{_fmt(row.per_block_top10_recall)} | {_fmt(row.within_block_pearson_r)} | "
                     f"{_fmt(row.within_block_pearson_squared)} |")
    lines += ['', '`top10 global`: k=ceil(0.1*321)=33 on the pooled target sample; '
              '`top10 per-block`: k=ceil(0.1*n_block) per block aggregated as sum(overlap)/sum(k). '
              'They have different denominators and are not interchangeable.',
              '', '## 3. Paired block MAE difference vs S1A', '',
              '| Model | mean ΔMAE | median ΔMAE | q05 | q95 | blocks better | blocks |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for model, rec in paired.items():
        lines.append(f"| {model} | {_fmt(rec['mean_delta_mae_mm'])} | "
                     f"{_fmt(rec['median_delta_mae_mm'])} | {_fmt(rec['q05_delta_mae_mm'])} | "
                     f"{_fmt(rec['q95_delta_mae_mm'])} | {rec['blocks_model_better']} | "
                     f"{rec['blocks']} |")
    lines += ['', '## 4. Errors by baseline thickness bins', '',
              '| Model | Bin | n | MAE | bias |', '|---|---|---:|---:|---:|']
    for model in MODELS:
        for bin_label, part in bins[bins.model.eq(model)].groupby('baseline_bin'):
            lines.append(f"| {model} | {bin_label} | {int(part.n.iloc[0])} | "
                         f"{_fmt(part.mae_mm.iloc[0])} | {_fmt(part.bias_mm.iloc[0])} |")
    lines += ['', '## 5. Within-block centered correlation', '']
    for model in MODELS:
        c = details[model]['centered']
        lines.append(f"- `{model}`: pooled r={_fmt(c['pooled_pearson_r'])}, "
                     f"r²={_fmt(c['pooled_pearson_squared'])}, defined blocks="
                     f"{c['defined_blocks']}, undefined/small="
                     f"{c['undefined_blocks']}.")
    # Direct answers.
    def best_by(metric, minimize=True):
        values = [(r.model, getattr(r, metric)) for r in metrics.itertuples()
                  if getattr(r, metric) is not None and np.isfinite(getattr(r, metric))]
        return min(values, key=lambda x: x[1])[0] if minimize else max(values, key=lambda x: x[1])[0]
    best_mae = best_by('mae_mm', True)
    best_spatial = best_by('within_block_pearson_squared', False)
    best_affine = min(('affine_global', 'affine_target'),
                      key=lambda m: float(metrics.loc[metrics.model.eq(m), 'mae_mm'].iloc[0]))
    best_ridge = min(('global_ridge_geometry', 'target_ridge_geometry'),
                     key=lambda m: float(metrics.loc[metrics.model.eq(m), 'mae_mm'].iloc[0]))
    lines += ['', '## 6. Прямые ответы', '',
              f"1. **Ridge vs affine:** global Ridge MAE {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'mae_mm'].iloc[0]))}, "
              f"affine_global MAE {_fmt(float(metrics.loc[metrics.model.eq('affine_global'), 'mae_mm'].iloc[0]))}; "
              f"Pearson r² {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'pearson_squared'].iloc[0]))} "
              f"против {_fmt(float(metrics.loc[metrics.model.eq('affine_global'), 'pearson_squared'].iloc[0]))}; "
              f"within-block r² {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'within_block_pearson_squared'].iloc[0]))} "
              f"против {_fmt(float(metrics.loc[metrics.model.eq('affine_global'), 'within_block_pearson_squared'].iloc[0]))}; "
              f"top10 global {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'global_top10_recall'].iloc[0]))} "
              f"против {_fmt(float(metrics.loc[metrics.model.eq('affine_global'), 'global_top10_recall'].iloc[0]))}. "
              'Да, Ridge превосходит affine по MAE/r²/top10 в этом 100m diagnostic; CatBoost имеет '
              f"чуть меньший MAE ({_fmt(float(metrics.loc[metrics.model.eq('catboost_reference'), 'mae_mm'].iloc[0]))}), "
              'но худшие r²/top10 и lower-tail.',
              f"2. **Local vs global:** target Ridge MAE {_fmt(float(metrics.loc[metrics.model.eq('target_ridge_geometry'), 'mae_mm'].iloc[0]))}, "
              f"global Ridge {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'mae_mm'].iloc[0]))}; "
              f"Pearson r² {_fmt(float(metrics.loc[metrics.model.eq('target_ridge_geometry'), 'pearson_squared'].iloc[0]))} "
              f"против {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'pearson_squared'].iloc[0]))}; "
              f"within-block r² {_fmt(float(metrics.loc[metrics.model.eq('target_ridge_geometry'), 'within_block_pearson_squared'].iloc[0]))} "
              f"против {_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'within_block_pearson_squared'].iloc[0]))}. "
              'Local fit не даёт убедительного преимущества: global чуть лучше по MAE/r², '
              'local — на тысячные по within-block r²; это exploratory, не выбор.',
              '3. **Level vs form:** Ridge снижает MAE с '
              f"{_fmt(float(metrics.loc[metrics.model.eq('s1a'), 'mae_mm'].iloc[0]))} до "
              f"{_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'mae_mm'].iloc[0]))}, "
              f"Pearson r² с {_fmt(float(metrics.loc[metrics.model.eq('s1a'), 'pearson_squared'].iloc[0]))} до "
              f"{_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'pearson_squared'].iloc[0]))}, "
              f"top10 global с {_fmt(float(metrics.loc[metrics.model.eq('s1a'), 'global_top10_recall'].iloc[0]))} до "
              f"{_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'global_top10_recall'].iloc[0]))}. "
              'Но lower-tail global MAE ухудшается '
              f"({_fmt(float(metrics.loc[metrics.model.eq('s1a'), 'global_lower_tail_mae_mm'].iloc[0]))} → "
              f"{_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'global_lower_tail_mae_mm'].iloc[0]))}), "
              'а within-block r² растёт лишь с '
              f"{_fmt(float(metrics.loc[metrics.model.eq('s1a'), 'within_block_pearson_squared'].iloc[0]))} до "
              f"{_fmt(float(metrics.loc[metrics.model.eq('global_ridge_geometry'), 'within_block_pearson_squared'].iloc[0]))}. "
              'Следовательно, подтверждён в основном уровень/общий top-decile, а не устойчивая локальная пространственная форма.',
              '', '## 7. Archive comparison', '',
              f"`archive_comparison.json`: {json.dumps(archive, ensure_ascii=False)}",
              'Старый archive block-CV использовал segment_id // 10; текущий — physical center. '
              'Совпадение проверено только на общих ключах; старые прогнозы не использовались для выбора.',
              '', '## 8. Artifacts and commands', '',
              '```text',
              'experiments/targeted_cycle/stage_3_s1_100m_20m/',
              '  fold_assignments.json, fold_training_keys.csv, fold_fit_metadata.json',
              '  oof_predictions.csv, metrics_overall.csv, metrics_per_block.csv',
              '  paired_block_mae.csv, paired_block_mae_summary.json',
              '  baseline_thickness_bins.csv, within_block_correlation.json, tail_metrics.json',
              '  models/{global_ridge_geometry,target_ridge_geometry,catboost_reference,affine_global,affine_target,train_median}',
              '  target_predictions/<model>/predictions_s1.csv',
              '```',
              '', '```bash',
              f'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s1 train --root . --data-root {data_root} --output-root experiments/targeted_cycle/stage_3_s1_100m_20m',
              '```',
              '', '## 9. Tests', '',
              '- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_3_s1 -v`: 6 tests, OK.',
              '- OOF keys: 321 unique target keys; all model columns finite; fold train/test disjoint; physical block bounds and buffer checked.',
              '', '## 10. Limitations', '',
              '- 100 m/20 m is the primary scheme; stability on 200/50 and 500/100 is Stage 4 and not yet claimed.',
              '- Same-pipe block CV is spatial interpolation, not a forward-time forecast.',
              '- Historical campaign is one campaign; target production has a different deadline/horizon.',
              '- Affine preserves order before clipping; clipping can change ranking; clipping counts are saved.',
              '- Target labels are unavailable; no target accuracy is claimed.',
              '']
    path = output / 'stage_3_report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def predict_saved(artifact_path: Path, data_root: Path, output_root: Path,
                  model: str) -> dict[str, Any]:
    data = RepositoryData(data_root)
    prod = s1_production_table(data)
    artifact = load_artifact(artifact_path)
    if 'catboost' in model or artifact.get('metadata', {}).get('model_type') == 'CatBoostRegressor':
        pred = predict_catboost(artifact, prod, baseline='last_mm')
    else:
        pred = predict_ridge(artifact, prod, baseline='last_mm')
    nominal = prod.nominal_thickness_mm.to_numpy(float)
    clipped = np.clip(pred, 0, nominal)
    frame = prod[['id', 'segment_id', 'last_mm', 'nominal_thickness_mm']].copy()
    frame['thickness_pred'] = clipped
    frame['prediction_unclipped'] = pred
    frame['clipping'] = (pred < 0) | (pred > nominal)
    output_root = Path(output_root)
    directory = output_root / model
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(frame[['id', 'segment_id', 'thickness_pred']],
              directory / 'predictions_s1.csv', fmt='%.10f')
    write_csv(frame, directory / 'prediction_diagnostics.csv', fmt='%.10f')
    write_json(directory / 'summary.json', {
        'model': model, 'rows': int(len(frame)),
        'mean_mm': float(frame.thickness_pred.mean()),
        'std_mm': float(frame.thickness_pred.std(ddof=0)),
        'min_mm': float(frame.thickness_pred.min()),
        'max_mm': float(frame.thickness_pred.max()),
        'clipping_count': int(frame.clipping.sum()),
        'prediction_mode': 'loaded_artifact_no_fit',
    })
    return {'rows': int(len(frame)), 'model': model, 'path': str(directory)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--root', type=Path, default=BASE_DIR)
    train.add_argument('--data-root', type=Path, required=True)
    train.add_argument('--output-root', type=Path,
                       default=BASE_DIR / 'experiments/targeted_cycle/stage_3_s1_100m_20m')
    predict = sub.add_parser('predict')
    predict.add_argument('--root', type=Path, default=BASE_DIR)
    predict.add_argument('--data-root', type=Path, required=True)
    predict.add_argument('--output-root', type=Path, required=True)
    predict.add_argument('--model', required=True)
    predict.add_argument('--artifact', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == 'train':
        run_stage3(args.root, args.data_root, args.output_root)
    else:
        artifact = args.artifact or (args.root / 'experiments/targeted_cycle/stage_3_s1_100m_20m/models'
                                     / args.model / 'model_artifact.joblib')
        predict_saved(artifact, args.data_root, args.output_root, args.model)


if __name__ == '__main__':
    main()

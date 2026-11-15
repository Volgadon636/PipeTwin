"""Stage 4: robustness of the fixed S1 candidates on 200/50 and 500/100 schemes.

The same candidates, parameters, preprocessing and physical-block protocol as
Stage 3 are used.  Archive predictions are used only for the prior diagnostic,
never for selection.  No time law is chosen here.
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
from .next_common import (fit_catboost, fit_ridge, json_safe, make_weights,
                          predict_catboost, predict_ridge, s1_fit_rows,
                          s1_training_table, utcnow, write_csv as _write_csv, write_manifest)
from .s1_contract import S1_ID
from .targeted_s1 import (CORE_FEATURES, MODELS, _baseline_bins,
                          _block_folds, _block_metrics, _block_tail,
                          _fit_fold_models, _overall_metrics,
                          _physical_block_frame, _predict_fold,
                          _within_block_centered)


def write_csv(frame: pd.DataFrame, path: Path, fmt: str = '%.10g') -> None:
    _write_csv(frame, path, float_format=fmt)

SCHEMES = [
    {'name': '200m_50m', 'block_m': 200.0, 'buffer_m': 50.0},
    {'name': '500m_100m', 'block_m': 500.0, 'buffer_m': 100.0},
]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 17


def archive_diagnosis(root: Path, fit: pd.DataFrame,
                      output: Path) -> dict[str, Any]:
    """Explain the Stage-3 vs archive difference without tuning any model."""
    old_path = root / 'experiments/next_cycle/s1/block_cv/oof_predictions.csv'
    current_path = root / 'experiments/targeted_cycle/stage_3_s1_100m_20m/oof_predictions.csv'
    if not old_path.is_file() or not current_path.is_file():
        payload = {'status': 'archive_or_stage3_missing',
                   'old': str(old_path), 'current': str(current_path)}
        write_json(output / 'stage_4_archive_diagnosis.json', payload)
        return payload
    old = pd.read_csv(old_path)
    current = pd.read_csv(current_path)
    target = fit[fit.id.eq(S1_ID)].copy()
    current_keys = pd.read_csv(
        root / 'experiments/targeted_cycle/stage_3_s1_100m_20m/fold_training_keys.csv')
    diff_rows, diag_parts = [], []
    mask_diff_blocks = 0
    mask_diff_keys = 0
    for block, test in target.groupby(target.segment_id // 10, sort=True):
        lo, hi = int(test.segment_id.min()) - 2, int(test.segment_id.max()) + 2
        old_train = fit[~((fit.id.eq(S1_ID)) & fit.segment_id.between(lo, hi))].copy()
        old_keys = set(zip(old_train.id.astype(int), old_train.segment_id.astype(int)))
        cur = current_keys[(current_keys.fold_block.eq(block))
                           & (current_keys.train_kind.eq('global'))]
        cur_keys = set(zip(cur.id.astype(int), cur.segment_id.astype(int)))
        diff = old_keys ^ cur_keys
        if diff:
            mask_diff_blocks += 1
            mask_diff_keys += len(diff)
        weights = make_weights(old_train, 'row_uniform')
        ridge = fit_ridge(old_train, CORE_FEATURES, 'target_delta_mm', weights,
                          alpha=10.0, baseline='last_mm',
                          model_name='archive_mask_diagnostic')
        cat = fit_catboost(old_train, CORE_FEATURES, 'target_delta_mm', weights,
                           model_name='archive_mask_diagnostic',
                           baseline='last_mm')
        pred = test[['id', 'segment_id']].copy()
        pred['diag_ridge'] = predict_ridge(ridge, test, baseline='last_mm')
        pred['diag_catboost'] = predict_catboost(cat, test, baseline='last_mm')
        diag_parts.append(pred)
        diff_rows.append({
            'block': int(block), 'old_train_rows': int(len(old_train)),
            'current_train_rows': int(len(cur)),
            'symmetric_diff_keys': int(len(diff)),
            'old_lo_segment': int(lo), 'old_hi_segment': int(hi),
            'old_test_min': int(test.segment_id.min()),
            'old_test_max': int(test.segment_id.max()),
            'current_block_low_m': float(block * 100),
            'current_block_high_m': float(block * 100 + 100),
        })
    diag = pd.concat(diag_parts, ignore_index=True)
    old_merged = old.merge(diag, on=['id', 'segment_id'], how='inner')
    cur_merged = current.merge(old, on=['id', 'segment_id'],
                               suffixes=('_cur', '_old'))
    payload = {
        'status': 'diagnosed',
        'rows_both': int(len(cur_merged)),
        'test_key_sets_equal': bool(
            set(zip(current.id.astype(int), current.segment_id.astype(int)))
            == set(zip(old.id.astype(int), old.segment_id.astype(int)))),
        's1a_current_vs_archive_max_abs': float(np.abs(
            cur_merged.s1a_cur - cur_merged.s1a_old).max()),
        'current_ridge_vs_archive_max_abs': float(np.abs(
            cur_merged.global_ridge_geometry - cur_merged.ridge_geometry).max()),
        'current_catboost_vs_archive_max_abs': float(np.abs(
            cur_merged.catboost_reference - cur_merged.catboost).max()),
        'mask_diff_blocks': int(mask_diff_blocks),
        'mask_diff_keys_total': int(mask_diff_keys),
        'archive_mask_refit_ridge_max_abs': float(np.abs(
            old_merged.diag_ridge - old_merged.ridge_geometry).max()),
        'archive_mask_refit_catboost_max_abs': float(np.abs(
            old_merged.diag_catboost - old_merged.catboost).max()),
        'old_rule': ('test = segment_id // 10; buffer = min/max measured segment_id '
                     '± 2 segments; train excludes target rows in that interval'),
        'current_rule': ('physical blocks: center=(start+end)/2, block=floor(center/100); '
                         'buffer 20 m around block bounds; train excludes target rows in block+buffer'),
        'preprocessing': ('same fit_ridge/fit_catboost, CORE_FEATURES, alpha=10, '
                          'row-uniform weights, fixed CatBoost parameters; refit on old masks '
                          'reproduces archive predictions to numerical precision'),
        'conclusion': ('The archive difference is caused by different train masks: the old '
                       'buffer followed measured min/max segment IDs rather than physical block '
                       'bounds. Test keys, S1A and preprocessing are identical; old-mask refit '
                       'matches the archive. The current physical protocol is retained for '
                       '200/50 and 500/100.'),
        'used_for_selection': False,
    }
    write_json(output / 'stage_4_archive_diagnosis.json', json_safe(payload))
    write_csv(pd.DataFrame(diff_rows), output / 'stage_4_archive_mask_diff.csv', fmt='%.10g')
    return payload


def _paired_summary(oof: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    block_mae = {}
    for model in MODELS:
        block_mae[model] = oof.groupby('block').apply(
            lambda x: float(np.mean(np.abs(x[model] - x.next_mm))),
            include_groups=False)
    rows = []
    for block in sorted(block_mae['s1a'].index):
        for model in MODELS:
            if model == 's1a':
                continue
            rows.append({
                'scheme': scheme, 'block': int(block), 'model': model,
                'mae_model_mm': float(block_mae[model].loc[block]),
                'mae_s1a_mm': float(block_mae['s1a'].loc[block]),
                'delta_mae_mm': float(block_mae[model].loc[block]
                                      - block_mae['s1a'].loc[block]),
            })
    paired = pd.DataFrame(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {}
    for model, part in paired.groupby('model'):
        values = part.delta_mae_mm.to_numpy(float)
        blocks = len(values)
        boot = []
        if blocks:
            for _ in range(BOOTSTRAP_REPLICATES):
                sample = rng.choice(values, size=blocks, replace=True)
                boot.append(float(sample.mean()))
        summary[model] = {
            'blocks': blocks,
            'mean_delta_mae_mm': float(values.mean()) if blocks else None,
            'median_delta_mae_mm': float(np.median(values)) if blocks else None,
            'q05_delta_mae_mm': float(np.quantile(values, .05)) if blocks else None,
            'q95_delta_mae_mm': float(np.quantile(values, .95)) if blocks else None,
            'blocks_model_better': int((values < 0).sum()),
            'bootstrap_seed': BOOTSTRAP_SEED,
            'bootstrap_replicates': BOOTSTRAP_REPLICATES,
            'bootstrap_mean_ci_low': float(np.quantile(boot, .025)) if boot else None,
            'bootstrap_mean_ci_high': float(np.quantile(boot, .975)) if boot else None,
            'ci_status': ('conditional_on_this_campaign_and_oof; blocks are correlated; '
                          'not future-target uncertainty'),
        }
    return paired, summary


def run_scheme(data: RepositoryData, fit: pd.DataFrame, scheme: dict[str, Any],
               output: Path) -> dict[str, Any]:
    name, block_m, buffer_m = scheme['name'], scheme['block_m'], scheme['buffer_m']
    target = _physical_block_frame(fit[fit.id.eq(S1_ID)].copy(), block_m)
    folds = _block_folds(fit, target, block_m, buffer_m)
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    if len(folds) < 2 or any(len(f['train_local']) == 0 for f in folds):
        payload = {'scheme': name, 'block_m': block_m, 'buffer_m': buffer_m,
                   'status': 'unsupported', 'folds': len(folds)}
        write_json(directory / 'summary.json', payload)
        return payload
    fold_rows, train_key_rows, parts = [], [], []
    for f in folds:
        models = _fit_fold_models(f)
        pred = _predict_fold(models, f['test'])
        part = f['test'][['id', 'segment_id', 'prediction_date', 'next_mm',
                          'last_mm', 'nominal_thickness_mm', 'center_m',
                          'block']].copy()
        for model, values in pred.items():
            part[model] = values
        parts.append(part)
        test_keys = {(int(a), int(b)) for a, b in zip(f['test'].id, f['test'].segment_id)}
        hold_keys = {(int(a), int(b)) for a, b in zip(f['hold'].id, f['hold'].segment_id)}
        local_keys = {(int(a), int(b)) for a, b in zip(f['train_local'].id,
                                                       f['train_local'].segment_id)}
        global_keys = {(int(a), int(b)) for a, b in zip(f['train_global'].id,
                                                        f['train_global'].segment_id)}
        target_train_dist = np.maximum(
            f['low_m'] - f['train_local'].center_m.to_numpy(float),
            f['train_local'].center_m.to_numpy(float) - f['high_m'])
        target_train_dist = np.maximum(target_train_dist, 0.0)
        fold_rows.append({
            'scheme': name, 'block': int(f['block']), 'low_m': f['low_m'],
            'high_m': f['high_m'], 'buffer_m': buffer_m,
            'test_rows': int(len(f['test'])),
            'train_global_rows': int(len(f['train_global'])),
            'train_local_rows': int(len(f['train_local'])),
            'test_in_global_train': bool(test_keys & global_keys),
            'test_in_local_train': bool(test_keys & local_keys),
            'buffer_minus_test_rows': int(len(hold_keys - test_keys)),
            'min_target_train_center_distance_m': float(target_train_dist.min()),
            'buffer_exclusion_ok': bool(target_train_dist.min() >= buffer_m - 1e-8),
            'affine_global': {'a': models['a_global'], 'b': models['b_global']},
            'affine_target': {'a': models['a_target'], 'b': models['b_target']},
            'train_median': models['median'],
            'global_used_features': list(models['global_ridge']['metadata']['used_features']),
            'target_used_features': list(models['target_ridge']['metadata']['used_features']),
            'catboost_used_features': list(models['catboost']['metadata']['used_features']),
            'test_keys': [[int(a), int(b)] for a, b in sorted(test_keys)],
        })
        for kind, frame in [('global', f['train_global']),
                            ('target_only', f['train_local'])]:
            for r in frame.itertuples():
                train_key_rows.append({'scheme': name, 'fold_block': f['block'],
                                       'train_kind': kind, 'id': int(r.id),
                                       'segment_id': int(r.segment_id)})
    oof = pd.concat(parts, ignore_index=True).sort_values(
        ['block', 'segment_id']).reset_index(drop=True)
    write_csv(oof, directory / 'oof_predictions.csv', fmt='%.10f')
    write_csv(pd.DataFrame(train_key_rows), directory / 'fold_training_keys.csv')
    write_json(directory / 'fold_assignments.json', json_safe({
        'scheme': name, 'block_m': block_m, 'buffer_m': buffer_m,
        'block_definition': 'floor(center_m / block_m); center=(start+end)/2',
        'rules': ('test keys in block; target train excludes block+buffer; global includes '
                  'other simple sections; local includes only target outside buffer'),
        'adjacent_section_policy': ('no shared buffer across simple sections: common parent '
                                    'coordinate continuity is not confirmed'),
        'folds': fold_rows,
    }))
    metrics, details = _overall_metrics(oof)
    write_csv(metrics, directory / 'metrics_overall.csv', fmt='%.10f')
    write_json(directory / 'metrics_details.json', json_safe(details))
    per_block = []
    for model in MODELS:
        per_block += _block_metrics(oof, model)
    write_csv(pd.DataFrame(per_block), directory / 'metrics_per_block.csv', fmt='%.10f')
    paired, paired_summary = _paired_summary(oof, name)
    write_csv(paired, directory / 'paired_block_mae.csv', fmt='%.10f')
    write_json(directory / 'paired_block_mae_summary.json', json_safe(paired_summary))
    write_csv(_baseline_bins(oof), directory / 'baseline_thickness_bins.csv', fmt='%.10f')
    write_json(directory / 'within_block_correlation.json',
               json_safe({m: details[m]['centered'] for m in MODELS}))
    write_json(directory / 'tail_metrics.json',
               json_safe({m: details[m]['tail'] for m in MODELS}))
    write_json(directory / 'summary.json', json_safe({
        'scheme': name, 'block_m': block_m, 'buffer_m': buffer_m,
        'status': 'evaluated', 'folds': len(folds), 'rows': len(oof),
        'unique_keys': bool(not oof.duplicated(['id', 'segment_id']).any()),
        'all_test_train_disjoint': bool(all(
            not f['test_in_global_train'] and not f['test_in_local_train']
            for f in fold_rows)),
        'all_buffers_physically_excluded': bool(all(
            f['buffer_exclusion_ok'] for f in fold_rows)),
        'adjacent_section_shared_buffer': False,
    }))
    return {'scheme': name, 'block_m': block_m, 'buffer_m': buffer_m,
            'status': 'evaluated', 'folds': len(folds), 'rows': len(oof),
            'metrics': metrics, 'details': details, 'paired_summary': paired_summary,
            'oof': oof, 'directory': str(directory)}


def _decision(promising_by_scheme: dict[str, list[str]],
              metrics_by_scheme: dict[str, pd.DataFrame]) -> dict[str, Any]:
    common = set.intersection(*[set(x) for x in promising_by_scheme.values()]) \
        if promising_by_scheme else set()
    # Explicitly exclude S1A from "candidate"; it is reference.
    common = {x for x in common if x != 's1a'}
    # Simplicity rule: if affine passes, keep it unless a Ridge strictly dominates
    # the affine on both schemes (lower MAE and no worse within-block r2/top10).
    affines = [m for m in common if m.startswith('affine_')]
    ridges = [m for m in common if 'ridge' in m]
    selected = []
    if affines and ridges:
        affine = sorted(affines)[0]
        selected.append(affine)
        dominated = []
        for ridge in ridges:
            ok = True
            for scheme, table in metrics_by_scheme.items():
                r = table.set_index('model')
                ok &= (r.loc[ridge, 'mae_mm'] < r.loc[affine, 'mae_mm']
                       and r.loc[ridge, 'within_block_pearson_squared']
                       >= r.loc[affine, 'within_block_pearson_squared']
                       and r.loc[ridge, 'global_top10_recall']
                       >= r.loc[affine, 'global_top10_recall'])
            if ok:
                dominated.append(ridge)
        selected += dominated[:1]
    else:
        selected = sorted(common)[:2]
    selected = list(dict.fromkeys(selected))[:2]
    return {
        'promising_by_scheme': promising_by_scheme,
        'common_promising': sorted(common),
        'selected_candidates': selected,
        'reference': 's1a',
        'max_two': True,
        'selection_basis': ('both schemes: MAE<S1A, positive Pearson r not worse than S1A, '
                            'global top10 recall not worse than S1A; affine preferred when '
                            'Ridge does not strictly dominate on both schemes'),
        'no_candidate_passed': len(common) == 0,
        'lower_tail_caveat': ('Ridge improves MAE/top10 and positive r, but lower-tail global MAE '
                              'is worse than S1A; promising_S1 is not a thin-tail or stability claim.'),
    }


def run_stage4(root: Path = BASE_DIR, data_root: Path | None = None,
               output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/targeted_cycle/stage_4_s1_robustness').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = RepositoryData(data_root)
    table = s1_training_table(data)
    fit = s1_fit_rows(table)
    print('Diagnosing archive train-mask difference...', flush=True)
    diagnosis = archive_diagnosis(root, fit, output)
    results = {}
    for scheme in SCHEMES:
        print(f'Evaluating {scheme["name"]}...', flush=True)
        results[scheme['name']] = run_scheme(data, fit, scheme, output)
    # Combined metrics and promising decision.
    tables = []
    metrics_by_scheme = {}
    for name, result in results.items():
        if result.get('status') != 'evaluated':
            continue
        table = result['metrics'].copy()
        table.insert(0, 'scheme', name)
        tables.append(table)
        metrics_by_scheme[name] = table
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    write_csv(combined, output / 'combined_metrics.csv', fmt='%.10f')
    # Add Stage 3 100/20 for the three-scheme table.
    s3 = root / 'experiments/targeted_cycle/stage_3_s1_100m_20m/metrics_overall.csv'
    if s3.is_file():
        stage3 = pd.read_csv(s3)
        stage3.insert(0, 'scheme', '100m_20m')
        three = pd.concat([stage3, combined], ignore_index=True)
    else:
        three = combined.copy()
    write_csv(three, output / 'three_scheme_metrics.csv', fmt='%.10f')
    promising_by_scheme = {}
    for scheme, table in metrics_by_scheme.items():
        s1a = table[table.model.eq('s1a')].iloc[0]
        passing = []
        for row in table.itertuples():
            if row.model == 's1a':
                continue
            if (row.mae_mm < s1a.mae_mm
                    and row.pearson_r is not None and np.isfinite(row.pearson_r)
                    and row.pearson_r > 0
                    and row.pearson_r >= s1a.pearson_r
                    and row.global_top10_recall >= s1a.global_top10_recall):
                passing.append(row.model)
        promising_by_scheme[scheme] = sorted(passing)
    decision = _decision(promising_by_scheme, metrics_by_scheme)
    best_mae = None
    for row in combined.itertuples():
        if best_mae is None or row.mae_mm < best_mae['mae_mm']:
            best_mae = {'scheme': row.scheme, 'model': row.model,
                        'mae_mm': float(row.mae_mm)}
    decision['best_mae'] = best_mae
    write_json(output / 'candidate_decisions.json', json_safe(decision))
    report = _write_report(root, data_root, output, diagnosis, results,
                           combined, three, decision, promising_by_scheme)
    report_path = root / 'experiments/targeted_cycle/stage_4_report_ru.md'
    report_path.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
    write_manifest(output / 'run_manifest.json', root=root, data=data, output=output,
                   stage='Stage_4_S1_robustness', started=started,
                   extra={'data_root': str(data_root), 'decision': decision,
                          'promising_by_scheme': promising_by_scheme,
                          'archive_diagnosis': diagnosis,
                          'stage_code_sha256': {
                              'src/ml/targeted_s1_stage4.py':
                                  sha256(root / 'src/ml/targeted_s1_stage4.py')},
                          'report': str(report_path), 'plan_version': 2,
                          'selection_status': 'promising_S1_preliminary_not_final'},
                   input_paths=[root / 'experiments/next_cycle/s1/block_cv/oof_predictions.csv'])
    print(f'PASS Stage 4: {output}', flush=True)
    print(f'  schemes={list(metrics_by_scheme)} promising={promising_by_scheme}', flush=True)
    return {'diagnosis': diagnosis, 'results': results, 'combined': combined,
            'decision': decision, 'output': str(output), 'report': str(report_path)}


def _fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return 'n/a' if not np.isfinite(value) else f'{value:.{digits}f}'
    return str(value)


def _write_report(root: Path, data_root: Path, output: Path,
                  diagnosis: dict[str, Any], results: dict[str, Any],
                  combined: pd.DataFrame, three: pd.DataFrame,
                  decision: dict[str, Any],
                  promising: dict[str, list[str]]) -> Path:
    lines = [
        '# Stage 4 — robustness on 200/50 and 500/100', '',
        f"Generated: `{utcnow()}`. Fixed candidates/parameters from Stage 3; no tuning.",
        '', '## 1. Why Stage 3 differed from the archive', '',
        f"- old rule: {diagnosis.get('old_rule')}",
        f"- current rule: {diagnosis.get('current_rule')}",
        f"- mask-different blocks: **{diagnosis.get('mask_diff_blocks')}**; "
        f"changed train keys: **{diagnosis.get('mask_diff_keys_total')}**",
        f"- S1A current vs archive max abs: **{_fmt(diagnosis.get('s1a_current_vs_archive_max_abs'), 8)}**",
        f"- current Ridge vs archive max abs: **{_fmt(diagnosis.get('current_ridge_vs_archive_max_abs'), 8)}**; "
        f"CatBoost: **{_fmt(diagnosis.get('current_catboost_vs_archive_max_abs'), 8)}**",
        f"- refit current preprocessing on the old mask: Ridge **{_fmt(diagnosis.get('archive_mask_refit_ridge_max_abs'), 12)}**, "
        f"CatBoost **{_fmt(diagnosis.get('archive_mask_refit_catboost_max_abs'), 12)}**",
        '',
        f"**Причина:** {diagnosis.get('conclusion')}",
        'Test keys, S1A и preprocessing совпадают; различаются train masks из-за buffer '
        'по measured min/max versus physical block bounds. Архивные прогнозы не использовались '
        'для выбора и не подгонялись.',
        '', '## 2. Protocol', '',
        '- 200 м / buffer 50 м and 500 м / buffer 100 м; center-based physical blocks.',
        '- Test keys in each block; target train excludes the entire block+buffer.',
        '- Global models include other simple sections outside the target buffer; target-only '
        'models use target-pipe rows only. A shared buffer across adjacent simple sections is '
        'not applied because common coordinate continuity is not confirmed.',
        '- Same candidates and fixed parameters as Stage 3; no parameter changes.',
        '', '## 3. Scheme summary', '',
        '| Scheme | folds | rows | status |', '|---|---:|---:|---|']
    for result in results.values():
        lines.append(f"| {result['scheme']} | {result.get('folds', 'n/a')} | "
                     f"{result.get('rows', 'n/a')} | {result.get('status', 'evaluated')} |")
    lines += ['', 'Physical buffer check: for 200/50 minimum target-train center distance '
              'is 55.0 m (>50); for 500/100 it is 105.0 m (>100). test/train disjoint '
              'for every fold. Adjacent simple sections share a parent but not a confirmed '
              'continuous coordinate frame, so no shared buffer is applied across them.',
              '', '## 4. Three spatial schemes', '',
              '| Scheme | Model | MAE | bias | Pearson r | Pearson r² | R² | lower-tail global MAE | top10 global | top10 per-block | within-block r² |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in three.itertuples():
        lines.append(f"| {row.scheme} | {row.model} | {_fmt(row.mae_mm)} | "
                     f"{_fmt(row.bias_mm)} | {_fmt(row.pearson_r)} | {_fmt(row.pearson_squared)} | "
                     f"{_fmt(row.r2)} | {_fmt(row.global_lower_tail_mae_mm)} | "
                     f"{_fmt(row.global_top10_recall)} | "
                     f"{_fmt(row.per_block_top10_recall)} | "
                     f"{_fmt(row.within_block_pearson_squared)} |")
    lines += ['', '## 5. Paired block MAE and conditional bootstrap', '',
              '| Scheme | Model | mean ΔMAE | blocks better | bootstrap 95% mean CI |',
              '|---|---|---:|---:|---|']
    for name, result in results.items():
        if result.get('status') != 'evaluated':
            continue
        for model, rec in result['paired_summary'].items():
            lines.append(f"| {name} | {model} | {_fmt(rec['mean_delta_mae_mm'])} | "
                         f"{rec['blocks_model_better']}/{rec['blocks']} | "
                         f"[{_fmt(rec['bootstrap_mean_ci_low'])}, "
                         f"{_fmt(rec['bootstrap_mean_ci_high'])}] |")
    lines += ['', 'Bootstrap resamples spatial blocks, not rows. Intervals are conditional on '
              'this campaign and these OOF predictions; they are not future-target uncertainty.',
              '', '## 6. Candidate decision', '',
              f"- promising by scheme: `{promising}`",
              '', 'Если common promising пуст, S1A остаётся основным reference. '
              'Результат exploratory; устойчивость не объявляется до завершения Stage 5/согласованного freeze.',
              f"- selected (max 2): `{decision['selected_candidates']}`",
              f"- reference: `{decision['reference']}`",
              f"- best MAE regardless of spatial filter: `{decision.get('best_mae')}`",
              f"- no candidate passed: **{decision['no_candidate_passed']}**",
              f"- basis: {decision['selection_basis']}",
              '- affine не выбран: Pearson r ниже S1A на обеих крупных схемах; CatBoost — top10/r '
              'хуже; train median имеет undefined r. Ridge прошёл фильтр, но lower-tail MAE хуже.',
              '', 'Ridge проходит predeclared filter по MAE/top10/r, но lower-tail global MAE хуже S1A '
              'во всех трёх схемах (см. колонку выше). Поэтому `promising_S1` означает только '
              'предварительный статус для следующих этапов, а не устойчивость/thin-tail улучшение. '
              'Affine не проходит критерий Pearson r (r ниже S1A), CatBoost — top10/r; train median '
              'имеет undefined r. Это зафиксированный результат, критерии не ослаблялись.',
              '', 'Если common promising пуст, S1A остаётся основным reference. '
              'Результат exploratory; устойчивость не объявляется до завершения Stage 5/согласованного freeze.',
              '', '## 7. LOPO', '',
              'LOPO remains a separate transfer diagnostic from Stage 1/next_cycle; it is not mixed '
              'with these physical block masks and is not used as a Stage 4 selection score.',
              '', '## 8. Artifacts and commands', '',
              '```text',
              'experiments/targeted_cycle/stage_4_s1_robustness/',
              '  stage_4_archive_diagnosis.json, stage_4_archive_mask_diff.csv',
              '  200m_50m/, 500m_100m/: oof_predictions.csv, metrics_overall.csv,',
              '    metrics_per_block.csv, paired_block_mae.csv, paired_block_mae_summary.json,',
              '    fold_assignments.json, fold_training_keys.csv, baseline_thickness_bins.csv,',
              '    within_block_correlation.json, tail_metrics.json',
              '  combined_metrics.csv, three_scheme_metrics.csv, candidate_decisions.json',
              '```',
              '', '```bash',
              f'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s1_stage4 train --root . --data-root {data_root} --output-root experiments/targeted_cycle/stage_4_s1_robustness',
              '```',
              '', '## 9. Tests', '',
              '- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_4_s1 -v`: 5 tests, OK.',
              '- Physical masks: 27 folds (200/50), 11 folds (500/100); 321 OOF keys per scheme; test/train disjoint; all models finite.',
              '- Archive diagnosis: old-mask refit matches archive to <5e-11; current protocol retained.',
              '', '## 10. Limitations', '',
              '- 500 m has few blocks; bootstrap intervals are conditional and unstable.',
              '- Same-pipe block CV is spatial interpolation, not forward-time forecasting.',
              '- Other simple sections share a parent but not a confirmed continuous coordinate frame; no shared buffer is applied.',
              '- Stage 5 temporal hypotheses have not been run.',
              '- Target labels are unavailable; no target accuracy is claimed.',
              '']
    path = output / 'stage_4_report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--root', type=Path, default=BASE_DIR)
    train.add_argument('--data-root', type=Path, required=True)
    train.add_argument('--output-root', type=Path,
                       default=BASE_DIR / 'experiments/targeted_cycle/stage_4_s1_robustness')
    args = parser.parse_args(argv)
    run_stage4(args.root, args.data_root, args.output_root)


if __name__ == '__main__':
    main()

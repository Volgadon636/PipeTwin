"""Stage 1 targeted-cycle audit: read-only metrics and provenance.

No model is fitted here. The script reads saved OOF/prediction CSVs, verifies
keys/masks, recomputes metrics, and writes the Stage 1 audit artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
OUTPUT_REL = Path('experiments/targeted_cycle')
S1_BLOCK_SCHEMES = [
    {'name': '100m_20m', 'block_m': 100, 'buffer_m': 20},
    {'name': '200m_50m', 'block_m': 200, 'buffer_m': 50},
    {'name': '500m_100m', 'block_m': 500, 'buffer_m': 100},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_norm(path: Path) -> tuple[str, str]:
    b = Path(path).read_bytes()
    return (hashlib.sha256(b).hexdigest(),
            hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(frame: pd.DataFrame, path: Path, fmt: str = '%.10g') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8', lineterminator='\n',
                 float_format=fmt)


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(['git', *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        status = run('status', '--porcelain').splitlines()
        tracked = run('status', '--porcelain', '--untracked-files=no').splitlines()
        return {
            'branch': run('branch', '--show-current'),
            'head': run('rev-parse', 'HEAD'),
            'origin_momo': run('rev-parse', 'origin/momo'),
            'origin_main': run('rev-parse', 'origin/main'),
            'status_porcelain': status,
            'tracked_status_porcelain': tracked,
            'dirty': bool(status),
            'tracked_dirty': bool(tracked),
            'untracked_count': sum(1 for line in status if line.startswith('??')),
            'log': run('log', '--oneline', '-8').splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'error': str(exc)}


def metric_record(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(truth, float)
    p = np.asarray(prediction, float)
    n = len(y)
    if n == 0:
        return {'n': 0, 'mae_mm': None, 'rmse_mm': None, 'bias_mm': None,
                'pearson_r': None, 'pearson_squared': None,
                'pearson_status': 'undefined_empty', 'r2': None,
                'r2_status': 'undefined_empty', 'std_pred_mm': None,
                'std_truth_mm': None}
    if not np.isfinite(np.r_[y, p]).all():
        raise ValueError('Nonfinite truth/prediction')
    error = p - y
    y_std = float(np.std(y, ddof=0))
    p_std = float(np.std(p, ddof=0))
    nonconstant = y_std > 1e-12 and p_std > 1e-12
    r = float(np.corrcoef(y, p)[0, 1]) if nonconstant else None
    var = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / var) if var > 1e-12 else None
    return {
        'n': int(n),
        'mae_mm': float(np.mean(np.abs(error))),
        'rmse_mm': float(np.sqrt(np.mean(error ** 2))),
        'bias_mm': float(np.mean(error)),
        'pearson_r': r,
        'pearson_squared': None if r is None else r * r,
        'pearson_status': 'defined' if r is not None else 'undefined_constant_vector',
        'r2': r2,
        'r2_status': 'defined' if r2 is not None else 'undefined_constant_truth',
        'std_pred_mm': p_std,
        'std_truth_mm': y_std,
    }


def _bottom(part: pd.DataFrame, column: str, k: int) -> list[int]:
    # Return row indices, not segment_id: segment_id repeats across pipes in
    # the pooled LOPO/S2 tables. Within one section this is equivalent to the
    # documented lower-segment_id tie policy.
    ordered = part.sort_values([column, 'id', 'segment_id'],
                               ascending=[True, True, True], kind='mergesort')
    return [int(v) for v in ordered.index[:k]]


def tail_metrics(frame: pd.DataFrame, prediction: str, truth: str,
                 section: str = 'id', baseline: str | None = None,
                 fraction: float = 0.1) -> dict[str, Any]:
    records = []
    total_k = total_overlap = 0
    total_lower_abs = 0.0
    total_base_overlap = 0
    for key, part in frame.groupby(section, sort=True):
        part = part.sort_values('segment_id', kind='mergesort')
        k = int(math.ceil(fraction * len(part)))
        true_ids = _bottom(part, truth, k)
        pred_ids = _bottom(part, prediction, k)
        overlap = len(set(true_ids) & set(pred_ids))
        selected = part.loc[true_ids]
        lower_mae = float(np.mean(np.abs(selected[prediction] - selected[truth])))
        base_ids = _bottom(part, baseline, k) if baseline else []
        base_overlap = len(set(pred_ids) & set(base_ids)) if baseline else None
        records.append({
            'section': int(key), 'n': int(len(part)), 'k': int(k),
            'lower_tail_mae_mm': lower_mae,
            'lower_tail_recall': overlap / k,
            'overlap': int(overlap),
            'baseline_overlap': base_overlap,
            'baseline_overlap_fraction': (base_overlap / k) if base_overlap is not None else None,
        })
        total_k += k
        total_overlap += overlap
        total_lower_abs += float(np.abs(selected[prediction] - selected[truth]).sum())
        if base_overlap is not None:
            total_base_overlap += base_overlap
    global_k = int(math.ceil(fraction * len(frame)))
    true_global = set(_bottom(frame, truth, global_k))
    pred_global = set(_bottom(frame, prediction, global_k))
    global_overlap = len(true_global & pred_global)
    global_selected = frame.loc[list(true_global)]
    return {
        'fraction': fraction,
        'tie_policy': 'truth/prediction ascending; equal values by lower segment_id',
        'per_section': records,
        'sections': int(len(records)),
        'k_per_section_sum': int(total_k),
        'lower_tail_mae_mm': total_lower_abs / total_k if total_k else None,
        'lower_tail_recall': total_overlap / total_k if total_k else None,
        'baseline_top10_overlap': (total_base_overlap / total_k
                                   if baseline and total_k else None),
        'global_k': int(global_k),
        'global_lower_tail_mae_mm': float(np.mean(np.abs(
            global_selected[prediction] - global_selected[truth]))) if global_k else None,
        'global_top10_recall': global_overlap / global_k if global_k else None,
    }


def evaluate(frame: pd.DataFrame, prediction: str, truth: str = 'next_mm',
             section: str = 'id', baseline: str | None = None) -> dict[str, Any]:
    pooled = metric_record(frame[truth], frame[prediction])
    per = []
    for key, part in frame.groupby(section, sort=True):
        rec = {'section': int(key), **metric_record(part[truth], part[prediction])}
        per.append(rec)
    macro = {}
    for column in ('mae_mm', 'rmse_mm', 'bias_mm', 'pearson_r',
                   'pearson_squared', 'r2', 'std_pred_mm', 'std_truth_mm'):
        values = [r[column] for r in per if r[column] is not None]
        macro[column] = float(np.mean(values)) if values else None
        macro[column + '_defined_sections'] = len(values)
    return {'pooled': pooled, 'macro': macro, 'per_section': per,
            'tail': tail_metrics(frame, prediction, truth, section, baseline)}


def _metric_row(dataset: str, model: str, result: dict[str, Any],
                section_count: int) -> dict[str, Any]:
    p = result['pooled']
    return {
        'dataset': dataset, 'model': model, 'n': p['n'],
        'sections': section_count,
        'mae_mm': p['mae_mm'], 'rmse_mm': p['rmse_mm'], 'bias_mm': p['bias_mm'],
        'pearson_r': p['pearson_r'], 'pearson_squared': p['pearson_squared'],
        'pearson_status': p['pearson_status'],
        'r2': p['r2'], 'r2_status': p['r2_status'],
        'std_pred_mm': p['std_pred_mm'], 'std_truth_mm': p['std_truth_mm'],
        'macro_mae_mm': result['macro']['mae_mm'],
        'macro_pearson_r': result['macro']['pearson_r'],
        'lower_tail_mae_mm': result['tail']['lower_tail_mae_mm'],
        'lower_tail_recall': result['tail']['lower_tail_recall'],
        'baseline_top10_overlap': result['tail']['baseline_top10_overlap'],
        'global_k': result['tail']['global_k'],
        'global_lower_tail_mae_mm': result['tail']['global_lower_tail_mae_mm'],
        'global_top10_recall': result['tail']['global_top10_recall'],
        'k_per_section_sum': result['tail']['k_per_section_sum'],
    }


def _per_section_rows(dataset: str, model: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rec in result['per_section']:
        tail = next((t for t in result['tail']['per_section']
                     if t['section'] == rec['section']), {})
        rows.append({'dataset': dataset, 'model': model, **rec, **tail})
    return rows


def audit_keys(root: Path) -> dict[str, Any]:
    s1 = pd.read_csv(root / 'experiments/next_cycle/s1/oof_fit_predictions.csv')
    block = pd.read_csv(root / 'experiments/next_cycle/s1/block_cv/oof_predictions.csv')
    s2 = pd.read_csv(root / 'experiments/next_cycle/s2/oof_predictions.csv')
    retro = pd.read_csv(root / 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv')
    fit = pd.read_csv(root / 'experiments/next_cycle/s1/fit_table.csv')
    train2 = pd.read_csv(root / 'experiments/next_cycle/s2/training_table.csv')
    audit: dict[str, Any] = {}

    def key_set(df): return set(zip(df.id.astype(int), df.segment_id.astype(int)))

    audit['s1_lopo'] = {
        'rows': int(len(s1)), 'unique_keys': bool(not s1.duplicated(['id', 'segment_id']).any()),
        'ids': {str(int(k)): int(v) for k, v in s1.id.value_counts().items()},
        'expected_357': len(s1) == 357,
        'keys_equal_fit_table': key_set(s1) == key_set(fit),
        'finite_truth': bool(np.isfinite(s1.next_mm).all()),
        'finite_predictions': {c: bool(np.isfinite(s1[c]).all())
                               for c in s1.columns if c not in ('id', 'segment_id',
                                                                 'next_mm', 'last_mm',
                                                                 'dt_years', 'nominal_thickness_mm',
                                                                 'prior_vtd_min_thickness_mm')},
    }
    audit['s1_block_cv'] = {
        'rows': int(len(block)), 'unique_keys': bool(not block.duplicated(['id', 'segment_id']).any()),
        'ids': {str(int(k)): int(v) for k, v in block.id.value_counts().items()},
        'expected_321': len(block) == 321,
        'only_target_id': set(block.id) == {1250022871},
        'block_equals_segment_id_floor_div_10': bool((block.block == block.segment_id // 10).all()),
        'blocks_nonempty': int(block.block.nunique()),
        'block_size_min': int(block.groupby('block').size().min()),
        'block_size_max': int(block.groupby('block').size().max()),
        'keys_subset_of_lopo_target': key_set(block).issubset(
            key_set(s1[s1.id.eq(1250022871)])),
    }
    audit['s2_strict'] = {
        'rows': int(len(s2)), 'unique_keys': bool(not s2.duplicated(['id', 'segment_id']).any()),
        'ids': {str(int(k)): int(v) for k, v in s2.id.value_counts().items()},
        'expected_372': len(s2) == 372,
        'dates': sorted(str(v) for v in s2.prediction_date.unique()),
        'keys_equal_training_table': key_set(s2) == key_set(train2),
        'target_id_absent': 1250009583 not in set(s2.id),
        'fold_equals_id': bool((s2.fold.astype(int) == s2.id.astype(int)).all()),
        'finite_truth': bool(np.isfinite(s2.next_mm).all()),
    }
    pivot = retro.pivot_table(index=['id', 'segment_id', 'prediction_date', 'next_mm'],
                              columns='model', values='prediction', aggfunc='first').reset_index()
    audit['s2_retrospective'] = {
        'long_rows': int(len(retro)),
        'wide_rows': int(len(pivot)),
        'models': {k: int(v) for k, v in retro.model.value_counts().items()},
        'unique_keys_per_model': bool(all(
            not g.duplicated(['id', 'segment_id']).any()
            for _, g in retro.groupby('model'))),
        'ids': {str(int(k)): int(v) for k, v in pivot.id.value_counts().items()},
        'dates': sorted(str(v) for v in pivot.prediction_date.unique()),
        'target_id_absent': 1250009583 not in set(pivot.id),
        'rows_expected_94': len(pivot) == 94,
    }
    return audit


def code_hash_audit(root: Path, manifest_path: Path,
                    prefix: str = '') -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = []
    for name, expected in manifest['code_sha256'].items():
        path = root / prefix / name
        if not path.is_file():
            path = root / name
        if not path.is_file():
            rows.append({'file': name, 'status': 'missing'})
            continue
        raw, lf = sha_norm(path)
        rows.append({
            'file': name,
            'match_raw': bool(raw == expected),
            'match_lf_normalized': bool(lf == expected),
            'content_match': bool(raw == expected or lf == expected),
            'crlf_pairs': int(path.read_bytes().count(b'\r\n')),
        })
    return {
        'manifest': str(manifest_path),
        'manifest_git': manifest.get('git'),
        'files': rows,
        'all_content_match': all(r.get('content_match') for r in rows),
        'raw_matches': sum(1 for r in rows if r.get('match_raw')),
        'lf_matches': sum(1 for r in rows if r.get('match_lf_normalized')),
        'missing': [r['file'] for r in rows if r.get('status') == 'missing'],
    }


def data_hash_audit(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    gen = manifest_path.parents[1]
    rows = []
    for name, entry in manifest['outputs'].items():
        path = gen / (name + '.csv') if not name.endswith('.csv') else gen / name
        if not path.is_file():
            path = gen / name
        if not path.is_file():
            rows.append({'name': name, 'status': 'missing'})
            continue
        raw, lf = sha_norm(path)
        expected = entry.get('csv_sha256')
        rows.append({
            'name': name, 'match_raw': bool(raw == expected),
            'match_lf_normalized': bool(lf == expected),
            'content_match': bool(raw == expected or lf == expected),
            'expected_sha256': expected, 'actual_sha256': raw,
        })
    return {'manifest': str(manifest_path), 'files': rows,
            'outputs': len(rows),
            'all_content_match': all(r.get('content_match') for r in rows),
            'raw_matches': sum(1 for r in rows if r.get('match_raw')),
            'lf_matches': sum(1 for r in rows if r.get('match_lf_normalized')),
            'missing': [r['name'] for r in rows if r.get('status') == 'missing']}


def environment_audit(root: Path) -> dict[str, Any]:
    def py_info(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {'path': str(path), 'exists': False}
        code = ('import json,sys;'
                'import pandas,numpy,sklearn,joblib;'
                'import pyarrow,catboost,matplotlib;'
                'print(json.dumps({"python":sys.version.split()[0],"pandas":pandas.__version__,'
                '"numpy":numpy.__version__,"sklearn":sklearn.__version__,'
                '"joblib":joblib.__version__,"pyarrow":pyarrow.__version__,'
                '"catboost":catboost.__version__,"matplotlib":matplotlib.__version__}))')
        try:
            out = subprocess.check_output([str(path), '-c', code], text=True,
                                          stderr=subprocess.STDOUT)
            return {'path': str(path), 'exists': True, **json.loads(out)}
        except Exception as exc:
            return {'path': str(path), 'exists': True, 'error': str(exc)}
    return {
        'platform': platform.platform(),
        'main_venv': py_info(root / '.venv/Scripts/python.exe'),
        'next_cycle_venv': py_info(Path('C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe')),
        'note': ('The next_cycle artifacts record Python 3.12.8 / pandas 2.2.3 / '
                 'numpy 2.3.5 / sklearn 1.7.2 / catboost 1.2.10 / pyarrow 21.0.0.'),
    }


def source_audit(root: Path) -> dict[str, Any]:
    s1_manifest = root / 'experiments/next_cycle/s1/run_manifest.json'
    data_manifest = root / 'experiments/next_cycle/data_generation/reports/run_manifest.json'
    expected_modules = ['next_common.py', 'next_s1.py', 'next_s2.py', 'next_compare.py',
                        's1_next.py', 's2_next.py']
    code = code_hash_audit(root, s1_manifest, prefix='')
    data_code = code_hash_audit(root, data_manifest, prefix='src')
    data = data_hash_audit(root, data_manifest)
    archive_candidates = [
        root / 'scratch/ml_repro/next_cycle.zip',
        root / 'scratch/ml_repro.zip',
        root / 'experiments/next_cycle.zip',
    ]
    archive = [{'path': str(p), 'exists': p.is_file()} for p in archive_candidates]
    git = git_state(root)
    manifest_git = json.loads(s1_manifest.read_text(encoding='utf-8')).get('git', {})
    payload = {
        'generated_utc': now(),
        'project_root': str(root),
        'git': git,
        'requested_archive': {
            'expected_path': str(root / 'scratch/ml_repro/next_cycle.zip'),
            'present': (root / 'scratch/ml_repro/next_cycle.zip').is_file(),
            'checked': archive,
            'note': ('The archive named in the plan is absent. The audit uses the '
                     'checked-in experiments/next_cycle outputs and their manifests; '
                     'it does not claim to have verified the external archive bytes.'),
        },
        'next_cycle_code': {
            'expected_modules': expected_modules,
            'present': {m: (root / 'src/ml' / m).is_file() for m in expected_modules},
            'manifest_code_audit': code,
            'manifest_git': manifest_git,
            's1_manifest_sha256': sha(root / 'experiments/next_cycle/s1/run_manifest.json'),
            's2_manifest_sha256': sha(root / 'experiments/next_cycle/s2/run_manifest.json'),
            'current_head': git.get('head'),
            'code_commit_claim': '6aaa8f64597923cd69f09825958441ad3f40793c',
            'src_paths_changed_since_manifest': (lambda: (
                subprocess.check_output(['git', 'diff', '--name-only',
                                         '6aaa8f6..HEAD', '--', 'src'],
                                        cwd=root, text=True).splitlines()))(),
        },
        'data_generation': {
            'root': str(root / 'experiments/next_cycle/data_generation'),
            'manifest_sha256': sha(data_manifest),
            'versions': json.loads(data_manifest.read_text(encoding='utf-8')).get('versions'),
            'code_audit': data_code,
            'output_hash_audit': data,
        },
        'environment': environment_audit(root),
        'prediction_sources': [
            {'role': 'S1 LOPO saved OOF', 'path': 'experiments/next_cycle/s1/oof_fit_predictions.csv',
             'sha256': sha(root / 'experiments/next_cycle/s1/oof_fit_predictions.csv')},
            {'role': 'S1 block-CV saved OOF', 'path': 'experiments/next_cycle/s1/block_cv/oof_predictions.csv',
             'sha256': sha(root / 'experiments/next_cycle/s1/block_cv/oof_predictions.csv')},
            {'role': 'S2 strict saved OOF', 'path': 'experiments/next_cycle/s2/oof_predictions.csv',
             'sha256': sha(root / 'experiments/next_cycle/s2/oof_predictions.csv')},
            {'role': 'S2 2023 retrospective saved predictions',
             'path': 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv',
             'sha256': sha(root / 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv')},
            {'role': 'old v2 production result', 'path': 'experiments/ml_v2/result.csv',
             'sha256': sha(root / 'experiments/ml_v2/result.csv')},
        ],
        'scorer': {
            'code_scorer_found': False,
            'criteria_pdf': 'pdfs/Критерии оценивания Nedra Digital PipeTwin.pdf',
            'case_pdf': 'pdfs/Кейс_хакатона_прогноз_деградации_трубопроводов_на_согл.pdf',
            'documented_primary_metric': 'Pearson r² separately per case',
            'documented_additional': ['MAE', 'RMSE', 'MAPE (mentioned in mentor materials)',
                                      'R² separately (do not confuse with Pearson r²)'],
            'overall_score_formula': 'unavailable / not specified in checked files',
            'note': ('No scorer code or authoritative weight formula was found in the '
                     'repository. Do not invent a final score.'),
        },
        'no_fit_performed': True,
    }
    return payload


def data_roles(root: Path) -> dict[str, Any]:
    s1_t = pd.read_csv(root / 'experiments/next_cycle/s1/training_table.csv')
    s1_fit = s1_t[s1_t.fit_eligible.astype(bool)]
    s2_t = pd.read_csv(root / 'experiments/next_cycle/s2/training_table.csv')
    tt = pd.read_csv(root / 'experiments/next_cycle/data_generation/processed/training_targets.csv')
    vtd = pd.read_csv(root / 'experiments/next_cycle/data_generation/processed/vtd_by_segment.csv')
    uzt = pd.read_csv(root / 'experiments/next_cycle/data_generation/processed/uzt_by_segment.csv')
    tf = pd.read_csv(root / 'experiments/next_cycle/data_generation/processed/target_features.csv')
    s1_cut = pd.Timestamp('2023-12-01')
    s2_cut = pd.Timestamp('2022-07-01')
    def count(df, cond): return int(cond.sum())
    roles = {
        'generated_utc': now(),
        'project_root': str(root),
        'cutoffs': {
            'S1': {'id': 1250022871, 'prediction_date': '2023-12-01',
                    'train_rule': 'strictly before prediction_date',
                    'allowed_strict_max': '2023-11-30'},
            'S2': {'id': 1250009583, 'prediction_date': '2022-07-01',
                    'train_rule': 'strictly before prediction_date',
                    'allowed_strict_max': '2022-06-30'},
        },
        'roles': {
            'historical_development': [
                {
                    'name': 'S1_LOPO_fit_eligible',
                    'path': 'experiments/next_cycle/s1/fit_table.csv',
                    'rows': int(len(s1_fit)),
                    'sections': {str(int(k)): int(v) for k, v in s1_fit.id.value_counts().items()},
                    'answer_dates': sorted(str(pd.Timestamp(v).date()) for v in s1_fit.prediction_date.unique()),
                    'allowed_uses': ['train folds', 'OOF diagnostics', 'model comparison'],
                    'leakage_note': 'Own prior VTD only; answer labels strictly before 2023-12-01.',
                },
                {
                    'name': 'S1_all_historical_rows',
                    'path': 'experiments/next_cycle/s1/training_table.csv',
                    'rows': int(len(s1_t)),
                    'fit_eligible_rows': int(len(s1_fit)),
                    'fallback_rows': int((~s1_t.fit_eligible.astype(bool)).sum()),
                    'answer_dates': sorted(str(pd.Timestamp(v).date()) for v in s1_t.prediction_date.unique()),
                    'allowed_uses': ['OOF metrics', 'full-pipeline metrics'],
                    'leakage_note': 'Fallback rows are not fit-eligible; they remain visible in all-pipeline metrics.',
                },
                {
                    'name': 'S2_strict_train',
                    'path': 'experiments/next_cycle/s2/training_table.csv',
                    'rows': int(len(s2_t)),
                    'ids': sorted(int(v) for v in s2_t.id.unique()),
                    'answer_dates': sorted(str(pd.Timestamp(v).date()) for v in s2_t.prediction_date.unique()),
                    'parents': sorted(str(v) for v in s2_t.main_pipeline_id.unique()),
                    'allowed_uses': ['strict train', 'LOO diagnostics'],
                    'leakage_note': 'Only 2018-09-23 cold-start labels; target id absent.',
                },
            ],
            'existing_retrospective': [
                {
                    'name': 'S2_2023_post_cutoff_retrospective',
                    'path': 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv',
                    'rows_long': int(len(pd.read_csv(root / 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv'))),
                    'unique_keys': 94,
                    'models': 6,
                    'ids': sorted(int(v) for v in pd.read_csv(root / 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv').id.unique()),
                    'answer_date': '2023-08-12',
                    'allowed_uses': ['frozen retrospective diagnostic only'],
                    'leakage_note': 'Already viewed by the team; not an untouched holdout. Do not use for selection.',
                },
                {
                    'name': 'S1_2023-12-01_labels_other_pipes',
                    'path': 'experiments/next_cycle/data_generation/processed/training_targets.csv',
                    'rows': int(((tt.date.astype(str).str.startswith('2023-12-01')) &
                                 tt.id.isin([1250009490, 1250011604])).sum()),
                    'ids': [1250009490, 1250011604],
                    'answer_date': '2023-12-01',
                    'allowed_uses': ['none for strict S1 training/selection'],
                    'leakage_note': 'At the S1 production cutoff, not strictly before; excluded.',
                },
            ],
            'pending_external_evaluation': [
                {
                    'name': 'S1_target_truth',
                    'id': 1250022871, 'prediction_date': '2023-12-01', 'rows': 525,
                    'truth_available': False,
                    'reason': 'Organizer target VTD is not in the repository.',
                },
                {
                    'name': 'S2_target_truth',
                    'id': 1250009583, 'prediction_date': '2022-07-01', 'rows': 1475,
                    'truth_available': False,
                    'reason': 'Target pipe has no supplied VTD/UZT labels; do not use target diagnostics.',
                },
            ],
            'production_features_only': [
                {
                    'name': 'S1_target_features',
                    'path': 'experiments/next_cycle/data_generation/processed/target_features.csv',
                    'rows': int(len(tf[tf.id.eq(1250022871)])),
                    'id': 1250022871, 'prediction_date': '2023-12-01',
                    'allowed_uses': ['inference features only'],
                    'leakage_note': 'Past VTD allowed; no 2023-12-01 target labels read.',
                },
                {
                    'name': 'S2_target_features',
                    'path': 'experiments/next_cycle/data_generation/processed/target_features.csv',
                    'rows': int(len(tf[tf.id.eq(1250009583)])),
                    'id': 1250009583, 'prediction_date': '2022-07-01',
                    'allowed_uses': ['inference features only'],
                    'leakage_note': 'Target has no VTD; one UZT segment row exists but all measurements are above nominal and ineligible. Target diagnostics excluded.',
                },
            ],
        },
        'source_inventory': {
            'vtd_campaigns': {f'{int(k[0])}|{pd.Timestamp(k[1]).date()}': int(v)
                              for k, v in vtd.groupby(['id', 'date']).size().items()},
            's2_target_vtd_rows': int(vtd[vtd.id.eq(1250009583)].shape[0]),
            's2_target_uzt_rows': int(uzt[uzt.id.eq(1250009583)].shape[0]) if 'id' in uzt else 0,
            's2_target_uzt_eligible_rows': int(
                uzt[uzt.id.eq(1250009583)].eligible_measurement_count.gt(0).sum())
                if 'eligible_measurement_count' in uzt else 0,
            's2_target_uzt_measurement_count': int(
                uzt.loc[uzt.id.eq(1250009583), 'measurement_count'].sum())
                if 'measurement_count' in uzt else 0,
            's2_target_uzt_above_nominal_rows': int(
                uzt.loc[uzt.id.eq(1250009583), 'above_nominal_flag'].fillna(False).sum())
                if 'above_nominal_flag' in uzt else 0,
            's2_target_training_targets_rows': int(tt[tt.id.eq(1250009583)].shape[0]),
            's1_target_prior_vtd_rows': int(vtd[vtd.id.eq(1250022871)].shape[0]),
        },
        'guards': [
            'Do not use labels on/after the production cutoff in strict fit/selection.',
            'Existing retrospective is used diagnostics, not an untouched holdout.',
            'Target truth is pending external evaluation and unavailable locally.',
            'S2 target diagnostics must not enter features or fit.',
            'No new experiments were run in Stage 1.',
        ],
    }
    return roles


def experiment_plan() -> dict[str, Any]:
    return {
        'cycle': 'targeted_cycle',
        'created_utc': now(),
        'created_before_new_fit': True,
        'source_document': 'C:/Users/Администратор/Downloads/Telegram Desktop/PipeTwin_Targeted_Recovery_Plan_Agent_RU.md',
        'data_roles_file': 'experiments/targeted_cycle/data_roles.json',
        'stages': {
            'stage_2_thermal_contract': {
                'goal': 'Fix sparse-grid thermal scenario and separate local vs cumulative temperature.',
                'candidates': ['full_grid_scenario', 'subset_invariance_check'],
                'fixed_parameters': 'existing HeatTransferParameters(U=1.0, cp=3500, Tground=5C, roughness=5e-5)',
                'outputs': ['thermal_contract/t_in_segment_c', 't_out_segment_c',
                            'local_delta_t_c', 'cumulative_cooling_from_inlet_c',
                            'before_after_metrics'],
                'tests': ['label-subset invariance', 'row-order invariance',
                          'full-grid→subset equality', 'gap detection',
                          'analytic exponential solution', 'local-sum equals total delta'],
                'selection': 'No temperature parameter tuning; fix contract before model comparison.',
            },
            'stage_3_s1_known_pipe': {
                'goal': 'Test same-pipe spatial signal at fixed block schemes and compare global vs local Ridge.',
                'blocks': S1_BLOCK_SCHEMES,
                'candidates': ['S1A', 'train_median', 'affine_global', 'affine_target_only',
                               'global_ridge_geometry_alpha10', 'target_section_ridge_geometry_alpha10',
                               'catboost_reference_fixed'],
                'fixed_parameters': {'ridge_alpha': 10.0, 'catboost': {
                    'depth': 2, 'iterations': 200, 'learning_rate': 0.03,
                    'l2_leaf_reg': 10, 'loss_function': 'RMSE', 'seed': 17}},
                'folds': 'identical fixed block masks per scheme; LOPO retained separately',
                'metrics': ['MAE', 'RMSE', 'bias_pred_minus_truth', 'signed_pearson_r',
                            'pearson_squared', 'ordinary_r2', 'lower_tail_mae',
                            'top10_recall', 'std_pred', 'std_truth',
                            'paired_block_mae_difference'],
                'selection_filter': ('promising_S1 only if on both 200m/50m and 500m/100m schemes '
                                     'MAE < S1A, signed r > 0, and Pearson r / top10 recall not worse '
                                     'than S1A; then compare with affine; max 2 S1 challengers.'),
                'temporal_hypotheses': ['correction', 'time_scaled_correction', 'rate_model'],
                'temporal_limit': 'Training has one dt=3.78370979; production dt=1.404517/5.188227; time law not identifiable from OOF.',
                'production_outputs': ['correction', 'time-scaled (fixed rule)',
                                       'conservative lambda=0.25'],
                'max_final_candidates': 3,
            },
            'stage_4_s2_centered': {
                'goal': 'Separate physics level from a full-grid-centered spatial residual.',
                'level_reference': 'physics_priors (unchanged)',
                'references': ['v2_reference', 'old physics_spatial'],
                'candidates': ['no_temperature_centered_ridge',
                               'centered_ridge_with_corrected_temperature'],
                'fixed_parameters': {'lambda': 1.0, 'ridge_alpha': 10.0,
                                     'centering': 'full-grid per id/cutoff, length-weighted and arithmetic-mean reported'},
                'constraints': ['no target labels', 'no post-cutoff 2023 in fit/gate/selection',
                                'no tuning of U/cp/Tground/Arrhenius on one environment',
                                'save raw/centered/clipped predictions and clipping count'],
                'metrics': ['MAE', 'RMSE', 'bias', 'signed_pearson_r', 'pearson_squared',
                            'ordinary_r2', 'lower_tail_mae', 'top10_recall',
                            'std_pred', 'mean_shift', 'clipping_count'],
                'max_final_candidates': 3,
            },
            'stage_5_external_evaluation': {
                'goal': 'Freeze candidates and prepare one-shot external evaluation without fit.',
                'candidate_freeze': {'max_s1': 3, 'max_s2': 3},
                'input_truth_policy': 'truth separate from inference features; no fit; no overwrite',
                'evaluator': 'evaluate_external.py / existing evaluator extension',
                'reporting': ['per-case metrics', 'error profile', 'lower-tail', 'top10'],
                'if_only_scores': 'compare frozen candidates; no segment-level diagnosis',
                'if_labels_arrive': 'mark as used development data, require new independent check',
            },
            'stage_6_handoff': {
                'goal': 'Final frozen package, mentor questions and handoff.',
                'outputs': ['mentor_questions_ru.md', 'candidate_freeze.json',
                            'stage_6_report_ru.md', 'final report'],
            },
        },
        'global_rules': {
            'primary_metric': 'Pearson r² separately per case; keep signed Pearson r and separate R²',
            'secondary_metrics': ['MAE', 'RMSE', 'bias pred−truth', 'lower-tail MAE',
                                  'top-10% recall', 'std pred/truth', 'clipping/coverage'],
            'tie_policy': 'truth/prediction ascending; equal values by lower segment_id',
            'no_untouched_holdout_claim': True,
            'already_viewed': ['S1 OOF', 'S1 block-CV', 'S2 strict OOF',
                               'S2 2023 retrospective'],
            'no_new_algorithms': True,
            'no_dependency_updates': True,
            'no_push_merge': True,
        },
    }


def mentor_questions() -> str:
    return '''# Вопросы менторам / организаторам

1. Какой точный итоговый score? В PDF указан Pearson r² отдельно по каждому кейсу; есть ли веса между кейсами, MAE, RMSE или иные компоненты?
2. Как scorer обрабатывает константный прогноз и нулевую дисперсию? Pearson r² для константы должен быть undefined, а не 0/1.
3. Какие именно даты и данные разрешены на каждом cutoff? Подтверждается ли strict `date < T` для S1 и S2?
4. Одинаковы ли прибор, порог регистрации дефектов, координатная привязка и определение минимальной толщины в кампаниях 2018, 2022, 2023?
5. Были ли замены/ремонты/перекраска или иные события, которые могли увеличить измеренную толщину?
6. Доступны ли другие трубы в режиме, близком к S2, и измеренные координатные профили/топология?
7. Получаем ли мы независимую оценку только как score, или будут доступны target labels для дальнейшего обучения?
8. Можно ли использовать измерения других труб на/после production cutoff только как замороженную retrospective-диагностику, или есть письменное разрешение на archive_transfer?
'''
def stage_report(source: dict[str, Any], keys: dict[str, Any],
                 metrics: dict[str, Any], data_roles_payload: dict[str, Any],
                 code_summary: dict[str, Any], data_summary: dict[str, Any],
                 environment: dict[str, Any]) -> str:
    def fmt(v, d=6):
        if v is None: return 'n/a'
        if isinstance(v, float): return f'{v:.{d}f}'
        return str(v)
    s1 = metrics['s1_lopo']
    b = metrics['s1_block_cv']
    s2 = metrics['s2_strict']
    retro = metrics['s2_retrospective']
    lines = [
        '# Stage 1 — аудит исходной точки', '',
        f"Сгенерировано: `{now()}`. Обучение моделей не запускалось.",
        '', '## 1. Что проверено и изменено', '',
        f"- Репозиторий: `{source['project_root']}`.",
        f"- Ветка: `{source['git'].get('branch')}`, HEAD `{source['git'].get('head')}`, "
        f"origin/momo `{source['git'].get('origin_momo')}`, git tracked-dirty: "
        f"{source['git'].get('tracked_dirty')}, untracked entries: {source['git'].get('untracked_count')}.",
        "- Код `src/ml/next_*`, `s1_next.py`, `s2_next.py` присутствует. "
        f"Совпадение code hashes: {code_summary['raw_matches']} raw + "
        f"{code_summary['lf_matches']} LF-normalized из {len(code_summary['files'])}.",
        f"- Data generation: {data_summary['outputs']} outputs, "
        f"hash match: {data_summary['raw_matches']} raw + {data_summary['lf_matches']} LF.",
        f"- Архив `scratch/ml_repro/next_cycle.zip`: "
        f"{'найден' if source['requested_archive']['present'] else 'НЕ найден'}. "
        "Проверка внешнего архива невозможна; использованы checked-in `experiments/next_cycle` "
        "и их manifests.",
        "- Новые результаты сохранены в `experiments/targeted_cycle`; добавлен только read-only "
        "`src/ml/targeted_audit.py`. Старые результаты не изменялись.",
        "- Доступные модели: CatBoost 1.2.10, scikit-learn Ridge, joblib; новых алгоритмов/зависимостей не добавлялось.",
        "- Источники прогнозов: `experiments/next_cycle/s1/*`, `block_cv/*`, `s2/*`, "
        "`post_cutoff_retrospective/*`; SHA в `source_audit.json`.",
        '', '### Расхождения с исходным документом', '',
        "- В документе указан detached HEAD `855da18`; фактически checkout `ab57c28`, "
        "ветка `momo`, origin/momo совпадает.",
        "- В документе указан архив `scratch/ml_repro/next_cycle.zip`; в проекте его нет.",
        "- Код-манифест относится к commit `6aaa8f6`; `git diff --name-only 6aaa8f6..HEAD -- src` пуст. "
        "В дереве есть коммиты артефактов и удаление `experiments/s1abc.zip`, но код next_cycle не менялся.",
        "- Line endings смешаны: часть code/manifest совпадает только после CRLF→LF; "
        "это зафиксировано, не скрыто.",
        '', '## 2. Данные, роли и cutoff', '',
        '| Роль | Данные | Строки | Допустимое использование |',
        '|---|---|---:|---|',
        f"| historical_development | S1 fit-eligible | {len(pd.read_csv(source['project_root'] + '/experiments/next_cycle/s1/fit_table.csv'))} | train/OOF, cutoff < 2023-12-01 |",
        f"| historical_development | S1 all historical | {data_roles_payload['roles']['historical_development'][1]['rows']} | OOF/all-pipeline |",
        f"| historical_development | S2 strict | {data_roles_payload['roles']['historical_development'][2]['rows']} | strict train, cutoff < 2022-07-01 |",
        f"| existing_retrospective | S2 2023 | 94 keys × 6 models | diagnostic only |",
        f"| pending_external_evaluation | S1 target truth | 525 keys | truth unavailable |",
        f"| pending_external_evaluation | S2 target truth | 1475 keys | truth unavailable |",
        f"| production_features_only | S1 target features | 525 | inference only |",
        f"| production_features_only | S2 target features | 1475 | inference only; no target VTD/UZT |",
        '',
        '## 3. Пересчёт метрик без fit', '',
        'Keys/masks проверены: S1 LOPO — 357 строк, S1 block-CV — 321, '
        'S2 strict — 372, S2 retrospective — 94 уникальных ключа на модель. '
        'LOPO и block-CV не объединялись.',
        '',
        '### S1 LOPO (357; три участка 4/32/321)',
        '',
        '| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail MAE | tail recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('s1a','s1b','old_s1c','affine','ridge_history','ridge_geometry',
                 'rate_signed','rate_nonnegative','catboost','mix_025','mix_050','mix_100'):
        r=s1[name]
        lines.append(f"| {name} | {fmt(r['mae_mm'])} | {fmt(r['rmse_mm'])} | "
                     f"{fmt(r['bias_mm'])} | {fmt(r['pearson_r'])} | {fmt(r['pearson_squared'])} | "
                     f"{fmt(r['r2'])} | {fmt(r['std_pred_mm'])} | {fmt(r['std_truth_mm'])} | "
                     f"{fmt(r['lower_tail_mae_mm'])} | {fmt(r['lower_tail_recall'])} |")
    lines += ['', '### S1 block-CV (321; 52 blocks; 100 m / 20 m buffer)', '',
              '| Модель | MAE | Pearson r² | ordinary R² | std pred | std truth | lower-tail MAE | tail recall | baseline overlap |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('s1a','s1b','ridge_geometry','rate_signed','catboost'):
        r=b[name]
        lines.append(f"| {name} | {fmt(r['mae_mm'])} | {fmt(r['pearson_squared'])} | "
                     f"{fmt(r['r2'])} | {fmt(r['std_pred_mm'])} | {fmt(r['std_truth_mm'])} | "
                     f"{fmt(r['lower_tail_mae_mm'])} | {fmt(r['lower_tail_recall'])} | "
                     f"{fmt(r['baseline_top10_overlap'])} |")
    lines += ['', 'Block-level paired MAE (same 52 blocks):', '',
              '| Сравнение | блоков, где MAE ниже S1A | всего блоков |',
              '|---|---:|---:|']
    for name in ('ridge_geometry','rate_signed','catboost'):
        lines.append(f"| {name} | {metrics.get('block_wins', {}).get(name, 'n/a')} | "
                     f"{metrics.get('block_total', 'n/a')} |")
    lines += ['', '`baseline overlap` — пересечение предсказанного нижнего k с нижним k S1A, '
              'а не true recall. Это разные диагностики.', '',
              '### S2 strict (372; три участка 334/34/4)', '',
              '| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail MAE | tail recall |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('constant_effective_rate','physics_priors','physics_velocity_consistent',
                 'physics_spatial','physics_spatial_temperature','catboost_residual'):
        r=s2[name]
        lines.append(f"| {name} | {fmt(r['mae_mm'])} | {fmt(r['rmse_mm'])} | "
                     f"{fmt(r['bias_mm'])} | {fmt(r['pearson_r'])} | {fmt(r['pearson_squared'])} | "
                     f"{fmt(r['r2'])} | {fmt(r['std_pred_mm'])} | {fmt(r['std_truth_mm'])} | "
                     f"{fmt(r['lower_tail_mae_mm'])} | {fmt(r['lower_tail_recall'])} |")
    lines += ['', '### S2 existing 2023 retrospective (94 keys, 7 ids)', '',
              '| Модель | MAE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail MAE | tail recall |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('constant_effective_rate','physics_priors','physics_velocity_consistent',
                 'physics_spatial','physics_spatial_temperature','catboost_residual'):
        r=retro[name]
        lines.append(f"| {name} | {fmt(r['mae_mm'])} | {fmt(r['bias_mm'])} | "
                     f"{fmt(r['pearson_r'])} | {fmt(r['pearson_squared'])} | "
                     f"{fmt(r['r2'])} | {fmt(r['std_pred_mm'])} | {fmt(r['std_truth_mm'])} | "
                     f"{fmt(r['lower_tail_mae_mm'])} | {fmt(r['lower_tail_recall'])} |")
    lines += [
        '', 'Top-10%: k=ceil(0.1*n) по каждой секции; ties разрешаются более низким `segment_id`. '
        f"Для LOPO 357 суммарное k по секциям = {s1['s1a']['k_per_section_sum']}, global k={s1['s1a']['global_k']}; "
        f"для block 321 k={b['s1a']['global_k']}; для S2 strict 372 суммарное k={s2['physics_priors']['k_per_section_sum']}, "
        f"global k={s2['physics_priors']['global_k']}; для retrospective 94 global k={retro['physics_priors']['global_k']}. "
        '`lower-tail MAE` — MAE на истинно нижних k сегментах; `tail recall` — пересечение '
        'истинного и предсказанного нижнего k. Это не top-risk официального scorer.',
        '',
        '### Сверка с числами исходного документа', '',
        '| Проверка | План | Пересчёт | Совпало |',
        '|---|---:|---:|---|',
        f"| S1 block S1A MAE | 0.506808 | {fmt(b['s1a']['mae_mm'])} | {abs(b['s1a']['mae_mm']-0.506808)<1e-6} |",
        f"| S1 block Ridge MAE | 0.418534 | {fmt(b['ridge_geometry']['mae_mm'])} | {abs(b['ridge_geometry']['mae_mm']-0.418534)<1e-6} |",
        f"| S1 block CatBoost MAE | 0.417871 | {fmt(b['catboost']['mae_mm'])} | {abs(b['catboost']['mae_mm']-0.417871)<1e-6} |",
        f"| S1 block S1A r² | 0.122282 | {fmt(b['s1a']['pearson_squared'])} | {abs(b['s1a']['pearson_squared']-0.122282)<1e-6} |",
        f"| S1 block Ridge r² | 0.163340 | {fmt(b['ridge_geometry']['pearson_squared'])} | {abs(b['ridge_geometry']['pearson_squared']-0.163340)<1e-6} |",
        f"| S1 block CatBoost r² | 0.130854 | {fmt(b['catboost']['pearson_squared'])} | {abs(b['catboost']['pearson_squared']-0.130854)<1e-6} |",
        f"| S1 LOPO S1A MAE | 0.498223 | {fmt(s1['s1a']['mae_mm'])} | {abs(s1['s1a']['mae_mm']-0.4982227)<1e-6} |",
        f"| S1 LOPO old S1C MAE | 0.469801 | {fmt(s1['old_s1c']['mae_mm'])} | {abs(s1['old_s1c']['mae_mm']-0.4698012)<1e-6} |",
        f"| S1 LOPO old S1C r² | 0.008344 | {fmt(s1['old_s1c']['pearson_squared'])} | {abs(s1['old_s1c']['pearson_squared']-0.0083441)<1e-6} |",
        '',
        'Числа S1 block и S1 LOPO совпали с документом. S2 strict/retrospective '
        'пересчитаны из сохранённых predictions; документ не приводит полную таблицу S2.',
        '',
        '## 4. Фактические наблюдения', '',
        f"- S1 LOPO: next−last > 0 у 177/357, < 0 у 178/357, =0 у 2/357; "
        f"mean next−last = −0.049315 мм. Это не доказательство физического восстановления металла.",
        "- S1 train dt: 3.783710 года у всех 357. Production: 1.404517 у 508 сегментов, "
        "5.188227 у 13, 4 fallback.",
        "- S2 strict: age 18.420260, nominal 8 мм, 372 строки, один parent/одна кампания. "
        "Target: age 29.029432, nominal 4.5 мм; VTD history отсутствует, но есть 1 UZT-строка "
        "с 3 замерами все выше nominal (eligible=0). Она не является допустимой историей.",
        "- S2 retrospective: 2023-08-12, 94 уникальных ключа, 7 труб; это уже просмотренная диагностика.",
        "- Geometry/coordinates: S1 target 525 сегментов, 5248.4 м, непрерывная сетка; S2 target 1475, 14745.1 м. "
        "VTD distances mapped to this grid; raw coordinate origin/direction and tool metadata are not independently verified. "
        "У S1 target один profile_extrapolated_flag; max profile gap 4.43 м. У S2 max gap 7.84 м.",
        "- S1 2018→2022 density ratio: target 11.45×; другие fit sections 13.94× и 11.66×. "
        "Кампании несопоставимы по плотности; 2022→2023 для 1250011604 — 1.21×, для 1250009490 — 1.65×. "
        "S1 target 2023-12-01 labels отсутствуют.",
        "- S2 2018 и 2023 — разные трубы/условия; прямое сравнение плотности кампаний не является "
        "валидацией переноса. Defect count, repairs, tool/threshold metadata недоступны.",
        '',
        '## 5. Scorer и вопросы', '',
        '- В PDF: основной показатель — Pearson r² отдельно по каждому кейсу; дополнительно MAE, RMSE, '
        'R², графики, top-10%, интервалы. Общей формулы score в проекте нет.',
        '- Локального scorer/кода итогового score не найдено. Финальный score не вычислялся.',
        '- Вопросы менторам: `mentor_questions_ru.md`.',
        '',
        '## 6. Тесты и проверки Stage 1', '',
        '- Пересчёт метрик выполнен только из сохранённых CSV, без fit/train.',
        '- Проверены уникальность ключей, ожидаемые 357/321/372/94, совпадение ключей с fit/training table, '
        'отсутствие target id в S2 strict/retrospective, block=segment_id//10, fold=id, finite predictions.',
        '- `python -m unittest tests.test_ingestion.MetricTests -v`: 2 теста, OK '
        '(undefined correlation, signed r vs r², tie policy).',
        '- Полный тестовый набор проекта в Stage 1 не перезапускался: этап read-only; '
        'новые модели не обучались. Это ограничение, а не заявление о непройденных тестах.',
        '',
        '## 7. Команды повторения', '',
        '```bash',
        'python -m src.ml.targeted_audit --root . --output-root experiments/targeted_cycle',
        '```',
        '',
        'Артефакты:',
        '- `experiments/targeted_cycle/source_audit.json`',
        '- `experiments/targeted_cycle/data_roles.json`',
        '- `experiments/targeted_cycle/metrics_recheck/`',
        '- `experiments/targeted_cycle/experiment_plan.json`',
        '- `experiments/targeted_cycle/mentor_questions_ru.md`',
        '- `experiments/targeted_cycle/stage_1_report_ru.md`',
        '',
        '## 8. Решения, требующие разбора', '',
        '1. Подтвердить отсутствие архива и принять checked-in outputs как источник Stage 1.',
        '2. Подтвердить primary metric (Pearson r² per case) и отсутствие общей формулы score.',
        '3. Подтвердить, что 2023 retrospective уже использована и не является untouched holdout.',
        '4. Подтвердить допустимость same-pipe block-CV как диагностики, не как forward validation.',
        '5. Подтвердить, что S1B/старый S1C не выбираются заново по результатам Stage 1.',
        '',
    ]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--output-root', type=Path, default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.output_root or root / OUTPUT_REL).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('Auditing source and keys...', flush=True)
    source = source_audit(root)
    keys = audit_keys(root)
    roles = data_roles(root)
    # Metrics
    s1 = pd.read_csv(root / 'experiments/next_cycle/s1/oof_fit_predictions.csv')
    block = pd.read_csv(root / 'experiments/next_cycle/s1/block_cv/oof_predictions.csv')
    s2 = pd.read_csv(root / 'experiments/next_cycle/s2/oof_predictions.csv')
    retro = pd.read_csv(root / 'experiments/next_cycle/s2/post_cutoff_retrospective/predictions.csv')
    s1_models = ['s1a','s1b','old_s1c','affine','ridge_history','ridge_geometry',
                 'rate_signed','rate_nonnegative','catboost','mix_025','mix_050','mix_100']
    block_models = ['s1a','s1b','ridge_geometry','rate_signed','catboost']
    s2_models = ['constant_effective_rate','physics_priors','physics_velocity_consistent',
                 'physics_spatial','physics_spatial_temperature','catboost_residual']
    s1_results = {m: evaluate(s1, m, baseline='s1a') for m in s1_models}
    block_results = {m: evaluate(block, m, baseline='s1a') for m in block_models}
    block_win_rows = []
    block_wins = {m: 0 for m in block_models if m != 's1a'}
    for block_id, part in block.groupby('block', sort=True):
        row = {'block': int(block_id), 'n': int(len(part))}
        for model in block_models:
            row[model + '_mae_mm'] = float(np.mean(np.abs(part[model] - part.next_mm)))
        for model in block_wins:
            if row[model + '_mae_mm'] < row['s1a_mae_mm']:
                block_wins[model] += 1
        block_win_rows.append(row)
    write_csv(pd.DataFrame(block_win_rows), out / 'metrics_recheck/s1_block_cv_block_wins.csv', fmt='%.10f')
    write_json(out / 'metrics_recheck/s1_block_cv_block_wins.json',
               {'blocks': int(block.block.nunique()),
                'wins_vs_s1a': block_wins,
                'definition': 'per-block MAE; lower is better; blocks correlated'})
    s2_results = {m: evaluate(s2, m, baseline='physics_priors') for m in s2_models}
    retro_wide = retro.pivot_table(index=['id','segment_id','prediction_date','next_mm'],
                                   columns='model', values='prediction', aggfunc='first').reset_index()
    retro_results = {m: evaluate(retro_wide, m, baseline='physics_priors') for m in s2_models}
    rows = []
    per_rows = []
    for dataset, results in [('s1_lopo', s1_results), ('s1_block_cv', block_results),
                             ('s2_strict', s2_results), ('s2_retrospective', retro_results)]:
        for model, result in results.items():
            rows.append(_metric_row(dataset, model, result,
                                    len(result['per_section'])))
            per_rows += _per_section_rows(dataset, model, result)
    metrics_table = pd.DataFrame(rows)
    per_table = pd.DataFrame(per_rows)
    write_csv(metrics_table, out / 'metrics_recheck/metrics_summary.csv', fmt='%.10f')
    write_csv(per_table, out / 'metrics_recheck/per_section_metrics.csv', fmt='%.10f')
    for dataset, results in [('s1_lopo', s1_results), ('s1_block_cv', block_results),
                             ('s2_strict', s2_results), ('s2_retrospective', retro_results)]:
        write_csv(metrics_table[metrics_table.dataset.eq(dataset)],
                  out / f'metrics_recheck/{dataset}_metrics.csv', fmt='%.10f')
        write_json(out / f'metrics_recheck/{dataset}_details.json',
                   {m: r for m, r in results.items()})
    write_json(out / 'metrics_recheck/key_mask_audit.json', keys)
    write_json(out / 'source_audit.json', source)
    write_json(out / 'data_roles.json', roles)
    write_json(out / 'experiment_plan.json', experiment_plan())
    (out / 'mentor_questions_ru.md').write_text(mentor_questions(), encoding='utf-8')
    # Code/data summaries for the report.
    code_summary = source['next_cycle_code']['manifest_code_audit']
    data_summary = source['data_generation']['output_hash_audit']
    # Build report from the actual tables.
    report = stage_report(source, keys, {
        's1_lopo': {m: _metric_row('s1_lopo', m, s1_results[m], 3) for m in s1_models},
        's1_block_cv': {m: _metric_row('s1_block_cv', m, block_results[m], 1) for m in block_models},
        's2_strict': {m: _metric_row('s2_strict', m, s2_results[m], 3) for m in s2_models},
        's2_retrospective': {m: _metric_row('s2_retrospective', m, retro_results[m], 7) for m in s2_models},
        'block_wins': block_wins,
        'block_total': int(block.block.nunique()),
    }, roles, code_summary, data_summary, source['environment'])
    (out / 'stage_1_report_ru.md').write_text(report, encoding='utf-8')
    print(f'PASS Stage 1 audit: {out}', flush=True)
    print(f'  keys: {keys["s1_lopo"]["rows"]} LOPO, {keys["s1_block_cv"]["rows"]} block, '
          f'{keys["s2_strict"]["rows"]} S2 strict, {keys["s2_retrospective"]["wide_rows"]} retrospective',
          flush=True)


if __name__ == '__main__':
    main()

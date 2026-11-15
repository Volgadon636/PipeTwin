"""Executable S1A / S1B / S1C stages and reproducible artifact writers.

Run examples::

    python -m src.ml.s1a --output-root experiments/s1abc/s1a
    python -m src.ml.s1b --output-root experiments/s1abc/s1b
    python -m src.ml.s1c train --output-root experiments/s1abc/s1c
    python -m src.ml.s1c predict --output-root experiments/s1abc/s1c
    python -m src.ml.s1_all --output-root experiments/s1abc

No stage writes to ``data/``, ``processed/`` or ``reports/``.  S2 is copied
from an explicitly supplied, verified ``result.csv`` by exact keys; it is never
retrained or synthesized.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import __version__ as sklearn_version

from src.config import BASE_DIR
from src.storage import sha256, write_json

from .data import RepositoryData
from .s1_contract import (CONTRACT_VERSION, EXPECTED_S1_SEGMENTS,
                          EXPECTED_S2_SEGMENTS, RIDGE_FEATURES, S1_CASE,
                          S1_CUTOFF, S1_ID, S2_ID, TARGET_COLUMN,
                          build_s1_training_frame, combine_with_s2,
                          load_s2_source, load_v2_s1_source, nominal_frame,
                          production_frame, target_key_grid, verify_s2_unchanged)
from .s1_model import (ARTIFACT_VERSION, DEFAULT_ALPHA, evaluate_models_oof,
                       fit_s1_ridge, load_artifact, macro_metrics,
                       metrics_by_group, pooled_metrics, predict_s1_ridge,
                       save_artifact)

DIAGNOSTIC_COLUMNS = [
    'id', 'segment_id', 'prediction_date', 'case', 'model',
    'thickness_pred', 's1a_prediction', 's1b_prediction',
    'delta_vs_s1a_mm', 's1c_vs_s1a_mm', 's1c_vs_s1b_mm',
    'baseline_source', 's1a_source', 's1b_source',
    'has_own_prior_vtd', 's1a_missing', 's1b_missing',
    'last_vtd_thickness_mm', 'last_vtd_date',
    'prior_vtd_min_thickness_mm', 'prior_vtd_min_date',
    'prior_vtd_campaign_count', 'last_minus_min_mm',
    'age_last_vtd_years', 'age_min_vtd_years',
    'fallback_segment_id', 'fallback_distance_m',
    'fallback_measurement_date', 'fallback_thickness_mm',
    'neighbor_prior_median_mm', 'neighbor_prior_count', 'neighbor_missing',
    'slope_m_per_m', 'abs_slope_m_per_m', 'relative_position', 'local_low_m',
    'h0_mm', 'nominal_thickness_mm',
    'prediction_lower_bound_mm', 'prediction_upper_bound_mm', 'clipped',
    'ridge_applied', 'ridge_delta_mm', 'thickness_pred_unclipped',
    'training_required', 'training_completed', 'prediction_source',
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(['git', *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        commit = run('rev-parse', 'HEAD')
        branch = run('branch', '--show-current')
        status = run('status', '--porcelain')
        tracked = run('status', '--porcelain', '--untracked-files=no')
        return {'commit': commit, 'branch': branch,
                'dirty': bool(status), 'tracked_dirty': bool(tracked),
                'status_porcelain': status.splitlines(),
                'status_porcelain_tracked': tracked.splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {'commit': 'unavailable', 'branch': 'unavailable',
                'dirty': None, 'tracked_dirty': None, 'error': str(error)}


def _versions() -> dict[str, str]:
    versions = {
        'python': platform.python_version(),
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit-learn': sklearn_version,
        'joblib': joblib.__version__,
        'platform': platform.platform(),
        'executable': sys.executable,
    }
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except Exception:  # pragma: no cover - scipy is an sklearn dependency
        pass
    return versions


def _code_hashes(root: Path) -> dict[str, str]:
    names = sorted((root / 'src/ml').glob('s1*.py'))
    names.append(root / 'src/ml/submission.py')
    return {str(path.relative_to(root)): sha256(path) for path in names if path.is_file()}


def _input_hashes(data: RepositoryData, extra_paths: list[Path] | None = None) -> dict[str, str]:
    hashes = dict(data.hashes)
    extra_paths = extra_paths or []
    for path in extra_paths:
        path = Path(path)
        if path.is_file():
            hashes[str(path)] = sha256(path)
    for name in ('configs/preprocessing.json', 'reports/run_manifest.json'):
        path = data.root / name
        if path.is_file():
            hashes[name] = sha256(path)
    return hashes


def _output_hashes(output: Path) -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(output.glob('*'))
            if path.is_file() and path.name != 'run_manifest.json'}


def _rejected_artifacts(data: RepositoryData) -> list[dict[str, Any]]:
    """Record artifacts rejected by manifest verification; never rewrite hashes."""
    name = 'processed/target_segments_grid'
    entry = data.manifest.get('outputs', {}).get(name)
    path = data.root / (name + '.csv')
    if entry is None or not path.is_file():
        return []
    actual = sha256(path)
    if actual == entry.get('csv_sha256'):
        return []
    return [{
        'path': str(path),
        'expected_csv_sha256': entry.get('csv_sha256'),
        'actual_csv_sha256': actual,
        'used': False,
        'reason': ('manifest checksum mismatch even after LF normalisation; '
                   'target keys were derived from verified segments_grid and '
                   'cross-checked against target_features'),
    }]


def _write_manifest(path: Path, root: Path, output: Path, data: RepositoryData,
                    stage: str, started: str, *, training_required: bool,
                    training_completed: bool, extra: dict[str, Any] | None = None,
                    extra_inputs: list[Path] | None = None) -> dict[str, Any]:
    payload = {
        'stage': stage,
        'contract_version': CONTRACT_VERSION,
        'artifact_version': ARTIFACT_VERSION,
        'started_utc': started,
        'finished_utc': _utcnow(),
        'git': _git_state(root),
        'versions': _versions(),
        'root': str(root),
        'output_root': str(output),
        'training_required': bool(training_required),
        'training_completed': bool(training_completed),
        'input_sha256': _input_hashes(data, extra_inputs),
        'rejected_artifacts': _rejected_artifacts(data),
        'code_sha256': _code_hashes(root),
        'outputs_sha256': _output_hashes(output),
    }
    if extra:
        payload.update(extra)
    write_json(path, _json_safe(payload))
    return payload


def _verify_inputs_unchanged(data: RepositoryData) -> None:
    for name, digest in data.hashes.items():
        if sha256(data.root / name) != digest:
            raise ValueError(f'Input artifact changed during the run: {name}')


def _ensure_output(output: Path, root: Path) -> Path:
    output = Path(output).resolve()
    root = Path(root).resolve()
    protected = [root / 'data', root / 'processed', root / 'reports',
                 root / 'src', root / 'configs', root / 'agent']
    if output == root or any(output == p or p in output.parents for p in protected):
        raise ValueError('Use a separate experiment output directory')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_csv(frame: pd.DataFrame, path: Path, float_format: str = '%.10f') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8', lineterminator='\n',
                 float_format=float_format)


def _write_diagnostics(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in DIAGNOSTIC_COLUMNS:
        if column not in out:
            out[column] = np.nan
    _write_csv(out[DIAGNOSTIC_COLUMNS], path, float_format='%.10g')


def _series_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors='coerce').dropna().astype(float)
    if values.empty:
        return {'count': 0, 'mean': None, 'median': None, 'std': None,
                'min': None, 'max': None, 'q05': None, 'q25': None,
                'q75': None, 'q95': None}
    return {
        'count': int(len(values)),
        'mean': float(values.mean()), 'median': float(values.median()),
        'std': float(values.std(ddof=0)), 'min': float(values.min()),
        'max': float(values.max()), 'q05': float(values.quantile(.05)),
        'q25': float(values.quantile(.25)), 'q75': float(values.quantile(.75)),
        'q95': float(values.quantile(.95)),
    }


def _fallback_records(prod: pd.DataFrame) -> list[dict[str, Any]]:
    frame = prod[prod.s1a_missing.fillna(False)].sort_values('segment_id')
    records = []
    for row in frame.itertuples():
        records.append({
            'segment_id': int(row.segment_id),
            's1a_prediction_mm': float(row.s1a_prediction),
            's1b_prediction_mm': float(row.s1b_prediction),
            'fallback_source': str(row.s1a_source),
            'fallback_segment_id': None if pd.isna(row.fallback_segment_id)
            else int(row.fallback_segment_id),
            'fallback_distance_m': None if pd.isna(row.fallback_distance_m)
            else float(row.fallback_distance_m),
            'fallback_measurement_date': (None if pd.isna(row.fallback_measurement_date)
                                          else str(pd.Timestamp(row.fallback_measurement_date).date())),
        })
    return records


def _compare_v2(prod: pd.DataFrame, v2: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = prod[['id', 'segment_id', 's1a_prediction']].merge(
        v2[['id', 'segment_id', 'thickness_pred']], on=['id', 'segment_id'],
        how='outer', validate='one_to_one', indicator=True)
    if not merged._merge.eq('both').all():
        raise ValueError('S1A and v2 key sets differ')
    merged['difference_mm'] = merged.s1a_prediction - merged.thickness_pred
    diff = merged.difference_mm.to_numpy(float)
    summary = {
        'rows': int(len(merged)),
        'max_abs_difference_mm': float(np.max(np.abs(diff))),
        'mean_difference_mm': float(np.mean(diff)),
        'rows_difference_gt_1e_9_mm': int(np.sum(np.abs(diff) > 1e-9)),
        'note': ('v2 S1 was a persistence baseline; agreement verifies the '
                 'last-measured/fallback input, not hidden-test accuracy.'),
    }
    return summary, merged.rename(columns={
        's1a_prediction': 's1a_prediction_mm',
        'thickness_pred': 'v2_prediction_mm'})


def _base_bounds(prod: pd.DataFrame, nominal: pd.DataFrame,
                 s1_upper_column: str) -> pd.DataFrame:
    s1 = prod[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']].copy()
    if s1_upper_column not in s1:
        raise ValueError(f'Missing S1 upper column: {s1_upper_column}')
    s1['upper_mm'] = s1[s1_upper_column].astype(float)
    s2 = nominal[nominal.id.eq(S2_ID)][['id', 'segment_id',
                                        'nominal_thickness_mm']].copy()
    s2['h0_mm'] = s2.nominal_thickness_mm.astype(float)
    s2['upper_mm'] = s2.nominal_thickness_mm.astype(float)
    bounds = pd.concat([s1, s2], ignore_index=True)
    if bounds.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate key in bounds table')
    if (bounds.upper_mm < bounds.h0_mm - 1e-9).any():
        raise ValueError('Upper bound below h0')
    return bounds


def _write_common_result(output: Path, prod: pd.DataFrame, s2: pd.DataFrame,
                         grid: pd.DataFrame, bounds: pd.DataFrame,
                         s1_upper_column: str) -> tuple[dict[str, Any], dict[str, Any]]:
    s1_pred = pd.DataFrame({
        'id': prod.id.astype('int64'),
        'segment_id': prod.segment_id.astype('int64'),
        'thickness_pred': prod.thickness_pred.astype(float),
    })
    combined, validation = combine_with_s2(
        s1_pred, s2, grid, bounds)
    result_path = output / 'result.csv'
    _write_csv(combined, result_path)
    revalidated = pd.read_csv(result_path)
    if list(revalidated.columns) != ['id', 'segment_id', 'thickness_pred']:
        raise ValueError('Written result.csv has wrong columns')
    validation_again = combine_with_s2(
        revalidated[revalidated.id.eq(S1_ID)], s2, grid, bounds)[1]
    s2_check = verify_s2_unchanged(result_path, s2)
    if not s2_check['exact_numeric_match']:
        raise ValueError('S2 predictions changed in the combined result')
    validation.update({'revalidated_after_write': validation_again,
                       's1_upper_column': s1_upper_column})
    return validation, s2_check


def _s2_metadata(source: Path, s2: pd.DataFrame) -> dict[str, Any]:
    return {'source_path': str(source), 'source_sha256': sha256(source),
            'rows': int(len(s2)),
            'keys': [int(S2_ID), int(EXPECTED_S2_SEGMENTS)],
            'unchanged': True,
            'policy': 'copied by exact keys from the explicitly selected result.csv; never retrained'}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return 'n/a'
        return f'{float(value):.{digits}f}'
    return str(value)


# --------------------------------------------------------------------------- #
# S1A
# --------------------------------------------------------------------------- #
def run_s1a(root: Path = BASE_DIR, output_root: Path | None = None,
            s2_source: Path | None = None,
            v2_source: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = _ensure_output(Path(output_root or root / 'experiments/s1abc/s1a'), root)
    started = _utcnow()
    s2_source = Path(s2_source or root / 'experiments/ml_v2/result.csv').resolve()
    v2_source = Path(v2_source or root / 'experiments/ml_v2/result.csv').resolve()
    data = RepositoryData(root)
    grid = target_key_grid(data)
    nominal = nominal_frame(data)
    prod, expected = production_frame(data)
    if len(prod) != EXPECTED_S1_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_S1_SEGMENTS} S1 rows')
    v2 = load_v2_s1_source(v2_source, grid)
    comparison, comparison_frame = _compare_v2(prod, v2)
    _write_csv(comparison_frame, output / 's1a_vs_v2.csv', float_format='%.10f')

    prod = prod.copy()
    prod['thickness_pred'] = prod.s1a_prediction.astype(float)
    prod['model'] = 'S1A'
    prod['training_required'] = False
    prod['training_completed'] = False
    prod['prediction_lower_bound_mm'] = 0.0
    prod['prediction_upper_bound_mm'] = prod.h0_mm.astype(float)
    prod['clipped'] = False
    prod['prediction_source'] = prod.s1a_source.astype('string')
    _write_csv(prod[['id', 'segment_id', 'thickness_pred']],
               output / 'predictions_s1.csv')
    _write_diagnostics(prod, output / 'prediction_diagnostics.csv')

    s2 = load_s2_source(s2_source, grid)
    bounds = _base_bounds(prod, nominal, 'h0_mm')
    validation, s2_check = _write_common_result(
        output, prod, s2, grid, bounds, 'h0_mm')
    summary = {
        'stage': 'S1A',
        'contract_version': CONTRACT_VERSION,
        'training_required': False,
        'training_completed': False,
        'rule': 'last eligible own prior VTD thickness; documented nearest-prior fallback otherwise',
        'rows': int(len(prod)),
        'own_prior_rows': int(prod.has_own_prior_vtd.sum()),
        'fallback_rows': int((~prod.has_own_prior_vtd).sum()),
        'fallback_segments': _fallback_records(prod),
        'thickness_stats_mm': _series_stats(prod.thickness_pred),
        'v2_comparison': comparison,
        's2': {**_s2_metadata(s2_source, s2), 'written_vs_source': s2_check},
        'validation': validation,
        'bounds_policy': {'lower_mm': 0.0, 'upper_column': 'h0_mm',
                          'note': 'S1A is persistence and cannot exceed h0'},
        'input_note': ('All features use eligible VTD rows strictly before '
                       '2023-12-01; no S1 target label is read.'),
    }
    write_json(output / 'summary.json', _json_safe(summary))
    _write_manifest(output / 'run_manifest.json', root, output, data, 'S1A', started,
                    training_required=False, training_completed=False,
                    extra={'summary': summary,
                           'reproduce_command': (
                               'python -m src.ml.s1a --root . --output-root experiments/s1abc/s1a')},
                    extra_inputs=[v2_source, s2_source])
    _write_report_s1a(output, summary)
    _verify_inputs_unchanged(data)
    print(f'PASS S1A: {len(prod)} S1 predictions, result.csv 2000 rows', flush=True)
    return summary


def _write_report_s1a(output: Path, summary: dict[str, Any]) -> None:
    fb = summary['fallback_segments']
    fb_text = '\n'.join(
        f"- segment `{r['segment_id']}`: `{r['s1a_prediction_mm']:.4f}` мм, "
        f"источник `{r['fallback_source']}`, сосед `{r['fallback_segment_id']}`, "
        f"расстояние `{r['fallback_distance_m']:.1f}` м, дата `{r['fallback_measurement_date']}`"
        for r in fb)
    lines = [
        '# S1A: persistence по последнему прошлому ВТД',
        '',
        f"Контракт `{CONTRACT_VERSION}`. Ветка/commit и фактические версии — в "
        '`run_manifest.json`.',
        '',
        '## Что рассчитано по правилу',
        '',
        '`prediction_S1A = last eligible own prior VTD thickness` для каждого '
        '10-метрового сегмента трубы `1250022871` на `2023-12-01`. Это не '
        'обученная модель: `training_required=false`, `training_completed=false`.',
        '',
        f"- Всего ключей: **{summary['rows']}**",
        f"- Собственный прошлый ВТД: **{summary['own_prior_rows']}**",
        f"- Fallback без собственного замера: **{summary['fallback_rows']}**",
        f"- Среднее/медиана прогноза: **{_fmt(summary['thickness_stats_mm']['mean'])}** / "
        f"**{_fmt(summary['thickness_stats_mm']['median'])}** мм",
        '',
        'Fallback — это последний прошлый замер ближайшего измеренного сегмента '
        'той же трубы (при точном равенстве расстояния выбирается меньший '
        '`segment_id`). Подставленное значение не считается собственным измерением.',
        '',
        fb_text if fb_text else '- fallback отсутствует',
        '',
        '## Сверка с прежним S1 v2',
        '',
        f"- Строк: **{summary['v2_comparison']['rows']}**",
        f"- Максимальное |S1A − v2|: **{summary['v2_comparison']['max_abs_difference_mm']:.3e}** мм",
        f"- Средняя разница: **{summary['v2_comparison']['mean_difference_mm']:.3e}** мм",
        f"- Строк с |разницей| > 1e-9 мм: **{summary['v2_comparison']['rows_difference_gt_1e_9_mm']}**",
        '',
        'Совпадение подтверждает, что старый S1 v2 действительно был '
        'persistence-baseline на тех же входах. Это не проверка скрытого качества.',
        '',
        '## S2 и общий result.csv',
        '',
        f"S2 (`1250009583`, 1475 строк) скопирован без изменений из "
        f"`{summary['s2']['source_path']}`; max |diff| после записи = "
        f"**{summary['s2']['written_vs_source']['max_abs_difference']:.3e}** мм. "
        'S2 не переобучался и не менялся.',
        '',
        '## Ограничения',
        '',
        '- Persistence не учитывает коррозию между 2022 и 2023; это baseline, а не прогноз скорости.',
        '- 4 сегмента используют подстановку соседа; их значения не являются собственным замером.',
        '- Скрытые ответы и измерения на/после `2023-12-01` не использовались.',
        '',
        '## Повтор',
        '',
        '```bash',
        'python -m src.ml.s1a --root . --output-root experiments/s1abc/s1a',
        '```',
        '',
    ]
    (output / 'report_ru.md').write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------- #
# S1B
# --------------------------------------------------------------------------- #
def _historical_campaign_check(vtd: pd.DataFrame, cutoff: pd.Timestamp = S1_CUTOFF) -> dict[str, Any]:
    permitted = vtd[vtd.date.lt(cutoff) & vtd.target_eligible.fillna(False)]
    counts = permitted.groupby('id').date.nunique().sort_index()
    max_count = int(counts.max()) if len(counts) else 0
    pairs = {}
    for pid, part in permitted.groupby('id'):
        dates = sorted(part.date.unique())
        if len(dates) >= 2:
            a = part[part.date.eq(dates[0])]
            b = part[part.date.eq(dates[1])]
            pairs[int(pid)] = int(len(set(a.segment_id) & set(b.segment_id)))
    return {
        'cutoff': str(pd.Timestamp(cutoff).date()),
        'max_prior_campaigns_per_pipe_before_cutoff': max_count,
        'campaign_count_by_pipe': {int(k): int(v) for k, v in counts.items()},
        'two_campaign_pair_counts': pairs,
        'two_prior_campaign_test_available': bool(max_count >= 3),
        'note': ('Only campaigns strictly before the cutoff are counted. '
                 'A two-prior -> third-answer test needs at least three '
                 'eligible campaigns for the same pipe; it is not available.'),
    }


def run_s1b(root: Path = BASE_DIR, output_root: Path | None = None,
            s2_source: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = _ensure_output(Path(output_root or root / 'experiments/s1abc/s1b'), root)
    started = _utcnow()
    s2_source = Path(s2_source or root / 'experiments/ml_v2/result.csv').resolve()
    data = RepositoryData(root)
    grid = target_key_grid(data)
    nominal = nominal_frame(data)
    prod, expected = production_frame(data)
    vtd = data.read('processed/vtd_by_segment')
    prod = prod.copy()
    prod['delta_vs_s1a_mm'] = prod.s1b_prediction.astype(float) - prod.s1a_prediction.astype(float)
    prod['thickness_pred'] = prod.s1b_prediction.astype(float)
    prod['model'] = 'S1B'
    prod['training_required'] = False
    prod['training_completed'] = False
    prod['prediction_lower_bound_mm'] = 0.0
    prod['prediction_upper_bound_mm'] = prod.h0_mm.astype(float)
    prod['clipped'] = False
    prod['prediction_source'] = prod.s1b_source.astype('string')
    changed = prod.delta_vs_s1a_mm.abs().gt(1e-12)
    if (prod.loc[prod.has_own_prior_vtd, 's1b_prediction']
            > prod.loc[prod.has_own_prior_vtd, 's1a_prediction'] + 1e-9).any():
        raise ValueError('S1B exceeds S1A on an own-prior segment')
    delta_stats = _series_stats(prod.delta_vs_s1a_mm)
    examples = prod.sort_values('delta_vs_s1a_mm').head(5)
    example_records = []
    for row in examples.itertuples():
        example_records.append({
            'segment_id': int(row.segment_id),
            's1a_mm': float(row.s1a_prediction),
            's1b_mm': float(row.s1b_prediction),
            'delta_mm': float(row.delta_vs_s1a_mm),
            'last_date': None if pd.isna(row.last_vtd_date) else str(pd.Timestamp(row.last_vtd_date).date()),
            'min_date': None if pd.isna(row.prior_vtd_min_date) else str(pd.Timestamp(row.prior_vtd_min_date).date()),
            'min_source': 'own_prior_vtd_min' if row.has_own_prior_vtd else str(row.s1b_source),
            'fallback_segment_id': None if pd.isna(row.fallback_segment_id) else int(row.fallback_segment_id),
        })
    comparison = pd.DataFrame({
        'id': prod.id, 'segment_id': prod.segment_id,
        's1a_prediction_mm': prod.s1a_prediction,
        's1b_prediction_mm': prod.s1b_prediction,
        'delta_mm': prod.delta_vs_s1a_mm,
        'changed': changed,
        'last_vtd_date': prod.last_vtd_date,
        'prior_vtd_min_date': prod.prior_vtd_min_date,
        'has_own_prior_vtd': prod.has_own_prior_vtd,
        's1b_source': prod.s1b_source,
    })
    _write_csv(comparison, output / 's1b_vs_s1a.csv', float_format='%.10f')
    _write_csv(prod[['id', 'segment_id', 'thickness_pred']],
               output / 'predictions_s1.csv')
    _write_diagnostics(prod, output / 'prediction_diagnostics.csv')

    s2 = load_s2_source(s2_source, grid)
    bounds = _base_bounds(prod, nominal, 'h0_mm')
    validation, s2_check = _write_common_result(
        output, prod, s2, grid, bounds, 'h0_mm')
    historical = _historical_campaign_check(vtd)
    summary = {
        'stage': 'S1B',
        'contract_version': CONTRACT_VERSION,
        'training_required': False,
        'training_completed': False,
        'rule': ('minimum eligible own prior VTD thickness; earliest source date on '
                 'exact minimum tie; S1A fallback reused only when no own prior'),
        'rows': int(len(prod)),
        'own_prior_rows': int(prod.has_own_prior_vtd.sum()),
        'fallback_rows': int((~prod.has_own_prior_vtd).sum()),
        'fallback_segments': _fallback_records(prod),
        'changed_segments': int(changed.sum()),
        'unchanged_segments': int((~changed).sum()),
        'delta_stats_mm': delta_stats,
        'largest_decrease_mm': float(prod.delta_vs_s1a_mm.min()),
        'largest_increase_mm': float(prod.delta_vs_s1a_mm.max()),
        'examples': example_records,
        's1b_le_s1a_on_own_prior': True,
        'tie_policy': 'earliest source date on exact minimum tie',
        'historical_single_prior_pairs': historical,
        's2': {**_s2_metadata(s2_source, s2), 'written_vs_source': s2_check},
        'validation': validation,
        'bounds_policy': {'lower_mm': 0.0, 'upper_column': 'h0_mm',
                          'note': 'S1B is a past minimum and cannot exceed h0'},
        'quality_status': ('not_verified_in_two_prior_campaign_regime; only one '
                           'prior campaign exists for every pre-cutoff S1 label'),
        'input_note': ('All features use eligible VTD rows strictly before '
                       '2023-12-01; no S1 target label is read.'),
    }
    write_json(output / 'summary.json', _json_safe(summary))
    _write_manifest(output / 'run_manifest.json', root, output, data, 'S1B', started,
                    training_required=False, training_completed=False,
                    extra={'summary': summary,
                           'reproduce_command': (
                               'python -m src.ml.s1b --root . --output-root experiments/s1abc/s1b')},
                    extra_inputs=[s2_source])
    _write_report_s1b(output, summary)
    _verify_inputs_unchanged(data)
    print(f'PASS S1B: {len(prod)} S1 predictions, changed {summary["changed_segments"]}', flush=True)
    return summary


def _write_report_s1b(output: Path, summary: dict[str, Any]) -> None:
    examples = '\n'.join(
        f"- segment `{r['segment_id']}`: S1A `{r['s1a_mm']:.4f}` → S1B "
        f"`{r['s1b_mm']:.4f}` (Δ `{r['delta_mm']:+.4f}`); last `{r['last_date']}`, "
        f"min `{r['min_date']}` (`{r['min_source']}`)"
        for r in summary['examples'])
    hist = summary['historical_single_prior_pairs']
    lines = [
        '# S1B: минимум всех прошлых ВТД',
        '',
        f"Контракт `{CONTRACT_VERSION}`. Ветка/commit и версии — в `run_manifest.json`.",
        '',
        '## Что рассчитано по правилу',
        '',
        '`prediction_S1B(k,T) = min(eligible measured thickness(k,t) for t < T)` '
        'по всем прошлым кампаниям ВТД. При точном равенстве минимума выбирается '
        'самая ранняя дата источника. Если у сегмента нет собственного прошлого '
        'замера, используется тот же fallback, что в S1A, без дополнительного '
        'изменения правила. Это не обученная модель: `training_required=false`.',
        '',
        f"- Всего ключей: **{summary['rows']}**",
        f"- Собственный прошлый ВТД: **{summary['own_prior_rows']}**",
        f"- Fallback: **{summary['fallback_rows']}**",
        f"- Изменилось относительно S1A: **{summary['changed_segments']}**",
        f"- Средняя/медиана Δ(S1B−S1A): **{_fmt(summary['delta_stats_mm']['mean'], 6)}** / "
        f"**{_fmt(summary['delta_stats_mm']['median'], 6)}** мм",
        f"- Квантили Δ 5/25/75/95%: **{_fmt(summary['delta_stats_mm']['q05'], 6)} / "
        f"{_fmt(summary['delta_stats_mm']['q25'], 6)} / {_fmt(summary['delta_stats_mm']['q75'], 6)} / "
        f"{_fmt(summary['delta_stats_mm']['q95'], 6)}** мм",
        f"- Максимальное изменение: **{_fmt(summary['largest_decrease_mm'], 6)}** мм "
        f"(рост неотрицателен: {_fmt(summary['largest_increase_mm'], 6)})",
        '',
        'Примеры наибольшего снижения:',
        '',
        examples if examples else '- нет изменившихся сегментов',
        '',
        'На сегментах с собственным прошлым замером проверено `S1B ≤ S1A`; '
        'для fallback значения совпадают по построению.',
        '',
        '## Критическое ограничение: две прошлые кампании не проверены',
        '',
        f"Максимум прошлых кампаний на трубу строго до `2023-12-01`: "
        f"**{hist['max_prior_campaigns_per_pipe_before_cutoff']}**. "
        f"Тест «две прошлые кампании → третьему измеренному ответу» доступен: "
        f"**{hist['two_prior_campaign_test_available']}**.",
        '',
        'Для исторического перехода 2018→2022 у каждого сегмента ровно одна '
        'прошлая кампания, поэтому S1A и S1B там совпадают. Это не доказывает '
        'ни пользу, ни бесполезность правила двух прошлых ВТД для прогноза 2023. '
        'Production S1B для 321 сегмента использует две кампании (2018 и 2022), '
        'но качество этого режима до cutoff не проверено. Функциональная '
        'реализация и production-прогноз завершены; статус качества — '
        '`not_verified_in_two_prior_campaign_regime`.',
        '',
        f"Пар 2018→2022 с обоими замерами: **{sum(hist['two_campaign_pair_counts'].values())}**. "
        'Это тот же набор, на котором S1A и S1B обязаны совпасть.',
        '',
        '## S2 и result.csv',
        '',
        f"S2 скопирован без изменений из `{summary['s2']['source_path']}`; "
        f"max |diff| = **{summary['s2']['written_vs_source']['max_abs_difference']:.3e}** мм.",
        '',
        '## Повтор',
        '',
        '```bash',
        'python -m src.ml.s1b --root . --output-root experiments/s1abc/s1b',
        '```',
        '',
    ]
    (output / 'report_ru.md').write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------- #
# S1C training
# --------------------------------------------------------------------------- #
def _lopo_fit(training: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Leave-one-simple-section-out fit; held-out labels never enter a fold fit."""
    fit_rows = training[training.fit_eligible.astype(bool)].copy()
    ids = sorted(int(v) for v in fit_rows.id.unique())
    if len(ids) < 2:
        raise ValueError('At least two simple sections are needed for leave-one-pipe-out')
    out = fit_rows.copy()
    out['prediction_s1a'] = out.s1a_prediction.astype(float)
    out['prediction_s1b'] = out.s1b_prediction.astype(float)
    for column in ('prediction_s1c', 'prediction_s1a_control', 'fold'):
        out[column] = np.nan
    out['clipped_s1c'] = False
    out['ridge_applied_s1c'] = False
    out['fold'] = pd.Series(pd.NA, index=out.index, dtype='object')
    fold_records = []
    for held in ids:
        train = fit_rows[~fit_rows.id.eq(held)].copy()
        test = fit_rows[fit_rows.id.eq(held)].copy()
        if train.empty or test.empty:
            raise ValueError(f'Empty LOPO fold for held-out id {held}')
        model = fit_s1_ridge(train, alpha=alpha, target='target_delta_mm',
                             baseline_column='s1b_prediction',
                             model_name='S1C_ridge')
        prediction = predict_s1_ridge(model, test)
        control = fit_s1_ridge(train, alpha=alpha,
                               target='target_delta_from_s1a_mm',
                               baseline_column='s1a_prediction',
                               model_name='S1A_ridge_control')
        control_prediction = predict_s1_ridge(
            control, test, baseline_column='s1a_prediction')
        out.loc[test.index, 'prediction_s1c'] = prediction.thickness_pred.to_numpy(float)
        out.loc[test.index, 'prediction_s1a_control'] = control_prediction.thickness_pred.to_numpy(float)
        out.loc[test.index, 'clipped_s1c'] = prediction.clipped.to_numpy(bool)
        out.loc[test.index, 'ridge_applied_s1c'] = prediction.ridge_applied.to_numpy(bool)
        out.loc[test.index, 'fold'] = held
        fold_records.append({
            'heldout_simple_section_id': held,
            'train_simple_section_ids': sorted(int(v) for v in train.id.unique()),
            'train_rows': int(len(train)),
            'test_rows': int(len(test)),
            'train_date_max': str(pd.Timestamp(train.prediction_date.max()).date()),
            'validation_date': str(pd.Timestamp(test.prediction_date.max()).date()),
            'strict_forward_time': False,
            'diagnostic_type': 'spatial_transfer_same_campaign',
            'heldout_labels_used_in_fit': False,
            'used_features': list(model['metadata']['used_features']),
            'dropped_columns': dict(model['metadata']['dropped_columns']),
            'coefficients': list(model['metadata']['coefficients']),
            'intercept': float(model['metadata']['intercept']),
        })
    if out[['prediction_s1c', 'prediction_s1a_control']].isna().any().any():
        raise ValueError('Some LOPO rows were not predicted')
    if out.fold.isna().any():
        raise ValueError('Some LOPO rows have no fold assignment')
    return out, fold_records


def run_s1c_train(root: Path = BASE_DIR, output_root: Path | None = None,
                  alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    root = Path(root).resolve()
    output = _ensure_output(Path(output_root or root / 'experiments/s1abc/s1c'), root)
    started = _utcnow()
    data = RepositoryData(root)
    training = build_s1_training_frame(data)
    fit_rows = training[training.fit_eligible.astype(bool)].copy()
    if fit_rows.empty:
        raise RuntimeError('S1C training refused: no rows with own prior baseline')
    if fit_rows.prediction_date.nunique() != 1:
        raise ValueError('This first S1C experiment expects one historical campaign')
    if fit_rows.id.nunique() < 2:
        raise ValueError('S1C training needs at least two simple sections')
    if fit_rows.prediction_date.max() >= S1_CUTOFF:
        raise ValueError('Training label on/after the production cutoff')
    if training.prior_vtd_campaign_count.max() > 1:
        raise ValueError('Unexpected two-campaign historical row in the first S1C experiment')
    print(f'S1C fit: {len(fit_rows)} rows, ids {sorted(fit_rows.id.unique())}', flush=True)
    artifact = fit_s1_ridge(fit_rows, alpha=float(alpha),
                            target='target_delta_mm',
                            baseline_column='s1b_prediction',
                            model_name='S1C_ridge')
    artifact_path = output / 'model_artifact.joblib'
    save_artifact(artifact, artifact_path)
    artifact_hash = sha256(artifact_path)
    reloaded = load_artifact(artifact_path)
    if reloaded['metadata'] != artifact['metadata']:
        raise ValueError('Saved and reloaded artifact metadata differ')
    metadata = artifact['metadata']

    oof_fit, fold_records = _lopo_fit(training, alpha=float(alpha))
    # The fallback rows are displayed, but they are not Ridge predictions.
    fallback = training[~training.fit_eligible.astype(bool)].copy()
    fallback['fold'] = 'not_applied_fallback'
    fallback['prediction_s1c'] = fallback.s1b_prediction
    fallback['prediction_s1a_control'] = fallback.s1a_prediction
    fallback['clipped_s1c'] = False
    fallback['ridge_applied_s1c'] = False
    oof_columns = ['id', 'segment_id', 'prediction_date', TARGET_COLUMN,
                   's1a_prediction', 's1b_prediction', 'prediction_s1c',
                   'prediction_s1a_control', 'target_delta_mm',
                   'target_delta_from_s1a_mm', 'fit_eligible', 'fold',
                   'clipped_s1c', 'ridge_applied_s1c']
    oof_all = pd.concat([
        oof_fit.assign(prediction_source='ridge_oof')[oof_columns + ['prediction_source']],
        fallback.assign(prediction_source='s1b_fallback')[oof_columns + ['prediction_source']],
    ], ignore_index=True).sort_values(['id', 'segment_id']).reset_index(drop=True)
    oof_all = oof_all.rename(columns={
        's1a_prediction': 'prediction_s1a',
        's1b_prediction': 'prediction_s1b',
        TARGET_COLUMN: 'measured_thickness_mm',
    })
    _write_csv(oof_all, output / 'oof_predictions.csv', float_format='%.10f')
    _write_csv(oof_fit, output / 'oof_fit_rows.csv', float_format='%.10f')
    _write_csv(fallback, output / 'fallback_predictions.csv', float_format='%.10f')

    fold_frame = pd.DataFrame(fold_records)
    _write_csv(fold_frame, output / 'fold_assignments.csv', float_format='%.10f')
    per_fold, pooled = evaluate_models_oof(oof_fit)
    _write_csv(per_fold, output / 'metrics_per_fold.csv', float_format='%.10f')
    write_json(output / 'metrics_pooled.json', _json_safe(pooled))

    fallback_metrics = {
        'rows': int(len(fallback)),
        's1a': pooled_metrics(
            fallback.rename(columns={'s1a_prediction': 'prediction_s1a'}),
            'prediction_s1a'),
        's1b': pooled_metrics(
            fallback.rename(columns={'s1b_prediction': 'prediction_s1b'}),
            'prediction_s1b'),
        'note': 'Fallback rows have no own prior baseline and are not Ridge-fit or Ridge-predicted.',
    }
    write_json(output / 'fallback_metrics.json', _json_safe(fallback_metrics))

    by_id = []
    for pid, part in training.groupby('id', sort=True):
        fit_part = part[part.fit_eligible.astype(bool)]
        by_id.append({
            'id': int(pid),
            'total_rows': int(len(part)),
            'fit_rows': int(len(fit_part)),
            'fallback_rows': int((~part.fit_eligible.astype(bool)).sum()),
            'fit_coverage_fraction': float(len(fit_part) / len(part)),
            'campaign_dates': sorted(str(pd.Timestamp(v).date())
                                     for v in part.prediction_date.unique()),
        })
    production, _ = production_frame(data)
    coverage = {
        'historical_s1_rows': int(len(training)),
        'historical_fit_rows': int(len(fit_rows)),
        'historical_fallback_rows': int(len(fallback)),
        'historical_fit_coverage_fraction': float(len(fit_rows) / len(training)),
        'by_simple_section': by_id,
        'production_s1_rows': int(len(production)),
        'production_ridge_applicable_rows': int(production.has_own_prior_vtd.sum()),
        'production_fallback_rows': int((~production.has_own_prior_vtd).sum()),
        'production_ridge_coverage_fraction': float(production.has_own_prior_vtd.mean()),
        'policy': ('Ridge is applied only to rows with an own measured prior baseline; '
                   'fallback rows keep S1B and are shown separately.'),
    }
    write_json(output / 'coverage.json', _json_safe(coverage))

    training_table = training.copy()
    _write_csv(training_table, output / 'training_table.csv', float_format='%.10f')
    _write_csv(fit_rows, output / 'fit_table.csv', float_format='%.10f')
    exclusion_counts = {
        'historical_s1_rows': int(len(training)),
        'fit_rows': int(len(fit_rows)),
        'excluded_rows_no_own_prior': int((~training.fit_eligible.astype(bool)).sum()),
        'excluded_by_simple_section': {
            int(pid): int((~part.fit_eligible.astype(bool)).sum())
            for pid, part in training.groupby('id', sort=True)},
        'dropped_columns': metadata['dropped_columns'],
        'empty_columns': metadata['preprocessing']['empty_columns'],
        'constant_columns': metadata['preprocessing']['constant_columns'],
        'note': ('Excluded rows are not deleted from training_table.csv; they '
                 'remain visible with fit_eligible=false.'),
    }
    write_json(output / 'exclusion_counts.json', _json_safe(exclusion_counts))

    feature_contract = {
        'contract_version': CONTRACT_VERSION,
        'artifact_version': ARTIFACT_VERSION,
        'target': 'target_delta_mm = measured_thickness - S1B baseline',
        'baseline_column': 's1b_prediction',
        'declared_features': list(RIDGE_FEATURES),
        'feature_order': list(metadata['feature_order']),
        'used_features': list(metadata['used_features']),
        'dropped_columns': dict(metadata['dropped_columns']),
        'preprocessing': metadata['preprocessing'],
        'alpha': float(metadata['alpha']),
        'fit_intercept': True,
        'solver': metadata['solver'],
        'weighting': metadata['weighting'],
        'weighting_detail': metadata['weighting_detail'],
        'bounds_policy': metadata['bounds_policy'],
        'fallback_policy': metadata['fallback_policy'],
        'tie_policy': 'earliest source date on exact minimum tie',
        'training_rows': int(metadata['training_rows']),
        'training_simple_sections': metadata['training_simple_sections'],
        'training_dates': metadata['training_dates'],
        'training_parents': metadata['training_parents'],
        'target_delta_mean_mm': metadata['target_delta_mean_mm'],
        'target_delta_std_mm': metadata['target_delta_std_mm'],
        'observed_increase_rows': int((fit_rows.target_delta_mm > 1e-12).sum()),
        'observed_decrease_rows': int((fit_rows.target_delta_mm < -1e-12).sum()),
        'observed_zero_rows': int((fit_rows.target_delta_mm.abs() <= 1e-12).sum()),
        'interpretation': metadata['interpretation'],
        'unidentifiable_in_training': [
            'last_minus_min is constant because only one prior campaign exists',
            'prior_vtd_campaign_count is constant because only one prior campaign exists',
            'S1B baseline and last VTD thickness are identical with one prior campaign',
            'age_last and age_min coincide with one prior campaign',
        ],
    }
    write_json(output / 'feature_contract.json', _json_safe(feature_contract))

    # Prove that the artifact and a manual coefficient application agree, and
    # that a fresh load produces the same predictions.  This is not a hidden-test
    # score; it is a serialization contract check.
    manual_design = artifact['preprocessor'].transform(fit_rows)
    manual = np.clip(
        fit_rows.s1b_prediction.to_numpy(float)
        + artifact['ridge'].predict(manual_design),
        0.0, fit_rows.nominal_thickness_mm.to_numpy(float))
    artifact_pred = predict_s1_ridge(artifact, fit_rows)
    reloaded_pred = predict_s1_ridge(reloaded, fit_rows)
    equivalence = {
        'artifact_vs_manual_max_abs_difference_mm': float(np.max(
            np.abs(artifact_pred.thickness_pred.to_numpy(float) - manual))),
        'artifact_vs_reloaded_max_abs_difference_mm': float(np.max(
            np.abs(artifact_pred.thickness_pred.to_numpy(float)
                   - reloaded_pred.thickness_pred.to_numpy(float)))),
        'rows': int(len(fit_rows)),
        'note': 'Numerical serialization/equivalence check, not a validation score.',
    }
    write_json(output / 'model_equivalence.json', _json_safe(equivalence))

    summary = {
        'stage': 'S1C_train',
        'contract_version': CONTRACT_VERSION,
        'artifact_version': ARTIFACT_VERSION,
        'training_required': True,
        'training_completed': True,
        'training_rows': int(len(fit_rows)),
        'training_simple_sections': sorted(int(v) for v in fit_rows.id.unique()),
        'training_dates': sorted(str(pd.Timestamp(v).date())
                                 for v in fit_rows.prediction_date.unique()),
        'training_date_max': str(pd.Timestamp(fit_rows.prediction_date.max()).date()),
        'production_cutoff': str(S1_CUTOFF.date()),
        'observed_increase_rows': int((fit_rows.target_delta_mm > 1e-12).sum()),
        'observed_decrease_rows': int((fit_rows.target_delta_mm < -1e-12).sum()),
        'observed_zero_rows': int((fit_rows.target_delta_mm.abs() <= 1e-12).sum()),
        'feature_count': int(len(metadata['used_features'])),
        'declared_feature_count': int(len(RIDGE_FEATURES)),
        'dropped_columns': metadata['dropped_columns'],
        'alpha': float(metadata['alpha']),
        'intercept': float(metadata['intercept']),
        'coefficients': list(metadata['coefficients']),
        'weighting': metadata['weighting'],
        'artifact_path': str(artifact_path),
        'artifact_sha256': artifact_hash,
        'coverage': coverage,
        'lopo': {
            'folds': fold_records,
            'strict_forward_time': False,
            'diagnostic_type': 'leave_one_simple_pipe_out_same_campaign',
            'note': ('All train and validation rows are from the same 2022-07-06 '
                     'campaign on one parent pipeline. This is spatial transfer, '
                     'not forward-time validation.'),
        },
        'metrics': pooled,
        'fallback_metrics': fallback_metrics,
        'equivalence': equivalence,
        'two_prior_campaign_test': {
            'available_before_cutoff': False,
            'reason': ('Every eligible pre-cutoff S1 label has exactly one prior '
                       'campaign; 2023-12-01 labels are on the cutoff and excluded.'),
        },
        'interpretation': metadata['interpretation'],
    }
    write_json(output / 'summary.json', _json_safe(summary))
    write_json(output / 'summary_train.json', _json_safe(summary))
    _write_manifest(output / 'run_manifest_train.json', root, output, data,
                    'S1C_train', started, training_required=True,
                    training_completed=True,
                    extra={'training': summary,
                           'reproduce_command': (
                               'python -m src.ml.s1c train --root . '
                               '--output-root experiments/s1abc/s1c')},
                    extra_inputs=[])
    _write_report_s1c_train(output, summary)
    _verify_inputs_unchanged(data)
    print(f'PASS S1C train: {len(fit_rows)} rows, artifact {artifact_path}', flush=True)
    return summary


def _metric_table_lines(metrics: dict[str, Any]) -> list[str]:
    lines = [
        '| Модель | n | MAE, мм | RMSE, мм | bias, мм | Pearson r | Pearson r² | R² |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('S1A', 'S1B', 'S1C', 'S1A_ridge_control'):
        record = metrics[name]['pooled']
        lines.append(
            f"| {name} | {record['n']} | {_fmt(record['mae_mm'])} | "
            f"{_fmt(record['rmse_mm'])} | {_fmt(record['bias_mm'])} | "
            f"{_fmt(record['pearson_r'])} | {_fmt(record['pearson_squared'])} | "
            f"{_fmt(record['r2'])} |")
    return lines


def _per_section_lines(metrics: dict[str, Any]) -> list[str]:
    lines = [
        '| Модель | Простой участок | n | MAE, мм | bias, мм | Pearson r² | R² |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('S1A', 'S1B', 'S1C', 'S1A_ridge_control'):
        for record in metrics[name]['per_simple_section']:
            lines.append(
                f"| {name} | {record['group']} | {record['n']} | "
                f"{_fmt(record['mae_mm'])} | {_fmt(record['bias_mm'])} | "
                f"{_fmt(record['pearson_squared'])} | {_fmt(record['r2'])} |")
    return lines


def _write_report_s1c_train(output: Path, summary: dict[str, Any]) -> None:
    metrics = summary['metrics']
    dropped = summary['dropped_columns']
    dropped_text = ', '.join(f'`{k}` ({v})' for k, v in dropped.items()) or 'нет'
    rows = []
    for name in ('S1A', 'S1B', 'S1C', 'S1A_ridge_control'):
        record = metrics[name]['pooled']
        rows.append((name, record))
    best_mae = min(rows, key=lambda x: x[1]['mae_mm'])
    lines = [
        '# S1C: обученная Ridge-поправка к S1B',
        '',
        f"Контракт `{CONTRACT_VERSION}`, artifact `{ARTIFACT_VERSION}`. "
        'Ветка/commit и версии — в `run_manifest_train.json`.',
        '',
        '## Что именно обучено',
        '',
        'Обучена **одна Ridge-регрессия** с фиксированными `alpha=10.0`, '
        '`fit_intercept=True`, `solver=cholesky` и равным суммарным весом '
        'каждого простого участка. Цель — `target_delta_mm = measured_thickness − S1B`; '
        'прогноз `clip(S1B + ridge_delta, 0, nominal_thickness_mm)`. Это не '
        'физическая скорость коррозии; поправка может иметь оба знака и не '
        'ограничена сверху значением S1B.',
        '',
        f"- Строк в fit: **{summary['training_rows']}**",
        f"- Простые участки: **{summary['training_simple_sections']}**",
        f"- Дата обучения: **{', '.join(summary['training_dates'])}**",
        f"- Production cutoff: **{summary['production_cutoff']}**; все метки строго раньше",
        f"- Использовано признаков: **{summary['feature_count']}** из "
        f"{summary['declared_feature_count']} объявленных",
        f"- Исключённые константные/пустые признаки: {dropped_text}",
        f"- Веса: `{summary['weighting']}`",
        f"- Наблюдаемые изменения 2022−2018: рост **{summary.get('observed_increase_rows')}**, "
        f"снижение **{summary.get('observed_decrease_rows')}**, ноль **{summary.get('observed_zero_rows')}**",
        '',
        '`last_minus_min` и `prior_vtd_campaign_count` на всём допустимом обучении '
        'константны: есть только одна прошлая кампания (2018). Их эффект обучить '
        'нельзя; они исключены train-only политикой. `S1B` и последний замер там '
        'совпадают, а `age_last` и `age_min` коллинеарны. Это ограничение '
        'идентифицируемости, а не скрытая настройка.',
        '',
        '## Проверка: leave-one-simple-pipe-out',
        '',
        '3 фолда; в каждом fold обучение исключает весь проверяемый простой участок. '
        'Все train и validation строки относятся к одной кампании `2022-07-06` '
        'одного parent pipeline `MAIN-2`. Это **spatial-transfer diagnostic**, '
        'а не forward-time прогноз. Строгих forward-time folds до cutoff нет.',
        '',
        'Pooled (одни и те же 357 ключей для всех моделей):',
        '',
        *_metric_table_lines(metrics),
        '',
        'По простым участкам:',
        '',
        *_per_section_lines(metrics),
        '',
        f"Лучшая MAE в этом diagnostic: **{best_mae[0]}** ({_fmt(best_mae[1]['mae_mm'])} мм). "
        'Это не статистически значимое сравнение: три простых участка одного '
        'parent pipeline не являются независимыми трубопроводами. Macro MAE '
        'по участкам: S1A/S1B 0.4095 мм, S1C 0.4347 мм; pooled MAE: 0.4982 → '
        '0.4698 мм. Эти сводки отвечают на разные вопросы и не дают '
        'устойчивого выигрыша S1C.',
        '',
        '## Вывод по качеству',
        '',
        'S1C снизил MAE относительно S1A/S1B (0.4982 → 0.4698 мм), но Pearson r² '
        'упал (0.1125 → 0.0083), а обычный R² остался отрицательным (-0.1473). '
        'Контроль `S1A_ridge_control` дал те же числа, потому что при одной '
        'прошлой кампании S1B и последний замер совпадают. Это **не доказанное '
        'улучшение**: поправка в основном сжимает уровень, а не восстанавливает '
        'ранжирование. Артефакт сохранён как экспериментальный; S1C не объявляется '
        'победителем и не заменяет production result автоматически.',
        '',
        '## Coverage и fallback',
        '',
        f"- Исторические S1-строки: **{summary['coverage']['historical_s1_rows']}**",
        f"- Вошли в Ridge: **{summary['coverage']['historical_fit_rows']}** "
        f"({_fmt(100 * summary['coverage']['historical_fit_coverage_fraction'], 2)}%)",
        f"- Без собственного baseline: **{summary['coverage']['historical_fallback_rows']}** "
        '(видны в `training_table.csv` с `fit_eligible=false`)',
        f"- Production: Ridge применим к **{summary['coverage']['production_ridge_applicable_rows']}** "
        f"из **{summary['coverage']['production_s1_rows']}**, fallback — "
        f"**{summary['coverage']['production_fallback_rows']}**",
        '',
        'Fallback-сегменты не получают Ridge-поправку: для них сохраняется S1B. '
        'Метрики fallback сохранены отдельно в `fallback_metrics.json`.',
        '',
        '## Артефакт и воспроизведение',
        '',
        f"- Полный artifact: `{summary['artifact_path']}`",
        f"- SHA256: `{summary['artifact_sha256']}`",
        '- В artifact входят fitted preprocessor (median imputer, scaler), Ridge, '
        'intercept, coefficients, feature order, dropped columns, alpha, weighting, '
        'bounds/fallback policies и contract version.',
        '',
        '```bash',
        'python -m src.ml.s1c train --root . --output-root experiments/s1abc/s1c',
        'python -m src.ml.s1c predict --root . --output-root experiments/s1abc/s1c',
        '```',
        '',
        '## Ограничения',
        '',
        '- Один historical campaign, один parent pipeline, три простых участка.',
        '- Нет строгого forward-time fold: train date max = validation date.',
        '- Перенос S1C на production S1B с двумя прошлыми кампаниями (2018 и 2022) '
        'не проверен: в обучении была только одна кампания.',
        '- `alpha=10` зафиксирован заранее, а не подобран; это инженерный старт.',
        '- Скрытые ответы, измерения на/после cutoff и внешние MAE 0.551 / r² 0.221 '
        'не использовались.',
        '',
    ]
    (output / 'report_training_ru.md').write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------- #
# S1C prediction from the saved artifact
# --------------------------------------------------------------------------- #
def run_s1c_predict(root: Path = BASE_DIR, output_root: Path | None = None,
                    artifact_path: Path | None = None,
                    s2_source: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = _ensure_output(Path(output_root or root / 'experiments/s1abc/s1c'), root)
    started = _utcnow()
    artifact_path = Path(artifact_path or output / 'model_artifact.joblib').resolve()
    s2_source = Path(s2_source or root / 'experiments/ml_v2/result.csv').resolve()
    artifact = load_artifact(artifact_path)
    metadata = artifact['metadata']
    data = RepositoryData(root)
    grid = target_key_grid(data)
    nominal = nominal_frame(data)
    prod, expected = production_frame(data)
    prediction = predict_s1_ridge(artifact, prod)
    prediction['s1c_vs_s1a_mm'] = (
        prediction.thickness_pred.astype(float) - prediction.s1a_prediction.astype(float))
    prediction['s1c_vs_s1b_mm'] = (
        prediction.thickness_pred.astype(float) - prediction.s1b_prediction.astype(float))
    prediction['model'] = metadata['model_name']
    prediction['training_required'] = True
    prediction['training_completed'] = True
    if int(prediction.ridge_applied.sum()) != int(prod.has_own_prior_vtd.sum()):
        raise ValueError('Ridge applicability does not match own-prior coverage')
    if prediction.loc[~prediction.ridge_applied, 'ridge_delta_mm'].abs().max() > 0:
        raise ValueError('Fallback rows received a nonzero Ridge delta')
    _write_csv(prediction[['id', 'segment_id', 'thickness_pred']],
               output / 'predictions_s1.csv')
    _write_diagnostics(prediction, output / 'prediction_diagnostics.csv')
    comparison = pd.DataFrame({
        'id': prediction.id, 'segment_id': prediction.segment_id,
        's1a_prediction_mm': prediction.s1a_prediction,
        's1b_prediction_mm': prediction.s1b_prediction,
        's1c_prediction_mm': prediction.thickness_pred,
        's1c_vs_s1a_mm': prediction.s1c_vs_s1a_mm,
        's1c_vs_s1b_mm': prediction.s1c_vs_s1b_mm,
        'ridge_applied': prediction.ridge_applied,
        'clipped': prediction.clipped,
        'prediction_source': prediction.prediction_source,
    })
    _write_csv(comparison, output / 'comparison_production.csv', float_format='%.10f')

    s2 = load_s2_source(s2_source, grid)
    bounds = _base_bounds(prediction, nominal, 'nominal_thickness_mm')
    validation, s2_check = _write_common_result(
        output, prediction, s2, grid, bounds, 'nominal_thickness_mm')
    clipped = prediction.clipped.fillna(False).astype(bool)
    summary = {
        'stage': 'S1C_predict',
        'contract_version': CONTRACT_VERSION,
        'artifact_version': metadata['artifact_version'],
        'training_required': True,
        'training_completed': True,
        'artifact_path': str(artifact_path),
        'artifact_sha256': sha256(artifact_path),
        'training_rows': int(metadata['training_rows']),
        'training_simple_sections': metadata['training_simple_sections'],
        'training_dates': metadata['training_dates'],
        'feature_count': len(metadata['used_features']),
        'rows': int(len(prediction)),
        'ridge_applied_rows': int(prediction.ridge_applied.sum()),
        'fallback_rows': int((~prediction.ridge_applied).sum()),
        'fallback_segments': _fallback_records(prediction),
        'clipped_rows': int(clipped.sum()),
        'above_h0_rows': int((prediction.thickness_pred
                              > prediction.h0_mm + 1e-9).sum()),
        'thickness_stats_mm': _series_stats(prediction.thickness_pred),
        's1c_vs_s1a_stats_mm': _series_stats(prediction.s1c_vs_s1a_mm),
        's1c_vs_s1b_stats_mm': _series_stats(prediction.s1c_vs_s1b_mm),
        'positive_vs_s1b_rows': int((prediction.s1c_vs_s1b_mm > 1e-12).sum()),
        'negative_vs_s1b_rows': int((prediction.s1c_vs_s1b_mm < -1e-12).sum()),
        'bounds_policy': metadata['bounds_policy'],
        'fallback_policy': metadata['fallback_policy'],
        's2': {**_s2_metadata(s2_source, s2), 'written_vs_source': s2_check},
        'validation': validation,
        'prediction_mode': 'loaded_saved_artifact_no_fit',
        'input_note': ('Predict reads no training_targets and calls no fit; '
                       'all features use VTD rows strictly before 2023-12-01.'),
    }
    write_json(output / 'summary.json', _json_safe(summary))
    _write_manifest(output / 'run_manifest.json', root, output, data,
                    'S1C_predict', started, training_required=True,
                    training_completed=True,
                    extra={'summary': summary,
                           'training_metadata': metadata,
                           'reproduce_command': (
                               'python -m src.ml.s1c predict --root . '
                               '--output-root experiments/s1abc/s1c')},
                    extra_inputs=[artifact_path, s2_source])
    _write_report_s1c_predict(output, summary)
    _verify_inputs_unchanged(data)
    print(f'PASS S1C predict: {len(prediction)} S1 predictions, '
          f'{summary["fallback_rows"]} fallback, result.csv 2000 rows', flush=True)
    return summary


def _write_report_s1c_predict(output: Path, summary: dict[str, Any]) -> None:
    metrics_path = output / 'metrics_pooled.json'
    metric_lines = ['Метрики обучения отсутствуют в этом output root: сначала выполните `s1c train`.']
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        metric_lines = _metric_table_lines(metrics)
    clipped_note = ('Нет значений, изменённых clipping.' if summary['clipped_rows'] == 0
                    else f"Clipping изменил **{summary['clipped_rows']}** значений.")
    lines = [
        '# S1C: production-прогноз из сохранённого артефакта',
        '',
        f"Контракт `{CONTRACT_VERSION}`. Commit/ветка/версии — в `run_manifest.json`. "
        'Этот запуск загружает `model_artifact.joblib` и **не вызывает fit**.',
        '',
        '## Обучение и artifact',
        '',
        f"- `training_completed=true`, строк обучения: **{summary['training_rows']}**",
        f"- Простые участки: **{summary['training_simple_sections']}**",
        f"- Даты обучения: **{', '.join(summary['training_dates'])}**",
        f"- Использовано признаков: **{summary['feature_count']}**",
        f"- Artifact: `{summary['artifact_path']}`",
        f"- SHA256: `{summary['artifact_sha256']}`",
        '',
        '## Production-прогноз',
        '',
        f"- S1-ключей: **{summary['rows']}**",
        f"- Ridge применён: **{summary['ridge_applied_rows']}**",
        f"- Fallback без Ridge: **{summary['fallback_rows']}**",
        f"- Среднее/медиана: **{_fmt(summary['thickness_stats_mm']['mean'], 6)}** / "
        f"**{_fmt(summary['thickness_stats_mm']['median'], 6)}** мм",
        f"- Мин/макс: **{_fmt(summary['thickness_stats_mm']['min'], 6)}** / "
        f"**{_fmt(summary['thickness_stats_mm']['max'], 6)}** мм",
        f"- Δ(S1C−S1B): mean **{_fmt(summary['s1c_vs_s1b_stats_mm']['mean'], 6)}**, "
        f"median **{_fmt(summary['s1c_vs_s1b_stats_mm']['median'], 6)}**, "
        f"min **{_fmt(summary['s1c_vs_s1b_stats_mm']['min'], 6)}**, "
        f"max **{_fmt(summary['s1c_vs_s1b_stats_mm']['max'], 6)}** мм",
        f"- Положительных/отрицательных поправок к S1B: **{summary['positive_vs_s1b_rows']}** / "
        f"**{summary['negative_vs_s1b_rows']}**",
        f"- {clipped_note}",
        f"- Прогнозов выше h0 (разрешено политикой nominal): " 
        f"**{summary['above_h0_rows']}**; выше nominal: **0**",
        f"- Bounds: `{summary['bounds_policy']}`",
        '',
        'Fallback-сегменты (Ridge не применён):',
        '',
        *[
            f"- segment `{r['segment_id']}`: `{r['s1b_prediction_mm']:.4f}` мм, "
            f"сосед `{r['fallback_segment_id']}`, дата `{r['fallback_measurement_date']}`"
            for r in summary['fallback_segments']
        ],
        '',
        '## Таблица качества на одинаковых исторических ключах',
        '',
        *metric_lines,
        '',
        'S1C не объявляется улучшением только потому, что fit прошёл. Сравнение '
        'с S1A, S1B и контрольной `S1A_ridge_control` находится в отчёте обучения; '
        'если преимущества нет, это отрицательный результат, а артефакт сохранён.',
        '',
        '## S2 и общий result.csv',
        '',
        f"S2 (`1250009583`, 1475 строк) скопирован без изменений из "
        f"`{summary['s2']['source_path']}`; max |diff| = "
        f"**{summary['s2']['written_vs_source']['max_abs_difference']:.3e}** мм. "
        'S2 не переобучался, его признаки и параметры не менялись.',
        '',
        '## Ограничения',
        '',
        '- S1C обучен на одном historical campaign и одном parent pipeline; '
        'LOPO по трём простым участкам — spatial transfer, не forward-time.',
        '- Production S1B использует две прошлые кампании, а обучение — только одну; '
        'перенос поправки в этот режим не проверен.',
        '- Скрытые ответы и измерения на/после cutoff не использовались.',
        '',
        '## Inference без обучения',
        '',
        '```bash',
        'python -m src.ml.s1c predict --root . --output-root experiments/s1abc/s1c',
        '```',
        '',
    ]
    (output / 'report_ru.md').write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------- #
# Sequential run
# --------------------------------------------------------------------------- #
def run_all(root: Path = BASE_DIR, output_root: Path | None = None,
            s2_source: Path | None = None, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    root = Path(root).resolve()
    base = _ensure_output(Path(output_root or root / 'experiments/s1abc'), root)
    started = _utcnow()
    s2_source = Path(s2_source or root / 'experiments/ml_v2/result.csv').resolve()
    s1a = run_s1a(root, base / 's1a', s2_source)
    s1b = run_s1b(root, base / 's1b', s2_source)
    s1c_train = run_s1c_train(root, base / 's1c', alpha)
    s1c_predict = run_s1c_predict(root, base / 's1c',
                                  base / 's1c/model_artifact.joblib', s2_source)
    comparison = write_metric_artifacts(root, base)
    sequence = {
        'contract_version': CONTRACT_VERSION,
        'started_utc': started,
        'finished_utc': _utcnow(),
        'order': ['S1A', 'S1B', 'S1C_train', 'S1C_predict'],
        'git': _git_state(root),
        'versions': _versions(),
        'stages': {
            's1a': {'summary': s1a, 'path': str(base / 's1a')},
            's1b': {'summary': s1b, 'path': str(base / 's1b')},
            's1c_train': {'summary': s1c_train, 'path': str(base / 's1c')},
            's1c_predict': {'summary': s1c_predict, 'path': str(base / 's1c')},
        },
        'result_paths': {
            's1a': str(base / 's1a/result.csv'),
            's1b': str(base / 's1b/result.csv'),
            's1c': str(base / 's1c/result.csv'),
        },
        's2_source': str(s2_source),
        's2_source_sha256': sha256(s2_source),
        'metric_comparison_path': str(base / 'model_comparison.csv'),
        'candidate_result_path': str(base / 'result.csv'),
        'candidate_result_sha256': comparison['candidate_result_sha256'],
        'reproduce_command': 'python -m src.ml.s1_all --root . --output-root experiments/s1abc',
    }
    write_json(base / 'sequence_manifest.json', _json_safe(sequence))
    lines = [
        '# S1A → S1B → S1C: последовательный запуск',
        '',
        f"Контракт `{CONTRACT_VERSION}`. Commit/ветка/версии — в `sequence_manifest.json`.",
        '',
        '| Stage | Что сделано | Обучение | Ключевой артефакт |',
        '|---|---|---|---|',
        f"| S1A | persistence: last prior VTD + fallback | нет | `{base / 's1a/result.csv'}` |",
        f"| S1B | минимум всех прошлых ВТД + тот же fallback | нет | `{base / 's1b/result.csv'}` |",
        f"| S1C train | Ridge-поправка к S1B, {s1c_train['training_rows']} строк | да | `{base / 's1c/model_artifact.joblib'}` |",
        f"| S1C predict | загрузка artifact без fit | да (artifact) | `{base / 's1c/result.csv'}` |",
        '',
        'S2 во всех `result.csv` скопирован без изменений из явно выбранного '
        f"`{s2_source}`; проверка max |diff| = 0 выполнена при записи.",
        '',
        '## Итоговая проверка S1C',
        '',
        *_metric_table_lines(s1c_train['metrics']),
        '',
        'S1C снизил MAE (0.4982 → 0.4698 мм), но Pearson r² упал '
        '(0.1125 → 0.0083), а обычный R² остался отрицательным. Это не доказанное '
        'улучшение: поправка в основном сжимает уровень, а не восстанавливает '
        'ранжирование. Артефакт сохранён как экспериментальный, production result '
        '`experiments/ml_v2/result.csv` не перезаписан.',
        '',
        'Таблица метрик рядом с каждым `result.csv`: `s1a/metrics_summary.csv`, '
        '`s1b/metrics_summary.csv`, `s1c/metrics_summary.csv`; общая — '
        '`model_comparison.csv` / `model_comparison.md`.',
        '',
        'Подробности и ограничения: `s1c/report_training_ru.md`, `s1c/report_ru.md`, '
        '`s1b/report_ru.md`, `s1a/report_ru.md`.',
        '',
        '## Повтор',
        '',
        '```bash',
        'python -m src.ml.s1_all --root . --output-root experiments/s1abc',
        '```',
        '',
    ]
    (base / 'report_ru.md').write_text('\n'.join(lines), encoding='utf-8')
    return sequence


# --------------------------------------------------------------------------- #
# Comparison table next to each result.csv
# --------------------------------------------------------------------------- #
def write_metric_artifacts(root: Path = BASE_DIR,
                           base_output: Path | None = None) -> dict[str, Any]:
    """Compute MAE/R² on the same 357 historical keys and write comparison tables.

    S1A/S1B are deterministic rules, so their values on these keys are not
    out-of-fold model predictions.  S1C and the S1A control are out-of-fold
    leave-one-simple-section-out predictions from ``oof_fit_rows.csv``.  The
    table is a diagnostic on observed 2022 labels, not a hidden-test score.
    """
    import shutil

    root = Path(root).resolve()
    base = Path(base_output or root / 'experiments/s1abc').resolve()
    data = RepositoryData(root)
    training = build_s1_training_frame(data)
    fit = training[training.fit_eligible.astype(bool)].copy()
    oof_path = base / 's1c' / 'oof_fit_rows.csv'
    if not oof_path.is_file():
        raise FileNotFoundError(f'Run S1C train first: {oof_path}')
    oof = pd.read_csv(oof_path)
    needed = ['id', 'segment_id', 'prediction_s1c', 'prediction_s1a_control']
    missing = set(needed) - set(oof.columns)
    if missing:
        raise ValueError(f'OOF table is missing columns: {sorted(missing)}')
    merged = fit[['id', 'segment_id', 'prediction_date', TARGET_COLUMN,
                  's1a_prediction', 's1b_prediction']].merge(
        oof[needed], on=['id', 'segment_id'], how='inner',
        validate='one_to_one', indicator=True)
    if not merged._merge.eq('both').all() or len(merged) != len(fit):
        raise ValueError('OOF keys do not match the fit-eligible historical keys')
    merged = merged.drop(columns='_merge')
    if len(merged) != 357:
        raise ValueError(f'Expected 357 comparison rows, got {len(merged)}')

    comparison = {
        'S1A': merged.assign(prediction=merged.s1a_prediction),
        'S1B': merged.assign(prediction=merged.s1b_prediction),
        'S1C': merged.assign(prediction=merged.prediction_s1c),
        'S1A_ridge_control': merged.assign(
            prediction=merged.prediction_s1a_control),
    }
    states = {
        'S1A': ('complete_rule_no_training', False, False,
                'rule_no_fit', 'last prior VTD / documented fallback'),
        'S1B': ('complete_rule_no_training', False, False,
                'rule_no_fit', 'minimum of all prior VTD / same fallback'),
        'S1C': ('fit_complete_oof', True, True,
                'leave_one_simple_section_out',
                'Ridge correction to S1B; OOF predictions'),
        'S1A_ridge_control': ('control_oof', True, True,
                               'leave_one_simple_section_out',
                               'same Ridge machinery on S1A baseline; control only'),
    }
    rows = []
    per_section_rows = []
    for model, frame in comparison.items():
        record = pooled_metrics(frame, 'prediction')
        state, required, completed, design, note = states[model]
        row = {
            'stage': model.lower() if model != 'S1A_ridge_control' else 's1a_control',
            'model': model,
            'completion_state': state,
            'training_required': required,
            'training_completed': completed,
            'evaluation_design': design,
            'n': int(record['n']),
            'coverage_of_comparison': 1.0,
            'historical_coverage_fraction': float(len(fit) / len(training)),
            'historical_rows_total': int(len(training)),
            'historical_rows_fit': int(len(fit)),
            'mae_mm': record['mae_mm'],
            'rmse_mm': record['rmse_mm'],
            'bias_mm': record['bias_mm'],
            'pearson_r': record['pearson_r'],
            'pearson_squared': record['pearson_squared'],
            'r2': record['r2'],
            'pearson_status': record['pearson_status'],
            'r2_status': record['r2_status'],
            'train_date_max': '2022-07-06',
            'validation_date': '2022-07-06',
            'result_path': str(base / ('s1c' if model.startswith('S1C') else
                                       's1a' if model == 'S1A' else
                                       's1b' if model == 'S1B' else 's1a') / 'result.csv'),
            'report_path': str(base / ('s1c/report_training_ru.md'
                                       if model in ('S1C', 'S1A_ridge_control')
                                       else 's1a/report_ru.md' if model == 'S1A'
                                       else 's1b/report_ru.md')),
            'artifact_path': (str(base / 's1c/model_artifact.joblib')
                              if model in ('S1C', 'S1A_ridge_control') else None),
            'note': note,
        }
        section_records = []
        for section, part in frame.groupby('id', sort=True):
            section_record = pooled_metrics(part, 'prediction')
            section_records.append({
                'group': str(int(section)),
                'n': int(section_record['n']),
                'mae_mm': section_record['mae_mm'],
                'rmse_mm': section_record['rmse_mm'],
                'bias_mm': section_record['bias_mm'],
                'pearson_r': section_record['pearson_r'],
                'pearson_squared': section_record['pearson_squared'],
                'r2': section_record['r2'],
            })
            per_section_rows.append({
                'model': model, 'id': int(section),
                'n': int(section_record['n']),
                'mae_mm': section_record['mae_mm'],
                'rmse_mm': section_record['rmse_mm'],
                'bias_mm': section_record['bias_mm'],
                'pearson_r': section_record['pearson_r'],
                'pearson_squared': section_record['pearson_squared'],
                'r2': section_record['r2'],
            })
        macro = macro_metrics(pd.DataFrame(section_records))
        row.update({
            'macro_groups': macro['groups'],
            'macro_mae_mm': macro['mae_mm'],
            'macro_bias_mm': macro['bias_mm'],
            'macro_pearson_squared': macro['pearson_squared'],
            'macro_r2': macro['r2'],
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    per_section = pd.DataFrame(per_section_rows)
    _write_csv(table, base / 'model_comparison.csv', float_format='%.10f')
    _write_csv(per_section, base / 'model_comparison_per_section.csv',
               float_format='%.10f')

    table_for_md = table.copy()
    md_lines = [
        '# Сравнение S1A / S1B / S1C на одинаковых исторических ключах',
        '',
        'Все значения ниже рассчитаны на **одних и тех же 357 ключах**: сегменты, '
        'у которых в обучающей кампании `2022-07-06` есть собственный прошлый '
        'замер (2018). S1A/S1B — правила без обучения; S1C и контроль — '
        'leave-one-simple-section-out OOF. Это spatial-transfer diagnostic на '
        'одной кампании, не forward-time и не скрытая проверка.',
        '',
        '`MAE` — это запрошенная MOE (mean absolute error), мм. `Pearson r²` — '
        'квадрат корреляции; `R²` — обычный коэффициент детерминации, он может '
        'быть отрицательным и не равен Pearson r².',
        '',
        '| Модель | Состояние | n | MAE, мм | RMSE, мм | bias, мм | Pearson r | Pearson r² | R² |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in table_for_md.itertuples():
        md_lines.append(
            f"| {row.model} | {row.completion_state} | {row.n} | "
            f"{_fmt(row.mae_mm)} | {_fmt(row.rmse_mm)} | {_fmt(row.bias_mm)} | "
            f"{_fmt(row.pearson_r)} | {_fmt(row.pearson_squared)} | {_fmt(row.r2)} |")
    md_lines += [
        '',
        '## Macro-сводка по простым участкам',
        '',
        '| Модель | Групп | macro MAE, мм | macro bias, мм | macro Pearson r² | macro R² |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in table_for_md.itertuples():
        md_lines.append(
            f"| {row.model} | {row.macro_groups} | {_fmt(row.macro_mae_mm)} | "
            f"{_fmt(row.macro_bias_mm)} | {_fmt(row.macro_pearson_squared)} | "
            f"{_fmt(row.macro_r2)} |")
    md_lines += [
        '',
        '## Как читать результат',
        '',
        '- В этих 357 парах измеренная толщина выросла в 177 случаях и снизилась '
        'в 178 (2 нулевых); знак сохраняется, отрицательные наблюдения не обнуляются.',
        '- S1A и S1B на этих 357 строках **обязаны совпасть**: в 2022 году у '
        'каждого сегмента была только одна прошлая кампания (2018). Это не '
        'доказывает бесполезность S1B в production-режиме двух кампаний; это '
        'показывает, что данный historical test не различает правила.',
        '- S1C и `S1A_ridge_control` также совпали: при одной прошлой кампании '
        '`S1B` и последний замер — одна и та же величина, а коллинеарные '
        'признаки делят коэффициент. Контроль нужен, чтобы не приписать эффект '
        'обучения выбору baseline.',
        '- S1C снизил MAE с 0.4982 до 0.4698 мм, но Pearson r² упал с 0.1125 до '
        '0.0083, а обычный R² остался отрицательным. Это **не доказанное '
        'улучшение**: модель в основном сжимает уровень, а не восстанавливает '
        'ранжирование. Артефакт сохранён, но S1C не объявляется победителем.',
        '- Три простых участка одного parent pipeline `MAIN-2` не являются '
        'независимыми трубопроводами; статистическая значимость не заявляется.',
        '',
        '## Файлы',
        '',
        '- `model_comparison.csv` / `model_comparison_per_section.csv`',
        '- `s1a/metrics_summary.csv`, `s1b/metrics_summary.csv`, `s1c/metrics_summary.csv`',
        '- `result.csv` в каждом stage directory; S2 во всех скопирован без изменений.',
        '',
    ]
    (base / 'model_comparison.md').write_text('\n'.join(md_lines), encoding='utf-8')
    # Duplicate the same table next to every result.csv.
    for stage in ('s1a', 's1b', 's1c'):
        stage_dir = base / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(table, stage_dir / 'metrics_summary.csv', float_format='%.10f')
        stage_lines = [
            f'# Метрики {stage.upper()} на общих 357 исторических ключах',
            '',
            'Полная таблица и ограничения: `../model_comparison.md`. '
            'S1A/S1B — правила без обучения, S1C — OOF Ridge. '
            '`MAE` = MOE, мм; `R²` — обычный R², не Pearson r².',
            '',
            '| Модель | Состояние | n | MAE, мм | RMSE, мм | bias, мм | Pearson r | Pearson r² | R² |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for row in table_for_md.itertuples():
            stage_lines.append(
                f"| {row.model} | {row.completion_state} | {row.n} | "
                f"{_fmt(row.mae_mm)} | {_fmt(row.rmse_mm)} | {_fmt(row.bias_mm)} | "
                f"{_fmt(row.pearson_r)} | {_fmt(row.pearson_squared)} | {_fmt(row.r2)} |")
        (stage_dir / 'metrics_summary.md').write_text('\n'.join(stage_lines),
                                                       encoding='utf-8')

    # Keep stage manifests honest after adding the comparison files.
    for stage in ('s1a', 's1b', 's1c'):
        stage_dir = base / stage
        for manifest_name in ('run_manifest.json', 'run_manifest_train.json'):
            manifest_path = stage_dir / manifest_name
            if not manifest_path.is_file():
                continue
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            payload['outputs_sha256'] = _output_hashes(stage_dir)
            payload['metrics_summary_sha256'] = sha256(
                stage_dir / 'metrics_summary.csv')
            payload['metrics_summary_md_sha256'] = sha256(
                stage_dir / 'metrics_summary.md')
            payload['metrics_summary_path'] = str(stage_dir / 'metrics_summary.csv')
            write_json(manifest_path, _json_safe(payload))

    # Candidate S1C submission at the sequence root, next to the comparison table.
    # This does not overwrite experiments/ml_v2/result.csv or the existing production choice.
    candidate = base / 'result.csv'
    shutil.copyfile(base / 's1c' / 'result.csv', candidate)
    if sha256(candidate) != sha256(base / 's1c' / 'result.csv'):
        raise ValueError('Candidate result copy hash mismatch')
    payload = {
        'generated_utc': _utcnow(),
        'scope': 'same_357_historical_keys',
        'notes': [
            'S1A and S1B are identical on this historical set because only one prior campaign exists.',
            'S1C and S1A_ridge_control are identical because S1B equals last VTD with one prior campaign.',
            'S1C lowers MAE but lowers Pearson r squared; no proven improvement is claimed.',
            'The top-level result.csv is a candidate copy of s1c/result.csv, not a promotion of experiments/ml_v2/result.csv.',
        ],
        'rows': _json_safe(table.to_dict(orient='records')),
        'per_section': _json_safe(per_section.to_dict(orient='records')),
        'candidate_result_sha256': sha256(candidate),
        'source_s1c_result_sha256': sha256(base / 's1c' / 'result.csv'),
    }
    write_json(base / 'model_comparison.json', payload)
    return payload

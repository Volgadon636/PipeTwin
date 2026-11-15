"""Compare next-cycle S1/S2 results and assemble an explicit candidate pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .s1_contract import S1_ID, S2_ID, combine_with_s2
from .next_common import json_safe, utcnow, write_csv


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ['id', 'segment_id', 'thickness_pred']:
        raise ValueError(f'{path}: expected exactly id,segment_id,thickness_pred')
    if frame.duplicated(['id', 'segment_id']).any():
        raise ValueError(f'{path}: duplicate keys')
    if not np.isfinite(frame.thickness_pred.to_numpy(float)).all():
        raise ValueError(f'{path}: nonfinite predictions')
    return frame


def assemble_pair(root: Path, output_root: Path,
                  s1_candidate: str = 's1a',
                  s2_candidate: str = 'physics_priors') -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output_root or root / 'experiments/next_cycle/assembled').resolve()
    s1_dir = root / 'experiments/next_cycle/s1/candidates' / s1_candidate
    s2_dir = root / 'experiments/next_cycle/s2/candidates' / s2_candidate
    s1 = _read_predictions(s1_dir / 'predictions_s1.csv')
    s2 = _read_predictions(s2_dir / 'predictions_s2.csv')
    data = RepositoryData(root / 'experiments/next_cycle/data_generation')
    target_grid = data.read('processed/target_segments_grid')[['id', 'segment_id']].copy()
    s1_diag = pd.read_csv(s1_dir / 'prediction_diagnostics.csv')
    s2_diag = pd.read_csv(s2_dir / 'prediction_diagnostics.csv')
    s1_bounds = s1_diag[['id', 'segment_id', 'h0_mm',
                         'nominal_thickness_mm']].merge(
        s1, on=['id', 'segment_id'], how='left', validate='one_to_one')
    # Use the exact prediction CSV (not the lower-precision diagnostics) for
    # the validating upper bound of the fixed conservative rules.
    s1_bounds['upper_mm'] = np.where(
        s1_candidate in ('s1a', 's1b'),
        np.maximum(s1_bounds.h0_mm, s1_bounds.thickness_pred),
        s1_bounds.nominal_thickness_mm)
    s2_bounds = s2_diag[['id', 'segment_id', 'nominal_thickness_mm']].copy()
    s2_bounds['h0_mm'] = s2_bounds.nominal_thickness_mm
    s2_bounds['upper_mm'] = s2_bounds.nominal_thickness_mm
    bounds = pd.concat([s1_bounds[['id', 'segment_id', 'h0_mm',
                                  'nominal_thickness_mm', 'upper_mm']],
                        s2_bounds[['id', 'segment_id', 'h0_mm',
                                  'nominal_thickness_mm', 'upper_mm']]],
                       ignore_index=True)
    if bounds.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate bounds key')
    combined, validation = combine_with_s2(s1, s2, target_grid, bounds)
    directory = output / f'{s1_candidate}__{s2_candidate}'
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(combined, directory / 'result.csv', float_format='%.10f')
    written = pd.read_csv(directory / 'result.csv')
    s1_written = written[written.id.eq(S1_ID)][['id', 'segment_id', 'thickness_pred']]
    s1_written = s1_written.sort_values(['id', 'segment_id']).reset_index(drop=True)
    s1_expected = s1.sort_values(['id', 'segment_id']).reset_index(drop=True)
    s2_written = written[written.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']]
    s2_written = s2_written.sort_values(['id', 'segment_id']).reset_index(drop=True)
    s2_expected = s2.sort_values(['id', 'segment_id']).reset_index(drop=True)
    s1_diff = float(np.abs(s1_written.thickness_pred.to_numpy(float)
                           - s1_expected.thickness_pred.to_numpy(float)).max())
    s2_diff = float(np.abs(s2_written.thickness_pred.to_numpy(float)
                           - s2_expected.thickness_pred.to_numpy(float)).max())
    if s1_diff != 0.0 or s2_diff != 0.0:
        raise ValueError(f'Assembly changed a fixed half: S1 {s1_diff}, S2 {s2_diff}')
    summary = {
        'assembled_utc': utcnow(),
        's1_candidate': s1_candidate,
        's2_candidate': s2_candidate,
        'rows': int(len(written)),
        'unique_keys': bool(not written.duplicated(['id', 'segment_id']).any()),
        's1_rows': int((written.id.eq(S1_ID)).sum()),
        's2_rows': int((written.id.eq(S2_ID)).sum()),
        's1_max_abs_diff_vs_candidate': s1_diff,
        's2_max_abs_diff_vs_candidate': s2_diff,
        's1_artifact_sha256': (sha256(s1_dir / 'model_artifact.joblib')
                               if (s1_dir / 'model_artifact.joblib').is_file() else None),
        's2_artifact_sha256': (sha256(s2_dir / 'model_artifact.joblib')
                               if (s2_dir / 'model_artifact.joblib').is_file() else None),
        'result_sha256': sha256(directory / 'result.csv'),
        'validation': validation,
        'note': ('Explicit candidate pair. This is not a claim of a best production result '
                 'and does not overwrite experiments/ml_v2/result.csv.'),
    }
    write_json(directory / 'assembly.json', json_safe(summary))
    write_json(directory / 'model_ids.json', {
        's1': s1_candidate, 's2': s2_candidate,
        'policy': 'explicit IDs only; no automatic winner',
    })
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.{digits}f}'
    return str(value)


def write_overall_report(root: Path, output_root: Path,
                         s1_candidate: str = 's1a',
                         s2_candidate: str = 'physics_priors') -> Path:
    root = Path(root).resolve()
    base = Path(output_root or root / 'experiments/next_cycle').resolve()
    s1_metrics = _load_json(base / 's1/metrics_pooled.json')
    s2_metrics = _load_json(base / 's2/metrics_pooled.json')
    s2_post = _load_json(base / 's2/post_cutoff_retrospective/metrics.json')
    s1_sel = _load_json(base / 's1/selected_candidates.json')
    s2_sel = _load_json(base / 's2/selected_candidates.json')
    data_audit = _load_json(base / 'audit/data_audit.json')
    fit = s1_metrics['fit_eligible_357']
    allp = s1_metrics['all_pipeline_561']
    lines = [
        '# PipeTwin next cycle: S1 diagnostics and S2 strict candidates', '',
        f"Generated: `{utcnow()}`. Branch/commit — в `experiment_registry.json` и manifests.",
        '',
        'Это exploratory-цикл. Ни один кандидат не объявляется доказанным production winner; '
        '`experiments/ml_v2/result.csv` не перезаписан. Push/deploy не выполнялись.',
        '',
        '## Данные и воспроизводимость', '',
        f"- Согласованная генерация: `{data_audit['data_generation_root']}`.",
        f"- S1: 525 ключей, `1250022871`, `2023-12-01`; S2: 1475, `1250009583`, `2022-07-01`.",
        f"- S1ABC reproduction: max abs diff result.csv = "
        f"`{data_audit['reproduction_of_s1abc']['s1c']['max_abs_thickness_diff']}`; "
        f"artifact hash `{data_audit['reproduction_of_s1abc']['s1c']['artifact_sha256_new']}`.",
        '- Старый `target_segments_grid.csv` был отвергнут по checksum (лишний `delta_t_c`); '
        'новая генерация manifest-verified и используется для всех новых сравнений.',
        '',
        '## S1: что проверено и что изменилось', '',
        'Основная spatial-цель: within-section Pearson / Pearson², не только MAE. '
        'LOPO по трём простым участкам одного parent — spatial-transfer диагностика. '
        'Строгих forward folds нет: у каждого допустимого S1-ответа только одна прошлая кампания.',
        '',
        '| S1 model | MAE | RMSE | bias | Pearson r | Pearson r² | ordinary R² | within-section r² | top-10% recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('s1a', 's1b', 'old_s1c', 'ridge_history', 'ridge_geometry',
                 'rate_signed', 'rate_nonnegative', 'catboost', 'mix_025',
                 'mix_050', 'mix_100', 'affine', 'train_median'):
        rec = fit[name]
        ws = rec['within_section_centered']
        lines.append(
            f"| {name} | {_fmt(rec['pooled']['mae_mm'])} | {_fmt(rec['pooled']['rmse_mm'])} | "
            f"{_fmt(rec['pooled']['bias_mm'])} | {_fmt(rec['pooled']['pearson_r'])} | "
            f"{_fmt(rec['pooled']['pearson_squared'])} | {_fmt(rec['pooled']['r2'])} | "
            f"{_fmt(ws['pearson_squared'])} | {_fmt(rec['topk']['pooled_recall'])} |")
    lines += [
        '', f"S1 default: **{s1_sel['default']}**. {s1_sel['default_reason']}. "
        f"best_spatial=`{s1_sel['best_spatial']}`, best_mae=`{s1_sel['best_mae']}`.",
        '',
        'Почему старый S1C терял сигнал (подтверждено этим циклом):', '',
        '1. `s1b_prediction` и `last_vtd_thickness_mm` были идентичны во всём fit; Ridge делил '
        'эффект между дублями, а в production эти величины уже различались. Новая core-модель '
        'использует один last baseline.',
        '2. Горизонт в train был константным (3.7837 года), поэтому старый learner не мог '
        'масштабировать поправку по времени. Rate-модель вынесена отдельно; при константном dt '
        'signed-rate OOF алгебраически совпал с delta-моделью, а production различается из-за '
        'фактического dt 1.4045/5.1882.',
        '3. `last_minus_min` и campaign count были константны; их эффект не идентифицируется.',
        '4. Равный вес простых участков давал Kish ESS ≈31.65; row-uniform/inverse-sqrt '
        'проверены как ablation. Ни одна схема не восстановила within-section сигнал S1A.',
        '5. Block-CV на целевой трубе (100 м + 20 м buffer) показывает потенциальный same-pipe '
        'эффект: within-section r² у ridge_geometry/rate_signed выше, чем у S1A; но LOPO между '
        'трубами этого не подтверждает. Это spatial interpolation, не будущий прогноз.',
        '',
        f"All-pipeline 561 (fallback = S1B): S1A MAE `{_fmt(allp['s1a']['pooled']['mae_mm'])}`, "
        f"old S1C `{_fmt(allp['old_s1c']['pooled']['mae_mm'])}`, "
        f"ridge_geometry `{_fmt(allp['ridge_geometry']['pooled']['mae_mm'])}`, "
        f"catboost `{_fmt(allp['catboost']['pooled']['mae_mm'])}`; "
        'within-section сигнал при этом падает, поэтому S1A оставлен default.',
        '',
        '## S2: strict cold-start', '',
        'Strict train: только `2018-09-23` (372 строки, 3 простых участка, один parent). '
        '2023-08-12 кампании — только frozen retrospective, не fit/selection/gate.',
        '',
        '| S2 model | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | pred std |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ('constant_effective_rate', 'physics_priors',
                 'physics_velocity_consistent', 'physics_spatial',
                 'physics_spatial_temperature', 'catboost_residual'):
        rec = s2_metrics[name]
        ws = rec['within_section_centered']
        lines.append(
            f"| {name} | {_fmt(rec['pooled']['mae_mm'])} | {_fmt(rec['pooled']['rmse_mm'])} | "
            f"{_fmt(rec['pooled']['bias_mm'])} | {_fmt(rec['pooled']['pearson_r'])} | "
            f"{_fmt(rec['pooled']['pearson_squared'])} | {_fmt(rec['pooled']['r2'])} | "
            f"{_fmt(ws['pearson_squared'])} | {_fmt(rec['pooled']['prediction_std_mm'])} |")
    lines += [
        '', f"S2 default: **{s2_sel['default']}**; level_winner=`{s2_sel['level_winner']}`, "
        f"spatial_winner=`{s2_sel['spatial_winner']}`.",
        'Уровень: physics воспроизводит v2 production mean 2.809539; constant даёт 1.539584. '
        'Пространственный рисунок: только spatial candidates дают ненулевой std, но strict OOF '
        'within-section r² мал (0.004–0.012), а frozen 2023 retrospective ухудшает MAE по сравнению '
        'с physics-only. Поэтому spatial correction не объявлена улучшением.',
        '',
        'Temperature: `physics_spatial_temperature` добавляет scenario cooling из `heat_transfer.py` '
        'и сдвигает target mean до 2.5907; на strict OOF и 2023 retrospective это не улучшает MAE. '
        'Velocity-consistent variant после refit r_ref численно совпал с physics (компенсация), '
        'поэтому сам по себе не доказывает пользу.',
        '',
        'Frozen post-cutoff 2023 retrospective (не selection): physics MAE '
        f"`{_fmt(s2_post['physics_priors']['pooled']['mae_mm'])}`, bias "
        f"`{_fmt(s2_post['physics_priors']['pooled']['bias_mm'])}`; spatial MAE "
        f"`{_fmt(s2_post['physics_spatial']['pooled']['mae_mm'])}`; catboost MAE "
        f"`{_fmt(s2_post['catboost_residual']['pooled']['mae_mm'])}`. "
        'Physics level переносится лучше constant, но систематически смещён.',
        '',
        '## Explicit assembly', '',
        f"`{s1_candidate}` + `{s2_candidate}`: "
        f"`experiments/next_cycle/assembled/{s1_candidate}__{s2_candidate}/result.csv`. "
        'S1 и S2 половинки проверены на точное сохранение; 2000 уникальных ключей. '
        'Это candidate pair, не «лучший» result.',
        '',
        '## Artifacts', '',
        '- `experiments/next_cycle/audit/data_audit.json`',
        '- `experiments/next_cycle/s1/` — OOF, folds, weight ablation, block CV, candidates, figures, report',
        '- `experiments/next_cycle/s2/` — strict OOF, eligibility, domain shift, retrospective, candidates, figures, report',
        '- `experiments/next_cycle/selected_candidates.json` (aggregate)',
        '- `experiments/next_cycle/s1/selected_candidates.json`, `s2/selected_candidates.json`',
        '- `experiments/next_cycle/experiment_registry.json`',
        '',
        '## Limitations', '',
        '- Three S1 simple sections and one S2 parent: no independent forward-time validation.',
        '- S1 production S1B uses two campaigns while training has one; S1C transfer remains unverified.',
        '- S2 target has no labels; target accuracy is not computed. Domain shift (4.5 mm / 29 yr / cold-dry) is severe.',
        '- Temperature scenario is a steady-state proxy from mean inputs, not a measured 29-year profile.',
        '- Post-cutoff 2023 labels are used only for the frozen retrospective; no strict-model selection.',
        '',
        '## Commands', '',
        '```bash',
        'python -m src.data_prep --output-root experiments/next_cycle/data_generation',
        'python -m src.ml.next_s1 train --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s1',
        'python -m src.ml.next_s2 train --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s2',
        'python -m src.ml.next_compare --root . --output-root experiments/next_cycle --s1-candidate s1a --s2-candidate physics_priors',
        '```', '',
    ]
    path = base / 'report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def run_compare(root: Path = BASE_DIR, output_root: Path | None = None,
                s1_candidate: str = 's1a',
                s2_candidate: str = 'physics_priors') -> dict[str, Any]:
    root = Path(root).resolve()
    base = Path(output_root or root / 'experiments/next_cycle').resolve()
    assembly = assemble_pair(root, base / 'assembled', s1_candidate, s2_candidate)
    report = write_overall_report(root, base, s1_candidate, s2_candidate)
    payload = {
        'generated_utc': utcnow(),
        's1_default': _load_json(base / 's1/selected_candidates.json')['default'],
        's2_default': _load_json(base / 's2/selected_candidates.json')['default'],
        'explicit_pair': {'s1': s1_candidate, 's2': s2_candidate},
        'assembly': assembly,
        'report_path': str(report),
        'note': 'No automatic production replacement; no push/deploy.',
    }
    write_json(base / 'selected_candidates.json', json_safe(payload))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/next_cycle')
    parser.add_argument('--s1-candidate', default='s1a')
    parser.add_argument('--s2-candidate', default='physics_priors')
    args = parser.parse_args(argv)
    run_compare(args.root, args.output_root, args.s1_candidate, args.s2_candidate)


if __name__ == '__main__':
    main()

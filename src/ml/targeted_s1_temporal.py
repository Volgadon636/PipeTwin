"""Stage 5: fixed temporal interpretations of the selected S1 Ridge.

No model is fitted here.  The existing Stage 3 full-train Ridge artifact is
loaded and applied to production features with actual per-segment dt.  OOF
diagnostics use the saved Stage 3 block-CV predictions.
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
from .next_common import (evaluate_predictions, json_safe, load_artifact,
                          s1_fit_rows, s1_production_table, s1_training_table,
                          utcnow, write_csv as _write_csv, write_manifest)
from .s1_contract import S1_ID
from .targeted_s1 import _block_tail


def write_csv(frame: pd.DataFrame, path: Path, fmt: str = '%.10g') -> None:
    _write_csv(frame, path, float_format=fmt)

VARIANTS = ['correction', 'time_scaled', 'conservative_0_25']
MODEL_BASES = ['global_ridge_geometry', 'target_ridge_geometry']
STAGE3 = Path('experiments/targeted_cycle/stage_3_s1_100m_20m')


def _stats(values) -> dict[str, Any]:
    x = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(float)
    if len(x) == 0:
        return {'count': 0, 'mean': None, 'median': None, 'std': None,
                'min': None, 'max': None, 'q05': None, 'q25': None,
                'q75': None, 'q95': None}
    return {'count': int(len(x)), 'mean': float(x.mean()),
            'median': float(np.median(x)), 'std': float(x.std(ddof=0)),
            'min': float(x.min()), 'max': float(x.max()),
            'q05': float(np.quantile(x, .05)), 'q25': float(np.quantile(x, .25)),
            'q75': float(np.quantile(x, .75)), 'q95': float(np.quantile(x, .95))}


def _load_artifact(root: Path, base: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / STAGE3 / 'models' / base / 'model_artifact.joblib'
    if not path.is_file():
        raise FileNotFoundError(f'Stage 3 artifact not found: {path}')
    artifact = load_artifact(path)
    return artifact, {'path': str(path), 'sha256': sha256(path),
                      'metadata': artifact['metadata']}


def _g_production(artifact: dict[str, Any], prod: pd.DataFrame) -> np.ndarray:
    own = prod.has_own_prior_vtd.astype(bool).to_numpy()
    g = np.zeros(len(prod), float)
    if own.any():
        g[own] = artifact['estimator'].predict(
            artifact['preprocessor'].transform(prod.loc[own]))
    return g


def _apply_variants(prod: pd.DataFrame, artifact: dict[str, Any],
                    dt_train: float) -> pd.DataFrame:
    own = prod.has_own_prior_vtd.astype(bool).to_numpy()
    last = prod.last_mm.to_numpy(float)
    nominal = prod.nominal_thickness_mm.to_numpy(float)
    dt = prod.dt_years.to_numpy(float)
    g = _g_production(artifact, prod)
    ratio = np.where(own & np.isfinite(dt) & (dt_train > 0), dt / dt_train, 0.0)
    fallback = prod.s1b_prediction.to_numpy(float)
    frame = prod[['id', 'segment_id', 'last_mm', 'dt_years',
                  'has_own_prior_vtd', 's1b_prediction',
                  'nominal_thickness_mm']].copy()
    frame['model_base'] = artifact['metadata']['model_name']
    frame['g_raw'] = g
    frame['dt_train_years'] = dt_train
    frame['dt_ratio_to_train'] = ratio
    for name, multiplier in [('correction', 1.0),
                             ('time_scaled', ratio),
                             ('conservative_0_25', 0.25)]:
        raw = last + multiplier * g
        raw = np.where(own, raw, fallback)
        clipped = np.clip(raw, 0.0, nominal)
        frame[name + '_raw'] = raw
        frame[name] = clipped
        frame[name + '_clipped'] = (raw < 0) | (raw > nominal)
        frame[name + '_above_last_raw'] = raw > last + 1e-12
        frame[name + '_above_last'] = clipped > last + 1e-12
        frame[name + '_above_nominal_raw'] = raw > nominal + 1e-12
        frame[name + '_above_nominal'] = clipped > nominal + 1e-12
        frame[name + '_abs_delta_last'] = np.abs(clipped - last)
        frame[name + '_abs_delta_last_raw'] = np.abs(raw - last)
        frame[name + '_fallback'] = ~own
    frame['fallback'] = ~own
    return frame


def _variant_summary(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    out = {}
    for name in VARIANTS:
        out[name] = {
            'model_base': model,
            'abs_delta_last': _stats(frame[name + '_abs_delta_last']),
            'abs_delta_last_raw': _stats(frame[name + '_abs_delta_last_raw']),
            'above_last_raw': int(frame[name + '_above_last_raw'].sum()),
            'above_last_final': int(frame[name + '_above_last'].sum()),
            'above_nominal_raw': int(frame[name + '_above_nominal_raw'].sum()),
            'above_nominal_final': int(frame[name + '_above_nominal'].sum()),
            'clipping_count': int(frame[name + '_clipped'].sum()),
            'fallback_rows': int(frame[name + '_fallback'].sum()),
            'mean_raw': float(frame[name + '_raw'].mean()),
            'mean_final': float(frame[name].mean()),
            'std_prediction': float(frame[name].std(ddof=0)),
            'std_abs_delta_last': float(frame[name + '_abs_delta_last'].std(ddof=0)),
            'mean_shift_after_clipping': float(frame[name].mean()
                                               - frame[name + '_raw'].mean()),
            'min_final': float(frame[name].min()),
            'max_final': float(frame[name].max()),
        }
    return out


def _difference(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    d = (a - b).to_numpy(float)
    return {'count': int(len(d)), 'mean': float(d.mean()),
            'median': float(np.median(d)), 'std': float(d.std(ddof=0)),
            'min': float(d.min()), 'max': float(d.max()),
            'q05': float(np.quantile(d, .05)),
            'q95': float(np.quantile(d, .95)),
            'nonzero_rows': int((np.abs(d) > 1e-12).sum())}


def _thin_diagnostics(frame: pd.DataFrame, n: int = 53) -> pd.DataFrame:
    order = frame.sort_values(['last_mm', 'segment_id'], kind='mergesort')
    thin = order.head(n).copy()
    thin['thin_rank'] = np.arange(1, len(thin) + 1)
    for name in VARIANTS:
        thin[name + '_delta_vs_last'] = thin[name] - thin.last_mm
    thin['s1a_top_reference'] = True
    return thin


def _tail_metrics(oof: pd.DataFrame, pred: str) -> dict[str, Any]:
    return _block_tail(oof, pred)


def _oof_table(oof: pd.DataFrame, dt_train: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Stage 3 OOF predictions are clipped.  For time-ratio 1 (all historical
    # rows) correction == time_scaled; conservative is a fixed shrinkage.
    ratio = (oof.dt_years.to_numpy(float) / dt_train)
    if not np.allclose(ratio, 1.0, atol=1e-12):
        raise ValueError('Not all Stage 3 OOF rows have dt == dt_train')
    rows, details = [], {}
    frame = oof.copy()
    for base in MODEL_BASES:
        if base not in frame:
            continue
        frame[base + '_correction'] = frame[base]
        frame[base + '_time_scaled'] = frame[base]
        frame[base + '_conservative_0_25'] = np.clip(
            frame.last_mm + 0.25 * (frame[base] - frame.last_mm), 0,
            frame.nominal_thickness_mm)
        for name in VARIANTS:
            col = f'{base}_{name}'
            rec = evaluate_predictions(frame, col)
            tail = _tail_metrics(frame, col)
            details[col] = {'pooled': rec['pooled'], 'tail': tail}
            rows.append({
                'model_base': base, 'variant': name, 'n': rec['pooled']['n'],
                'mae_mm': rec['pooled']['mae_mm'],
                'rmse_mm': rec['pooled']['rmse_mm'],
                'bias_mm': rec['pooled']['bias_mm'],
                'pearson_r': rec['pooled']['pearson_r'],
                'pearson_squared': rec['pooled']['pearson_squared'],
                'r2': rec['pooled']['r2'],
                'global_top10_recall': tail['global_top10_recall'],
                'global_lower_tail_mae_mm': tail['global_lower_tail_mae_mm'],
                'per_block_top10_recall': tail['per_block_top10_recall'],
                'per_block_lower_tail_mae_mm': tail['per_block_lower_tail_mae_mm'],
            })
    table = pd.DataFrame(rows)
    return table, details


def run_stage5(root: Path = BASE_DIR, data_root: Path | None = None,
               output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/targeted_cycle/stage_5_temporal').resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    data = RepositoryData(data_root)
    fit = s1_fit_rows(s1_training_table(data))
    prod = s1_production_table(data)
    dt_values = fit.dt_years.round(12).value_counts()
    if len(dt_values) != 1:
        raise RuntimeError(f'Expected one common train dt, got {dt_values.to_dict()}')
    dt_train = float(dt_values.index[0])
    dt_own = prod.loc[prod.has_own_prior_vtd.astype(bool), 'dt_years']
    if not np.isfinite(dt_own).all() or (dt_own <= 0).any():
        raise ValueError('Invalid production dt')
    dt_counts = {str(round(float(k), 12)): int(v)
                 for k, v in dt_own.round(12).value_counts().items()}
    artifacts = {}
    references = {}
    for base in MODEL_BASES:
        artifact, ref = _load_artifact(root, base)
        artifacts[base] = artifact
        references[base] = ref
    production = {}
    summaries = {}
    for base in MODEL_BASES:
        frame = _apply_variants(prod, artifacts[base], dt_train)
        production[base] = frame
        summaries[base] = _variant_summary(frame, base)
        write_csv(frame, output / 'production' / f'{base}_diagnostics.csv',
                  fmt='%.10f')
        for name in VARIANTS:
            pred = frame[['id', 'segment_id', name]].rename(
                columns={name: 'thickness_pred'})
            write_csv(pred, output / 'production' / f'{base}_{name}_predictions.csv',
                      fmt='%.10f')
    # Differences among time interpretations (global Ridge main; target appendix).
    diffs = {}
    for base in MODEL_BASES:
        frame = production[base]
        diffs[base] = {
            'time_scaled_minus_correction': _difference(
                frame.time_scaled, frame.correction),
            'conservative_minus_correction': _difference(
                frame.conservative_0_25, frame.correction),
            'time_scaled_minus_conservative': _difference(
                frame.time_scaled, frame.conservative_0_25),
        }
    for base in MODEL_BASES:
        for name in VARIANTS:
            summaries[base][name]['max_abs_difference_vs_correction'] = float(
                np.max(np.abs(production[base][name].to_numpy(float)
                              - production[base]['correction'].to_numpy(float))))
    # Thin-tail behavior.
    thin = _thin_diagnostics(production['global_ridge_geometry'], n=53)
    write_csv(thin, output / 'thin_segments_global_ridge.csv', fmt='%.10f')
    # OOF comparison from Stage 3 saved predictions.
    oof_path = root / STAGE3 / 'oof_predictions.csv'
    if not oof_path.is_file():
        raise FileNotFoundError(oof_path)
    oof = pd.read_csv(oof_path)
    oof = oof.merge(fit[['id', 'segment_id', 'dt_years']],
                    on=['id', 'segment_id'], how='left', validate='one_to_one')
    if oof.dt_years.isna().any():
        raise ValueError('Stage 3 OOF missing dt_years')
    oof_table, oof_details = _oof_table(oof, dt_train)
    write_csv(oof_table, output / 'oof_temporal_metrics.csv', fmt='%.10f')
    write_json(output / 'oof_temporal_details.json', json_safe(oof_details))
    # Proposal: reference + at most two challengers, no winner.
    proposal = {
        'status': 'awaiting_user_freeze_choice',
        'reference': 's1a',
        'challengers': [
            {
                'id': 'global_ridge_geometry_correction',
                'base_model': 'global_ridge_geometry',
                'temporal_assumption': 'h_last + g(X)',
                'model_path': references['global_ridge_geometry']['path'],
                'model_sha256': references['global_ridge_geometry']['sha256'],
            },
            {
                'id': 'global_ridge_geometry_time_scaled',
                'base_model': 'global_ridge_geometry',
                'temporal_assumption': 'h_last + (dt_target/dt_train)*g(X)',
                'model_path': references['global_ridge_geometry']['path'],
                'model_sha256': references['global_ridge_geometry']['sha256'],
            },
        ],
        'sensitivity_not_frozen': [
            'conservative_0_25 (fixed shrinkage; not selected by OOF)',
            'target_ridge_geometry variants (local fit; no clear advantage in Stage 4)',
        ],
        'selection_basis': ('Common train dt permits the time-scaled expression; OOF cannot '
                            'distinguish correction from time-scaled because all historical '
                            'dt equal. No temporal winner is declared.'),
        'no_unconditional_prediction_gt_last_ban': True,
        'bounds': '[0, nominal_thickness_mm]',
    }
    write_json(output / 'proposal.json', json_safe(proposal))
    write_json(output / 'dt_audit.json', json_safe({
        'train_rows': int(len(fit)), 'dt_train_years': dt_train,
        'train_unique_dt': {str(k): int(v) for k, v in dt_values.items()},
        'train_dates': sorted(str(pd.Timestamp(v).date())
                              for v in fit.prediction_date.unique()),
        'production_own_rows': int(len(dt_own)),
        'production_dt_counts': dt_counts,
        'fallback_rows': int((~prod.has_own_prior_vtd.astype(bool)).sum()),
        'dt_train_common': True,
        'note': 'dt_target is computed from each segment own last VTD date; fallback rows have no own dt.',
    }))
    write_json(output / 'production_summary.json', json_safe(summaries))
    write_json(output / 'differences.json', json_safe(diffs))
    report = _report(output, dt_train, dt_counts, summaries, diffs, thin,
                     oof_table, proposal)
    report_path = root / 'experiments/targeted_cycle/stage_5_report_ru.md'
    report_path.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
    write_manifest(output / 'run_manifest.json', root=root, data=data, output=output,
                   stage='Stage_5_temporal', started=started,
                   extra={'data_root': str(data_root), 'dt_train': dt_train,
                          'dt_counts': dt_counts, 'proposal': proposal,
                          'report': str(report_path), 'plan_version': 2,
                          'stage_code_sha256': {
                              'src/ml/targeted_s1_temporal.py':
                                  sha256(root / 'src/ml/targeted_s1_temporal.py')},
                          'no_model_fit': True},
                   input_paths=[root / STAGE3 / 'models/global_ridge_geometry/model_artifact.joblib',
                                root / STAGE3 / 'models/target_ridge_geometry/model_artifact.joblib',
                                root / STAGE3 / 'oof_predictions.csv'])
    print(f'PASS Stage 5: {output}', flush=True)
    print(f'  dt_train={dt_train:.12f}, production dt={dt_counts}', flush=True)
    return {'dt_train': dt_train, 'dt_counts': dt_counts,
            'summaries': summaries, 'differences': diffs,
            'proposal': proposal, 'output': str(output), 'report': str(report_path)}


def _fmt(value, digits=6):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return 'n/a' if not np.isfinite(value) else f'{value:.{digits}f}'
    return str(value)


def _report(output: Path, dt_train: float, dt_counts: dict[str, int],
            summaries: dict[str, Any], diffs: dict[str, Any],
            thin: pd.DataFrame, oof: pd.DataFrame,
            proposal: dict[str, Any]) -> Path:
    lines = [
        '# Stage 5 — temporal interpretations of the fixed S1 Ridge', '',
        f"Generated: `{utcnow()}`. No model fit, no parameter tuning. "
        'Stage 3/4 artifacts are loaded read-only.', '',
        '## 1. Actual horizons', '',
        f"- training rows: **357**, unique train dt: **{dt_train:.12f} years**.",
        f"- production own-prior segments: **{sum(dt_counts.values())}**. "
        f"dt counts: `{dt_counts}`.",
        '- production dt is computed from each segment own last VTD date; 4 fallback rows have no own dt.',
        '- common train dt is true, so `dt_target/dt_train` is defined; the OOF cannot identify a time law.',
        '', '## 2. Historical OOF temporal variants', '',
        'Stage 3 OOF all have the same dt. Therefore correction and time-scaled are numerically identical; '
        'conservative is a fixed 0.25 shrinkage. The OOF table cannot choose between 1 and 2.', '',
        '| Base | Variant | n | MAE | bias | Pearson r | Pearson r² | R² | top10 global | lower-tail global MAE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in oof.itertuples():
        lines.append(f"| {row.model_base} | {row.variant} | {row.n} | {_fmt(row.mae_mm)} | "
                     f"{_fmt(row.bias_mm)} | {_fmt(row.pearson_r)} | "
                     f"{_fmt(row.pearson_squared)} | {_fmt(row.r2)} | "
                     f"{_fmt(row.global_top10_recall)} | {_fmt(row.global_lower_tail_mae_mm)} |")
    lines += ['', 'The correction and time-scaled OOF predictions are identical by construction; '
              'this is not evidence that either time law is correct.', '',
              '## 3. Production variants (global and target Ridge)', '',
              '| Base | Variant | mean | std pred | std |pred-last| | min | max | above last raw | above nominal raw | clipping | fallback | mean shift after clip |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for base, by_variant in summaries.items():
        for name in VARIANTS:
            s = by_variant[name]
            lines.append(f"| {base} | {name} | {_fmt(s['mean_final'])} | "
                         f"{_fmt(s['std_prediction'])} | {_fmt(s['std_abs_delta_last'])} | "
                         f"{_fmt(s['min_final'])} | {_fmt(s['max_final'])} | "
                         f"{s['above_last_raw']} | {s['above_nominal_raw']} | "
                         f"{s['clipping_count']} | {s['fallback_rows']} | "
                         f"{_fmt(s['mean_shift_after_clipping'])} |")
    lines += ['', 'No unconditional ban on prediction > last was applied. Raw predictions and clipped '
              'predictions are both saved; only the established `[0, nominal]` bounds are enforced.',
              '', '### |prediction − last| distribution', '',
              '| Base | Variant | mean | median | q05 | q95 | max |',
              '|---|---|---:|---:|---:|---:|---:|']
    for base, by_variant in summaries.items():
        for name in VARIANTS:
            s = by_variant[name]['abs_delta_last']
            lines.append(f"| {base} | {name} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
                         f"{_fmt(s['q05'])} | {_fmt(s['q95'])} | {_fmt(s['max'])} |")
    lines += ['', '## 4. Differences between temporal variants', '',
              '| Base | Comparison | mean | median | q05 | q95 | max | nonzero rows |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for base, pairs in diffs.items():
        for key, d in pairs.items():
            lines.append(f"| {base} | {key} | {_fmt(d['mean'])} | {_fmt(d['median'])} | "
                         f"{_fmt(d['q05'])} | {_fmt(d['q95'])} | {_fmt(d['max'])} | "
                         f"{d['nonzero_rows']} |")
    lines += ['', '### Maximum absolute difference vs correction', '',
              '| Base | Variant | max abs difference, mm |', '|---|---|---:|']
    for base, by_variant in summaries.items():
        for name in VARIANTS:
            lines.append(f"| {base} | {name} | "
                         f"{_fmt(by_variant[name]['max_abs_difference_vs_correction'])} |")
    lines += ['', '## 5. Thin-tail behavior', '',
              f"Thin set: **{len(thin)}** lowest `last_mm` segments; tie order by segment_id.", '',
              '| Base variant | Thin mean pred | Thin mean Δ vs last | Thin above last |',
              '|---|---:|---:|---:|']
    for name in VARIANTS:
        lines.append(f"| global_ridge_geometry_{name} | "
                     f"{_fmt(float(thin[name].mean()))} | "
                     f"{_fmt(float(thin[name + '_delta_vs_last'].mean()))} | "
                     f"{int((thin[name] > thin.last_mm + 1e-12).sum())} |")
    lines += ['', 'Thin-tail file: `thin_segments_global_ridge.csv`. Fallback rows are explicitly marked; '
              'Ridge is not applied to them.', '',
              '## 6. Proposal for freeze choice', '',
              f"- Reference: **{proposal['reference']}**.",
              '- Challengers (max 2):']
    for c in proposal['challengers']:
        lines.append(f"  - `{c['id']}`: `{c['temporal_assumption']}` on `{c['base_model']}`")
    lines += [f"- Sensitivity, not frozen: {proposal['sensitivity_not_frozen']}",
              f"- Basis: {proposal['selection_basis']}",
              '- OOF does not select a time law; the proposal is for our explicit choice, not a winner claim.',
              '', '## 7. Artifacts and commands', '',
              '```text',
              'experiments/targeted_cycle/stage_5_temporal/',
              '  dt_audit.json, oof_temporal_metrics.csv, oof_temporal_details.json',
              '  production/<base>_diagnostics.csv, production/<base>_<variant>_predictions.csv',
              '  production_summary.json, differences.json, thin_segments_global_ridge.csv',
              '  proposal.json, run_manifest.json',
              '```',
              '', '```bash',
              'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s1_temporal --root . --data-root C:/Users/Администратор/Documents/PipeTwin_next/experiments/next_cycle/data_generation --output-root experiments/targeted_cycle/stage_5_temporal',
              '```',
              '', '## 8. Tests', '',
              '- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_5_temporal -v`: 6 tests, OK.',
              '- Full Stage 2–5 test run: 26 tests, OK.',
              '- Проверены actual dt, fallback без Ridge, bounds, raw/clipped columns, выше-last без запрета, OOF identity correction/time-scaled и proposal <= 2.',
              '', '## 9. Limitations', '',
              '- Only one historical dt; OOF is identical for correction and time-scaled.',
              '- Historical campaign is one campaign; target horizon differs.',
              '- Production graph is not validation; target labels are unavailable.',
              '- Stage 5 does not choose a winner and does not run Stage 6.',
              '']
    path = output / 'stage_5_report_ru.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/targeted_cycle/stage_5_temporal')
    args = parser.parse_args(argv)
    run_stage5(args.root, args.data_root, args.output_root)


if __name__ == '__main__':
    main()


def recalc_stage5(output_root: Path | None = None) -> dict[str, Any]:
    """Recompute summary/report from saved diagnostics; predictions untouched."""
    output = Path(output_root or Path('experiments/targeted_cycle/stage_5_temporal')).resolve()
    dt = json.loads((output / 'dt_audit.json').read_text(encoding='utf-8'))
    summaries = {}
    for base in MODEL_BASES:
        frame = pd.read_csv(output / 'production' / f'{base}_diagnostics.csv')
        summaries[base] = _variant_summary(frame, base)
        for name in VARIANTS:
            summaries[base][name]['max_abs_difference_vs_correction'] = float(
                np.max(np.abs(frame[name].to_numpy(float)
                              - frame['correction'].to_numpy(float))))
    diffs = {}
    for base in MODEL_BASES:
        frame = pd.read_csv(output / 'production' / f'{base}_diagnostics.csv')
        diffs[base] = {
            'time_scaled_minus_correction': _difference(frame.time_scaled, frame.correction),
            'conservative_minus_correction': _difference(
                frame.conservative_0_25, frame.correction),
            'time_scaled_minus_conservative': _difference(
                frame.time_scaled, frame.conservative_0_25),
        }
    write_json(output / 'production_summary.json', json_safe(summaries))
    write_json(output / 'differences.json', json_safe(diffs))
    oof = pd.read_csv(output / 'oof_temporal_metrics.csv')
    thin = pd.read_csv(output / 'thin_segments_global_ridge.csv')
    proposal = json.loads((output / 'proposal.json').read_text(encoding='utf-8'))
    report = _report(output, float(dt['dt_train_years']), dt['production_dt_counts'],
                     summaries, diffs, thin, oof, proposal)
    top = output.parent / 'stage_5_report_ru.md'
    top.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
    write_json(output / 'recalc.json', {
        'method': 'recalculated from saved production diagnostics; no model fit, no prediction change',
        'std_policy': 'std(prediction) and std(|prediction-last|) reported separately',
        'max_abs_difference': {base: {name: summaries[base][name]['max_abs_difference_vs_correction']
                                     for name in VARIANTS} for base in MODEL_BASES},
        'rows': {base: int(len(pd.read_csv(output / 'production' / f'{base}_diagnostics.csv')))
                 for base in MODEL_BASES},
    })
    return {'summaries': summaries, 'differences': diffs, 'report': str(top)}

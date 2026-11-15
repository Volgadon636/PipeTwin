"""External evaluator for frozen PipeTwin candidate pairs.

Truth is supplied separately and is never merged into inference features.  The
evaluator does not fit, tune or rewrite predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.storage import sha256, write_json
from src.ml.next_common import evaluate_predictions
from src.ml.targeted_audit import tail_metrics

CASE_IDS = {1250022871: 'S1', 1250009583: 'S2'}
CASE_DATES = {1250022871: pd.Timestamp('2023-12-01'),
              1250009583: pd.Timestamp('2022-07-01')}
EXPECTED = {1250022871: 525, 1250009583: 1475}


def _json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is pd.NaT:
        return None
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


def _validate_predictions(frame: pd.DataFrame) -> None:
    if list(frame.columns) != ['id', 'segment_id', 'thickness_pred']:
        raise ValueError('Predictions must have exactly id,segment_id,thickness_pred')
    if frame.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate prediction keys')
    if len(frame) != 2000:
        raise ValueError(f'Expected 2000 predictions, got {len(frame)}')
    for pid, expected in EXPECTED.items():
        n = int(frame.id.eq(pid).sum())
        if n != expected:
            raise ValueError(f'Expected {expected} {CASE_IDS[pid]} rows, got {n}')
    if not np.isfinite(frame.thickness_pred.to_numpy(float)).all():
        raise ValueError('Nonfinite predictions')


def _validate_truth(truth: pd.DataFrame) -> pd.DataFrame:
    required = {'id', 'segment_id', 'thickness_measured'}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f'Truth is missing columns: {sorted(missing)}')
    frame = truth.copy()
    frame['id'] = pd.to_numeric(frame.id, errors='raise').astype('int64')
    frame['segment_id'] = pd.to_numeric(frame.segment_id, errors='raise').astype('int64')
    frame['thickness_measured'] = pd.to_numeric(frame.thickness_measured, errors='raise')
    if frame.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate truth keys')
    if not np.isfinite(frame.thickness_measured).all():
        raise ValueError('Nonfinite truth thickness')
    if (frame.thickness_measured <= 0).any():
        raise ValueError('Truth thickness must be positive mm')
    if not frame.id.isin(CASE_IDS).all():
        raise ValueError('Truth contains an unknown case id')
    if 'date' in frame:
        dates = pd.to_datetime(frame.date, format='ISO8601', errors='raise')
        for pid, expected_date in CASE_DATES.items():
            part = dates[frame.id.eq(pid)]
            if len(part) and not part.eq(expected_date).all():
                raise ValueError(f'{CASE_IDS[pid]} truth date does not match {expected_date.date()}')
        frame['date'] = dates
    return frame


def evaluate(predictions_path: Path, truth_path: Path, output_root: Path) -> dict[str, Any]:
    predictions_path, truth_path = Path(predictions_path), Path(truth_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(predictions_path)
    _validate_predictions(pred)
    truth = _validate_truth(pd.read_csv(truth_path))
    expected_keys = set(zip(pred.id, pred.segment_id))
    truth_keys = set(zip(truth.id, truth.segment_id))
    if expected_keys != truth_keys:
        raise ValueError(f'Truth keys differ: missing={len(expected_keys-truth_keys)}, '
                         f'extra={len(truth_keys-expected_keys)}')
    merged = pred.merge(truth, on=['id', 'segment_id'], how='inner', validate='one_to_one')
    if len(merged) != 2000:
        raise ValueError(f'Joined rows {len(merged)} != 2000')
    merged['case'] = merged.id.map(CASE_IDS)
    metrics = {}
    per_case_rows = []
    for case, part in merged.groupby('case', sort=True):
        rec = evaluate_predictions(part, 'thickness_pred', truth='thickness_measured',
                                   section='id')
        tail = tail_metrics(part, 'thickness_pred', truth='thickness_measured',
                            section='id')
        metrics[case] = {'pooled': rec['pooled'], 'macro': rec['macro'],
                         'per_pipe': rec['per_section'], 'tail': tail}
        per_case_rows.append({'case': case, **rec['pooled'],
                              'global_top10_recall': tail['global_top10_recall'],
                              'global_lower_tail_mae_mm': tail['global_lower_tail_mae_mm']})
    overall = evaluate_predictions(merged, 'thickness_pred', truth='thickness_measured',
                                   section='id')
    # Error table and baseline bins.
    merged['error_mm'] = merged.thickness_pred - merged.thickness_measured
    merged['abs_error_mm'] = merged.error_mm.abs()
    bins = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    merged['baseline_bin'] = pd.cut(merged.thickness_pred, bins=bins, right=False,
                                    include_lowest=True)
    bin_rows = []
    for (case, bin_label), part in merged.groupby(['case', 'baseline_bin'], observed=True):
        bin_rows.append({'case': case, 'prediction_bin': str(bin_label),
                         'n': int(len(part)), 'mae_mm': float(part.abs_error_mm.mean()),
                         'bias_mm': float(part.error_mm.mean())})
    # Constant / near-constant diagnostics per case.
    const = {}
    for case, part in merged.groupby('case'):
        v = part.thickness_pred.to_numpy(float)
        const[case] = {'n': int(len(v)), 'nunique': int(len(np.unique(v))),
                       'std_mm': float(v.std(ddof=0)), 'min': float(v.min()),
                       'max': float(v.max()),
                       'is_constant': bool(np.ptp(v) <= 1e-12),
                       'near_constant': bool(v.std(ddof=0) <= 1e-6)}
    payload = {
        'predictions_path': str(predictions_path),
        'predictions_sha256': sha256(predictions_path),
        'truth_path': str(truth_path),
        'truth_sha256': sha256(truth_path),
        'rows': int(len(merged)),
        'case_counts': {k: int(v) for k, v in merged.case.value_counts().items()},
        'keys_match': True,
        'finite_predictions': True,
        'truth_unit': 'mm',
        'dates_checked': 'truth.date must match case dates when supplied',
        'metrics': _json(metrics),
        'overall': _json({'pooled': overall['pooled'], 'macro': overall['macro'],
                          'per_pipe': overall['per_section']}),
        'per_case': _json(per_case_rows),
        'baseline_bins': _json(bin_rows),
        'constant_diagnostics': _json(const),
        'no_fit': True,
        'truth_usage': 'evaluation only; separate from inference features',
    }
    write_json(output / 'external_metrics.json', payload)
    merged[['id', 'segment_id', 'case', 'thickness_pred', 'thickness_measured',
            'error_mm', 'abs_error_mm']].to_csv(
        output / 'external_errors.csv', index=False, encoding='utf-8',
        lineterminator='\n', float_format='%.10f')
    lines = [
        '# External evaluation', '',
        f"Predictions: `{predictions_path}`; truth: `{truth_path}`.",
        '',
        '| Case | n | MAE | RMSE | bias | Pearson r | Pearson r² | R² | top10 recall | lower-tail MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for case in ('S1', 'S2'):
        r = metrics[case]['pooled']
        tail = metrics[case]['tail']
        lines.append(f"| {case} | {r['n']} | {r['mae_mm']:.6f} | {r['rmse_mm']:.6f} | "
                     f"{r['bias_mm']:+.6f} | {r['pearson_r'] if r['pearson_r'] is not None else 'n/a'} | "
                     f"{r['pearson_squared'] if r['pearson_squared'] is not None else 'n/a'} | "
                     f"{r['r2'] if r['r2'] is not None else 'n/a'} | "
                     f"{tail['global_top10_recall']:.6f} | {tail['global_lower_tail_mae_mm']:.6f} |")
    lines += ['', 'No fit, tuning or prediction rewrite was performed. '
              'Truth is separate from inference features.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate(args.predictions, args.truth, args.output_root)


if __name__ == '__main__':
    main()

"""Test scenario-specific residual intervals using nested section holdouts.

Only labels before each production cutoff enter the calibration experiment.
The final point forecasts are read unchanged from the verified production CSV.
These are empirically tested CV residual intervals, not guaranteed conformal
coverage under independent samples: sections share one parent pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.ml.data import RepositoryData
from src.ml.next_common import add_local_prior_features
from src.ml.production_models import (
    FORECAST_KEYS, H31_PARAMETERS, cold_training, fit_s1_models, fit_s2_models,
    predict_s1_models, predict_s2_models, s1_prediction_frame, s1_training,
)
from src.ml.s1_contract import S1_CUTOFF, S1_ID, S2_ID, enrich_s1
from src.ml.targeted_s2 import S2_CUTOFF
from src.ml.windows import make_snapshot

FRONTEND = ROOT / 'frontend'
DEFAULT_OUTPUT = FRONTEND / 'calibration/intervals'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_radius(errors, alpha=.2):
    values = np.asarray(errors, dtype=float)
    if not 0 < alpha < 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative residuals and 0 < alpha < 1')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        raise ValueError('Too few calibration observations for a finite interval')
    return float(np.sort(values)[rank - 1])


def bounded_interval(prediction, nominal, radius):
    pred, upper = np.asarray(prediction, float), np.asarray(nominal, float)
    if radius < 0 or not np.isfinite(radius) or not np.isfinite(np.r_[pred, upper]).all():
        raise ValueError('Invalid interval inputs')
    if (pred < 0).any() or (pred > upper).any() or (upper <= 0).any():
        raise ValueError('Point forecast must satisfy its nominal bounds')
    return np.maximum(0., pred - radius), np.minimum(upper, pred + radius)


def interval_metrics(frame):
    covered = frame.next_mm.between(frame.lower_mm, frame.upper_mm)
    return {'n': len(frame), 'coverage': float(covered.mean()),
            'covered': int(covered.sum()),
            'mean_interval_width_mm': float((frame.upper_mm - frame.lower_mm).mean())}


def nested_intervals(train, predict_fold, alpha):
    """Outer labels never enter fitting or inner residual calibration."""
    if train.id.nunique() < 3 or train.duplicated(FORECAST_KEYS).any():
        raise ValueError('Need at least three sections with unique historical labels')
    outer_rows, audits = [], []
    for held_id, held in train.groupby('id', sort=True):
        development = train.loc[train.id.ne(held_id)]
        inner_rows = []
        for inner_id, calibration in development.groupby('id', sort=True):
            fit = development.loc[development.id.ne(inner_id)]
            inner_rows.append(predict_fold(fit, calibration))
        inner = pd.concat(inner_rows, ignore_index=True)
        q = residual_radius((inner.next_mm - inner.prediction_mm).abs(), alpha)
        forecast = predict_fold(development, held)
        lower, upper = bounded_interval(forecast.prediction_mm, forecast.nominal_thickness_mm, q)
        outer_rows.append(forecast.assign(lower_mm=lower, upper_mm=upper,
                                         calibration_radius_mm=q, outer_held_id=int(held_id)))
        audits.append({'held_id': int(held_id),
                       'fit_ids': sorted(int(pid) for pid in development.id.unique()),
                       'calibration_ids': sorted(int(pid) for pid in inner.id.unique()),
                       'calibration_n': len(inner), 'radius_mm': q,
                       'calibration_date_max': str(inner.prediction_date.max().date())})
    out = pd.concat(outer_rows, ignore_index=True)
    if len(out) != len(train) or out.duplicated(FORECAST_KEYS).any():
        raise ValueError('Nested validation changed historical keys')
    return out, audits


def s1_inputs(data):
    raw = data.read('data/cleaned/vtd')
    geometry = data.read('processed/segments_grid')
    train = s1_training(data, raw, geometry)
    frames = {}
    for (pid, cutoff), _ in train.groupby(['id', 'prediction_date'], sort=True):
        snapshot, _ = make_snapshot(data, {'id': int(pid), 'prediction_date': str(cutoff.date()),
                                            'case': 'between_vtd'})
        enriched = enrich_s1(snapshot, data.read('processed/vtd_by_segment'),
                             data.read('processed/profile_by_segment'), cutoff)
        enriched = add_local_prior_features(enriched.assign(last_mm=enriched.s1a_prediction))
        frames[pid, cutoff], _ = s1_prediction_frame(enriched, raw, geometry.loc[geometry.id.eq(pid)])
    fits = {}

    def predict(fit, held):
        cache_key = tuple(sorted(int(pid) for pid in fit.id.unique()))
        if cache_key not in fits:
            fits[cache_key] = fit_s1_models(fit)
        parts = []
        for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
            frame = frames[key]
            forecast = predict_s1_models(fits[cache_key], frame)
            output = frame[FORECAST_KEYS + ['nominal_thickness_mm']].merge(
                forecast[FORECAST_KEYS + ['H18']], on=FORECAST_KEYS, validate='one_to_one')
            parts.append(labels[FORECAST_KEYS + ['next_mm']].merge(
                output.rename(columns={'H18': 'prediction_mm'}), on=FORECAST_KEYS, validate='one_to_one'))
        return pd.concat(parts, ignore_index=True)

    return train, predict


def s2_inputs(data):
    train, grids, intervals = cold_training(data)
    fits = {}

    def predict(fit, held):
        cache_key = tuple(sorted(int(pid) for pid in fit.id.unique()))
        if cache_key not in fits:
            fits[cache_key] = fit_s2_models(fit, intervals, H31_PARAMETERS)
        parts = []
        for key, labels in held.groupby(['id', 'prediction_date'], sort=True):
            forecast = predict_s2_models(fits[cache_key], grids[key], intervals)
            output = grids[key][FORECAST_KEYS + ['nominal_thickness_mm']].merge(
                forecast[FORECAST_KEYS + ['H31']], on=FORECAST_KEYS, validate='one_to_one')
            parts.append(labels[FORECAST_KEYS + ['next_mm']].merge(
                output.rename(columns={'H31': 'prediction_mm'}), on=FORECAST_KEYS, validate='one_to_one'))
        return pd.concat(parts, ignore_index=True)

    return train, predict


def run(data_root, output, alpha=.2):
    output = output.resolve()
    if not output.is_relative_to(FRONTEND.resolve()) or output.exists():
        raise ValueError('Choose a new output directory inside frontend')
    data = RepositoryData(data_root)
    production = ROOT / 'production/h18_h31_verified/result.csv'
    original_hash = digest(production)
    final = pd.read_csv(production)
    grid = data.read('processed/target_features')
    summaries, target_parts, validation_parts, audit = {}, [], [], {}
    for case, pid, cutoff, method, prepare in [
            ('S1', S1_ID, S1_CUTOFF, 'H18', s1_inputs),
            ('S2', S2_ID, S2_CUTOFF, 'H31', s2_inputs)]:
        train, predict = prepare(data)
        if not train.prediction_date.lt(cutoff).all() or train.id.eq(S2_ID).any():
            raise ValueError('Calibration must respect production cutoff and S2 exclusion')
        held, folds = nested_intervals(train, predict, alpha)
        q = residual_radius((held.next_mm - held.prediction_mm).abs(), alpha)
        rows = final.loc[final.id.eq(pid)].merge(
            grid.loc[grid.id.eq(pid), ['id', 'segment_id', 'nominal_thickness_mm']],
            on=['id', 'segment_id'], validate='one_to_one')
        lower, upper = bounded_interval(rows.thickness_pred, rows.nominal_thickness_mm, q)
        target_parts.append(rows[['id', 'segment_id', 'thickness_pred']].assign(
            lower_mm=lower, upper_mm=upper, case=case, model=method,
            prediction_date=str(cutoff.date())))
        validation_parts.append(held.assign(case=case, model=method))
        summaries[case] = {'model': method, 'cutoff': str(cutoff.date()),
                           'calibration_n': len(held), 'calibration_radius_mm': q,
                           'calibration_dates': sorted(str(date.date()) for date in train.prediction_date.unique()),
                           'section_count': int(train.id.nunique()),
                           'parent_count': int(train.main_pipeline_id.nunique()),
                           'nested_validation': interval_metrics(held),
                           'by_profile': [{'id': int(key), **interval_metrics(part)}
                                          for key, part in held.groupby('id', sort=True)]}
        audit[case] = folds
        print(case, json.dumps(summaries[case], ensure_ascii=False), flush=True)
    if digest(production) != original_hash:
        raise ValueError('Production forecast changed during interval experiment')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.intervals-', dir=output.parent) as temp:
        stage = Path(temp) / 'output'
        stage.mkdir()
        pd.concat(target_parts, ignore_index=True).to_csv(stage / 'target_intervals.csv', index=False,
                                                         float_format='%.15g')
        pd.concat(validation_parts, ignore_index=True).to_csv(stage / 'nested_validation.csv', index=False,
                                                             float_format='%.15g')
        metadata = {'method': 'nested_section_cv_absolute_residual', 'nominal_coverage': 1 - alpha,
                    'alpha': alpha, 'seed': 17, 'cases': summaries, 'folds': audit,
                    'limitations': 'Section holdouts share one parent pipeline. Coverage is empirical, '
                                   'not an independent-parent or forward-time guarantee. Target labels are unavailable.',
                    'result_sha256': original_hash,
                    'preprocessing_manifest_sha256': digest(data.root / 'reports/run_manifest.json'),
                    'input_sha256': data.hashes,
                    'code_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                                    for path in [*sorted((ROOT / 'src').rglob('*.py')), Path(__file__)]},
                    'output_sha256': {name: digest(stage / name)
                                      for name in ['target_intervals.csv', 'nested_validation.csv']}}
        (stage / 'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + '\n', encoding='utf-8')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'scratch/odo_alignment_verified')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--alpha', type=float, default=.2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_root, args.output, args.alpha)


if __name__ == '__main__':
    main()

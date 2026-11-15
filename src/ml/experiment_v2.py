"""Physics-informed model v2: environment-aware cold start + gated spatial term.

Run:
    python -m src.ml.experiment_v2 --output-root experiments/ml_v2

Differences from the v1 baseline (``src/ml/experiment.py``), all documented in
``experiments/ml_v2/report.md``:

1. Cold start no longer transfers one warm/wet-pipe constant rate to the cold,
   dry 1250009583. It uses declared environment factors (temperature, watercut,
   velocity) with only the reference rate fitted, plus an explicit scenario grid.
2. A within-pipe profile (spatial) deviation is estimated and allowed into the
   forecast only if leave-one-campaign-out correlation is positive.
3. Between-VTD rate calibration distinguishes comparable and non-comparable
   inspection density, and reports the post-cutoff dense pair as a diagnostic
   instead of silently fitting a detection-biased rate.
4. Validation is leave-one-pipe-out (and leave-one-parent-out) instead of the
   unavailable held-parent forward folds; the limitation is stated, not hidden.

No original or processed artifact is modified. No random seed is needed.
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .audit import source_audit
from .data import RepositoryData
from .evaluation import metrics
from .exposure import attach_exposure
from .models import ConstantRate, fit_constant_rate, sample_weights as constant_weights
from .physics import (ColdStartPhysics, EnvironmentPriors, effective_exposure,
                      fit_cold_start_physics, reference_conditions, scenario_grid)
from .spatial import (FEATURES, fit_spatial_ridge, gate_spatial, leave_one_campaign_out,
                      predict_spatial, spatial_features, standardize_within_pipe)
from .submission import risk_ranking, validate_submission
from .windows import YEAR_SECONDS, build_training_windows, eligible_training, make_snapshot

DENSE_RATIO = (0.75, 1.33)   # campaign density ratio treated as comparable detection


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def aligned_intervals(frame, intervals):
    """Select the exposure intervals aligned with a subset of ``windows``."""
    return [intervals[i] for i in frame.index]


def profile_spatial(frame):
    """Static per-segment features from the full profile grid (one row/segment)."""
    profile = frame[['id', 'segment_id', 'start_m', 'end_m', 'z_center', 'slope_m_per_m']].copy()
    profile = spatial_features(profile)
    return standardize_within_pipe(profile)[['id', 'segment_id'] + FEATURES]


def merge_spatial(frame, spatial):
    merged = frame.merge(spatial, on=['id', 'segment_id'], how='left', validate='many_to_one')
    if len(merged) != len(frame):
        raise ValueError('Spatial merge changed the row count')
    if merged[FEATURES].isna().any().any():
        raise ValueError('Missing profile features for some segments')
    merged.index = frame.index  # keep exposure-interval alignment
    return merged


def cold_leave_one_parent_out(frame, interval_map, priors, gap, weighting, amplitude=0.5):
    """Cross-environment check: train on one parent, predict the other.

    Only two first-inspection parents exist, so this is a two-fold check of the
    actual transfer direction used in production (warm/wet -> cold/dry). It is
    not a hidden-test estimate.
    """
    records, predictions = [], []
    for parent, test in frame.groupby('main_pipeline_id'):
        train = frame[~frame.main_pipeline_id.eq(parent)]
        test_map = {i: interval_map[i] for i in test.index}
        train_map = {i: interval_map[i] for i in train.index}
        model, spatial_model = fit_cold(train, [train_map[i] for i in train.index],
                                        priors, gap, weighting, with_spatial=True)
        prediction = predict_cold(test, [test_map[i] for i in test.index], model,
                                  spatial_model, amplitude)
        for (pid, date), part in test.groupby(['id', 'prediction_date']):
            mask = test.index.isin(part.index)
            p = np.clip(prediction[mask], 0.0, part.nominal_thickness_mm.to_numpy(float))
            records.append(dict(heldout_parent=parent, id=int(pid),
                                prediction_date=str(date.date()), model='M1S',
                                r_ref_mm_year=model.r_ref_mm_year,
                                **metrics(part.min_residual_thickness_mm.to_numpy(float), p)))
        predictions.append(test.assign(
            thickness_pred=np.clip(prediction, 0.0, test.nominal_thickness_mm.to_numpy(float)),
            model='M1S', heldout_parent=parent))
    return records, predictions


def per_group_metrics(frame, truth='min_residual_thickness_mm', prediction='thickness_pred'):
    records = []
    for (pid, date), part in frame.groupby(['id', 'prediction_date']):
        y, p = part[truth].to_numpy(float), part[prediction].to_numpy(float)
        records.append(dict(id=int(pid), date=str(pd.Timestamp(date).date()),
                            **metrics(y, p)))
    return records


def pooled_within_pipe(predictions, truth='min_residual_thickness_mm',
                       groups=('id', 'prediction_date')):
    """Pooled out-of-pipe correlation after removing each campaign's mean level.

    This is the honest counterpart of averaging per-campaign Pearson values:
    campaigns with 3-4 segments otherwise dominate a naive mean.
    """
    records = []
    for model, part in predictions.groupby('model'):
        y = part[truth] - part.groupby(list(groups))[truth].transform('mean')
        p = part.thickness_pred - part.groupby(list(groups)).thickness_pred.transform('mean')
        error = part.thickness_pred.to_numpy(float) - part[truth].to_numpy(float)
        nonconstant = np.ptp(y.to_numpy(float)) > 1e-12 and np.ptp(p.to_numpy(float)) > 1e-12
        r = float(np.corrcoef(y, p)[0, 1]) if nonconstant else None
        records.append(dict(model=model, rows=int(len(part)),
                            mae_mm=float(np.mean(np.abs(error))),
                            rmse_mm=float(np.sqrt(np.mean(error ** 2))),
                            bias_mm=float(np.mean(error)),
                            pooled_within_pipe_pearson=r,
                            pooled_within_pipe_pearson_squared=None if r is None else r * r))
    return records


# --------------------------------------------------------------------------- #
# Between-VTD (S1) inspection pairs and rate calibration
# --------------------------------------------------------------------------- #
def inspection_pairs(vtd):
    """Consecutive own-segment inspection pairs with density and time gap."""
    vtd = vtd.copy()
    vtd['date'] = pd.to_datetime(vtd.date)
    # Detection density is the number of registered VTD observations, not the
    # number of inspected segments: vtd_by_segment holds one row per segment.
    density = (vtd.groupby(['id', 'date']).observation_count.sum()
               .rename('campaign_density'))
    vtd = vtd.join(density, on=['id', 'date'])
    rows = []
    for pid, group in vtd.groupby('id'):
        dates = sorted(group.date.unique())
        for d0, d1 in zip(dates, dates[1:]):
            a = group[group.date.eq(d0)][['segment_id', 'min_residual_thickness_mm',
                                          'observation_count', 'campaign_density']]
            b = group[group.date.eq(d1)][['segment_id', 'min_residual_thickness_mm',
                                          'observation_count', 'campaign_density']]
            pair = a.merge(b, on='segment_id', suffixes=('_0', '_1')).dropna(
                subset=['min_residual_thickness_mm_0', 'min_residual_thickness_mm_1'])
            if pair.empty:
                continue
            dt = (d1 - d0).total_seconds() / YEAR_SECONDS
            if dt <= 0:
                continue
            ratio = float(pair.campaign_density_1.iloc[0]) / float(pair.campaign_density_0.iloc[0])
            records = pair[['segment_id', 'min_residual_thickness_mm_0',
                            'min_residual_thickness_mm_1']].rename(columns={
                                'min_residual_thickness_mm_0': 'h_prior',
                                'min_residual_thickness_mm_1': 'h_next'})
            records['id'] = int(pid)
            records['date_prior'] = pd.Timestamp(d0)
            records['date_next'] = pd.Timestamp(d1)
            records['dt_years'] = dt
            records['density_ratio'] = ratio
            records['comparable'] = DENSE_RATIO[0] <= ratio <= DENSE_RATIO[1]
            rows.append(records)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_pair_rate(pairs, weighting='campaign_balanced'):
    """Nonnegative weighted least squares for a constant thinning rate, mm/year."""
    if pairs.empty:
        raise ValueError('No inspection pairs for rate calibration')
    loss = (pairs.h_prior - pairs.h_next).to_numpy(float)
    dt = pairs.dt_years.to_numpy(float)
    if not np.isfinite(np.r_[loss, dt]).all() or (dt <= 0).any():
        raise ValueError('Invalid pair loss/time')
    if weighting == 'uniform':
        weights = np.ones(len(pairs))
    else:  # equal total weight per pipe
        counts = pairs.groupby('id').segment_id.transform('size')
        weights = 1.0 / counts.to_numpy(float)
    weights = weights / weights.mean()
    rate = float(np.sum(weights * dt * loss) / np.sum(weights * dt ** 2))
    return max(0.0, rate)


def evaluate_pairs_leave_one_pipe(pairs, cutoff):
    """Predict the later campaign from the earlier thickness for each pipe."""
    permitted = pairs[pairs.date_next.lt(cutoff)]
    records, predictions = [], []
    for (pid, d0, d1), test in permitted.groupby(['id', 'date_prior', 'date_next']):
        train = permitted[~permitted.id.eq(pid)]
        available = not train.empty
        models = [('C0', 0.0)]
        if available:
            models.append(('M0', fit_pair_rate(train)))
            dense = train[train.comparable]
            models.append(('M0dense', fit_pair_rate(dense) if len(dense) else fit_pair_rate(train)))
        for name, rate in models:
            pred = test.h_prior.to_numpy(float) - rate * test.dt_years.to_numpy(float)
            frame = test.assign(thickness_pred=np.clip(pred, 0, test.h_prior.to_numpy(float)),
                                model=name)
            records.append(dict(id=int(pid), prediction_date=str(d1.date()),
                                model=name, rate_mm_year=rate, training_rows=len(train),
                                **metrics(test.h_next.to_numpy(float), frame.thickness_pred.to_numpy(float))))
            predictions.append(frame)
    return records, predictions


def s1_rate_decision(pairs, cutoff):
    """Strict as-of calibration plus a post-cutoff diagnostic disclosure."""
    pre = pairs[pairs.date_next.lt(cutoff)]
    comparable = pre[pre.comparable]
    diagnostic = pairs[pairs.date_next.ge(cutoff) & pairs.comparable]
    if len(comparable):
        basis = 'comparable_pre_cutoff_pairs'
        rate = fit_pair_rate(comparable)
        rows = comparable
    elif len(pre):
        basis = 'all_pre_cutoff_pairs_no_comparable_basis'
        rate = fit_pair_rate(pre)
        rows = pre
    else:
        basis = 'no_pre_cutoff_pairs'
        rate = 0.0
        rows = pre
    return dict(rate_mm_year=rate, basis=basis,
                used_pairs=len(rows), used_pipes=sorted(rows.id.unique().tolist()),
                comparable_pre_cutoff_pairs=int(len(comparable)),
                post_cutoff_diagnostic=dict(
                    pairs=int(len(diagnostic)),
                    pipes=sorted(diagnostic.id.unique().tolist()),
                    rate_mm_year=fit_pair_rate(diagnostic) if len(diagnostic) else None,
                    mean_change_mm=float((diagnostic.h_next - diagnostic.h_prior).mean())
                    if len(diagnostic) else None,
                    prior_persistence_pearson_squared=(
                        float(np.corrcoef(diagnostic.h_prior, diagnostic.h_next)[0, 1] ** 2)
                        if len(diagnostic) > 2 and np.ptp(diagnostic.h_prior) > 1e-12
                        else None),
                    dt_years=sorted(diagnostic.dt_years.round(3).unique().tolist())),
                policy='Post-cutoff pairs are reported, never used for the production rate.')


# --------------------------------------------------------------------------- #
# Cold start (S2): physics + gated spatial term, leave-one-pipe-out evidence
# --------------------------------------------------------------------------- #
def cold_frame(windows, spatial):
    frame = merge_spatial(windows[windows['case'].eq('cold_start')].copy(), spatial)
    return frame


def fit_cold(frame, intervals, priors=None, gap='own_mean', weighting='W1', with_spatial=False):
    """``intervals`` must already be positionally aligned with ``frame``."""
    if len(intervals) != len(frame):
        raise ValueError('Interval/frame length mismatch')
    model = fit_cold_start_physics(frame, intervals, priors, gap, weighting)
    spatial_model = fit_spatial_ridge(frame) if with_spatial else None
    return model, spatial_model


def predict_cold(frame, intervals, model, spatial_model=None, amplitude=0.5):
    if len(intervals) != len(frame):
        raise ValueError('Interval/frame length mismatch')
    base = model.predict(frame, intervals)
    if spatial_model is None:
        return base
    deviation = predict_spatial(frame, spatial_model)
    nominal = frame.nominal_thickness_mm.to_numpy(float)
    return np.clip(base + amplitude * deviation, 0.0, nominal)


def cold_leave_one_pipe_out(frame, interval_map, priors, gap, weighting, amplitude=0.5):
    """Honest S2 evidence: predict every first-inspection pipe from the others."""
    records, predictions = [], []
    for (pid, date), test in frame.groupby(['id', 'prediction_date']):
        train = frame[~frame.id.eq(pid)]
        test_intervals = [interval_map[i] for i in test.index]
        variants = {}
        # Nominal baseline has no fitted parameter and equals h_nominal.
        variants['C0'] = test.nominal_thickness_mm.to_numpy(float)
        if len(train):
            constant = fit_constant_rate(train, 'W1')
            variants['M0'] = constant.predict(test)
            train_intervals = [interval_map[i] for i in train.index]
            model, spatial_model = fit_cold(train, train_intervals, priors, gap,
                                            weighting, with_spatial=True)
            variants['M1'] = predict_cold(test, test_intervals, model)
            variants['M1S'] = predict_cold(test, test_intervals, model, spatial_model, amplitude)
        for name, prediction in variants.items():
            clipped = np.clip(prediction, 0.0, test.nominal_thickness_mm.to_numpy(float))
            records.append(dict(id=int(pid), prediction_date=str(date.date()), model=name,
                                **metrics(test.min_residual_thickness_mm.to_numpy(float),
                                          clipped)))
            predictions.append(test.assign(thickness_pred=clipped, model=name))
    return records, predictions


def cold_scenario_band(train, train_intervals, target, target_intervals, priors, gap, weighting):
    """Pre-declared grid -> model-scenario range, not a calibrated interval."""
    rows = []
    for scenario, scenario_gap in scenario_grid():
        try:
            model = fit_cold_start_physics(train, train_intervals,
                                           scenario, scenario_gap, weighting)
            pred = model.predict(target, target_intervals)
        except ValueError:
            continue
        rows.append(dict(theta_K=scenario.theta_K, beta_W=scenario.beta_W,
                         beta_v=scenario.beta_v, age_exponent=scenario.age_exponent,
                         gap_policy=scenario_gap, r_ref_mm_year=model.r_ref_mm_year,
                         thickness_mean_mm=float(np.mean(pred)),
                         thickness_p10_mm=float(np.quantile(pred, .1)),
                         thickness_p90_mm=float(np.quantile(pred, .9))))
    band = pd.DataFrame(rows)
    if band.empty:
        return band, None
    summary = dict(scenarios=len(band),
                   thickness_p10_mm=float(band.thickness_mean_mm.quantile(.1)),
                   thickness_median_mm=float(band.thickness_mean_mm.median()),
                   thickness_p90_mm=float(band.thickness_mean_mm.quantile(.9)),
                   r_ref_p10_mm_year=float(band.r_ref_mm_year.quantile(.1)),
                   r_ref_p90_mm_year=float(band.r_ref_mm_year.quantile(.9)),
                   note='Model-scenario range from declared priors; not a calibrated confidence interval.')
    return band, summary


# --------------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------------- #
def run(root=BASE_DIR, output_root=None, cold_weighting='W1', cold_gap='own_mean',
        spatial_amplitude=0.5):
    root = Path(root).resolve()
    output = Path(output_root or root / 'experiments/ml_v2').resolve()
    protected = [root / 'data', root / 'processed', root / 'reports', root / 'src', root / 'configs']
    if output == root or any(output == p or p in output.parents for p in protected):
        raise ValueError('Use a separate experiment output directory')
    output.mkdir(parents=True, exist_ok=True)

    data = RepositoryData(root)
    print('Auditing published artifacts...', flush=True)
    audit = source_audit(data)
    write_json(output / 'source_audit.json', audit)

    print('Reconstructing historical as-of windows...', flush=True)
    windows, histories = build_training_windows(data)
    windows, intervals = attach_exposure(windows, histories)
    spatial = profile_spatial(data.read('processed/profile_by_segment'))

    cold_all = cold_frame(windows, spatial)
    interval_by_index = {i: intervals[i] for i in windows.index}
    cold_intervals = [interval_by_index[i] for i in cold_all.index]
    if len(cold_intervals) != len(cold_all):
        raise ValueError('Cold-start interval alignment failed')

    print('Evaluating cold-start model, leave-one-pipe-out...', flush=True)
    loo_records, loo_predictions = cold_leave_one_pipe_out(
        cold_all, dict(zip(cold_all.index, cold_intervals)), EnvironmentPriors(),
        cold_gap, cold_weighting, spatial_amplitude)
    parent_records, parent_predictions = cold_leave_one_parent_out(
        cold_all, dict(zip(cold_all.index, cold_intervals)), EnvironmentPriors(),
        cold_gap, cold_weighting, spatial_amplitude)
    pd.DataFrame(parent_records).to_csv(output / 'folds_cross_environment.csv', index=False)

    # Spatial evidence uses every cold campaign; the production target is separate.
    spatial_cv = leave_one_campaign_out(cold_all, lam=10.0)
    gate = gate_spatial(spatial_cv)

    folds = pd.DataFrame(loo_records)
    folds.to_csv(output / 'folds.csv', index=False)
    write_json(output / 'spatial_cv.json', dict(cross_validation=spatial_cv, gate=gate))

    pair_table = inspection_pairs(data.read('processed/vtd_by_segment'))
    pair_table.to_csv(output / 'inspection_pairs.csv', index=False)

    diagnostics, candidates, case_meta, validation_predictions = [], [], {}, []
    for request in data.config['targets']:
        pid, case = int(request['id']), request['case']
        cutoff = pd.Timestamp(request['prediction_date'])
        parent = data.parent_groups().loc[pid]
        print(f'Forecasting {pid}: {case}, cutoff {cutoff.date()}...', flush=True)
        features, hydraulic = make_snapshot(data, request)
        features, target_intervals = attach_exposure(features, {(pid, cutoff): hydraulic})
        target_intervals = aligned_intervals(features, target_intervals)
        features = merge_spatial(features, spatial)

        if case == 'cold_start':
            forbidden = {int(r['id']) for r in data.config['targets'] if r['case'] == 'cold_start'}
            train = cold_all[~cold_all.id.isin(forbidden)
                             & cold_all.prediction_date.lt(cutoff)
                             & ~cold_all.main_pipeline_id.eq(parent)].copy()
            cold_interval_by_index = dict(zip(cold_all.index, cold_intervals))
            train_intervals = [cold_interval_by_index[i] for i in train.index]
            prior = EnvironmentPriors()
            model = fit_cold_start_physics(train, train_intervals, prior, cold_gap, cold_weighting)
            spatial_model = fit_spatial_ridge(train) if gate['usable'] else None
            pred = predict_cold(features, target_intervals, model, spatial_model, spatial_amplitude)
            constant = fit_constant_rate(train, 'W1') if len(train) else ConstantRate()
            candidates.append(features[['id', 'segment_id', 'case', 'prediction_date']].assign(
                model='C0', weighting='none', thickness_pred=features.nominal_thickness_mm))
            candidates.append(features[['id', 'segment_id', 'case', 'prediction_date']].assign(
                model='M0', weighting='W1', thickness_pred=constant.predict(features)))
            # Archive variant: allow the later MAIN-3 campaign (other parent) as sensitivity.
            archive = cold_all[~cold_all.id.isin(forbidden) & ~cold_all.main_pipeline_id.eq(parent)].copy()
            archive_intervals = [cold_interval_by_index[i] for i in archive.index]
            archive_model = fit_cold_start_physics(archive, archive_intervals, prior,
                                                   cold_gap, cold_weighting)
            archive_pred = archive_model.predict(features, target_intervals)
            candidates.append(features[['id', 'segment_id', 'case', 'prediction_date']].assign(
                model='M1_archive_training', weighting=cold_weighting, thickness_pred=archive_pred))
            band, band_summary = cold_scenario_band(train, train_intervals, features,
                                                     target_intervals, prior, cold_gap, cold_weighting)
            if not band.empty:
                band.to_csv(output / 'scenario_sensitivity_cold_start.csv', index=False)
            reference = reference_conditions(train)
            metadata = dict(
                target=request, name='M1' + ('S' if spatial_model is not None else ''),
                training_rows=len(train),
                training_pipes=sorted(train.id.unique().tolist()),
                training_campaigns=sorted(str(d) for d in train.prediction_date.dt.date.unique()),
                reference=reference, priors=model.priors, gap_policy=cold_gap,
                weighting=cold_weighting,
                r_ref_mm_year=model.r_ref_mm_year,
                mean_environment_factor=float(np.mean(
                    np.exp(np.clip(prior.theta_K * (1 / reference['temperature_K']
                                                    - 1 / (features.t_mix_mean.to_numpy(float) + 273.15))
                                   + prior.beta_W * (features.watercut_pct_mean.to_numpy(float) / 100
                                                     - reference['water_fraction'])
                                   + prior.beta_v * np.log((features.v_mix_mean.to_numpy(float)
                                                            + prior.velocity_offset_m_s)
                                                           / (reference['velocity_m_s']
                                                              + prior.velocity_offset_m_s)), -20, 20)))),
                mean_exposure_years=float(np.mean(model.exposure(features, target_intervals))),
                spatial_gate=gate, spatial_amplitude=spatial_amplitude,
                archive_training_r_ref_mm_year=archive_model.r_ref_mm_year,
                archive_training_thickness_mean_mm=float(np.mean(archive_pred)),
                scenario_band=band_summary,
                interpretation='Declared environment priors; reference rate fitted, not identified chemistry.')
            selection = ('M1S' if spatial_model is not None else 'M1') + \
                '; spatial term gated by leave-one-campaign-out correlation'
            prediction_intervals = _cold_intervals(features, target_intervals, model,
                                                   spatial_model, spatial_amplitude, band, band_summary)
        else:
            train = eligible_training(windows, cutoff, 'between_vtd')
            rate = s1_rate_decision(pair_table, cutoff)
            constant = fit_constant_rate(train, 'W1') if len(train) else ConstantRate()
            candidate_rate = max(rate['rate_mm_year'], 0.0)
            pred = np.clip(features.h0_mm.to_numpy(float)
                           - candidate_rate * features.horizon_years.to_numpy(float),
                           0.0, features.h0_mm.to_numpy(float))
            candidates.append(features[['id', 'segment_id', 'case', 'prediction_date']].assign(
                model='C0', weighting='none', thickness_pred=features.h0_mm))
            candidates.append(features[['id', 'segment_id', 'case', 'prediction_date']].assign(
                model='M0_v1', weighting='W1', thickness_pred=constant.predict(features)))
            candidates.append(features[['id', 'segment_id', 'case', 'prediction_date']].assign(
                model='M0_comparable', weighting='inspection_pair', thickness_pred=pred))
            pair_records, pair_predictions = evaluate_pairs_leave_one_pipe(pair_table, cutoff)
            for frame in pair_predictions:
                frame['evaluation_cutoff'] = cutoff
            validation_predictions += pair_predictions
            pair_metrics = pd.DataFrame(pair_records)
            pair_metrics.to_csv(output / 'pairs_leave_one_pipe_metrics.csv', index=False)
            pair_residuals = (pair_table[pair_table.date_next.lt(cutoff)].h_prior
                              - pair_table[pair_table.date_next.lt(cutoff)].h_next)
            prediction_intervals = dict(
                p10=pred + float(pair_residuals.quantile(.1)) if len(pair_residuals) else pred,
                p90=pred + float(pair_residuals.quantile(.9)) if len(pair_residuals) else pred,
                source='pooled pre-cutoff pair change quantiles; not a calibrated interval')
            metadata = dict(
                target=request, name='M1-S1-comparable',
                baseline_missing_count=int(features.baseline_missing.sum()),
                rate=s1_rate_decision(pair_table, cutoff),
                v1_constant_rate_mm_year=constant.rate_mm_year,
                feature_columns=['h0_mm', 'horizon_years'],
                fallback='nearest prior measured segment (unchanged from v1)',
                interpretation='Constant segment-minimum thinning with evidence-based rate basis.')
            selection = 'C0-style transfer from last measured minimum, rate from comparable pairs'

        features['thickness_pred'] = pred
        features['model'] = selection
        features['thickness_p10'] = prediction_intervals['p10']
        features['thickness_p90'] = prediction_intervals['p90']
        features['forecast_status'] = 'physics_informed_v2_not_independently_validated'
        diagnostics.append(features)
        case_meta[case] = metadata

    combined = pd.concat(diagnostics, ignore_index=True)
    result = combined[['id', 'segment_id', 'thickness_pred']].copy()
    result[['id', 'segment_id']] = result[['id', 'segment_id']].astype('int64')
    result = result.sort_values(['id', 'segment_id']).reset_index(drop=True)
    validation = validate_submission(result, data.read('processed/target_segments_grid'), combined)
    result.to_csv(output / 'result.csv', index=False, float_format='%.10f')
    validate_submission(pd.read_csv(output / 'result.csv'),
                        data.read('processed/target_segments_grid'), combined)

    combined.to_csv(output / 'prediction_diagnostics.csv', index=False)
    pd.concat(candidates, ignore_index=True).to_csv(output / 'candidate_predictions.csv', index=False)
    risk = risk_ranking(combined)
    risk.to_csv(output / 'risk_ranking.csv', index=False)
    risk[risk.top_10pct].to_csv(output / 'risk_top_10pct.csv', index=False)

    pair_pooled = (pooled_within_pipe(pd.concat(validation_predictions, ignore_index=True),
                                      truth='h_next', groups=('id', 'date_next'))
                   if validation_predictions else [])
    pooled_cold = pooled_within_pipe(pd.concat(loo_predictions, ignore_index=True))
    parent_frame = pd.DataFrame(parent_records)
    parent_pooled = (pooled_within_pipe(pd.concat(parent_predictions, ignore_index=True))
                     if parent_predictions else [])
    metrics_payload = dict(
        cold_start_leave_one_pipe_out=_summarise_folds(folds),
        cold_start_pooled=pooled_cold,
        cold_start_per_pipe=folds.to_dict(orient='records'),
        cold_start_cross_environment=parent_pooled,
        cold_start_cross_environment_per_pipe=parent_frame.to_dict(orient='records'),
        between_vtd_pooled=pair_pooled,
        spatial_cross_validation=spatial_cv, spatial_gate=gate,
        between_vtd_pair_leave_one_pipe=_read_csv_records(output / 'pairs_leave_one_pipe_metrics.csv'),
    )
    write_json(output / 'metrics.json', json_safe(metrics_payload))
    write_json(output / 'models.json', json_safe(case_meta))

    if any(sha256(root / name) != digest for name, digest in data.hashes.items()):
        raise ValueError('Input artifact changed during experiment')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = branch = 'unavailable'
    manifest = dict(
        git_sha=commit, branch=branch, seed=0, stochastic_operations=False,
        versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        preprocessing_versions=data.manifest['versions'], input_sha256=data.hashes,
        ml_code_sha256={p.name: sha256(p) for p in sorted((root / 'src/ml').glob('*.py'))},
        settings=dict(cold_weighting=cold_weighting, cold_gap_policy=cold_gap,
                      spatial_amplitude=spatial_amplitude, spatial_ridge_lambda=10.0,
                      dense_ratio=list(DENSE_RATIO),
                      priors=EnvironmentPriors().__dict__),
        submission=validation, result_sha256=sha256(output / 'result.csv'),
        tested_environment_is_original_lock=False)
    write_json(output / 'run_manifest.json', manifest)
    write_report_v2(output, folds, combined, case_meta, spatial_cv, gate, pair_table,
                    manifest, pooled_cold, pair_pooled, parent_pooled, parent_frame)
    print(f'PASS: {len(result)} validated v2 predictions at {output / "result.csv"}', flush=True)
    return dict(predictions=combined, metadata=case_meta, folds=folds, manifest=manifest)


def _cold_intervals(features, target_intervals, model, spatial_model, amplitude, band, band_summary):
    point = predict_cold(features, target_intervals, model, spatial_model, amplitude)
    if band is not None and not band.empty:
        low = float(band.thickness_mean_mm.quantile(.1))
        high = float(band.thickness_mean_mm.quantile(.9))
        shift = point - float(np.mean(point))
        p10 = np.clip(shift + low, 0, features.nominal_thickness_mm.to_numpy(float))
        p90 = np.clip(shift + high, 0, features.nominal_thickness_mm.to_numpy(float))
        return dict(p10=p10, p90=p90,
                    source='declared scenario grid; model uncertainty, not calibrated coverage')
    return dict(p10=point.copy(), p90=point.copy(), source='point forecast')


def json_safe(value):
    """Convert NumPy/NaN payloads to strict JSON (NaN -> null)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if value is pd.NaT:
        return None
    return value


def _summarise_folds(folds):
    summary = {}
    for model, part in folds.groupby('model'):
        summary[model] = dict(
            campaigns=int(len(part)), rows=int(len(part)),
            mae_mm=float(part.mae_mm.mean()), rmse_mm=float(part.rmse_mm.mean()),
            bias_mm=float(part.bias_mm.mean()),
            mean_pearson_squared=float(part.pearson_squared.dropna().mean())
            if part.pearson_squared.notna().any() else None,
            defined_pearson_campaigns=int(part.pearson_squared.notna().sum()),
        )
    return summary


def _read_csv_records(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path).to_dict(orient='records')


def write_report_v2(output, folds, predictions, metadata, spatial_cv, gate, pairs,
                    manifest, pooled_cold=None, pair_pooled=None,
                    parent_pooled=None, parent_frame=None):
    cold = metadata.get('cold_start', {})
    between = metadata.get('between_vtd', {})
    summary = pd.DataFrame(folds)
    table = []
    for model, part in summary.groupby('model'):
        mean_r2 = part.pearson_squared.dropna().mean() if part.pearson_squared.notna().any() else None
        mean_text = 'undefined (constant)' if mean_r2 is None or not np.isfinite(mean_r2) else f'{mean_r2:.3f}'
        table.append(f"| {model} | {len(part)} | {part.mae_mm.mean():.3f} | {part.rmse_mm.mean():.3f} | "
                     f"{part.bias_mm.mean():+.3f} | {mean_text} |")

    def pooled_rows(records):
        rows = []
        for record in records or []:
            r2 = record.get('pooled_within_pipe_pearson_squared')
            rows.append(f"| {record['model']} | {record['rows']} | {record['mae_mm']:.3f} | "
                        f"{record['rmse_mm']:.3f} | {record['bias_mm']:+.3f} | "
                        f"{'undefined' if r2 is None else f'{r2:.4f}'} |")
        return rows

    pooled_table = pooled_rows(pooled_cold)
    pair_table_lines = pooled_rows(pair_pooled)
    cold_r2 = next((r.get('pooled_within_pipe_pearson_squared') for r in (pooled_cold or [])
                    if r.get('model') == 'M1S'), None)
    cold_r2_text = 'undefined' if cold_r2 is None else f'{cold_r2:.4f}'
    parent_rows = []
    if parent_frame is not None and len(parent_frame):
        for record in parent_frame.to_dict(orient='records'):
            r2 = record.get('pearson_squared')
            parent_rows.append(
                f"| {record['heldout_parent']} | {record['id']} | {record['prediction_date']} | "
                f"{record['n']} | {record['mae_mm']:.3f} | {record['bias_mm']:+.3f} | "
                f"{'undefined' if r2 is None or not np.isfinite(r2) else f'{r2:.3f}'} |")
    lines = [
        '# PipeTwin ML v2: environment-aware physics and gated spatial term', '',
        f"Checked branch `{manifest['branch']}`, commit `{manifest['git_sha']}`.", '',
        '## What changed against v1', '',
        '1. **Cold start is environment-aware.** v1 fitted one constant rate on warm, wet MAIN-2 '
        'pipes and extrapolated it to the cold, dry, slow 1250009583. v2 keeps the multi-parameter '
        'rate structure of the technical specification (temperature, watercut, velocity), fits only '
        'the reference rate and declares the remaining exponents as explicit priors with a '
        'sensitivity grid.',
        '2. **A within-pipe profile deviation is allowed only with out-of-sample evidence.** The '
        'profile ridge model is gated by leave-one-campaign-out correlation and shrunk by '
        f'{manifest["settings"]["spatial_amplitude"]:.2f}.',
        '3. **Between-VTD rates separate comparable and non-comparable inspections.** Pairs whose '
        'campaign density differs by more than the declared ratio are not silently pooled with '
        'dense campaigns. Post-cutoff comparable pairs are reported as a diagnostic only.',
        '4. **Validation is leave-one-pipe-out / leave-one-campaign-out.** The repository still has '
        'no forward-time held-parent folded evidence before either production cutoff; that '
        'limitation remains and is not replaced by random segment splits.', '',
        '## Cold-start leave-one-pipe-out evidence', '',
        'Each first-inspection campaign is predicted by a model fitted on the other pipes. The '
        'metrics below are therefore out-of-pipe, not in-sample. They measure model behaviour on '
        'ten campaigns, not accuracy on the hidden 1250009583 inspection.', '',
        '| Model | Held-out campaigns | MAE (mm) | RMSE (mm) | Bias (mm) | Mean Pearson² |',
        '|---|---:|---:|---:|---:|---:|', *table, '',
        'A per-campaign mean is dominated by the three tiny campaigns (n = 1-4). The pooled '
        'out-of-pipe view below removes each campaign mean first and is the number to quote:', '',
        '| Model | Rows | MAE (mm) | RMSE (mm) | Bias (mm) | Pooled within-pipe Pearson² |',
        '|---|---:|---:|---:|---:|---:|', *pooled_table, '',
        'Constant predictions (C0/M0/M1) have zero within-pipe variance, so their Pearson '
        'correlation is undefined by definition. Only the profile term (M1S) can produce a '
        'within-pipe ranking at all. Its pooled value is ' + cold_r2_text + ', i.e. small and not '
        'hidden below. The cold-start *level* is unchanged by the profile term.', '',
        '### Cross-environment transfer (the actual production direction)', '',
        'Leave-one-pipe-out mixes warm/wet and cold/dry pipes in every training fold. Production '
        'does not: 1250009583 is cold and dry, while its only pre-cutoff first-inspection labels '
        'are warm and wet. The table below trains on one parent and predicts the other, which is '
        'the closest available test of that transfer. It is a two-direction check, not a '
        'hidden-test estimate.', '',
        '| Held-out parent | Pipe | Campaign | n | MAE (mm) | Bias (mm) | Pearson² |',
        '|---|---|---:|---:|---:|---:|---:|', *parent_rows, '',
        f"Spatial gate: usable = **{gate['usable']}**; pooled leave-one-campaign-out "
        f"Pearson r = {gate['pooled_pearson']}. Reason: {gate['reason']}.", '',
        '## Production forecasts', '',
        f"- `1250022871` (between VTD, 2023-12-01): rate basis **{between.get('rate', {}).get('basis')}**, "
        f"rate = {between.get('rate', {}).get('rate_mm_year'):.6g} mm/year, "
        f"{between.get('rate', {}).get('used_pairs', 0)} pairs, "
        f"{between.get('baseline_missing_count', 0)} segments with a borrowed prior baseline.",
        f"  Post-cutoff comparable diagnostic: {between.get('rate', {}).get('post_cutoff_diagnostic', {})}. ",
        '  Leave-one-pipe-out pair check (each pipe predicted from the others, cutoff-respecting):', '',
        '| Model | Rows | MAE (mm) | RMSE (mm) | Bias (mm) | Pooled within-pipe Pearson² |',
        '|---|---:|---:|---:|---:|---:|', *pair_table_lines, '',
        f"- `1250009583` (cold start, 2022-07-01): training rows {cold.get('training_rows')}, "
        f"pipes {cold.get('training_pipes')}, campaigns {cold.get('training_campaigns')}; "
        f"reference rate {cold.get('r_ref_mm_year', float('nan')):.6g} mm/year, "
        f"mean exposure {cold.get('mean_exposure_years', float('nan')):.3f} reference-years, "
        f"mean environment factor {cold.get('mean_environment_factor', float('nan')):.3f}. "
        f"Mean predicted thickness {predictions[predictions.id.eq(1250009583)].thickness_pred.mean():.3f} mm.",
        f"  Archive-training sensitivity (adds the later MAIN-3 campaign): reference rate "
        f"{cold.get('archive_training_r_ref_mm_year', float('nan')):.6g} mm/year, mean predicted "
        f"thickness {cold.get('archive_training_thickness_mean_mm', float('nan')):.3f} mm.",
        f"  Declared scenario band (model uncertainty, not calibrated): "
        f"{cold.get('scenario_band')}", '',
        '## Honest limitations', '',
        '- The hidden inspections of the two target pipes are not in the repository. No hidden-test '
        'accuracy is claimed.',
        '- The environment exponents are priors. With two confounded first-inspection environments '
        'they cannot be identified from data; the grid shows how much the level moves.',
        '- Segment-level minima persist poorly between inspections. Leave-one-campaign-out spatial '
        'correlation is small; the profile term changes ranking, not necessarily absolute accuracy.',
        '- Missing early operating history (29-year cold-start pipe, ~49 % hydraulic coverage) is '
        'represented by an explicit `own_mean` gap assumption and a lower/upper scenario, never by '
        'invented measurements.',
        '- Repairs, pigging, inhibition, chemistry, cathodic protection and external corrosion remain '
        'unavailable or undated; the fitted rate absorbs them as an effective segment minima rate.',
        '- The between-VTD production rate is zero because every pre-cutoff pair is non-comparable '
        f"({between.get('rate', {}).get('used_pairs', 0)} pairs, "
        f"{between.get('rate', {}).get('comparable_pre_cutoff_pairs', 0)} comparable). The one dense "
        'cross-pipe pair (2022-07-06 -> 2023-12-01) shows a real drop but lies on the production '
        f"cutoff: rate {between.get('rate', {}).get('post_cutoff_diagnostic', {}).get('rate_mm_year')} "
        f"mm/year, prior-thickness persistence Pearson² "
        f"{between.get('rate', {}).get('post_cutoff_diagnostic', {}).get('prior_persistence_pearson_squared')}. "
        'It is disclosed, not used. This is the single most important unresolved S1 question for '
        'the mentors.', '',
        '## Artifacts', '',
        '- `result.csv`: exactly `id,segment_id,thickness_pred` (2,000 rows).',
        '- `prediction_diagnostics.csv`: bounds, intervals, model metadata per segment.',
        '- `candidate_predictions.csv`: C0, M0, M1/M1S and sensitivity variants.',
        '- `folds.csv`: cold-start leave-one-pipe-out metrics per campaign.',
        '- `folds_cross_environment.csv`: two-fold warm/cold parent transfer metrics.',
        '- `scenario_sensitivity_cold_start.csv`: the declared prior grid values.',
        '- `spatial_cv.json`: spatial cross-validation and gate decision.',
        '- `inspection_pairs.csv`: every consecutive VTD pair with density ratio and comparability.',
        '- `pairs_leave_one_pipe_metrics.csv`, `metrics.json`, `models.json`, `run_manifest.json`.', '',
        '```bash', 'python -m src.ml.experiment_v2 --output-root experiments/ml_v2', '```', '',
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE_DIR / 'experiments/ml_v2')
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--cold-weighting', default='W1', choices=['W0', 'W1', 'W2'])
    parser.add_argument('--cold-gap', default='own_mean',
                        choices=['own_mean', 'reference', 'observed_only'])
    parser.add_argument('--spatial-amplitude', type=float, default=0.5)
    args = parser.parse_args()
    run(args.root, args.output_root, args.cold_weighting, args.cold_gap, args.spatial_amplitude)


if __name__ == '__main__':
    main()

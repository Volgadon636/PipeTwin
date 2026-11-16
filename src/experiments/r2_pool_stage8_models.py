"""Shape transformations and historical thermal scenarios for H36–H40."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.experiments.r2_pool_stage5 import INTEGRATED, thermal_exposure
from src.ml.evaluation import metrics
from src.ml.targeted_s2 import _merge_thermal, _thermal_features
from src.ml.windows import YEAR_SECONDS

CALIBRATION = 'crossfit_isotonic_shape'
ROBUST = 'robust_iqr_cap'
SOFT = 'soft_physical_bounds'
GROUND = 'early_ground_temperature'
REPLAY = 'early_first_observed_year'
SCENARIOS = 'equal_early_thermal_scenarios'


def select_by_inner_r_squared(frame, variants, min_rows, fallback):
    """Choose among fixed variants using complete inner profile scores."""
    detail, eligible = {}, {}
    for name in variants:
        rows = []
        for (pid, date), profile in frame.groupby(['id', 'prediction_date'], sort=True):
            result = metrics(profile.next_mm, profile[name]) if len(profile) >= min_rows else {}
            rows.append({'id': int(pid), 'date': str(date.date()), 'n': len(profile),
                         'r_squared': result.get('pearson_squared')})
        complete = bool(rows) and all(row['r_squared'] is not None for row in rows)
        score = float(np.mean([row['r_squared'] for row in rows])) if complete else None
        detail[name] = {'mean_r_squared': score, 'profiles': rows}
        if score is not None:
            eligible[name] = score
    if not eligible:
        return fallback, {'status': 'fallback_no_complete_score', 'scores': detail, 'selected': fallback}
    best = max(eligible.values())
    chosen = next(name for name in variants if name in eligible and best - eligible[name] <= 1e-12)
    return chosen, {'status': 'selected_on_inner_oof', 'scores': detail, 'selected': chosen}


def fit_shape_calibration(oof, bins, min_rows):
    parts, omitted = [], []
    for (pid, date), part in oof.groupby(['id', 'prediction_date'], sort=True):
        p, y = part.baseline.to_numpy(float), part.next_mm.to_numpy(float)
        if len(part) < min_rows or min(p.std(), y.std()) <= 1e-12:
            omitted.append({'id': int(pid), 'date': str(date.date()), 'n': len(part)})
            continue
        parts.append(pd.DataFrame({'x': (p - p.mean()) / p.std(), 'y': (y - y.mean()) / y.std(),
                                   'w': 1. / len(part)}))
    audit = {'used_profiles': len(parts), 'omitted_profiles': omitted}
    if not parts:
        return None, {**audit, 'status': 'identity_insufficient_profiles'}
    train = pd.concat(parts, ignore_index=True)
    # Quantile binning limits effective degrees of freedom on the small groups.
    edges = np.unique(np.quantile(train.x, np.linspace(0., 1., bins + 1)))
    group = np.searchsorted(edges[1:-1], train.x, side='right')
    aggregated = (train.assign(wx=train.x * train.w, wy=train.y * train.w)
                  .groupby(group)[['w', 'wx', 'wy']].sum())
    if len(aggregated) < 2:
        return None, {**audit, 'status': 'identity_insufficient_knots'}
    model = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(
        aggregated.wx / aggregated.w, aggregated.wy / aggregated.w, sample_weight=aggregated.w)
    if np.ptp(model.y_thresholds_) <= 1e-12:
        return None, {**audit, 'status': 'identity_flat_isotonic_map'}
    return model, {**audit, 'status': 'fitted',
                   'x_knots': model.X_thresholds_.tolist(), 'y_knots': model.y_thresholds_.tolist()}


def shape_calibration(base, length, nominal, model, strength):
    values = np.asarray(base, dtype=float)
    if model is None:
        return values
    loc = np.average(values, weights=length)
    width = float(np.sqrt(np.average((values - loc)**2, weights=length)))
    if width <= 1e-12:
        return values
    standard = (values - loc) / width
    shape = (1. - strength) * standard + strength * model.predict(standard)
    shape -= np.average(shape, weights=length)
    sd = float(np.sqrt(np.average(shape**2, weights=length)))
    if sd <= 1e-12:
        return values
    return np.clip(loc + width * shape / sd, 0., nominal)


def shape_bounds(raw, base, nominal, config):
    q1, q3 = np.quantile(base, [.25, .75])
    if q3 - q1 <= 1e-12:
        robust = base
    else:
        spread = config['iqr_factor'] * (q3 - q1)
        robust = np.clip(np.clip(base, q1 - spread, q3 + spread), 0., nominal)
    tau = config['softness_mm']
    soft = tau * (np.logaddexp(0., raw / tau) - np.logaddexp(0., (raw - nominal) / tau))
    return {ROBUST: robust, SOFT: np.clip(soft, 0., nominal)}


def thermal_name(conductance):
    return f'thermal_conductance_{conductance:g}'


def centered_temperature(full, params, conductance):
    setup = replace(params, overall_heat_transfer_w_m2_k=conductance)
    thermal = _thermal_features(full, setup)
    # The frozen grid already contains default scenario columns.
    current = _merge_thermal(full.drop(columns=thermal.columns.difference(['id', 'segment_id',
                                                                          'prediction_date'])), thermal)
    values = (current.t_in_segment_c + current.t_out_segment_c) / 2.
    valid = current.scenario_valid.fillna(False) & np.isfinite(values)
    if not valid.all():
        return np.full(len(full), np.nan)
    return (values - np.average(values, weights=current.length_m)).to_numpy(float)


def early_thermal_features(full, intervals, params, config):
    observed, audit = thermal_exposure(full, intervals, params)
    value = observed[INTEGRATED]
    if audit['thermal_status'] != 'observed_intervals_only':
        return {GROUND: value, REPLAY: value}, {**audit, 'leading_gap_years': None,
                                               'first_window_observed_years': None}
    inlet = intervals.t_start_usable
    density = intervals.density_liquid_work_usable
    velocity = intervals.velocity_m_s
    duration = intervals.duration_years
    good = (np.isfinite(inlet) & np.isfinite(density) & np.isfinite(velocity) & np.isfinite(duration)
            & inlet.gt(-273.15) & density.gt(0) & velocity.gt(0) & duration.gt(0))
    valid = intervals.loc[good].copy()
    first = valid.interval_start.min()
    history_start = full.start_date.iloc[0]
    end = first + pd.Timedelta(days=config['first_window_days'])
    first_year = valid.loc[valid.interval_start.lt(end)].copy()
    first_year['interval_end'] = first_year.interval_end.clip(upper=end)
    first_year['duration_years'] = (first_year.interval_end - first_year.interval_start).dt.total_seconds() / YEAR_SECONDS
    leading = max(0., (first - history_start).total_seconds() / YEAR_SECONDS)
    initial, _ = thermal_exposure(full, first_year, params)
    first_duration = float(first_year.duration_years.sum())
    replayed = value + initial[INTEGRATED] * (leading / first_duration)
    info = {**audit, 'leading_gap_years': leading, 'first_window_observed_years': first_duration,
            'history_start': str(history_start), 'first_admissible_interval': str(first),
            'thermal_shape_difference_std_c_year': float(np.std(replayed - value))}
    return {GROUND: value, REPLAY: replayed}, info

"""Explicit bounded-hold exposure assumption; missing history stays unknown."""
import numpy as np
import pandas as pd

from .windows import YEAR_SECONDS

FIELDS = ['t_mix_usable', 'watercut_pct_usable', 'v_liquid_true_usable']


def exposure_intervals(rows, start, end, max_hold_days=1.0):
    """Carry a QA-valid timestamp forward at most max_hold_days, never backward.

    This is a sensitivity assumption, NOT evidence that one calculation describes
    a full day. Intervals end at the next timestamp (including an invalid row),
    the cutoff, or the hold cap. Start clipping permits a valid pre-start reading
    only within that same explicit cap. No fill across the remaining gaps.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start or max_hold_days <= 0:
        raise ValueError('Invalid exposure horizon or hold cap')
    if rows.id.nunique() > 1:
        raise ValueError('Exposure intervals require one pipe')
    part = rows[rows.date.lt(end)].sort_values('date').copy()
    if part.date.duplicated().any():
        raise ValueError('Duplicate hydraulic timestamp')
    next_date = part.date.shift(-1).fillna(end)
    part['interval_end'] = pd.concat([
        next_date, part.date + pd.Timedelta(days=max_hold_days)], axis=1).min(axis=1).clip(upper=end)
    part['interval_start'] = part.date.clip(lower=start)
    valid = (part[FIELDS].notna().all(axis=1) & part.t_mix_usable.gt(-273.15)
             & part.watercut_pct_usable.between(0, 100) & part.v_liquid_true_usable.ge(0)
             & part.interval_end.gt(part.interval_start))
    part = part[valid].copy()
    part['duration_years'] = (part.interval_end - part.interval_start).dt.total_seconds() / YEAR_SECONDS
    part['temperature_K'] = part.t_mix_usable + 273.15
    part['water_fraction'] = part.watercut_pct_usable / 100
    part['velocity_m_s'] = part.v_liquid_true_usable
    horizon = (end - start).total_seconds() / YEAR_SECONDS
    supported = float(part.duration_years.sum())
    previous = start
    longest = 0.0
    for row in part.itertuples():
        longest = max(longest, (row.interval_start - previous).total_seconds() / 86400)
        previous = row.interval_end
    longest = max(longest, (end - previous).total_seconds() / 86400)
    return part, dict(observed_years=supported, unobserved_years=max(0.0, horizon-supported),
                      coverage_fraction=supported/horizon, longest_gap_days=longest,
                      exposure_policy=f'bounded_forward_hold_{max_hold_days:g}_days')


def attach_exposure(windows, histories, max_hold_days=1.0):
    frame = windows.copy()
    cache, intervals, summaries = {}, [], []
    for row in frame.itertuples():
        key = (int(row.id), row.start_date, row.prediction_date)
        if key not in cache:
            cache[key] = exposure_intervals(histories[(int(row.id), row.prediction_date)],
                                           row.start_date, row.prediction_date, max_hold_days)
        interval, summary = cache[key]
        intervals.append(interval)
        summaries.append(summary)
    for column in summaries[0]:
        frame[column] = [s[column] for s in summaries]
    return frame, intervals


def integrated_years(intervals, reference, coefficients, gap_policy='reference', horizon_years=None):
    """Integral of F_T*F_W*F_V; r0 later converts years to mm.

    Coefficients: [theta_T / 1000 K, beta_W, beta_v]. References are fitted
    exclusively on training intervals. Gap reference means F=1 on unknown time;
    observed_only omits unknown time, and is a sensitivity case, not zero corrosion.
    """
    if gap_policy not in ('reference', 'observed_only'):
        raise ValueError('Unknown gap policy')
    if reference['temperature_K'] <= 0 or reference['velocity_offset_m_s'] <= 0:
        raise ValueError('Reference temperature and velocity offset must be positive')
    t = intervals.temperature_K.to_numpy(float)
    w = intervals.water_fraction.to_numpy(float)
    v = intervals.velocity_m_s.to_numpy(float)
    if (t <= 0).any() or (v < 0).any() or ((w < 0) | (w > 1)).any():
        raise ValueError('Invalid physical units or inputs')
    x = np.column_stack([1000 * (1/reference['temperature_K'] - 1/t),
                         w-reference['water_fraction'],
                         np.log((v+reference['velocity_offset_m_s']) /
                                (reference['velocity_m_s']+reference['velocity_offset_m_s']))])
    value = float(np.dot(np.exp(np.clip(x @ coefficients, -20, 20)), intervals.duration_years))
    if gap_policy == 'reference':
        if horizon_years is None:
            raise ValueError('Reference gap filling needs the full horizon')
        value += max(0.0, horizon_years - float(intervals.duration_years.sum()))
    return value

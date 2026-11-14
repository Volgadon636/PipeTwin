"""Cutoff-first hydraulic quality rules; no future observations used to validate a run."""
import numpy as np
import pandas as pd

from .config import HYDRAULIC_FEATURES


def persistent_calendar_days(df, mask, days=3):
    """Mark anomalous rows belonging to runs of >=days distinct consecutive dates.

    The caller must already have removed observations on/after the prediction date.
    Multiple calculations on one day cannot satisfy this rule.
    """
    tmp = df[['id', 'date']].copy()
    tmp['day'] = tmp.date.dt.normalize()
    tmp['candidate'] = mask.fillna(False).to_numpy()
    daily = tmp.groupby(['id', 'day'], as_index=False).candidate.any().sort_values(['id', 'day'])
    previous = daily.groupby('id').candidate.shift(fill_value=False)
    adjacent = daily.groupby('id').day.diff().eq(pd.Timedelta(days=1))
    start = ~(daily.candidate & previous & adjacent)
    daily['run'] = start.cumsum()
    sizes = daily.groupby('run').candidate.transform('size')
    daily['persistent'] = daily.candidate & sizes.ge(days)
    keys = pd.MultiIndex.from_frame(tmp[['id', 'day']])
    stable = daily.set_index(['id', 'day']).persistent.reindex(keys).to_numpy(dtype=bool)
    return pd.Series(stable, index=df.index) & mask.fillna(False)


def hydraulic_asof(tech, pipe_ids, cutoff, config):
    cutoff = pd.Timestamp(cutoff)
    subset = tech.id.isin(pipe_ids) & tech.date.lt(cutoff) & tech.record_eligible
    df = tech.loc[subset].sort_values(['id', 'date', 'source_row']).copy().reset_index(drop=True)
    audit_rows = []
    flow = df.q_liquid.gt(0)
    limits = config['soft_limits']

    def apply_rule(field, candidate, rule, hard=False):
        candidate = candidate.fillna(False)
        kept = pd.Series(False, index=df.index) if hard else persistent_calendar_days(df, candidate, config['persistent_days'])
        remove = candidate & ~kept
        for i in df.index[candidate]:
            audit_rows.append({'id': int(df.at[i, 'id']), 'source_row': int(df.at[i, 'source_row']),
                               'date': df.at[i, 'date'], 'prediction_date': cutoff,
                               'field': field, 'rule': rule, 'original_value': float(df.at[i, field]),
                               'action': 'keep_persistent' if kept[i] else 'set_missing'})
        df.loc[remove, field + '_usable'] = np.nan

    for field in HYDRAULIC_FEATURES:
        df[field + '_usable'] = df[field].astype(float)
    for field in ('q_liquid', 'q_oil', 'v_liquid_true', 'v_mix', 'density_liquid_work', 'density_oil'):
        apply_rule(field, df[field].lt(0), 'negative_physical_quantity', hard=True)
    apply_rule('watercut_pct', df.watercut_pct.notna() & ~df.watercut_pct.between(0,100), 'watercut_outside_0_100', hard=True)
    for field in ('p_start', 'p_end'):
        apply_rule(field, flow & df[field].eq(0), 'zero_pressure_during_flow')
        apply_rule(field, df[field].gt(limits['pressure_high']), 'high_pressure_review_threshold')
    apply_rule('q_liquid', df.q_liquid.gt(limits['liquid_flow_high']), 'high_flow_review_threshold')
    # These outputs depend on the same suspect liquid-flow input; otherwise a removed
    # one-day flow spike would survive in the model through oil flow or velocity.
    for field in ('q_oil', 'v_liquid_true', 'v_mix'):
        apply_rule(field, df.q_liquid.gt(limits['liquid_flow_high']), 'paired_with_high_liquid_flow')
    for field in ('v_liquid_true', 'v_mix'):
        apply_rule(field, df[field].gt(limits['velocity_high']), 'high_velocity_review_threshold')
    for field in ('t_mix', 't_start', 't_end'):
        apply_rule(field, flow & df[field].eq(0), 'zero_temperature_during_flow')
        apply_rule(field, df[field].lt(limits['temperature_low']), 'low_temperature_review_threshold')
        apply_rule(field, df[field].gt(limits['temperature_high']), 'high_temperature_review_threshold')
    # t_mix's exact meaning is unresolved: do not impose a boundary-temperature model.
    for field in ('density_liquid_work', 'density_oil'):
        apply_rule(field, flow & df[field].eq(0), 'zero_density_during_flow', hard=True)
        apply_rule(field, flow & df[field].gt(0) & df[field].lt(limits['density_low']), 'low_density_review_threshold')
        apply_rule(field, df[field].gt(limits['density_high']), 'high_density_review_threshold')
    df['prediction_date'] = cutoff
    audit = pd.DataFrame(audit_rows, columns=['id','source_row','date','prediction_date','field','rule','original_value','action'])
    return df, audit


def hydraulic_summary(df, pipe_ids):
    """Daily medians then equal-weight daily statistics; do not extrapolate across gaps."""
    results = []
    for pid in pipe_ids:
        group = df[df.id.eq(pid)].copy()
        group['day'] = group.date.dt.normalize()
        cols = [f'{x}_usable' for x in HYDRAULIC_FEATURES]
        daily = group.groupby('day')[cols].median()
        record = {'id': int(pid), 'hydraulic_records_count': len(group), 'hydraulic_days_count': len(daily),
                  'hydraulic_source_date_min': group.date.min(), 'hydraulic_source_date_max': group.date.max()}
        span = (daily.index.max()-daily.index.min()).days+1 if len(daily) else 0
        record['hydraulic_coverage_fraction'] = len(daily)/span if span else np.nan
        record['hydraulic_max_gap_days'] = daily.index.to_series().diff().dt.days.max()
        for field in HYDRAULIC_FEATURES:
            values = daily[field + '_usable']
            record[field + '_mean'] = values.mean()
            record[field + '_std'] = values.std()
            record[field + '_valid_days'] = int(values.notna().sum())
        results.append(record)
    return pd.DataFrame(results)

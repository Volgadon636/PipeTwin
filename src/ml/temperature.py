"""Opt-in T-only exposure v1; never changes the legacy joint/model registry.

Input histories must come from the cutoff-specific hydraulic QA (``*_usable``),
not from snapshot means or spatial/fallback temperatures. Bounded holds are an
assumption about support, not continuous measurements. Missing history is unknown.
"""
from types import MappingProxyType

import numpy as np
import pandas as pd

from .features import FeatureDescriptor, FeatureRegistry
from .windows import YEAR_SECONDS

R_J_MOL_K = 8.314462618
REFERENCE_TEMPERATURE_K = 293.15
EA_CANDIDATES = MappingProxyType({
    'EA00_null': 0.0, 'EA25_low': 25.0, 'EA40_mid': 40.0,
    'EA55_elevated': 55.0, 'EA70_high': 70.0,
})
THERMAL_DEFINITIONS = {
    'thermal_exposure_observed_years': ('years', 'Sum of duration_years * F_T over valid bounded T intervals'),
    'thermal_exposure_reference_years': ('years', 'Observed exposure + unobserved years; null unless reference gap assumption declared'),
    'thermal_observed_years': ('years', 'Sum of valid bounded T interval durations'),
    'thermal_unobserved_years': ('years', 'Full horizon minus supported duration; unknown history'),
    'thermal_coverage_fraction': ('fraction', 'Supported duration / full horizon'),
    'thermal_longest_gap_days': ('days', 'Longest unsupported gap, including leading and trailing gaps'),
    'thermal_factor_observed_mean': ('dimensionless', 'Observed exposure / supported duration; null for no support'),
    'thermal_history_missing': ('bool', 'No supported temperature intervals'),
    'thermal_history_incomplete': ('bool', 'Some horizon duration has no supported temperature'),
}
THERMAL_FEATURE_COLUMNS = tuple(THERMAL_DEFINITIONS)
THERMAL_METADATA_COLUMNS = (
    'thermal_theta_K', 'thermal_ea_kJ_mol', 'thermal_candidate_id',
    'thermal_gap_policy', 'thermal_exposure_policy', 'thermal_profile_source',
)


def thermal_factor(temperature_K, ea_kj_mol):
    """Dimensionless exp((1000*Ea/R)*(1/293.15 - 1/T)); no silent clipping.

The low-level formula permits nonnegative Ea for unit/legacy checks. Feature
generation below restricts Ea to the predeclared team hypotheses.
"""
    if not np.isfinite(ea_kj_mol) or ea_kj_mol < 0:
        raise ValueError('Activation energy must be finite and nonnegative (kJ/mol)')
    temperature = np.asarray(temperature_K, dtype=float)
    if not np.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError('Temperature must be finite and positive (Kelvin)')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        factor = np.exp((1000.0 * ea_kj_mol / R_J_MOL_K) *
                        (1.0 / REFERENCE_TEMPERATURE_K - 1.0 / temperature))
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError('Thermal factor overflow/underflow; no clipping applied')
    return factor


def temperature_intervals(rows, start, end, max_hold_days=1.0):
    """T-only counterpart of exposure_intervals, with the same interval bounds.

    W/v are neither required nor read. Invalid next temperatures terminate prior
    holds. Duplicate pre-cutoff timestamps are rejected, never averaged. A valid
    0 Celsius survives; upstream hydraulic QA owns operating-context checks.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(start) or pd.isna(end) or end <= start or
            not np.isfinite(max_hold_days) or max_hold_days <= 0):
        raise ValueError('Invalid exposure horizon or hold cap')
    part = rows[['id', 'date', 't_mix_usable']].copy()
    part['date'] = pd.to_datetime(part.date)
    if part.date.isna().any():
        raise ValueError('Missing hydraulic timestamp')
    part = part[part.date.lt(end)].sort_values('date').reset_index(drop=True)
    if part.id.isna().any() or part.id.nunique() > 1:
        raise ValueError('Temperature intervals require one known pipe')
    if part.date.duplicated().any():
        raise ValueError('Duplicate hydraulic timestamp')
    next_date = part.date.shift(-1).fillna(end)
    part['interval_end'] = pd.concat([
        next_date, part.date + pd.Timedelta(days=max_hold_days)], axis=1).min(axis=1).clip(upper=end)
    part['interval_start'] = part.date.clip(lower=start)
    temperature = part.t_mix_usable.to_numpy(dtype=float, na_value=np.nan)
    valid = (np.isfinite(temperature) & (temperature > -273.15) &
             part.interval_end.gt(part.interval_start))
    part = part[valid].copy()
    part['duration_years'] = (part.interval_end - part.interval_start).dt.total_seconds() / YEAR_SECONDS
    part['temperature_K'] = part.t_mix_usable + 273.15
    horizon = (end - start).total_seconds() / YEAR_SECONDS
    supported = float(part.duration_years.sum())
    previous, longest = start, 0.0
    for row in part.itertuples():
        longest = max(longest, (row.interval_start - previous).total_seconds() / 86400)
        previous = row.interval_end
    longest = max(longest, (end - previous).total_seconds() / 86400)
    return part, dict(observed_years=supported, unobserved_years=max(0.0, horizon-supported),
                      coverage_fraction=min(1.0, supported/horizon), longest_gap_days=longest,
                      exposure_policy=f'bounded_forward_hold_{max_hold_days:g}_days')


def temperature_features(rows, start, end, *, ea_kj_mol, gap_policy, max_hold_days=1.0):
    """Compute the full audited contract for one pipe/window and declared policy.

    ``reference`` explicitly assumes F=1 on gaps. ``observed_only`` leaves the
    reference integral null. Both retain coverage/missingness; neither claims
    unobserved time has zero corrosion. No candidate is implicitly selected.
    """
    candidates = {ea: name for name, ea in EA_CANDIDATES.items()}
    if ea_kj_mol not in candidates:
        raise ValueError('Choose an energy from the predeclared EA_CANDIDATES')
    if gap_policy not in ('reference', 'observed_only'):
        raise ValueError('Explicit gap_policy must be reference or observed_only')
    intervals, summary = temperature_intervals(rows, start, end, max_hold_days)
    observed = summary['observed_years']
    gap = summary['unobserved_years']
    factors = thermal_factor(intervals.temperature_K.to_numpy(float), ea_kj_mol)
    exposure = float((intervals.duration_years.to_numpy(float) * factors).sum())
    reference = exposure + gap if gap_policy == 'reference' else np.nan
    if gap_policy == 'reference' and ea_kj_mol == 0:
        reference = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / YEAR_SECONDS
    return {
        'thermal_exposure_observed_years': exposure,
        'thermal_exposure_reference_years': reference,
        'thermal_observed_years': observed,
        'thermal_unobserved_years': gap,
        'thermal_coverage_fraction': summary['coverage_fraction'],
        'thermal_longest_gap_days': summary['longest_gap_days'],
        'thermal_factor_observed_mean': exposure / observed if observed else np.nan,
        'thermal_history_missing': observed == 0,
        'thermal_history_incomplete': gap > 0,
        'thermal_theta_K': 1000.0 * ea_kj_mol / R_J_MOL_K,
        'thermal_ea_kJ_mol': float(ea_kj_mol),
        'thermal_candidate_id': candidates[ea_kj_mol],
        'thermal_gap_policy': gap_policy,
        'thermal_exposure_policy': summary['exposure_policy'],
        'thermal_profile_source': 'section_t_mix',
    }


class TemperatureFeatureRegistry(FeatureRegistry):
    """Separate, opt-in registry using the existing descriptor/report contract.

    Ea/theta/candidate/policies are metadata, deliberately excluded from numeric
    model feature selection. All are attached alongside the feature columns.
    Each instance binds one candidate, policy and cutoff-specific histories;
    create a fresh instance if histories change. Cached summaries are per window.
    """

    def __init__(self, histories, *, ea_kj_mol, gap_policy, max_hold_days=1.0):
        # Validate parameters even when applying to an empty window table.
        if ea_kj_mol not in EA_CANDIDATES.values():
            raise ValueError('Choose an energy from the predeclared EA_CANDIDATES')
        if gap_policy not in ('reference', 'observed_only'):
            raise ValueError('Explicit gap_policy must be reference or observed_only')
        if not np.isfinite(max_hold_days) or max_hold_days <= 0:
            raise ValueError('Invalid hold cap')
        self.histories = histories
        self.parameters = dict(ea_kj_mol=ea_kj_mol, gap_policy=gap_policy, max_hold_days=max_hold_days)
        self._cache = {}
        super().__init__()

    def _register_default_features(self):
        for name, (unit, description) in THERMAL_DEFINITIONS.items():
            self.register(FeatureDescriptor(
                feature_name=name, formula_version='v1.0_t_only_bounded_arrhenius',
                source_columns=['t_mix_usable', 'date', 'start_date', 'prediction_date'],
                category='thermal_exposure', description=f'{description} [{unit}]',
                compute_fn=lambda frame, column=name: self._column(frame, column)))

    def _records(self, frame):
        records = []
        for row in frame.itertuples():
            cutoff = pd.Timestamp(row.prediction_date)
            key = (int(row.id), pd.Timestamp(row.start_date), cutoff)
            if key not in self._cache:
                # Missing map entries are integration mistakes, not empty histories.
                history = self.histories[(int(row.id), cutoff)]
                if not history.id.eq(row.id).all():
                    raise ValueError('History pipe does not match the feature window')
                if 'prediction_date' in history and not history.prediction_date.eq(cutoff).all():
                    raise ValueError('History QA cutoff does not match the feature window')
                self._cache[key] = temperature_features(history, key[1], cutoff, **self.parameters)
            records.append(self._cache[key])
        return records

    def _column(self, frame, name):
        values = [record[name] for record in self._records(frame)]
        dtype = bool if name.startswith('thermal_history_') else float
        return pd.Series(values, index=frame.index, dtype=dtype)

    def apply(self, df, feature_names=None):
        # Always attach coverage/missingness, even for a caller selecting one feature.
        names = list(dict.fromkeys(list(feature_names or THERMAL_FEATURE_COLUMNS) +
                                  list(THERMAL_FEATURE_COLUMNS)))
        out, reports = super().apply(df, names)
        records = self._records(df)
        for name in THERMAL_METADATA_COLUMNS:
            dtype = float if name in ('thermal_theta_K', 'thermal_ea_kJ_mol') else 'string'
            out[name] = pd.Series([record[name] for record in records], index=df.index, dtype=dtype)
        return out, reports


def attach_temperature_features(windows, histories, *, ea_kj_mol, gap_policy, max_hold_days=1.0):
    """Return a copy enriched with T-only columns plus FeatureReport audit rows."""
    return TemperatureFeatureRegistry(
        histories, ea_kj_mol=ea_kj_mol, gap_policy=gap_policy,
        max_hold_days=max_hold_days).apply(windows)

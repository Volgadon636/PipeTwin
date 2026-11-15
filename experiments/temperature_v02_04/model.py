"""Opt-in temperature rate model; does not register with production models."""
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from src.ml.models import sample_weights
from src.ml.temperature import (EA_CANDIDATES, R_J_MOL_K, REFERENCE_TEMPERATURE_K,
                                temperature_intervals, thermal_factor)
from src.ml.windows import YEAR_SECONDS, eligible_training
from experiments.temperature_v02_03_1.run import fit_rate, exposure_and_derivative

POLICIES = {'observed_only': 'observed_only', 'reference_gap': 'reference'}
WINDOW_KEYS = ['id', 'start_date', 'prediction_date']


def validate_parameters(ea_kj_mol, gap_policy, hold_days):
    if ea_kj_mol not in EA_CANDIDATES.values():
        raise ValueError('Ea must be a fixed declared grid value')
    if gap_policy not in POLICIES:
        raise ValueError('Declare observed_only or reference_gap')
    if not np.isfinite(hold_days) or hold_days <= 0 or hold_days > 1:
        raise ValueError('Hold must be positive and at most one day')


def eligible_cohort(windows, request, parents):
    excluded = parents.loc[request['id']] if request['case'] == 'cold_start' else None
    return eligible_training(windows, request['prediction_date'], request['case'], excluded).reset_index(drop=True)


class ExposureCache:
    """Freeze bounded intervals once, independent of candidate and gap policy."""
    def __init__(self, histories):
        self.histories = histories
        self.intervals = {}
        self.summaries = {}

    def window(self, pid, start, end, hold):
        key = (int(pid), pd.Timestamp(start), pd.Timestamp(end), float(hold))
        if key not in self.intervals:
            rows = self.histories[(key[0], key[2])]
            if (not rows.id.eq(pid).all() or not rows.date.lt(key[2]).all() or
                    ('prediction_date' in rows and not rows.prediction_date.eq(key[2]).all())):
                raise ValueError('History id/as-of QA cutoff mismatch')
            ints, summary = temperature_intervals(rows, key[1], key[2], hold)
            columns = ['id','date','interval_start','interval_end','duration_years','temperature_K']
            checksum = hashlib.sha256(ints[columns].to_csv(index=False,float_format='%.17g').encode()).hexdigest()
            self.intervals[key] = ints
            self.summaries[key] = dict(**summary, interval_sha256=checksum,
                thermal_usable_rows=len(ints), thermal_valid_days=int(ints.date.dt.normalize().nunique()),
                thermal_temp_min_C=ints.t_mix_usable.min(), thermal_temp_max_C=ints.t_mix_usable.max(),
                thermal_source_date_min=ints.date.min(), thermal_source_date_max=ints.date.max(),
                thermal_fallback_fraction=0.)
        return self.intervals[key], self.summaries[key]

    def attach(self, frame, *, ea_kj_mol=40., gap_policy='reference_gap', hold_days=1.):
        validate_parameters(ea_kj_mol, gap_policy, hold_days)
        candidate = next(k for k,v in EA_CANDIDATES.items() if v==ea_kj_mol)
        records = []
        values = {}
        for row in frame.itertuples():
            key = (row.id, pd.Timestamp(row.start_date), pd.Timestamp(row.prediction_date))
            if key not in values:
                ints, summary = self.window(*key, hold_days)
                horizon = (key[2]-key[1]).total_seconds()/YEAR_SECONDS
                e, _, eo, ho, hg = exposure_and_derivative(ints,horizon,ea_kj_mol,POLICIES[gap_policy])
                factors = thermal_factor(ints.temperature_K.to_numpy(float),ea_kj_mol)
                values[key] = dict(
                    exposure_years=e, thermal_exposure_observed_years=eo,
                    thermal_exposure_reference_years=e if gap_policy=='reference_gap' else np.nan,
                    thermal_observed_years=ho, thermal_unobserved_years=hg,
                    thermal_coverage_fraction=summary['coverage_fraction'],
                    thermal_longest_gap_days=summary['longest_gap_days'],
                    thermal_factor_observed_mean=eo/ho if ho else np.nan,
                    thermal_history_missing=ho==0, thermal_history_incomplete=hg>0,
                    thermal_theta_K=1000*ea_kj_mol/R_J_MOL_K, thermal_ea_kJ_mol=ea_kj_mol,
                    thermal_reference_temperature_K=REFERENCE_TEMPERATURE_K,
                    thermal_candidate_id=candidate, thermal_gap_policy=POLICIES[gap_policy],
                    thermal_exposure_policy=summary['exposure_policy'], thermal_profile_source='section_t_mix',
                    thermal_factor_p95_observed=float(np.quantile(factors,.95)) if len(factors) else np.nan,
                    thermal_factor_max_observed=float(factors.max()) if len(factors) else np.nan,
                    **{k:v for k,v in summary.items() if k.startswith('thermal_') or k=='interval_sha256'})
            horizon = (key[2]-key[1]).total_seconds()/YEAR_SECONDS
            if not np.isclose(row.horizon_years,horizon,rtol=1e-13,atol=1e-13):
                raise ValueError('Horizon does not match window dates')
            records.append(values[key])
        out = frame.copy()
        for name in records[0] if records else []:
            out[name] = [r[name] for r in records]
        return out


@dataclass(frozen=True)
class TemperatureRateModel:
    case: str
    ea_kj_mol: float = 40.
    gap_policy: str = 'reference_gap'
    hold_days: float = 1.
    rate_mm_year: float | None = None

    def __post_init__(self):
        validate_parameters(self.ea_kj_mol,self.gap_policy,self.hold_days)
        if self.case not in ('between_vtd','cold_start'):
            raise ValueError('Unknown production scenario')
        if self.rate_mm_year is not None and (not np.isfinite(self.rate_mm_year) or self.rate_mm_year < 0):
            raise ValueError('Rate must be finite and nonnegative')

    def predict(self, enriched):
        if self.rate_mm_year is None:
            raise ValueError('Unavailable fit: no candidate forecast can be exported')
        expected = [('case',self.case),('thermal_ea_kJ_mol',self.ea_kj_mol),
                    ('thermal_gap_policy',POLICIES[self.gap_policy]),
                    ('thermal_exposure_policy',f'bounded_forward_hold_{self.hold_days:g}_days')]
        if any(not enriched[col].eq(value).all() for col,value in expected):
            raise ValueError('Prediction features do not match fitted scenario/energy/policy/hold')
        h0 = enriched.h0_mm.to_numpy(float)
        exposure = enriched.exposure_years.to_numpy(float)
        if not np.isfinite(np.r_[h0,exposure]).all() or (h0<=0).any() or (exposure<0).any():
            raise ValueError('Invalid baseline or exposure')
        with np.errstate(over='ignore',invalid='ignore'):
            loss = self.rate_mm_year*exposure
        if not np.isfinite(loss).all():
            raise ValueError('Nonfinite predicted loss; no overflow clipping')
        return np.clip(h0-loss,0,h0)


def fit_temperature_rate(train, cache, *, ea_kj_mol=40., gap_policy='reference_gap', hold_days=1.):
    if train.empty or train['case'].nunique()!=1:
        raise ValueError('Fit requires one nonempty eligible scenario')
    if train[['main_pipeline_id','campaign_id','id','segment_id','start_date','prediction_date']].isna().any().any():
        raise ValueError('Missing training provenance')
    features = cache.attach(train,ea_kj_mol=ea_kj_mol,gap_policy=gap_policy,hold_days=hold_days)
    w = sample_weights(features,'W1')
    observed_loss = (features.h0_mm-features.min_residual_thickness_mm).to_numpy(float)
    fit = fit_rate(observed_loss,features.exposure_years.to_numpy(float),w)
    model = TemperatureRateModel(str(train['case'].iloc[0]),ea_kj_mol,gap_policy,hold_days,fit['r_ref_mm_year'])
    return model, features, fit

"""Small deterministic models. No empirical coefficient is a universal law."""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .exposure import integrated_years


def sample_weights(train, regime='W1'):
    if train.empty:
        raise ValueError('Cannot fit without eligible training labels')
    if regime == 'W0':
        weights = np.ones(len(train))
    elif regime in ('W1', 'W2'):
        # Equal parent-pipeline totals, equal campaigns within each parent.
        counts = train.groupby(['main_pipeline_id', 'campaign_id']).segment_id.transform('size')
        campaigns = train.groupby('main_pipeline_id').campaign_id.transform('nunique')
        weights = 1 / (counts.to_numpy(float) * campaigns.to_numpy(float))
        if regime == 'W2':
            # Fixed sensitivity multiplier, not tuned on held-out labels.
            threshold = float(train.min_residual_thickness_mm.quantile(.1))
            weights *= np.where(train.min_residual_thickness_mm.le(threshold), 2., 1.)
    else:
        raise ValueError(f'Unknown weighting: {regime}')
    return weights / weights.mean()


@dataclass
class ConstantRate:
    rate_mm_year: float = 0.0
    name: str = 'C0'
    weighting: str = 'none'

    def predict(self, frame):
        return np.clip(frame.h0_mm.to_numpy(float) - self.rate_mm_year * frame.horizon_years.to_numpy(float),
                       0, frame.h0_mm.to_numpy(float))

    def metadata(self):
        return asdict(self)


def fit_constant_rate(train, weighting='W1'):
    weights = sample_weights(train, weighting)
    years = train.horizon_years.to_numpy(float)
    loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)
    if not np.isfinite(np.r_[years, loss]).all() or (years <= 0).any():
        raise ValueError('Invalid training horizon/label')
    # Nonnegative WLS optimum in mm space, before output clipping. Retain
    # negative observed loss: clipping labels would bias the fitted rate upward.
    rate = max(0., float(np.sum(weights * years * loss) / np.sum(weights * years**2)))
    return ConstantRate(rate, 'M0', weighting)


@dataclass
class PhysicalModel:
    rate_mm_year: float
    reference: dict
    coefficients: list
    gap_policy: str
    objective: float

    def predict(self, frame, intervals):
        if len(frame) != len(intervals):
            raise ValueError('Exposure/prediction alignment mismatch')
        exposure = np.array([integrated_years(i, self.reference, np.array(self.coefficients),
                             self.gap_policy, t) for i, t in zip(intervals, frame.horizon_years)])
        return np.clip(frame.h0_mm.to_numpy(float) - self.rate_mm_year * exposure,
                       0, frame.h0_mm.to_numpy(float))

    def metadata(self):
        result = asdict(self)
        result['theta_T_K'] = 1000 * self.coefficients[0]
        result['name'] = 'M1'
        return result


def fit_physical_model(train, intervals, weighting='W1', gap_policy='reference', minimum_groups=3):
    """Regularized coordinate search, deliberately gated by independent groups.

    The minimum group count is an engineering guard, not proof of identifiability.
    Steps and L2 penalty are fixed experiment settings; no holdout tuning occurs.
    """
    if train.main_pipeline_id.nunique() < minimum_groups:
        raise ValueError(f'M1 needs at least {minimum_groups} independent parent pipelines')
    if len(intervals) != len(train):
        raise ValueError('Exposure/label alignment mismatch')
    observed = pd.concat(intervals, ignore_index=True).drop_duplicates(['id', 'date'])
    if observed.empty:
        raise ValueError('No valid exposure intervals')
    reference = {name: float(observed[name].median())
                 for name in ('temperature_K', 'water_fraction', 'velocity_m_s')}
    reference['velocity_offset_m_s'] = .1  # numerical regularizer; m/s
    weights = sample_weights(train, weighting)
    loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)

    def score(coef):
        exp = np.array([integrated_years(i, reference, coef, gap_policy, t)
                        for i, t in zip(intervals, train.horizon_years)])
        denom = np.sum(weights * exp**2)
        rate = max(0., float(np.sum(weights * exp * loss) / denom)) if denom > 0 else 0.
        error = float(np.average((loss - rate * exp)**2, weights=weights) + .1 * np.sum(coef**2))
        return error, rate

    coef = np.zeros(3)
    best, rate = score(coef)
    for step in (1., .25, .05):
        for _ in range(20):
            improved = False
            for axis in range(3):
                for sign in (-1, 1):
                    candidate = coef.copy()
                    candidate[axis] += sign * step
                    if candidate[0] < 0 or np.max(np.abs(candidate)) > 10:
                        continue
                    objective, candidate_rate = score(candidate)
                    if objective < best - 1e-12:
                        coef, best, rate = candidate, objective, candidate_rate
                        improved = True
            if not improved:
                break
    return PhysicalModel(rate, reference, coef.tolist(), gap_policy, best)

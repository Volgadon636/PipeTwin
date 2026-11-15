"""Environment-aware cold-start thinning with declared engineering priors.

The repository has only two independent *first-inspection* environments:

* MAIN-2 2018-09-23 — warm (~11 C), wet (~87 % watercut), moving (~0.6 m/s);
* MAIN-3 2023-08-12 — cold (~2 C), dry (~0.08 % watercut), slow (~0.13 m/s).

Fitting separate temperature, watercut and velocity exponents to two
confounded groups would be pseudo-identification. Instead the exponents are
**declared priors** (configurable), only the reference rate ``r_ref`` is fitted,
and a pre-declared grid is reported as sensitivity. No coefficient here is a
universal corrosion law; ``r_ref`` absorbs unknown chemistry, inhibition,
external corrosion and the fact that a 10 m segment minimum is an extreme
statistic rather than a mean wall loss.

Semi-empirical form (spec equations 7-8), per valid exposure interval:

    F(t) = exp(theta * (1/T_ref - 1/T) + beta_W * (W - W_ref)
               + beta_v * ln((v + v0) / (v_ref + v0)))
    loss = r_ref * (sum_t F(t) dt + unknown_gap_term) * age**(p-1)

``T`` is Kelvin, ``W`` is watercut as a fraction, ``v`` is m/s, ``dt`` years.
``p`` is an age exponent (1 = linear); it is a sensitivity scenario, not a
fitted law.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

THERMO_CLIP = 20.0


@dataclass(frozen=True)
class EnvironmentPriors:
    """Declared, non-fitted exponents and references (all dimensionless)."""

    theta_K: float = 2500.0          # effective Arrhenius sensitivity, K
    beta_W: float = 0.30             # watercut log-sensitivity
    beta_v: float = 0.10             # velocity power
    velocity_offset_m_s: float = 0.10
    age_exponent: float = 1.0

    def __post_init__(self):
        if self.theta_K < 0:
            raise ValueError('theta_K must be nonnegative (monotone warming assumption)')
        if self.velocity_offset_m_s <= 0:
            raise ValueError('velocity offset must be positive')
        if self.age_exponent <= 0:
            raise ValueError('age exponent must be positive')


def reference_conditions(train):
    """Train-only reference environment. Never call on the target pipe."""
    observed = train[['t_mix_mean', 'watercut_pct_mean', 'v_mix_mean']].dropna()
    if observed.empty:
        raise ValueError('No complete environment rows for reference conditions')
    return dict(
        temperature_K=float(observed.t_mix_mean.median()) + 273.15,
        water_fraction=float(observed.watercut_pct_mean.median()) / 100.0,
        velocity_m_s=float(observed.v_mix_mean.median()),
    )


def environment_factor(frame, reference, priors):
    """Multiplicative F_T * F_W * F_V for each row's as-of aggregates."""
    return _factor_from_arrays(frame.t_mix_mean.to_numpy(float) + 273.15,
                               frame.watercut_pct_mean.to_numpy(float) / 100.0,
                               frame.v_mix_mean.to_numpy(float), reference, priors)


def _factor_from_arrays(t, w, v, reference, priors):
    if not np.isfinite(np.r_[t, w, v]).all():
        raise ValueError('Environment factors need finite T, watercut and velocity')
    if np.any(t <= 0) or np.any(w < 0) or np.any(w > 1) or np.any(v < 0):
        raise ValueError('Invalid physical environment values')
    exponent = (
        priors.theta_K * (1.0 / reference['temperature_K'] - 1.0 / t)
        + priors.beta_W * (w - reference['water_fraction'])
        + priors.beta_v * np.log((v + priors.velocity_offset_m_s)
                                 / (reference['velocity_m_s'] + priors.velocity_offset_m_s))
    )
    return np.exp(np.clip(exponent, -THERMO_CLIP, THERMO_CLIP))


def effective_exposure(windows, intervals, reference, priors, gap_policy='own_mean'):
    """Years-equivalent exposure, in reference-rate units.

    ``intervals`` is the aligned list returned by ``attach_exposure``.
    ``gap_policy``:

    * ``own_mean``  — unobserved years run at the pipe's own observed mean
      factor (central case; assumes the observed regime is representative);
    * ``reference`` — unobserved years run at F = 1 (warm/wet reference);
    * ``observed_only`` — unobserved years contribute nothing (lower bound, not
      a measurement claim).

    The age exponent enters as ``exposure * age**(p-1)`` so that a constant
    environment reproduces ``r_ref * F * age**p``.
    """
    if gap_policy not in ('own_mean', 'reference', 'observed_only'):
        raise ValueError(f'Unknown gap policy: {gap_policy}')
    if len(intervals) != len(windows):
        raise ValueError('Exposure/window alignment mismatch')
    factors = environment_factor(windows, reference, priors)
    values = []
    for factor, interval, age in zip(factors, intervals, windows.age_years.to_numpy(float)):
        if age <= 0:
            raise ValueError('Nonpositive age')
        duration = float(interval.duration_years.sum())
        if duration > 0:
            interval_factor = _factor_from_arrays(interval.temperature_K.to_numpy(float),
                                                  interval.water_fraction.to_numpy(float),
                                                  interval.velocity_m_s.to_numpy(float),
                                                  reference, priors)
            observed = float(np.dot(interval_factor, interval.duration_years))
        else:
            observed = 0.0
        gap = max(0.0, age - duration)
        if gap_policy == 'reference':
            observed += gap
        elif gap_policy == 'own_mean':
            # Unobserved years run at this pipe's own observed mean environment;
            # with no observed interval use the as-of aggregate factor.
            observed += gap * (observed / duration if duration > 0 else factor)
        values.append(observed * age ** (priors.age_exponent - 1.0))
    return np.asarray(values, float)


def sample_weights(train, regime='W1'):
    """Equal total weight per parent pipeline, then per campaign."""
    if regime == 'W0':
        weights = np.ones(len(train))
    elif regime in ('W1', 'W2'):
        counts = train.groupby(['main_pipeline_id', 'campaign_id']).segment_id.transform('size')
        campaigns = train.groupby('main_pipeline_id').campaign_id.transform('nunique')
        weights = 1.0 / (counts.to_numpy(float) * campaigns.to_numpy(float))
        if regime == 'W2':
            threshold = float(train.min_residual_thickness_mm.quantile(.1))
            weights *= np.where(train.min_residual_thickness_mm.le(threshold), 2.0, 1.0)
    else:
        raise ValueError(f'Unknown weighting: {regime}')
    return weights / weights.mean()


@dataclass
class ColdStartPhysics:
    """``h = clip(h_nominal - r_ref * exposure, 0, h_nominal)``."""

    r_ref_mm_year: float
    reference: dict
    priors: dict
    gap_policy: str
    weighting: str
    training_rows: int = 0
    training_groups: int = 0

    name: str = 'M1'

    def predict(self, frame, intervals):
        exposure = effective_exposure(frame, intervals, self.reference,
                                      EnvironmentPriors(**self.priors), self.gap_policy)
        nominal = frame.nominal_thickness_mm.to_numpy(float)
        return np.clip(nominal - self.r_ref_mm_year * exposure, 0.0, nominal)

    def exposure(self, frame, intervals):
        return effective_exposure(frame, intervals, self.reference,
                                  EnvironmentPriors(**self.priors), self.gap_policy)

    def metadata(self):
        result = asdict(self)
        result['name'] = self.name
        result['interpretation'] = (
            'Effective segment-minimum thinning rate under declared environment priors; '
            'not an electrochemical CO2/H2S constant')
        return result


def fit_cold_start_physics(train, intervals, priors=None, gap_policy='own_mean',
                           weighting='W1', reference=None):
    """Fit only the nonnegative reference rate; priors stay declared."""
    priors = priors or EnvironmentPriors()
    if len(intervals) != len(train):
        raise ValueError('Exposure/label alignment mismatch')
    reference = reference or reference_conditions(train)
    exposure = effective_exposure(train, intervals, reference, priors, gap_policy)
    loss = (train.nominal_thickness_mm - train.min_residual_thickness_mm).to_numpy(float)
    if not np.isfinite(np.r_[exposure, loss]).all() or (exposure <= 0).any():
        raise ValueError('Invalid training exposure or label')
    weights = sample_weights(train, weighting)
    denominator = float(np.sum(weights * exposure ** 2))
    rate = max(0.0, float(np.sum(weights * exposure * loss) / denominator))
    groups = train.groupby(['main_pipeline_id', 'campaign_id']).ngroups if len(train) else 0
    return ColdStartPhysics(rate, reference, asdict(priors), gap_policy, weighting,
                            training_rows=len(train), training_groups=groups)


def scenario_grid():
    """Pre-declared cold-start sensitivity grid (no held-out selection)."""
    for theta in (0.0, 2500.0, 5000.0):
        for beta_w in (0.0, 0.3, 0.6):
            for beta_v in (0.0, 0.1, 0.2):
                for exponent in (0.75, 1.0, 1.25):
                    for gap in ('own_mean', 'reference', 'observed_only'):
                        yield EnvironmentPriors(theta_K=theta, beta_W=beta_w,
                                                 beta_v=beta_v, age_exponent=exponent), gap

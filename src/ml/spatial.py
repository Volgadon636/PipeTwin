"""Regularized within-pipe profile model and leave-one-campaign-out evidence.

A 10 m segment minimum has genuine within-pipe structure only if it is
reproducible across independent inspections. Earlier repository audits found
that thickness *levels* barely persist between campaigns (2018 -> 2022 Pearson
r ~ 0.25 on the large pipe, ~0.08 from 2022 -> 2023 on another), so a spatial
model is only defensible as a **small** deviation from the pipe-level physics
forecast. This module estimates that deviation from static profile features,
centres everything within a pipe, and reports out-of-sample correlation across
held-out campaigns. It is deliberately gated: if the pooled held-out
correlation is not positive, the caller must keep the deviation at zero.

No random seeds, no hidden state, closed-form ridge. Only NumPy/pandas.
"""
import numpy as np
import pandas as pd

FEATURES = ['z', 'slope', 'abs_slope', 'position', 'local_low', 'pool_depth']
ROLLING_SEGMENTS = 21


def spatial_features(frame, rolling=ROLLING_SEGMENTS):
    """Static profile features, computed per pipe, in physical units.

    ``local_low`` = z - rolling median (local topographic relief);
    ``pool_depth`` = z - rolling minimum (depth below the local low point).
    """
    parts = []
    for _, group in frame.groupby('id', sort=True):
        group = group.sort_values('segment_id').copy()
        z = group.z_center.to_numpy(float)
        slope = group.slope_m_per_m.to_numpy(float)
        start = group.start_m.to_numpy(float)
        length = float(np.nanmax(group.end_m.to_numpy(float))) if len(group) else 1.0
        series = pd.Series(z)
        group['z'] = z
        group['slope'] = slope
        group['abs_slope'] = np.abs(slope)
        group['position'] = start / max(length, 1.0)
        group['local_low'] = z - series.rolling(rolling, center=True, min_periods=1).median().to_numpy()
        group['pool_depth'] = z - series.rolling(rolling, center=True, min_periods=1).min().to_numpy()
        parts.append(group)
    result = pd.concat(parts, ignore_index=True) if parts else frame.copy()
    for column in FEATURES:
        if result[column].isna().any() or not np.isfinite(result[column].to_numpy(float)).all():
            raise ValueError(f'Nonfinite spatial feature: {column}')
    return result


def standardize_within_pipe(frame):
    """Centre/scale each feature by pipe, then pool. Values are dimensionless."""
    result = frame.copy()
    for column in FEATURES:
        grouped = result.groupby('id')[column]
        mean = grouped.transform('mean')
        scale = grouped.transform('std').replace(0.0, np.nan)
        result[column] = ((result[column] - mean) / scale).fillna(0.0)
    return result


def fit_spatial_ridge(frame, lam=10.0, target='min_residual_thickness_mm', group='id'):
    """Closed-form ridge on within-pipe centred features and labels."""
    if frame[target].isna().any():
        raise ValueError('Spatial fit requires complete labels')
    if not lam > 0:
        raise ValueError('Ridge penalty must be positive')
    y = (frame[target] - frame.groupby(group)[target].transform('mean')).to_numpy(float)
    x = frame[FEATURES].to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite spatial design or label')
    beta = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)
    return dict(coefficients=beta.tolist(), features=list(FEATURES), ridge_lambda=float(lam))


def predict_spatial(frame, model):
    """Deviation from the pipe mean, in mm. Mean-zero by construction upstream."""
    if list(model['features']) != FEATURES:
        raise ValueError('Spatial feature contract mismatch')
    return frame[FEATURES].to_numpy(float) @ np.asarray(model['coefficients'], float)


def leave_one_campaign_out(frame, lam=10.0, target='min_residual_thickness_mm',
                           group='id', campaign='prediction_date'):
    """Fit on every other campaign, predict the held-out one.

    Returns per-campaign correlations and the pooled out-of-sample correlation.
    A campaign is a physical inspection; 10 m neighbours are never treated as
    independent evidence.
    """
    keys = frame[group].astype(str) + ':' + frame[campaign].astype(str)
    frame = frame.assign(_campaign=keys)
    groups = sorted(frame._campaign.unique())
    if len(groups) < 2:
        return dict(groups=[], pooled_pearson=None,
                    status='fewer_than_two_campaigns')
    records, truths, predictions = [], [], []
    for held in groups:
        train = frame[frame._campaign != held]
        test = frame[frame._campaign == held]
        model = fit_spatial_ridge(train, lam, target, group)
        y = test[target].to_numpy(float)
        y = y - y.mean()
        p = predict_spatial(test, model)
        if np.ptp(y) <= 1e-12 or np.ptp(p) <= 1e-12:
            record = dict(campaign=held, n=len(test), pearson=None, pearson_squared=None)
        else:
            r = float(np.corrcoef(y, p)[0, 1])
            record = dict(campaign=held, n=len(test), pearson=r, pearson_squared=r * r)
        records.append(record)
        truths.append(y)
        predictions.append(p)
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    if np.ptp(truth) <= 1e-12 or np.ptp(prediction) <= 1e-12:
        pooled = None
    else:
        pooled = float(np.corrcoef(truth, prediction)[0, 1])
    return dict(groups=records, pooled_pearson=pooled,
                pooled_pearson_squared=None if pooled is None else pooled * pooled,
                ridge_lambda=float(lam), status='evaluated')


def gate_spatial(cv, minimum_pooled_r=0.0):
    """A spatial correction may enter production only with positive evidence."""
    pooled = cv.get('pooled_pearson')
    usable = pooled is not None and pooled > minimum_pooled_r
    return dict(usable=bool(usable), pooled_pearson=pooled,
                reason=('positive out-of-sample correlation' if usable
                        else 'no positive leave-one-campaign-out evidence; deviation set to zero'))

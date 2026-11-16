"""Fixed research models for spatial covariance, Pearson loss, and pooling."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge

from src.experiments.r2_pool_stage6 import demean_training_profile
from src.ml.next_common import fit_ridge, make_weights, predict_ridge

GP = 'spatial_gaussian_process'
PEARSON = 'profile_pearson_loss'
NORMALIZED = 'normalized_training_residual'
CENTERED = 'centered_training_residual'


def spatial_covariance(x, u, x_ref, u_ref, config):
    distance = np.sqrt(3.) * np.abs(u[:, None] - u_ref) / config['length_fraction']
    feature_scale = x.shape[1] * config['feature_length_per_sqrt_dimension'] ** 2
    feature_distance = cdist(x, x_ref, metric='sqeuclidean') / feature_scale
    spatial = (1. + distance) * np.exp(-distance - .5 * feature_distance)
    return x @ x_ref.T + config['spatial_variance'] * spatial


def fit_gaussian_process(x, location, y, config):
    covariance = spatial_covariance(x, location, x, location, config)
    covariance.flat[::len(x) + 1] += config['noise_to_linear_variance']
    mean = float(y.mean())
    dual = cho_solve(cho_factor(covariance, lower=True), y - mean)
    return x, location, dual, mean


def predict_gaussian_process(model, x, location, config):
    x_ref, u_ref, dual, mean = model
    return mean + spatial_covariance(x, location, x_ref, u_ref, config) @ dual


def pearson_loss_and_gradient(coef, profiles):
    """Average of the differentiable profile losses and their analytic gradient."""
    losses = []
    gradient = np.zeros_like(coef)
    for x, y, base in profiles:
        pred = base + x @ coef
        denominator = np.dot(pred, pred) + 1e-12
        numerator = np.dot(y, pred)
        truth_norm_sq = np.dot(y, y)
        losses.append(1. - numerator**2 / (truth_norm_sq * denominator))
        d_r_squared = 2. * numerator / (truth_norm_sq * denominator) * (
            y - numerator / denominator * pred)
        gradient -= x.T @ d_r_squared
    return float(np.mean(losses)), gradient / len(profiles)


def fit_pearson_loss(train, x, truth, level, initial, config):
    profiles, group_audit = [], []
    for (pid, date), idx in train.groupby(['id', 'prediction_date'], sort=True).indices.items():
        y, p, design = truth[idx], level[idx], x[idx]
        used = len(idx) >= config['minimum_profile_rows'] and np.ptp(y) > 1e-12
        if used:
            profiles.append((design - design.mean(axis=0), y - y.mean(), p - p.mean()))
        group_audit.append({'id': int(pid), 'date': str(date.date()), 'rows': len(idx), 'used': bool(used)})
    norm = float(np.linalg.norm(initial))
    if not profiles or norm <= 1e-12:
        return initial, {'profiles': group_audit, 'status': 'fallback_no_nonconstant_training_direction',
                         'coefficient_norm': norm}
    initial_loss, _ = pearson_loss_and_gradient(initial, profiles)
    # Optimize a unit vector while keeping the scale of the paired Ridge.
    def objective(direction):
        loss, gradient = pearson_loss_and_gradient(norm * direction, profiles)
        return loss, norm * gradient

    result = minimize(
        objective, initial / norm, jac=True, method='SLSQP',
        constraints=[{'type': 'eq', 'fun': lambda d: float(d @ d - 1.), 'jac': lambda d: 2. * d}],
        options={'maxiter': config['maximum_iterations'], 'ftol': config['ftol']},
    )
    accepted = (result.success and np.isfinite(result.x).all() and np.isfinite(result.fun)
                and abs(float(np.linalg.norm(result.x)) - 1.) < 1e-7
                and result.fun <= initial_loss + 1e-10)
    coef = norm * result.x if accepted else initial
    return coef, {
        'profiles': group_audit, 'status': 'optimized' if accepted else 'fallback_to_linear',
        'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
        'iterations': int(result.nit), 'coefficient_norm': norm,
        'initial_mean_r_squared': 1. - initial_loss,
        'fitted_mean_r_squared': 1. - pearson_loss_and_gradient(coef, profiles)[0],
        'coefficients': coef.tolist(),
    }


def fit_normalized(train, features, response, config):
    centered = demean_training_profile(response, train).ravel()
    groups = [train.id.to_numpy(), train.prediction_date.to_numpy()]
    variance = (train.assign(centered_square=centered**2)
                .groupby(groups).centered_square.transform('mean').to_numpy(float))
    scale = np.maximum(np.sqrt(variance), config['minimum_scale_mm'])
    output_scale = float(np.sqrt(np.mean(scale**2)))
    targets = {NORMALIZED: (centered / scale, output_scale), CENTERED: (centered, 1.)}
    fits = {}
    for name, (target, multiplier) in targets.items():
        fitted = fit_ridge(train.assign(stage7_response=target), features, 'stage7_response',
                           make_weights(train, 'row_uniform'), alpha=config['ridge_alpha'], model_name=name)
        fits[name] = fitted, multiplier
    by_profile = train[['id', 'prediction_date']].assign(scale_mm=scale).drop_duplicates()
    return fits, by_profile


def fit_independent(train, features, level, residual, linear, position, setup):
    prep = linear['preprocessor']
    design = prep.transform(train)
    target = train[residual].to_numpy(float)
    gp = fit_gaussian_process(design, train[position].to_numpy(float), target, setup['H31'])
    direction, pearson_audit = fit_pearson_loss(
        train, design, train.next_mm.to_numpy(float), np.asarray(level, dtype=float),
        linear['estimator'].coef_, setup['H32'])
    scaled, profiles = fit_normalized(train, features, target, setup['H33'])
    audit = {
        GP: {'preprocessing': prep.metadata(), 'position': position,
             'training_rows': len(train), 'noise': setup['H31']['noise_to_linear_variance']},
        PEARSON: pearson_audit,
        NORMALIZED: {'profiles': profiles.to_dict('records'),
                     'models': {name: {'fit': fit['metadata'], 'output_scale': multiplier}
                                for name, (fit, multiplier) in scaled.items()}},
    }
    return {'linear': linear, 'position': position, 'gp': gp, 'pearson_coef': direction,
            'scaled': scaled, 'audit': audit}


def independent_predictions(models, frame, setup):
    linear = models['linear']
    x = linear['preprocessor'].transform(frame)
    location = frame[models['position']].to_numpy(float)
    result = {
        GP: predict_gaussian_process(models['gp'], x, location, setup['H31']),
        PEARSON: linear['estimator'].intercept_ + x @ models['pearson_coef'],
    }
    for name, (fit, multiplier) in models['scaled'].items():
        result[name] = multiplier * predict_ridge(fit, frame)
    return result


def joint_design(s1, s2, common_indices):
    shared_s1 = s1[:, common_indices] / np.sqrt(2.)
    shared_s2 = s2 / np.sqrt(2.)
    private_s1 = s1.copy()
    private_s1[:, common_indices] /= np.sqrt(2.)
    first = np.column_stack([shared_s1, private_s1, np.zeros((len(s1), s2.shape[1]))])
    second = np.column_stack([shared_s2, np.zeros((len(s2), s1.shape[1])), shared_s2])
    return first, second


def fit_joint_s1(train_s1, train_s2, residual_s2, linear, config):
    prep = linear['preprocessor']
    common = {name: column for name, column in config['shared_features'].items()
              if name in prep.used_features_}
    indices = [prep.used_features_.index(name) for name in common]
    x1 = prep.transform(train_s1)
    x2 = np.column_stack([
        (train_s2[column].fillna(prep.imputer_statistics_[name]).to_numpy(float)
         - prep.scaler_mean_[name]) / prep.scaler_scale_[name] for name, column in common.items()])
    d1, d2 = joint_design(x1, x2, indices)
    y1 = train_s1.proxy_residual.to_numpy(float)
    y2 = demean_training_profile(residual_s2, train_s2).ravel()
    mu1 = d1.mean(axis=0)
    fit = Ridge(alpha=config['ridge_alpha'], solver='cholesky', fit_intercept=False).fit(
        np.vstack([d1 - mu1, d2 - d2.mean(axis=0)]),
        np.r_[y1 - y1.mean(), y2 - y2.mean()],
        sample_weight=np.r_[np.ones(len(y1)), np.full(len(y2), len(y1) / len(y2))])
    intercept = float(y1.mean() - mu1 @ fit.coef_)
    audit = {'shared_features': common, 's1_rows': len(train_s1), 's2_rows': len(train_s2),
             's2_ids': sorted(int(pid) for pid in train_s2.id.unique()),
             's2_dates': sorted(str(date.date()) for date in train_s2.prediction_date.unique()),
             's1_weight_per_row': 1., 's2_weight_per_row': len(y1) / len(y2),
             'coefficients': fit.coef_.tolist(), 's1_intercept': intercept}
    return prep, indices, fit, intercept, audit


def predict_joint_s1(model, frame):
    prep, indices, fit, intercept, _ = model
    d1, _ = joint_design(prep.transform(frame), np.empty((0, len(indices))), indices)
    return fit.predict(d1) + intercept

"""Strict forward-time, held-parent-pipeline evaluation and honest metrics."""
import numpy as np

from .models import ConstantRate, fit_constant_rate
from .windows import eligible_training


def metrics(truth, predictions, thin_threshold=None):
    y, p = np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not np.isfinite(np.r_[y, p]).all():
        raise ValueError('Metrics require aligned finite nonempty arrays')
    error = p - y
    variance = float(np.sum((y-y.mean())**2))
    nonconstant = np.ptp(y) > 1e-12 and np.ptp(p) > 1e-12
    result = dict(n=len(y), mae_mm=float(np.mean(np.abs(error))),
                  rmse_mm=float(np.sqrt(np.mean(error**2))), bias_mm=float(error.mean()),
                  r2=float(1-np.sum(error**2)/variance) if variance > 1e-12 else None,
                  pearson_squared=float(np.corrcoef(y, p)[0, 1]**2) if nonconstant else None,
                  pearson_status='defined' if nonconstant else 'undefined_constant_vector',
                  mape_pct=float(100*np.mean(np.abs(error/y))) if np.all(y > .1) else None)
    thin = np.zeros(len(y), dtype=bool) if thin_threshold is None else y <= thin_threshold
    result.update(thin_threshold_mm=thin_threshold, thin_count=int(thin.sum()),
                  thin_mae_mm=float(np.mean(np.abs(error[thin]))) if thin.any() else None,
                  thin_recall=float(np.mean(p[thin] <= thin_threshold)) if thin.any() else None)
    return result


def evaluate_forward(windows, cutoff):
    """Only labels earlier than cutoff; each fit excludes the entire test parent.

    With no earlier training campaign, C0 can still be evaluated. Fitted models
    are reported as unavailable, not trained using same-day inspection outcomes.
    """
    permitted = windows[windows.prediction_date.lt(cutoff)]
    records, predictions = [], []
    for (case, parent, date), test in permitted.groupby(['case', 'main_pipeline_id', 'prediction_date']):
        train = eligible_training(permitted, date, case, excluded_parent=parent)
        base = dict(case=case, heldout_parent=parent, prediction_date=str(date.date()),
                    train_rows=len(train), test_rows=len(test),
                    train_parent_count=int(train.main_pipeline_id.nunique()),
                    train_date_max=None if train.empty else str(train.prediction_date.max().date()),
                    horizon_min_years=float(test.horizon_years.min()),
                    horizon_max_years=float(test.horizon_years.max()))
        models = [ConstantRate()]
        if not train.empty:
            models += [fit_constant_rate(train, regime) for regime in ('W0', 'W1', 'W2')]
            try:
                from .models import RidgeCorrosionModel, LightGBMCorrosionModel, CatBoostCorrosionModel, EnsembleCorrosionModel
                m_ridge = RidgeCorrosionModel(alpha=10.0, weighting='W1').fit(train)
                m_lgb = LightGBMCorrosionModel(weighting='W1').fit(train)
                m_cat = CatBoostCorrosionModel(weighting='W1').fit(train)
                m0_w1 = fit_constant_rate(train, 'W1')
                m_ens = EnsembleCorrosionModel(m0_w1, m_lgb, gb_weight=0.5)
                models += [m_ridge, m_lgb, m_cat, m_ens]
            except Exception:
                pass
        else:
            records.append({**base, 'model': 'M0', 'status': 'no_earlier_independent_training_group'})
        for model in models:
            p = model.predict(test)
            threshold = None if train.empty else float(train.min_residual_thickness_mm.quantile(.1))
            for subset, mask in [('all', np.ones(len(test), dtype=bool)),
                                 ('missing_prior_baseline', test.baseline_missing.to_numpy(bool))]:
                if mask.any():
                    records.append({**base, **model.metadata(), 'model': model.name, 'subset': subset,
                                    'status': 'evaluated',
                                    **metrics(test.min_residual_thickness_mm.to_numpy()[mask], p[mask], threshold)})
            out = test[['id', 'segment_id', 'case', 'prediction_date', 'main_pipeline_id',
                        'h0_mm', 'horizon_years', 'baseline_missing', 'min_residual_thickness_mm']].copy()
            out['thickness_pred'] = p
            out['model'] = model.name
            out['weighting'] = model.weighting
            predictions.append(out)
    return records, predictions


def masked_baseline_check(features, block_size=10):
    """Secondary spatial diagnostic for nearest-prior baseline reconstruction.

    Hide contiguous blocks of prior measured segments. Never use target-date
    labels; this tests interpolation error, not future corrosion accuracy.
    """
    known = features[features.has_previous_vtd].sort_values('segment_id')
    actual, estimated = [], []
    for _, hidden in known.groupby(known.segment_id // block_size):
        available = known[~known.index.isin(hidden.index)]
        if available.empty:
            continue
        for row in hidden.itertuples():
            neighbor = available.loc[(available.start_m-row.start_m).abs().idxmin()]
            actual.append(row.last_vtd_thickness_mm)
            estimated.append(neighbor.last_vtd_thickness_mm)
    return dict(method='100m_block_masked_prior_measurements', independent_validation=False,
                **metrics(actual, estimated)) if actual else dict(status='insufficient_prior_measurements')

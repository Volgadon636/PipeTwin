"""Strict forward-time, held-parent-pipeline evaluation and honest metrics."""
import numpy as np

from .models import ConstantRate, fit_constant_rate
from .windows import eligible_training


def _null_metrics(reason):
    return dict(n=0, mae_mm=None, rmse_mm=None, bias_mm=None, r2=None,
                r2_status='undefined', r2_reason=reason,
                pearson_r=None, pearson_squared=None,
                pearson_status='undefined', pearson_reason=reason,
                mape_pct=None, mape_status='undefined', mape_reason=reason,
                mape_eligible_pct=None, mape_eligible_count=0,
                mape_excluded_count=0, mape_policy='truth>0.1 mm; current evaluator suppresses MAPE otherwise',
                thin_threshold_mm=None, thin_count=0, thin_mae_mm=None,
                thin_recall=None, inverse_ranking_flag=False)


def metrics(truth, predictions, thin_threshold=None, mape_threshold=0.1):
    """Aligned held-out metrics; undefined scores are null, never fabricated.

    ``pearson_squared`` is the organizer metric (squared correlation), not R2.
    A perfect inverse ranking has signed Pearson r=-1 and squared value 1, so the
    signed r and an explicit inverse-ranking flag are Always reported together.
    Existing MAPE policy is preserved: ``mape_pct`` is null if ANY truth <= 0.1 mm.
    A separately labelled eligible-subset MAPE is also returned.
    """
    y, p = np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)
    if y.shape != p.shape:
        raise ValueError('Metrics require aligned arrays')
    if len(y) == 0:
        return _null_metrics('empty_sample')
    if not np.isfinite(np.r_[y, p]).all():
        raise ValueError('Metrics require finite arrays; report nonfinite inputs as a data error')
    error = p - y
    variance = float(np.sum((y-y.mean())**2))
    y_constant = np.ptp(y) <= 1e-12
    p_constant = np.ptp(p) <= 1e-12
    nonconstant = not y_constant and not p_constant
    if not nonconstant:
        reason = 'constant_truth' if y_constant else 'constant_prediction'
    else:
        reason = ''
    pearson_r = float(np.corrcoef(y, p)[0, 1]) if nonconstant else None
    pearson_squared = float(pearson_r ** 2) if pearson_r is not None else None
    result = dict(
        n=len(y), mae_mm=float(np.mean(np.abs(error))),
        rmse_mm=float(np.sqrt(np.mean(error**2))), bias_mm=float(error.mean()),
        r2=float(1-np.sum(error**2)/variance) if variance > 1e-12 else None,
        r2_status='defined' if variance > 1e-12 else 'undefined_constant_truth',
        r2_reason='' if variance > 1e-12 else 'constant_truth',
        pearson_r=pearson_r, pearson_squared=pearson_squared,
        pearson_status='defined' if nonconstant else 'undefined_constant_vector',
        pearson_reason=reason,
        mape_pct=float(100*np.mean(np.abs(error/y))) if np.all(y > mape_threshold) else None,
        mape_status='defined' if np.all(y > mape_threshold) else 'suppressed_near_zero_truth',
        mape_reason='' if np.all(y > mape_threshold) else f'truth has value <= {mape_threshold}',
        mape_eligible_pct=(float(100*np.mean(np.abs(error[y > mape_threshold] / y[y > mape_threshold])))
                           if np.any(y > mape_threshold) else None),
        mape_eligible_count=int(np.sum(y > mape_threshold)),
        mape_excluded_count=int(np.sum(y <= mape_threshold)),
        mape_policy=f'truth>{mape_threshold} mm; current evaluator suppresses MAPE otherwise',
        inverse_ranking_flag=bool(pearson_r is not None and pearson_r < 0),
    )
    thin = np.zeros(len(y), dtype=bool) if thin_threshold is None else y <= thin_threshold
    result.update(thin_threshold_mm=thin_threshold, thin_count=int(thin.sum()),
                  thin_mae_mm=float(np.mean(np.abs(error[thin]))) if thin.any() else None,
                  thin_recall=float(np.mean(p[thin] <= thin_threshold)) if thin.any() else None)
    return result


def top_fraction_overlap(truth, predictions, fraction=0.1, groups=None, tie_policy='stable_index'):
    """Top-k overlap/recall where truth supports ranking.

    Selection uses k=ceil(fraction*n) within each group. Ties are broken by stable
    input order for reproducibility; a constant predictor's overlap is reported
    alongside the tie count, not presented as spatial skill.
    """
    y = np.asarray(truth, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if y.shape != p.shape or len(y) == 0 or not np.isfinite(np.r_[y, p]).all():
        raise ValueError('Top-fraction overlap requires aligned finite nonempty arrays')
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    groups = np.zeros(len(y), dtype=int) if groups is None else np.asarray(groups)
    records, all_truth, all_pred = [], [], []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        k = int(np.ceil(fraction * len(idx)))
        truth_order = idx[np.argsort(y[idx], kind='stable')]
        pred_order = idx[np.argsort(p[idx], kind='stable')]
        truth_top, pred_top = set(truth_order[:k].tolist()), set(pred_order[:k].tolist())
        overlap = len(truth_top & pred_top)
        records.append({'group': str(group), 'n': int(len(idx)), 'k': k,
                        'overlap': int(overlap), 'overlap_fraction': overlap / k,
                        'truth_recall': overlap / k if k else None,
                        'prediction_tie_count': int((p[idx] == p[idx][0]).sum()) if len(idx) else 0,
                        'tie_policy': tie_policy})
        all_truth.extend(truth_top)
        all_pred.extend(pred_top)
    if len(np.unique(groups)) > 1:
        records.append({'group': '__pooled__', 'n': int(len(y)),
                        'k': int(np.ceil(fraction * len(y))),
                        'overlap': len(set(all_truth) & set(all_pred)),
                        'overlap_fraction': (len(set(all_truth) & set(all_pred)) /
                                             max(1, int(np.ceil(fraction * len(y))))),
                        'truth_recall': (len(set(all_truth) & set(all_pred)) /
                                         max(1, int(np.ceil(fraction * len(y))))),
                        'prediction_tie_count': None, 'tie_policy': tie_policy})
    return {'fraction': fraction, 'k_rule': 'ceil(fraction*n) per group',
            'tie_policy': tie_policy, 'groups': records,
            'warning': 'Ties do not establish spatial discrimination; use deterministic order only for reproducibility.'}


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

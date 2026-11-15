"""Isolated Stage 03.1 historical experiment. Run as a module from repo root."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.models import fit_constant_rate, sample_weights
from src.ml.temperature import (EA_CANDIDATES, R_J_MOL_K, REFERENCE_TEMPERATURE_K,
                                temperature_intervals, thermal_factor)
from src.ml.windows import YEAR_SECONDS, build_training_windows, eligible_training

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KEYS = ['main_pipeline_id', 'campaign_id', 'id', 'segment_id', 'case',
        'start_date', 'prediction_date']
WINDOW = ['main_pipeline_id', 'campaign_id', 'id', 'case', 'start_date', 'prediction_date']
COMPARE = ['case', 'hold_days', 'gap_policy']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def write_json(name, value):
    (OUT / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save(name, records):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT / name, index=False, float_format='%.17g')
    return frame


def interval_hash(intervals, include_id=True):
    cols = (['id'] if include_id else []) + [
        'date', 'interval_start', 'interval_end', 'duration_years', 'temperature_K']
    return hashlib.sha256(intervals[cols].to_csv(index=False, float_format='%.17g').encode()).hexdigest()


def exposure_and_derivative(intervals, horizon, ea, policy):
    """Derivative is with respect to Ea in kJ/mol; gap derivative is zero."""
    if policy not in ('observed_only', 'reference'):
        raise ValueError('Unknown gap policy')
    dt = intervals.duration_years.to_numpy(float)
    kelvin = intervals.temperature_K.to_numpy(float)
    factor = thermal_factor(kelvin, ea)
    observed = float(np.sum(dt * factor))
    support = float(dt.sum())
    if not np.isfinite(horizon) or horizon <= 0 or support > horizon + 1e-12:
        raise ValueError('Invalid horizon/support')
    gap = max(0., horizon - support)
    exposure = observed + gap if policy == 'reference' else observed
    if ea == 0 and policy == 'reference':
        exposure = horizon  # exact M0 identity, as in the Stage 02 API
    derivative = float(np.sum(dt * factor * (1000. / R_J_MOL_K) *
                              (1. / REFERENCE_TEMPERATURE_K - 1. / kelvin)))
    return exposure, derivative, observed, support, gap


def fit_rate(loss, exposure, weights):
    y, e, w = map(lambda x: np.asarray(x, float), (loss, exposure, weights))
    if (not len(y) or y.shape != e.shape or y.shape != w.shape or
            not np.isfinite(np.r_[y, e, w]).all() or (e < 0).any() or (w <= 0).any()):
        raise ValueError('Require aligned finite losses, nonnegative exposures and positive weights')
    with np.errstate(over='ignore', invalid='ignore'):
        denom = float(np.sum(w * e**2))
        numerator = float(np.sum(w * e * y))
        zero_sse = float(np.sum(w * y**2))
    if not np.isfinite([denom, numerator, zero_sse]).all():
        raise ValueError('Nonfinite rate-fit sums; no overflow clipping')
    if denom == 0:
        return dict(status='unavailable_zero_exposure_denominator', r_ref_mm_year=None,
                    denominator=0., weighted_sse_mm2=None, weighted_rmse_mm=None,
                    zero_rate_sse_mm2=zero_sse)
    rate = max(0., numerator / denom)
    sse = float(np.sum(w * (y - rate * e)**2))
    return dict(status='fitted', r_ref_mm_year=rate, denominator=denom,
                weighted_sse_mm2=sse, weighted_rmse_mm=float(np.sqrt(sse/w.sum())),
                zero_rate_sse_mm2=zero_sse)


def thickness_metrics(frame, predictions):
    result = metrics(frame.min_residual_thickness_mm.to_numpy(float), predictions)
    return {k: result[k] for k in ['n', 'mae_mm', 'rmse_mm', 'bias_mm',
                                  'pearson_squared', 'pearson_status']}


def shape(exposure, reference, weights):
    x = np.sqrt(weights) * np.asarray(exposure)
    b = np.sqrt(weights) * np.asarray(reference)
    nx, nb = np.linalg.norm(x), np.linalg.norm(b)
    if nx == 0 or nb == 0:
        return dict(status='undefined_zero_vector', cosine_similarity=None,
                    proportionality_residual=None, scale_vs_reference=None)
    scale = float(np.dot(x, b) / np.dot(b, b))
    return dict(status='defined', cosine_similarity=float(np.dot(x/nx, b/nb)),
                proportionality_residual=float(np.linalg.norm(x-scale*b)/nx),
                scale_vs_reference=scale)


def sensitivity(exposure, derivative, weights, rate):
    if rate is None or rate <= 0:
        return dict(status='degenerate_zero_or_unavailable_rate', numerical_rank=0,
                    effective_normalized_rank=0)
    matrix = np.sqrt(weights)[:, None] * rate * np.column_stack([exposure, derivative])
    raw = np.linalg.svd(matrix, compute_uv=False)
    norms = np.linalg.norm(matrix, axis=0)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    scaled = np.linalg.svd(normalized, compute_uv=False)
    return dict(status='defined' if (norms > 0).all() else 'degenerate_zero_column',
                column_log_rate_norm=float(norms[0]), column_ea_norm=float(norms[1]),
                normalized_column_cosine=float(normalized[:, 0] @ normalized[:, 1]),
                singular_value_1=float(raw[0]), singular_value_2=float(raw[1]) if len(raw)>1 else 0.,
                numerical_rank=int(np.linalg.matrix_rank(matrix)),
                normalized_singular_value_1=float(scaled[0]),
                normalized_singular_value_2=float(scaled[1]) if len(scaled)>1 else 0.,
                effective_normalized_rank=int(np.sum(scaled > scaled[0]*1e-6)))


def relative_change(value, comparator):
    if not np.isfinite(value) or not np.isfinite(comparator):
        return None, 'unavailable'
    if comparator == 0:
        return (0., 'equal_zero') if value == 0 else (None, 'undefined_zero_comparator')
    return float(value/comparator - 1.), 'defined'


def fit_ties(values):
    finite = {k: float(v) for k, v in values.items() if np.isfinite(v)}
    if not finite:
        return []
    minimum = min(finite.values())
    tolerance = 1e-12 if minimum == 0 else .01 * minimum
    return [k for k, v in finite.items() if v - minimum <= tolerance]


def mae_ties(values):
    finite = {k: float(v) for k, v in values.items() if np.isfinite(v)}
    if not finite:
        return []
    best = min(finite.values())
    return [k for k, v in finite.items() if v-best < max(.01, .01*v) or v == best]


def target_cohort(windows, request, parents):
    excluded = parents.loc[request['id']] if request['case'] == 'cold_start' else None
    return eligible_training(windows, request['prediction_date'], request['case'], excluded).reset_index(drop=True)


def make_folds(cohort, request, fingerprints):
    """Index sets always refer to this target's already eligible cohort."""
    audits, folds = [], []
    for (parent, date), test in cohort.groupby(['main_pipeline_id', 'prediction_date']):
        train = eligible_training(cohort, date, request['case'], excluded_parent=parent)
        key = f'independent:{parent}:{date.date()}'
        audits.append(dict(case=request['case'], fold_id=key, evidence_status='forward_time_held_parent_selection_diagnostic',
                           heldout_parent=parent, test_date=date, train_rows=len(train), test_rows=len(test),
                           train_date_max=train.prediction_date.max(), train_parent_count=train.main_pipeline_id.nunique(),
                           admissible=not train.empty, reason='admissible' if len(train) else 'no_earlier_independent_training_group'))
        if len(train):
            folds.append((key, 'forward_time_held_parent_selection_diagnostic', train.index, test.index))
    if folds:
        return audits, folds
    for (parent, campaign), group in cohort.groupby(['main_pipeline_id', 'campaign_id']):
        for section, test in group.groupby('id'):
            train = group[~group.id.eq(section)]
            shared = set(fingerprints.loc[train.index]) & set(fingerprints.loc[test.index])
            reason = ('fewer_than_three_sections' if group.id.nunique() < 3 else
                      'identical_trajectories_cross_split' if shared else
                      'empty_calibration_or_test' if train.empty or test.empty else 'admissible')
            key = f'section:{campaign}:{section}'
            audits.append(dict(case=request['case'], fold_id=key,
                               evidence_status='within_parent_same_campaign_diagnostic',
                               heldout_parent=parent, campaign_id=campaign, heldout_section=section,
                               test_date=test.prediction_date.max(), train_date_max=train.prediction_date.max(),
                               train_rows=len(train), test_rows=len(test),
                               train_parent_count=train.main_pipeline_id.nunique(),
                               identical_trajectories_cross_split=len(shared),
                               admissible=reason == 'admissible', reason=reason))
            if reason == 'admissible':
                folds.append((key, 'within_parent_same_campaign_diagnostic', train.index, test.index))
    return audits, folds


def association_records(observations, base):
    variables = ['t_mix_usable', 'elapsed_years', 'watercut_pct_usable', 'v_liquid_true_usable']
    records = []
    for a, b in itertools.combinations(variables, 2):
        pair = observations[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        defined = len(pair)>1 and pair[a].nunique()>1 and pair[b].nunique()>1
        records.append(dict(**base, variable_x=a, variable_y=b, observation_rows=len(observations),
                            paired_valid_rows=len(pair), x_missing=int(observations[a].isna().sum()),
                            y_missing=int(observations[b].isna().sum()),
                            pearson=float(pair[a].corr(pair[b])) if defined else None,
                            status='descriptive_only' if defined else 'undefined_insufficient_or_constant'))
    return records


def decision_for_case(case, fits, stability, grouped, shapes, sensitivities):
    fs, st = fits[fits['case'].eq(case)], stability[stability['case'].eq(case)]
    primary = st[st.hold_days.eq(1)]
    winners = []
    for policy in ('observed_only', 'reference'):
        part = primary[primary.gap_policy.eq(policy)]
        tied = mae_ties(dict(zip(part.candidate_id, part.equal_section_mae_mm)))
        winners.append(tied[0] if len(tied) == 1 else None)
    proposed = winners[0] if winners[0] is not None and winners[0] == winners[1] else None
    usable = int(primary.usable_section_count.min()) if len(primary) else 0
    shape_rows = shapes[shapes['case'].eq(case) & shapes.reference_candidate.eq('EA00_null') & ~shapes.candidate_id.eq('EA00_null')]
    any_shape = bool((shape_rows.proportionality_residual > 1e-6).any())
    rules = dict(grouped_transfer_available=usable >= 3,
                 unique_primary_best_both_policies=proposed is not None,
                 beats_null_beyond_tolerance=None, no_short_hold_reversal=None,
                 distinguishable_shape=None, full_effective_sensitivity_rank=None,
                 survives_delete_one_section=None)
    if proposed is not None:
        selected = st[st.candidate_id.eq(proposed)]
        rules['beats_null_beyond_tolerance'] = proposed != 'EA00_null' and all(
            float(primary[(primary.gap_policy.eq(p)) & primary.candidate_id.eq('EA00_null')].equal_section_mae_mm.iloc[0]) -
            float(primary[(primary.gap_policy.eq(p)) & primary.candidate_id.eq(proposed)].equal_section_mae_mm.iloc[0]) >
            max(.01, .01*float(primary[(primary.gap_policy.eq(p)) & primary.candidate_id.eq('EA00_null')].equal_section_mae_mm.iloc[0]))
            for p in ('observed_only', 'reference'))
        rules['no_short_hold_reversal'] = bool(selected.mae_practical_tie.all())
        own_shape = shape_rows[shape_rows.candidate_id.eq(proposed)]
        rules['distinguishable_shape'] = len(own_shape) == 6 and bool((own_shape.proportionality_residual>1e-6).all())
        own_sens = sensitivities[sensitivities['case'].eq(case) & sensitivities.candidate_id.eq(proposed)]
        rules['full_effective_sensitivity_rank'] = len(own_sens) == 6 and bool(own_sens.effective_normalized_rank.eq(2).all())
        survives = True
        for _, part in grouped[grouped['case'].eq(case) & grouped.hold_days.eq(1)].groupby('gap_policy'):
            for section in part.heldout_section.unique():
                means = part[~part.heldout_section.eq(section)].groupby('candidate_id').mae_mm.mean().to_dict()
                survives &= mae_ties(means) == [proposed]
        rules['survives_delete_one_section'] = bool(survives)
    selected = proposed if all(v is True for v in rules.values()) else None
    outcome = ('provisional_development_choice' if selected else
               'indeterminate' if usable < 3 or not any_shape else 'candidate_set_retained')
    return dict(outcome=outcome, selected_candidate=selected,
                selected_ea_kJ_mol=EA_CANDIDATES[selected] if selected else None,
                retained_candidates=[selected] if selected else list(EA_CANDIDATES),
                rule_results=rules, failed_rules=[k for k,v in rules.items() if v is False],
                not_assessed_without_unique_proposal=[k for k,v in rules.items() if v is None],
                any_numerically_distinguishable_shape=any_shape,
                primary_fit_ties={p: fit_ties(dict(zip(g.candidate_id, g.weighted_sse_mm2)))
                                  for p,g in fs[fs.hold_days.eq(1)].groupby('gap_policy')},
                interpretation='Internal selection evidence only; no physical or independent validation.')


def generate():
    protocol = json.loads((OUT/'preregistration.json').read_text())
    before = json.loads((OUT/'protected_before.json').read_text())
    assert digest(OUT/'preregistration.json') == before['preregistration_sha256'], 'Protocol changed without amendment'
    assert protocol['candidates_kJ_mol'] == dict(EA_CANDIDATES)
    assert protocol['R_J_mol_K'] == R_J_MOL_K and protocol['reference_temperature_K'] == REFERENCE_TEMPERATURE_K
    data = RepositoryData(ROOT)
    windows, histories = build_training_windows(data)
    parents = data.parent_groups()
    print('Built historical windows once; input checksum verification passed.', flush=True)
    all_fits, all_shapes, all_sens, all_predictions, all_grouped = [], [], [], [], []
    audits, folds_audit, interval_audit, exposures, associations, overlap = [], [], [], [], [], []
    intervals_out, cohort_summaries = [], []
    trajectory_rows, baseline_checks = [], []
    for request in protocol['targets']:
        case, cutoff = request['case'], pd.Timestamp(request['prediction_date'])
        cohort = target_cohort(windows, request, parents)
        assert cohort.prediction_date.lt(cutoff).all()
        if case == 'cold_start':
            assert not cohort.main_pipeline_id.eq(parents.loc[request['id']]).any()
        assert not cohort.id.eq(1250009583).any()
        required = cohort[['h0_mm','min_residual_thickness_mm','horizon_years']].to_numpy(float)
        assert np.isfinite(required).all()
        assert not cohort[KEYS].isna().any().any()
        cohort['history_id'] = cohort.groupby(WINDOW, sort=True).ngroup().map(lambda x: f'{case}:{x}')
        # Audit every reconstructed window against this target, with explicit exclusion reason.
        raw = windows[KEYS + ['baseline_source','baseline_missing','h0_mm','min_residual_thickness_mm','horizon_years']].copy()
        raw['target_case'], raw['target_id'], raw['production_cutoff'] = case, request['id'], cutoff
        raw['exclusion_reason'] = np.select([
            ~raw['case'].eq(case), ~raw.prediction_date.lt(cutoff),
            raw.main_pipeline_id.eq(parents.loc[request['id']]) & (case == 'cold_start'),
            raw.baseline_missing & (case == 'between_vtd')],
            ['different_case','label_not_before_target_cutoff','cold_start_target_parent','missing_prior_baseline'], default='included')
        raw['negative_observed_loss'] = raw.h0_mm.lt(raw.min_residual_thickness_mm)
        audits.append(raw)
        cohort_summaries.append(dict(case=case, target_id=request['id'], cutoff=cutoff,
            rows=len(cohort), parents=sorted(cohort.main_pipeline_id.unique()),
            campaigns=sorted(cohort.campaign_id.unique()), sections=int(cohort.id.nunique()),
            history_windows=int(cohort.history_id.nunique()), latest_label=cohort.prediction_date.max(),
            negative_observed_loss_rows=int(cohort.h0_mm.lt(cohort.min_residual_thickness_mm).sum()),
            baseline_sources=cohort.baseline_source.value_counts().to_dict(),
            exclusion_counts=raw.exclusion_reason.value_counts().to_dict()))
        print(clean(cohort_summaries[-1]), flush=True)
        weights = sample_weights(cohort, 'W1')
        loss = (cohort.h0_mm-cohort.min_residual_thickness_mm).to_numpy(float)
        fixed = {}
        fingerprint_by_hold = {}
        case_observations = []
        for hold in protocol['holds_days']:
            fingerprint = {}
            for key, group in cohort.groupby(WINDOW, sort=True):
                parent, campaign, pid, _, start, end = key
                hid = group.history_id.iloc[0]
                rows = histories[(int(pid), end)]
                assert rows.date.lt(end).all() and rows.prediction_date.eq(end).all() and rows.id.eq(pid).all()
                intervals, summary = temperature_intervals(rows, start, end, hold)
                fixed[(hid, hold)] = intervals
                fingerprint[hid] = interval_hash(intervals, include_id=False)
                horizon = float(group.horizon_years.iloc[0])
                base = dict(zip(WINDOW, key))
                meta = dict(**base, history_id=hid, hold_days=hold, segment_rows=len(group),
                            horizon_years=horizon, **summary, interval_rows=len(intervals),
                            interval_sha256=interval_hash(intervals), trajectory_sha256=fingerprint[hid])
                interval_audit.append(meta)
                intervals_out.append(intervals.assign(history_id=hid, case=case, hold_days=hold,
                                                     start_date=start, prediction_date=end))
                trajectory_rows.append(dict(**meta, temperature_mean_C=intervals.t_mix_usable.mean(),
                                            temperature_min_C=intervals.t_mix_usable.min(),
                                            temperature_max_C=intervals.t_mix_usable.max()))
                if hold == 1:
                    fields = ['id','date','watercut_pct_usable','v_liquid_true_usable']
                    obs = intervals[['id','date','t_mix_usable']].merge(rows[fields], on=['id','date'], validate='one_to_one')
                    obs['elapsed_years'] = (obs.date-start).dt.total_seconds()/YEAR_SECONDS
                    obs['history_id'] = hid
                    associations.extend(association_records(obs, dict(case=case, scope='window_contributing_timestamps', history_id=hid)))
                    case_observations.append(obs)
                for candidate, ea in EA_CANDIDATES.items():
                    for policy in protocol['gap_policies']:
                        e, de, eo, ho, hg = exposure_and_derivative(intervals, horizon, ea, policy)
                        exposures.append(dict(**base, history_id=hid, hold_days=hold, gap_policy=policy,
                            candidate_id=candidate, ea_kJ_mol=ea, exposure_years=e, derivative_years_per_kJ_mol=de,
                            observed_exposure_years=eo, observed_years=ho, unobserved_years=hg,
                            horizon_years=horizon, observed_factor_mean=eo/ho if ho else None,
                            history_missing=ho == 0, interval_sha256=meta['interval_sha256']))
                assert interval_hash(intervals) == meta['interval_sha256']
            fingerprint_by_hold[hold] = cohort.history_id.map(fingerprint)
        # Trajectory duplication is checked at every hold, not just at one day.
        for hold in protocol['holds_days']:
            fa, fold_sets = make_folds(cohort, request, fingerprint_by_hold[hold])
            folds_audit.extend(dict(**r, hold_days=hold, production_cutoff=cutoff) for r in fa)
            for ha, hb in itertools.combinations(cohort.history_id.unique(), 2):
                a, b = fixed[(ha, hold)], fixed[(hb, hold)]
                ga, gb = cohort[cohort.history_id.eq(ha)].iloc[0], cohort[cohort.history_id.eq(hb)].iloc[0]
                overlap.append(dict(case=case, hold_days=hold, history_a=ha, history_b=hb,
                    same_section=ga.id == gb.id, same_parent=ga.main_pipeline_id == gb.main_pipeline_id,
                    same_campaign=ga.campaign_id == gb.campaign_id,
                    shared_source_timestamps=len(set(a.date)&set(b.date)),
                    overlapping_window_days=max(0., (min(ga.prediction_date,gb.prediction_date)-max(ga.start_date,gb.start_date)).total_seconds()/86400),
                    identical_trajectory=interval_hash(a,False)==interval_hash(b,False)))
            edf = pd.DataFrame(exposures)
            edf = edf[edf['case'].eq(case) & edf.hold_days.eq(hold)]
            for policy in protocol['gap_policies']:
                vectors = {}
                for candidate in EA_CANDIDATES:
                    rows = edf[edf.gap_policy.eq(policy) & edf.candidate_id.eq(candidate)].set_index('history_id')
                    vectors[candidate] = rows.loc[cohort.history_id]
                for candidate, ea in EA_CANDIDATES.items():
                    v = vectors[candidate]
                    e, de = v.exposure_years.to_numpy(float), v.derivative_years_per_kJ_mol.to_numpy(float)
                    base = dict(case=case, target_id=request['id'], production_cutoff=cutoff,
                                hold_days=hold, gap_policy=policy, candidate_id=candidate, ea_kJ_mol=ea)
                    fit = fit_rate(loss, e, weights)
                    pred = None if fit['r_ref_mm_year'] is None else np.clip(cohort.h0_mm.to_numpy()-fit['r_ref_mm_year']*e, 0, cohort.h0_mm.to_numpy())
                    counts = dict(rows=len(cohort), parents=cohort.main_pipeline_id.nunique(),
                                  campaigns=cohort.campaign_id.nunique(), sections=cohort.id.nunique(),
                                  distinct_history_windows=cohort.history_id.nunique(),
                                  distinct_thermal_trajectories=fingerprint_by_hold[hold].nunique(),
                                  distinct_exposure_values=len(np.unique(e)),
                                  negative_observed_loss_rows=int((loss<0).sum()), zero_exposure_rows=int((e==0).sum()),
                                  missing_temperature_history_rows=int(v.history_missing.sum()))
                    coverage = dict(observed_support_fraction=float(np.average(v.observed_years/v.horizon_years,weights=weights)),
                                    unknown_share=float(np.average(v.unobserved_years/v.horizon_years,weights=weights)),
                                    reference_gap_exposure_share=float(np.sum(weights*v.unobserved_years)/np.sum(weights*e)) if policy=='reference' and np.sum(weights*e)>0 else None,
                                    negative_loss_weighted_sse_mm2=float(np.sum(weights[loss<0]*(loss[loss<0]-(fit['r_ref_mm_year'] or 0)*e[loss<0])**2)))
                    all_fits.append(dict(**base, **fit, **counts, **coverage,
                                         **(thickness_metrics(cohort,pred) if pred is not None else {})))
                    all_sens.append(dict(**base, **sensitivity(e,de,weights,fit['r_ref_mm_year'])))
                    for ref in ('EA00_null','EA40_mid'):
                        all_shapes.append(dict(**base, reference_candidate=ref,
                                               **shape(e,vectors[ref].exposure_years.to_numpy(float),weights)))
                    if candidate=='EA00_null' and policy=='reference':
                        m0 = fit_constant_rate(cohort,'W1')
                        np.testing.assert_array_equal(e,cohort.horizon_years.to_numpy(float))
                        assert fit['r_ref_mm_year'] == m0.rate_mm_year
                        np.testing.assert_array_equal(pred,m0.predict(cohort))
                        baseline_checks.append(dict(**base, rate_equal=True, exposure_equal=True, predictions_equal=True,
                                                    m0_rate_mm_year=m0.rate_mm_year))
                    if pred is not None:
                        p = cohort[KEYS+['history_id','h0_mm','min_residual_thickness_mm','baseline_source','baseline_missing']].copy()
                        for k,value in base.items():
                            p[k]=value
                        p['split']='historical_fit'; p['fold_id']='all_eligible'; p['evidence_status']='in_sample_diagnostic'
                        p['exposure_years']=e; p['observed_years']=v.observed_years.to_numpy(); p['unobserved_years']=v.unobserved_years.to_numpy()
                        p['history_missing']=v.history_missing.to_numpy(); p['r_ref_mm_year']=fit['r_ref_mm_year']
                        p['predicted_loss_mm']=fit['r_ref_mm_year']*e; p['h_pred_mm']=pred; p['observed_loss_mm']=loss
                        p['negative_observed_loss']=loss<0; p['calibration_rows']=len(cohort)
                        p['calibration_section_ids']=';'.join(map(str,sorted(cohort.id.unique())))
                        all_predictions.append(p)
                    for fold_id, status, train_idx, test_idx in fold_sets:
                        tr, te = cohort.loc[train_idx],cohort.loc[test_idx]
                        trw = sample_weights(tr,'W1')
                        fitted=fit_rate(loss[train_idx],e[train_idx],trw)
                        if fitted['r_ref_mm_year'] is None:
                            all_grouped.append(dict(**base,fold_id=fold_id,evidence_status=status,
                                heldout_section=int(te.id.iloc[0]) if te.id.nunique()==1 else None,
                                train_rows=len(tr),test_rows=len(te),status=fitted['status']))
                            continue
                        tp=np.clip(te.h0_mm.to_numpy()-fitted['r_ref_mm_year']*e[test_idx],0,te.h0_mm.to_numpy())
                        # Equal-section aggregation remains valid even if an independent fold contains several sections.
                        for sid, section in te.groupby('id'):
                            mask=te.id.eq(sid).to_numpy()
                            all_grouped.append(dict(**base, fold_id=fold_id,evidence_status=status,
                                heldout_section=int(sid),train_rows=len(tr),test_rows=len(section),
                                calibration_section_ids=';'.join(map(str,sorted(tr.id.unique()))),
                                r_ref_mm_year=fitted['r_ref_mm_year'], status='evaluated',
                                **thickness_metrics(section,tp[mask])))
                        p=te[KEYS+['history_id','h0_mm','min_residual_thickness_mm','baseline_source','baseline_missing']].copy()
                        for k,value in base.items(): p[k]=value
                        p['split']='held_out'; p['fold_id']=fold_id; p['evidence_status']=status
                        p['exposure_years']=e[test_idx]; p['observed_years']=v.observed_years.to_numpy()[test_idx]
                        p['unobserved_years']=v.unobserved_years.to_numpy()[test_idx]; p['history_missing']=v.history_missing.to_numpy()[test_idx]
                        p['r_ref_mm_year']=fitted['r_ref_mm_year']; p['predicted_loss_mm']=fitted['r_ref_mm_year']*e[test_idx]
                        p['h_pred_mm']=tp; p['observed_loss_mm']=loss[test_idx]; p['negative_observed_loss']=loss[test_idx]<0
                        p['calibration_rows']=len(tr); p['calibration_section_ids']=';'.join(map(str,sorted(tr.id.unique())))
                        all_predictions.append(p)
        obs=pd.concat(case_observations,ignore_index=True).sort_values('elapsed_years',ascending=False).drop_duplicates(['id','date'])
        associations.extend(association_records(obs,dict(case=case,scope='deduplicated_section_timestamp_earliest_window_start',history_id='all')))
        print(f'{case}: completed all 30 fits and permitted holdouts.',flush=True)
    fits=pd.DataFrame(all_fits)
    for _, group in fits.groupby(COMPARE):
        null=group[group.candidate_id.eq('EA00_null')].weighted_sse_mm2.iloc[0]
        minimum=group.weighted_sse_mm2.min()
        ties=fit_ties(dict(zip(group.candidate_id,group.weighted_sse_mm2)))
        for idx,row in group.iterrows():
            for label,comp in [('null',null),('minimum',minimum)]:
                change,status=relative_change(row.weighted_sse_mm2,comp)
                fits.loc[idx,f'relative_sse_vs_{label}']=change
                fits.loc[idx,f'relative_sse_vs_{label}_status']=status
            fits.loc[idx,'fit_practical_tie']=row.candidate_id in ties
            fits.loc[idx,'fit_rank']=group.weighted_sse_mm2.rank(method='min').loc[idx]
    grouped=pd.DataFrame(all_grouped)
    if grouped.empty:
        grouped=pd.DataFrame(columns=COMPARE+['candidate_id','fold_id','heldout_section','status','mae_mm','rmse_mm','bias_mm'])
    for _,group in grouped.groupby(COMPARE+['heldout_section']):
        ties=mae_ties(dict(zip(group.candidate_id,group.mae_mm)))
        null=group[group.candidate_id.eq('EA00_null')].mae_mm.iloc[0]
        for idx,row in group.iterrows():
            grouped.loc[idx,'section_mae_rank']=group.mae_mm.rank(method='min').loc[idx]
            grouped.loc[idx,'section_mae_practical_tie']=row.candidate_id in ties
            grouped.loc[idx,'paired_mae_delta_vs_null_mm']=row.mae_mm-null
    stability=[]
    for key,group in fits.groupby(COMPARE):
        base=dict(zip(COMPARE,key))
        g=grouped.copy()
        for col,value in base.items(): g=g[g[col].eq(value)]
        for _,row in group.iterrows():
            cg=g[g.candidate_id.eq(row.candidate_id) & g.status.eq('evaluated')]
            section_mean=cg.groupby('heldout_section')[['mae_mm','rmse_mm','bias_mm']].mean()
            stability.append(dict(**base,candidate_id=row.candidate_id,ea_kJ_mol=row.ea_kJ_mol,
                fit_rank=row.fit_rank,fit_practical_tie=row.fit_practical_tie,
                usable_section_count=len(section_mean),equal_section_mae_mm=section_mean.mae_mm.mean(),
                equal_section_rmse_mm=section_mean.rmse_mm.mean(),equal_section_bias_mm=section_mean.bias_mm.mean(),
                opposing_sections_vs_null=int((cg.paired_mae_delta_vs_null_mm>0).sum()) if len(cg) else 0))
    stability=pd.DataFrame(stability)
    for _,group in stability.groupby(COMPARE):
        tied=mae_ties(dict(zip(group.candidate_id,group.equal_section_mae_mm)))
        for idx,row in group.iterrows():
            stability.loc[idx,'mae_practical_tie']=row.candidate_id in tied
            stability.loc[idx,'mae_rank']=group.equal_section_mae_mm.rank(method='min').loc[idx]
    shapes=pd.DataFrame(all_shapes); sens=pd.DataFrame(all_sens)
    decisions={case:decision_for_case(case,fits,stability,grouped,shapes,sens) for case in fits['case'].unique()}
    shared_candidates={d['selected_candidate'] for d in decisions.values()}
    shared=next(iter(shared_candidates)) if len(shared_candidates)==1 and None not in shared_candidates else None
    decision=dict(cases=decisions,shared_recommendation=shared,
        shared_ea_kJ_mol=EA_CANDIDATES[shared] if shared else None,
        shared_retained_candidates=[shared] if shared else list(EA_CANDIDATES),
        limitations=['One permitted parent/campaign per case in this checkout; sections are dependent.',
                     'No independent forward-time held-parent validation; no population confidence intervals.',
                     'Observed-only descriptors omit unknown corrosion exposure; reference filling assumes F=1.',
                     'Negative thinning is retained; contradictory readings and baseline uncertainty can dominate fit.',
                     'No effective physical activation energy established.'],
        stage04_handoff=dict(go='Isolated development and prospective validation design',
                             no_go='Unique scientifically selected Ea, production promotion, or physical validation claim',
                             continuity_setting='EA40 only if labelled unselected software continuity',
                             missing_evidence='Independent earlier calibration and later held-parent campaigns; timestamped T histories with much denser support and varied thermal trajectories, paired W/v, repeated reliable thickness/baseline measurements.'))
    save('cohort_audit.csv',pd.concat(audits,ignore_index=True)); write_json('cohort_summary.json',cohort_summaries)
    save('fold_audit.csv',folds_audit); save('interval_audit.csv',interval_audit)
    save('intervals.csv',pd.concat(intervals_out,ignore_index=True)); save('trajectory_summary.csv',trajectory_rows)
    save('trajectory_overlap.csv',overlap); save('candidate_exposures.csv',exposures)
    save('candidate_fits.csv',fits)
    save('profile_objective.csv',fits[COMPARE+['candidate_id','ea_kJ_mol','r_ref_mm_year','weighted_sse_mm2','weighted_rmse_mm','relative_sse_vs_null','relative_sse_vs_minimum','fit_rank','fit_practical_tie']])
    save('exposure_shape_diagnostics.csv',shapes); save('sensitivity_matrix_diagnostics.csv',sens)
    save('diagnostic_predictions.csv',pd.concat(all_predictions,ignore_index=True))
    save('grouped_transfer_metrics.csv',grouped); save('robustness_summary.csv',stability)
    save('temperature_associations.csv',associations); save('m0_identity.csv',baseline_checks)
    write_json('decision.json',decision)
    # Dimensionless excess over null shows attenuation by gaps without averaging policies.
    ee=pd.DataFrame(exposures)
    contrasts=[]
    for key,g in ee.groupby(['case','history_id','hold_days']):
        for candidate in EA_CANDIDATES:
            a=g[g.candidate_id.eq(candidate)].set_index('gap_policy')
            null=g[g.candidate_id.eq('EA00_null')].set_index('gap_policy')
            obs=null.loc['observed_only','exposure_years']; ref=null.loc['reference','exposure_years']
            contrasts.append(dict(case=key[0],history_id=key[1],hold_days=key[2],candidate_id=candidate,
                observed_excess_vs_null_fraction=(a.loc['observed_only','exposure_years']-obs)/obs if obs else None,
                reference_excess_vs_null_fraction=(a.loc['reference','exposure_years']-ref)/ref,
                attenuation_factor_support_fraction=obs/ref))
    save('reference_gap_attenuation.csv',contrasts)
    plot(fits)
    write_reports(protocol,cohort_summaries,fits,stability,decision)
    decision = enrich_outputs()
    changed=[name for name,h in before['sha256'].items() if not (ROOT/name).is_file() or digest(ROOT/name)!=h]
    # A separately disclosed concurrent ignore-file edit is outside scientific inputs.
    scientific_changes=[name for name in changed if name!='.gitignore']
    if scientific_changes: raise ValueError(f'Protected scientific files changed: {scientific_changes}')
    write_json('protected_verification.json',dict(phase='after_fits_before_tests',
        protected_files_checked=len(before['sha256']),changed=changed,
        changed_scientific_protected_files=scientific_changes))
    write_json('run_manifest.json',dict(branch=before['branch'],head=before['head'],initial_status_short=before['initial_status_short'],
        command='python -X utf8 -B -m experiments.temperature_v02_03_1.run',python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,preregistration_sha256=digest(OUT/'preregistration.json'),
        input_sha256=data.hashes,source_sha256={name:h for name,h in before['sha256'].items() if name.startswith('src/')},
        output_scope=str(OUT),spec_reading='Focused preamble applied; prior Russian v1.1 use recorded as handoff only',
        runtime_scope='Historical windows once; no production snapshot reconstruction or forecast generation'))
    print(json.dumps(clean(decision['cases']),indent=2),flush=True)


def plot(fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,case in enumerate(['between_vtd','cold_start']):
        for j,policy in enumerate(['observed_only','reference']):
            ax=axes[i,j]; rateax=ax.twinx()
            part=fits[fits['case'].eq(case)&fits.gap_policy.eq(policy)&fits.hold_days.eq(1)].sort_values('ea_kJ_mol')
            ax.plot(part.ea_kJ_mol,part.weighted_sse_mm2,'o-',color='#245a81',label='W1 SSE')
            rateax.plot(part.ea_kJ_mol,part.r_ref_mm_year,'s--',color='#b35d16',label='Fitted rate')
            ax.set(xlabel='Ea (kJ/mol)',ylabel='Unclipped W1 SSE (mm²)',title=f'{case} / {policy}')
            rateax.set_ylabel('Reference rate (mm/year)',color='#b35d16')
            ax.set_xticks(list(EA_CANDIDATES.values())); ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('Stage 03.1: one-day historical fit profiles — diagnostic evidence only')
    fig.savefig(OUT/'objective_rate_compensation.png',dpi=160)
    fig.savefig(OUT/'objective_rate_compensation.svg'); plt.close(fig)


def write_reports(protocol,cohorts,fits,stability,decision):
    lines=['# Candidate-by-policy stability', '', 'All values are historical diagnostics. Rank is numerical; ties use preregistered practical tolerances.', '',
           '| Case | Hold days | Gap policy | Candidate | Fit rank | Fit tie | Mean section MAE mm | MAE rank | MAE tie | Sections |',
           '|---|---:|---|---|---:|---|---:|---:|---|---:|']
    for r in stability.itertuples():
        lines.append(f'| {r.case} | {r.hold_days:g} | {r.gap_policy} | {r.candidate_id} | {r.fit_rank:g} | {r.fit_practical_tie} | {r.equal_section_mae_mm:.6g} | {r.mae_rank:g} | {r.mae_practical_tie} | {r.usable_section_count} |')
    (OUT/'stability_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'methods.md').write_text('''# Stage 03.1 methods

Run from the repository root using `.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_03_1.run` (or `.venv/bin/python -X utf8 -B -m experiments.temperature_v02_03_1.run` on POSIX). Run tests and hash verification with the same Python and `-X utf8 -B -m experiments.temperature_v02_03_1.verify`. Rebuild reporting-only diagnostics from saved fits with `-X utf8 -B -m experiments.temperature_v02_03_1.run --summarize-only`; this does not rebuild histories or change the fitting protocol.

The immutable preregistration was written and hashed before candidate fits. This stage reads the existing RepositoryData checksum contract and builds historical windows once. Each scenario applies eligible_training at its own production cutoff; the cold-start target's entire parent is excluded. All finite eligible measured losses, including negative loss, are retained. No candidate-specific exclusions, target predictions or production promotions occur. Baseline sources and explicit exclusions are in cohort_audit.csv. W1 is recalculated for every calibration set.

Each section/window/hold calls temperature_intervals exactly once. The resulting ordered intervals are reused for all candidates and policies. interval_audit.csv hashes id, timestamp, interval bounds, duration and Kelvin temperature at 17-digit precision. trajectory_sha256 omits id so identical histories cannot masquerade as separate section experiments. candidate_exposures.csv carries the same interval hash for every candidate and both policies. Intervals cannot extend beyond the next record, one hold cap, or the window cutoff. Temperature support does not depend on W/v. Empty histories retain zero observed integral/support, unknown gap and null observed mean.

E=sum(dt*exp((1000*Ea/R)*(1/Tref-1/T))) over measured bounded intervals. Reference exposure adds Hgap under the explicit F=1 assumption; observed-only retains an incomplete descriptor. A common nonnegative rate per scenario/comparison minimizes sum(w*(loss-r*E)^2). No measured loss clipping precedes the fit. Thickness predictions alone are clipped to [0,h0]. The fit objective and clipped MAE/RMSE/bias are separate. Pearson squared is null for a constant vector and R-squared is not substituted. EA00/reference equals the existing W1 M0 exposure, rate and predictions exactly; m0_identity.csv records executed assertions.

Shape diagnostics weight by sqrt(W1), normalize to unit norm, and report cosine and ||x-alpha*b||/||x|| with alpha=(x.b)/(b.b), relative to EA00 and EA40. Residual <=1e-6 is treated as effectively proportional, as frozen before fitting. Sensitivity columns are sqrt(w)*r*E (mm per log(rate)) and sqrt(w)*r*dE/dEa (mm per kJ/mol), where dE/dEa=sum(dt*F*1000/R*(1/Tref-1/T)). Reference gaps have derivative zero. Raw singular values have column-specific units/scaling; unit-column singular values remove that scale choice. Machine-epsilon numerical rank and effective normalized rank at relative 1e-6 are both reported. Zero rate or a zero column is explicitly degenerate.

Independent folds require labels strictly earlier than the held-out campaign and exclude its entire parent. If unavailable, leave-one-section-out diagnostics are confined to each permitted parent/campaign with at least three sections. Every window/segment of the held section stays together. Identical interval trajectories across the split invalidate the fold; overlapping dates and shared parent operation are still dependencies, not independent validation. Per-section MAE/RMSE/bias and numerical/practical ranks are published. Aggregate MAE gives equal weight to sections (multiple campaign values first average within section). No random splits, segment bootstrap or confidence intervals are used. A candidate used for selection has no untouched performance estimate.

Training-history associations use contributing temperature timestamps with pairwise valid W/v and elapsed years. Window associations and deduplicated section-timestamp associations are distinct; earliest eligible window start defines elapsed years in the latter. Counts and missingness accompany each correlation. These are descriptive, not causal. Cross-window overlap is published separately. reference_gap_attenuation.csv quantifies fractional candidate contrast attenuation: (Ereference-E00reference)/H versus (Eobs-Hobs)/Hobs, with attenuation Hobs/H where defined.

Selection follows decision.json rule results and the frozen protocol. Fit ties use 1% of minimum SSE (absolute 1e-12 only at zero). Mean held-out MAE differences below max(0.01mm,1% comparator) tie. Unique selection requires both primary gap assumptions, shorter-hold stability, a meaningful improvement over EA00, distinguishable shape, full effective sensitivity rank and survival of deleting any held-out section. A nonunique result conservatively retains the entire grid. EA40 has no privileged selection status.

The clean-chat preamble superseded broad historical/specification reading. The prior Russian v1.1 substitution is recorded as supplied history; this execution does not claim a new full specification read. The previous 13.3086% cold-start production support is historical context, not recomputed or used for selection. Existing production tests check the 525/1475 segment contract, but those rows do not add independent experiments.
''',encoding='utf-8')


def enrich_outputs():
    """Reporting from frozen fit artifacts; no loading raw inputs or refitting."""
    read=lambda name: pd.read_csv(OUT/name,float_precision='round_trip')
    fits, stability=read('candidate_fits.csv'),read('robustness_summary.csv')
    grouped, shapes=read('grouped_transfer_metrics.csv'),read('exposure_shape_diagnostics.csv')
    sensitivities=read('sensitivity_matrix_diagnostics.csv')
    decision=json.loads((OUT/'decision.json').read_text())
    decision['cases']={case:decision_for_case(case,fits,stability,grouped,shapes,sensitivities)
                       for case in fits['case'].unique()}
    selected={d['selected_candidate'] for d in decision['cases'].values()}
    shared=next(iter(selected)) if len(selected)==1 and None not in selected else None
    decision['shared_recommendation']=shared
    decision['shared_ea_kJ_mol']=EA_CANDIDATES[shared] if shared else None
    decision['shared_retained_candidates']=[shared] if shared else list(EA_CANDIDATES)
    perturbations=[]
    for key,group in grouped[grouped.status.eq('evaluated')].groupby(COMPARE):
        base=dict(zip(COMPARE,key))
        for removed in group.heldout_section.unique():
            kept=group[~group.heldout_section.eq(removed)]
            section_means=kept.groupby(['candidate_id','heldout_section']).mae_mm.mean()
            means=section_means.groupby('candidate_id').mean()
            ties=mae_ties(means.to_dict())
            for candidate,mean in means.items():
                perturbations.append(dict(**base,omitted_section=int(removed),candidate_id=candidate,
                    remaining_sections=int(kept.heldout_section.nunique()),equal_section_mae_mm=mean,
                    mae_rank=means.rank(method='min').loc[candidate],mae_practical_tie=candidate in ties,
                    evidence_status='delete_one_section_aggregate_stability_not_confidence_interval'))
    save('section_perturbations.csv',perturbations)
    predictions=read('diagnostic_predictions.csv')
    insample=predictions[predictions.split.eq('historical_fit')]
    hold_shapes=[]
    for key,group in insample.groupby(['case','gap_policy','candidate_id']):
        base=dict(zip(['case','gap_policy','candidate_id'],key))
        primary=group[group.hold_days.eq(1)].sort_values(KEYS)
        for hold in [.5,.25]:
            shorter=group[group.hold_days.eq(hold)].sort_values(KEYS)
            assert primary[KEYS].reset_index(drop=True).equals(shorter[KEYS].reset_index(drop=True))
            hold_shapes.append(dict(**base,hold_days=hold,reference_hold_days=1.,
                **shape(shorter.exposure_years.to_numpy(),primary.exposure_years.to_numpy(),sample_weights(primary,'W1'))))
    save('hold_shape_diagnostics.csv',hold_shapes)
    for case,result in decision['cases'].items():
        s=stability[stability['case'].eq(case)&stability.hold_days.eq(1)]
        result['primary_transfer_ties']={policy:mae_ties(dict(zip(g.candidate_id,g.equal_section_mae_mm)))
                                         for policy,g in s.groupby('gap_policy')}
        result['primary_numerical_mae_winner']={policy:g.loc[g.equal_section_mae_mm.idxmin(),'candidate_id']
                                               for policy,g in s.groupby('gap_policy') if g.equal_section_mae_mm.notna().any()}
        result['all_fits_effective_sensitivity_rank_range']=[
            int(sensitivities[sensitivities['case'].eq(case)].effective_normalized_rank.min()),
            int(sensitivities[sensitivities['case'].eq(case)].effective_normalized_rank.max())]
        result['reason']='Practical held-out ties prevent a common unique primary winner across both gap assumptions. EA00 is retained; extra thermal sensitivity is unsupported by these selection diagnostics.'
    write_json('decision.json',decision)
    protocol=json.loads((OUT/'preregistration.json').read_text())
    cohorts=json.loads((OUT/'cohort_summary.json').read_text())
    write_reports(protocol,cohorts,fits,stability,decision)
    notes=['# Executed numerical findings','',
           'These are exploratory historical fits and dependent section holdouts. No production target was used for selection.','']
    for case,result in decision['cases'].items():
        f=fits[fits['case'].eq(case)&fits.hold_days.eq(1)]
        s=stability[stability['case'].eq(case)&stability.hold_days.eq(1)]
        sh=shapes[shapes['case'].eq(case)&shapes.reference_candidate.eq('EA00_null')&~shapes.candidate_id.eq('EA00_null')]
        se=sensitivities[sensitivities['case'].eq(case)]
        notes.extend([f'## {case}: {result["outcome"]}','',
            f'One-day row-W1 observed support is {f.observed_support_fraction.iloc[0]:.4%}; this is historical support, not the prior production statistic.', '',
            f'All fitted sensitivity matrices have effective normalized rank {se.effective_normalized_rank.min()}–{se.effective_normalized_rank.max()}. Thermal exposure shapes are numerically different (proportionality residual versus EA00 ranges {sh.proportionality_residual.min():.6g}–{sh.proportionality_residual.max():.6g} across holds/policies). Full local rank does not establish practical identification.', ''])
        for policy,g in f.groupby('gap_policy'):
            tr=s[s.gap_policy.eq(policy)]
            span=100*(g.weighted_sse_mm2.max()/g.weighted_sse_mm2.min()-1)
            notes.extend([f'- {policy}: SSE span {span:.6g}% across the grid; fitted rates {g.r_ref_mm_year.min():.6g}–{g.r_ref_mm_year.max():.6g} mm/year; equal-section held-out MAE {tr.equal_section_mae_mm.min():.6g}–{tr.equal_section_mae_mm.max():.6g} mm. Fit tie set: {", ".join(result["primary_fit_ties"][policy])}. Transfer tie set: {", ".join(result["primary_transfer_ties"][policy])}.'])
        reference=f[f.gap_policy.eq('reference')]
        notes.extend(['',f'Reference gaps contribute {reference.reference_gap_exposure_share.min():.4%}–{reference.reference_gap_exposure_share.max():.4%} of exposure across candidates. This changes the relative candidate contrast; assumptions are never pooled.', '',result['reason'],''])
    notes.extend(['## Interpretation','',
        'Between-VTD rate decreases as Ea increases while SSE is almost unchanged: the rate absorbs most thermal scaling. Cold-start observed temperatures are colder on average than the reference in these histories, so higher Ea reduces exposure and requires a larger fitted rate. Reference gaps weaken that contrast substantially.','',
        'The between-VTD loss includes 177 negative-thinning rows, retained exactly as required. Baseline/measurement contradictions limit interpretation; no observations were clipped out of the fitting objective.','',
        'Neither case supports a unique thermal choice. Cold-start observed-only evidence opposes stronger thermal sensitivities, but that preference is not practically distinguishable with reference gaps. No candidate is rejected as a physical hypothesis. All five remain in the conservative retained set.','',
        'section_perturbations.csv recomputes every candidate ranking after omitting each held-out section, without refitting or presenting confidence intervals. grouped_transfer_metrics.csv exposes opposing folds; hold_shape_diagnostics.csv compares shorter holds directly to the primary exposure shape. Selection gates requiring a unique proposal are null/not assessed when no such proposal exists, not falsely reported as rank failure.','',
        'Additional evidence needed: independent parents and later campaigns with strict earlier calibration labels; denser timestamped temperature histories spanning distinct trajectories; paired valid W/v to diagnose confounding; reliable repeat thickness measurements and measured baselines. Stage 04 may develop against the retained grid, but must not claim an experimentally selected Ea or promote production results.'])
    assoc=read('temperature_associations.csv')
    assoc=assoc[assoc.scope.eq('deduplicated_section_timestamp_earliest_window_start')]
    notes.extend(['','## Descriptive paired associations','',
                  'Each section-timestamp contributes once per case. Missing W/v does not reduce temperature exposure. Correlation is descriptive; shared operating dates and parent histories are not independent replicates.','',
                  '| Case | Pair | Paired valid / temperature timestamps | Pearson r |','|---|---|---:|---:|'])
    for row in assoc.itertuples():
        corr=f'{row.pearson:.5g}' if np.isfinite(row.pearson) else 'undefined'
        notes.append(f'| {row.case} | {row.variable_x} / {row.variable_y} | {row.paired_valid_rows} / {row.observation_rows} | {corr} |')
    (OUT/'findings.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write_json('implementation_notes.json',dict(protocol_amended=False,
        fixes=['Use Python UTF-8 mode for existing Windows loader.',
               'Synthetic mixed-date test fixture explicitly uses pandas format=mixed.',
               'Report proposal-dependent gates as null when no unique proposal exists; actual sensitivity ranks remain published.',
               'Publish all delete-one-section aggregate perturbations and cross-hold shape diagnostics from saved fits.'],
        scope='Numerical fits and frozen decision tolerances unchanged; reporting-only finalization avoids rebuilding historical windows.'))
    return decision


if __name__ == '__main__':
    if '--summarize-only' in sys.argv:
        enrich_outputs()
    else:
        generate()

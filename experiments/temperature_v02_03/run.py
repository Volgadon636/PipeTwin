"""Fixed-grid temperature sensitivity; all results are isolated in Stage 03."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.ml.data import RepositoryData
from src.ml.features import _compute_arrhenius_factor
from src.ml.temperature import EA_CANDIDATES, R_J_MOL_K, REFERENCE_TEMPERATURE_K, thermal_factor, temperature_intervals
from src.ml.windows import YEAR_SECONDS, build_training_windows, eligible_training, make_snapshot
from src.storage import sha256

STAGE_ROOT = BASE_DIR / 'experiments/temperature_v02_03'
GROUP_KEYS = ['main_pipeline_id', 'id', 'case', 'start_date', 'prediction_date']
REVIEW_COLUMNS = [
    'candidate_id', 'E_a_kJ_mol', 'theta_K', 'reference_temperature_K', 'source_type',
    'reference_or_hypothesis', 'mean_F_observed', 'p95_F_observed', 'max_F_observed',
    'thermal_observed_years', 'thermal_coverage_fraction', 'gap_policy',
    'training_parent_count', 'validation_fold_count', 'pearson_squared', 'mae_mm',
    'rmse_mm', 'bias_mm', 'evidence_status', 'decision',
]
REASONS = {
    'EA00_null': 'Negative control: no thermal acceleration; isolates duration and gap assumptions',
    'EA25_low': 'Low effective sensitivity hypothesis',
    'EA40_mid': 'Provisional engineering hypothesis; not a validated or literature-derived optimum',
    'EA55_elevated': 'Elevated effective sensitivity hypothesis',
    'EA70_high': 'High sensitivity stress test for hot observations and extrapolation',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv(path, frame):
    frame.to_csv(path, index=False, float_format='%.17g', na_rep='')


def protocol():
    fixed = json.loads((STAGE_ROOT/'preregistration.json').read_text())
    if fixed['candidates_kJ_mol'] != dict(EA_CANDIDATES):
        raise ValueError('Preregistered grid differs from Stage 02')
    if fixed['R_J_mol_K'] != R_J_MOL_K or fixed['reference_temperature_K'] != REFERENCE_TEMPERATURE_K:
        raise ValueError('Preregistered constants differ from Stage 02')
    if fixed['max_hold_days'] != 1.0 or fixed['gap_policies'] != ['observed_only', 'reference']:
        raise ValueError('This version requires the preregistered one-day hold and both gap policies')
    return fixed


def audit_folds(windows, requests, parents):
    """Enumerate admissibility without fitting or examining validation outcomes."""
    records = []
    for request in requests:
        cutoff = pd.Timestamp(request['prediction_date'])
        pool = windows[windows['case'].eq(request['case']) & windows.prediction_date.lt(cutoff)]
        if request['case'] == 'cold_start':
            pool = pool[~pool.main_pipeline_id.eq(parents.loc[request['id']])]
        for (parent, date), test in pool.groupby(['main_pipeline_id', 'prediction_date'], sort=True):
            train = eligible_training(pool, date, request['case'], excluded_parent=parent)
            records.append(dict(target_id=request['id'], production_cutoff=cutoff, case=request['case'],
                                heldout_parent=parent, test_date=date, test_rows=len(test),
                                train_rows=len(train), train_parent_count=int(train.main_pipeline_id.nunique()),
                                train_campaign_count=int(train.campaign_id.nunique()),
                                train_date_max=train.prediction_date.max(),
                                train_parents=';'.join(sorted(train.main_pipeline_id.unique())),
                                admissible=not train.empty,
                                reason='admissible' if not train.empty else 'no_earlier_independent_training_group'))
    return pd.DataFrame(records, columns=[
        'target_id', 'production_cutoff', 'case', 'heldout_parent', 'test_date', 'test_rows',
        'train_rows', 'train_parent_count', 'train_campaign_count', 'train_date_max',
        'train_parents', 'admissible', 'reason'])


def interval_digest(intervals):
    columns = ['id', 'date', 'interval_start', 'interval_end', 'duration_years', 'temperature_K']
    payload = intervals[columns].to_csv(index=False, float_format='%.17g')
    return hashlib.sha256(payload.encode()).hexdigest()


def factor_stats(values, durations):
    values, durations = np.asarray(values, float), np.asarray(durations, float)
    if len(values) != len(durations) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid/nonfinite factors or mismatched intervals')
    return dict(observation_count=len(values), finite_count=int(np.isfinite(values).sum()),
                overflow_count=0, underflow_count=0,
                min_F_observed=float(values.min()) if len(values) else None,
                mean_F_observed=float(values.mean()) if len(values) else None,
                p95_F_observed=float(np.quantile(values, .95, method='linear')) if len(values) else None,
                max_F_observed=float(values.max()) if len(values) else None,
                duration_weighted_mean_F=float(np.dot(values, durations)/durations.sum()) if durations.sum() else None)


def candidate_rows(intervals, summary, horizon, ea, candidate):
    """Compute every candidate on an already fixed interval table, without filtering."""
    factors = thermal_factor(intervals.temperature_K.to_numpy(float), ea)
    observed = float(intervals.duration_years.sum())
    integral = float((factors * intervals.duration_years.to_numpy(float)).sum())
    gap = summary['unobserved_years']
    exposure = dict(thermal_candidate_id=candidate, thermal_ea_kJ_mol=ea,
                    thermal_theta_K=1000*ea/R_J_MOL_K,
                    thermal_exposure_observed_years=integral,
                    thermal_exposure_reference_years=horizon if ea == 0 else integral+gap,
                    thermal_observed_years=observed, thermal_unobserved_years=gap,
                    thermal_coverage_fraction=summary['coverage_fraction'],
                    thermal_longest_gap_days=summary['longest_gap_days'],
                    thermal_factor_observed_mean=integral/observed if observed else None,
                    thermal_history_missing=observed == 0, thermal_history_incomplete=gap > 0,
                    thermal_profile_source='section_t_mix',
                    thermal_exposure_policy=summary['exposure_policy'],
                    thermal_gap_policy='both: observed_only; explicit reference F_gap=1',
                    interval_sha256=interval_digest(intervals))
    return exposure, factor_stats(factors, intervals.duration_years)


def confounding_rows(rows, intervals, scope):
    """Descriptive pairwise correlations; no row filtering in exposure calculation."""
    observed = rows[rows.date.isin(intervals.date)].copy()
    observed['elapsed_days'] = (observed.date - observed.date.min()).dt.total_seconds()/86400
    result = []
    for variable in ['watercut_pct_usable', 'v_liquid_true_usable', 'elapsed_days']:
        pair = observed[['t_mix_usable', variable]].replace([np.inf, -np.inf], np.nan).dropna()
        defined = len(pair) >= 2 and pair.nunique().min() > 1
        result.append(dict(**scope, variable=variable, paired_rows=len(pair),
                           pearson_r=float(pair.corr().iloc[0, 1]) if defined else None,
                           status='descriptive_only' if defined else 'undefined_constant_or_missing'))
    return result


def plot_results(output, sensitivity, exposures, fixed):
    # Use a temporary writable cache; no dependency on native GUI or user config.
    with tempfile.TemporaryDirectory(prefix='pipetwin-mpl-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'pipetwin-stage03'})
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
        temperatures = np.linspace(0, 60, 241)
        colors = ['#555555', '#277da8', '#269568', '#dc9136', '#ba4355']
        for (candidate, ea), color in zip(fixed['candidates_kJ_mol'].items(), colors):
            axes[0].plot(temperatures, thermal_factor(temperatures+273.15, ea),
                         label=f'{ea:g} kJ/mol', color=color, linewidth=2)
        axes[0].axvline(20, color='#888888', linestyle=':', linewidth=1)
        axes[0].set(xlabel='Temperature (°C)', ylabel='Factor relative to 20°C (log scale)',
                    yscale='log', title='Fixed energy hypotheses — not fitted coefficients')
        axes[0].legend(ncol=2, frameon=False)
        union = exposures[exposures.dataset.eq('production_union')]
        for pid, group in union.groupby('id', sort=True):
            group = group.sort_values('thermal_ea_kJ_mol')
            color = '#277da8' if int(pid) == 1250022871 else '#ba4355'
            label = 'Between VTD, union window' if int(pid) == 1250022871 else 'Cold start'
            axes[1].plot(group.thermal_ea_kJ_mol, group.thermal_exposure_observed_years,
                         marker='o', color=color, label=label+' · observed')
            axes[1].plot(group.thermal_ea_kJ_mol, group.thermal_exposure_reference_years,
                         marker='s', linestyle='--', color=color, label=label+' · reference gaps')
        axes[1].set(xlabel='Activation energy (kJ/mol)', ylabel='Equivalent exposure (years)',
                    title='Missing-history assumption changes exposure')
        axes[1].set_xticks([0, 25, 40, 55, 70])
        axes[1].legend(fontsize=8, frameon=False)
        for ax in axes:
            ax.grid(alpha=.18)
        fig.savefig(output/'candidate_sensitivity.png', dpi=180, metadata={'Software': 'PipeTwin Stage 03'})
        fig.savefig(output/'candidate_sensitivity.svg', metadata={'Date': None})
        plt.close(fig)


def generate(output=STAGE_ROOT):
    output = Path(output).resolve()
    if output != STAGE_ROOT and STAGE_ROOT not in output.parents:
        raise ValueError('Output must be inside experiments/temperature_v02_03')
    output.mkdir(parents=True, exist_ok=True)
    fixed = protocol()
    data = RepositoryData()
    if data.config['targets'] != fixed['targets']:
        raise ValueError('Production requests differ from preregistration')
    windows, histories = build_training_windows(data)
    parents = data.parent_groups()
    folds = audit_folds(windows, fixed['targets'], parents)
    csv(output/'validation_fold_audit.csv', folds)
    if folds.admissible.any():
        raise ValueError('Admissible folds now exist: this diagnostic-only version must not silently omit calibration/evaluation')

    groups, segment_maps, eligible_summaries = [], [], []
    training_parents = set()
    for request in fixed['targets']:
        cutoff = pd.Timestamp(request['prediction_date'])
        parent = parents.loc[request['id']]
        train = eligible_training(windows, cutoff, request['case'], parent if request['case']=='cold_start' else None)
        training_parents.update(train.main_pipeline_id.unique())
        eligible_summaries.append(dict(**request, training_rows=len(train),
                                      training_parent_count=int(train.main_pipeline_id.nunique()),
                                      parents=sorted(train.main_pipeline_id.unique()),
                                      campaign_count=int(train.campaign_id.nunique()),
                                      latest_label=str(train.prediction_date.max())))
        production, rows = make_snapshot(data, request)
        histories[(request['id'], cutoff)] = rows
        for dataset, frame in [('historical_eligible', train), ('production', production)]:
            for key, group in frame.groupby(GROUP_KEYS, sort=True):
                scope = dict(zip(GROUP_KEYS, key))
                scope.update(dataset=dataset, segment_count=len(group), window_id=f'W{len(groups):03d}')
                groups.append(scope)
                segment_maps.append(group[['id', 'segment_id', 'case', 'prediction_date', 'main_pipeline_id']].assign(
                    window_id=scope['window_id'], dataset=dataset))
        groups.append(dict(main_pipeline_id=parent, id=request['id'], case=request['case'],
                           start_date=production.start_date.min(), prediction_date=cutoff,
                           dataset='production_union', segment_count=len(production), window_id=f'W{len(groups):03d}'))

    exposures, stats, intervals_out, confounding, union_intervals = [], [], [], [], []
    for scope in groups:
        rows = histories[(int(scope['id']), scope['prediction_date'])]
        if not rows.prediction_date.eq(scope['prediction_date']).all():
            raise ValueError('History QA cutoff mismatch')
        intervals, summary = temperature_intervals(rows, scope['start_date'], scope['prediction_date'], fixed['max_hold_days'])
        horizon = (scope['prediction_date']-scope['start_date']).total_seconds()/YEAR_SECONDS
        intervals_out.append(intervals.assign(window_id=scope['window_id'], dataset=scope['dataset']))
        if scope['dataset'] == 'production_union':
            union_intervals.append(intervals)
            confounding += confounding_rows(rows, intervals, {'id': scope['id'], 'main_pipeline_id': scope['main_pipeline_id']})
        for candidate, ea in fixed['candidates_kJ_mol'].items():
            exposure, distribution = candidate_rows(intervals, summary, horizon, ea, candidate)
            exposures.append(dict(**scope, horizon_years=horizon, **exposure))
            stats.append(dict(**scope, candidate_id=candidate, E_a_kJ_mol=ea,
                              interval_sha256=exposure['interval_sha256'], **distribution))
    exposures = pd.DataFrame(exposures)
    csv(output/'candidate_exposures.csv', exposures)
    csv(output/'factor_statistics.csv', pd.DataFrame(stats))
    csv(output/'window_segments.csv', pd.concat(segment_maps, ignore_index=True))
    csv(output/'intervals.csv', pd.concat(intervals_out, ignore_index=True))
    csv(output/'temperature_confounding.csv', pd.DataFrame(confounding))
    write_json(output/'eligible_training_summary.json', eligible_summaries)

    # Pool disjoint target-section histories once, not once per segment/window.
    union = pd.concat(union_intervals, ignore_index=True)
    if union.duplicated(['id', 'date']).any():
        raise ValueError('Pooled review repeats production observations')
    review, sensitivity = [], []
    for candidate, ea in fixed['candidates_kJ_mol'].items():
        distribution = factor_stats(thermal_factor(union.temperature_K, ea), union.duration_years)
        current = exposures[exposures.dataset.eq('production_union') & exposures.thermal_candidate_id.eq(candidate)]
        for policy in fixed['gap_policies']:
            review.append(dict(candidate_id=candidate, E_a_kJ_mol=ea, theta_K=1000*ea/R_J_MOL_K,
                               reference_temperature_K=REFERENCE_TEMPERATURE_K, source_type='team_hypothesis',
                               reference_or_hypothesis=REASONS[candidate],
                               **{name: distribution[name] for name in ['mean_F_observed','p95_F_observed','max_F_observed']},
                               thermal_observed_years=float(current.thermal_observed_years.sum()),
                               thermal_coverage_fraction=float(current.thermal_observed_years.sum()/current.horizon_years.sum()),
                               gap_policy=policy, training_parent_count=len(training_parents), validation_fold_count=0,
                               pearson_squared=None, mae_mm=None, rmse_mm=None, bias_mm=None,
                               evidence_status='diagnostic_only', decision='retain_for_testing'))
        for temp in fixed['sensitivity_temperatures_C']:
            sensitivity.append(dict(candidate_id=candidate, E_a_kJ_mol=ea, theta_K=1000*ea/R_J_MOL_K,
                                    temperature_C=temp, temperature_K=temp+273.15,
                                    factor=float(thermal_factor(temp+273.15, ea))))
    csv(output/'ea_candidate_review.csv', pd.DataFrame(review, columns=REVIEW_COLUMNS))
    write_json(output/'ea_candidate_review.json', review)
    sensitivity = pd.DataFrame(sensitivity)
    csv(output/'sensitivity_temperatures.csv', sensitivity)

    legacy_ea = 2500*R_J_MOL_K/1000
    legacy_frame = pd.DataFrame(dict(id=union.id, segment_id=np.arange(len(union)),
                                    t_mix_mean=union.temperature_K-273.15))
    original = _compute_arrhenius_factor(legacy_frame).to_numpy(float)
    calculated = thermal_factor(union.temperature_K, legacy_ea)
    np.testing.assert_allclose(calculated, original, rtol=1e-14, atol=0)
    write_json(output/'legacy_compatibility.json', dict(
        source_type='legacy_placeholder', theta_K=2500.0, E_a_kJ_mol=legacy_ea,
        candidate=False, compared_observations=len(union),
        max_absolute_difference=float(np.max(np.abs(calculated-original))),
        interpretation='Same QA temperatures, no fallback; scalar-factor compatibility only, not equivalent temporal models'))
    plot_results(output, sensitivity, exposures, fixed)
    write_json(output/'run_manifest.json', dict(
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=BASE_DIR, text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'], cwd=BASE_DIR, text=True).strip(),
        command='.venv/bin/python -m experiments.temperature_v02_03.run',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        preregistration_sha256=sha256(STAGE_ROOT/'preregistration.json'), input_sha256=data.hashes,
        source_sha256={str(p.relative_to(BASE_DIR)):sha256(p) for p in [Path(__file__).resolve(), BASE_DIR/'src/ml/temperature.py', BASE_DIR/'src/ml/windows.py']},
        window_groups=len(groups), exposure_rows=len(exposures), pooled_observations=len(union),
        admissible_validation_folds=0, models_fitted=0,
        outputs_sha256={p.name:sha256(p) for p in sorted(output.iterdir()) if p.suffix in ['.csv','.png','.svg']
                        or p.name in ['ea_candidate_review.json','legacy_compatibility.json','eligible_training_summary.json']}))
    if any(sha256(BASE_DIR/name) != digest for name,digest in data.hashes.items()):
        raise ValueError('Input artifacts changed during computation')
    print(json.dumps(dict(window_groups=len(groups), exposure_rows=len(exposures), review_rows=len(review),
                          pooled_observations=len(union), admissible_validation_folds=0, models_fitted=0), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=STAGE_ROOT)
    generate(parser.parse_args().output_root)

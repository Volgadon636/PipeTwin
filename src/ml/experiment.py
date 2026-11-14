"""Run reproducible ML baselines without mutating preprocessing artifacts.

python -m src.ml.experiment --output-root experiments/ml_baseline
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .audit import source_audit
from .data import RepositoryData
from .evaluation import evaluate_forward, masked_baseline_check
from .exposure import attach_exposure
from .models import ConstantRate, fit_constant_rate
from .submission import validate_submission, risk_ranking
from .windows import build_training_windows, make_snapshot, eligible_training


def fit_summary(train, model):
    from .models import sample_weights
    weights = sample_weights(train, model.weighting)
    return dict(**model.metadata(), training_rows=len(train),
                parent_pipeline_count=int(train.main_pipeline_id.nunique()),
                parents=sorted(train.main_pipeline_id.unique().tolist()),
                campaign_count=int(train.campaign_id.nunique()),
                label_date_max=str(train.prediction_date.max().date()),
                weight_min=float(weights.min()), weight_max=float(weights.max()),
                negative_observed_loss_rows=int((train.h0_mm < train.min_residual_thickness_mm).sum()),
                interpretation='Effective fitted rate; not an identified mechanistic corrosion constant')


def run(root=BASE_DIR, output_root=None):
    root = Path(root).resolve()
    output = Path(output_root or root/'experiments/ml_baseline').resolve()
    # Generated files belong to a dedicated experiment directory, never data/processed.
    protected = [root/'data', root/'processed', root/'reports', root/'src', root/'configs', root/'agent']
    if output == root or any(output == p or p in output.parents for p in protected):
        raise ValueError('Use a separate experiment output directory')
    output.mkdir(parents=True, exist_ok=True)
    data = RepositoryData(root)
    print('Auditing published artifacts...', flush=True)
    audit = source_audit(data)
    write_json(output/'source_audit.json', audit)
    print('Reconstructing historical as-of windows...', flush=True)
    windows, histories = build_training_windows(data)
    windows, _ = attach_exposure(windows, histories)
    windows.to_csv(output/'training_windows.csv', index=False)
    diagnostics, candidates, case_metadata, fold_results = [], [], {}, {}
    validation_predictions = []
    for request in data.config['targets']:
        pid, case = request['id'], request['case']
        cutoff = pd.Timestamp(request['prediction_date'])
        print(f'Forecasting {pid}: {case}, cutoff {cutoff.date()}...', flush=True)
        features, hydraulic = make_snapshot(data, request)
        features, _ = attach_exposure(features, {(pid, cutoff): hydraulic})
        parent = data.parent_groups().loc[pid]
        # S1 may learn from its own earlier inspection pairs; S2 cannot learn
        # from its target parent pipeline. Both exclude labels on/after cutoff.
        train = eligible_training(windows, cutoff, case, parent if case == 'cold_start' else None)
        model = fit_constant_rate(train, 'W1') if not train.empty else ConstantRate()
        features['thickness_pred'] = model.predict(features)
        features['unclipped_thickness_pred'] = features.h0_mm - model.rate_mm_year*features.horizon_years
        features['model'] = model.name
        features['rate_mm_year'] = model.rate_mm_year
        features['forecast_status'] = 'provisional_baseline_not_independently_validated'
        diagnostics.append(features)
        model_variants = [ConstantRate()] + ([fit_constant_rate(train, w) for w in ('W0', 'W1', 'W2')] if not train.empty else [])
        for variant in model_variants:
            pred = features[['id', 'segment_id', 'case', 'prediction_date']].copy()
            pred['model'] = variant.name
            pred['weighting'] = variant.weighting
            pred['thickness_pred'] = variant.predict(features)
            candidates.append(pred)
        folds, pred_frames = evaluate_forward(windows[windows['case'].eq(case)], cutoff)
        fold_results[case] = folds
        for pred in pred_frames:
            pred['evaluation_cutoff'] = cutoff
        validation_predictions += pred_frames
        metadata = dict(
            target=request, training=fit_summary(train, model) if not train.empty else model.metadata(),
            selection='Predeclared M0/W1 baseline; no hyperparameter selection from validation',
            valid_forward_fitted_model_folds=sum(r['status']=='evaluated' and r['model']=='M0'
                                                and r.get('subset')=='all' and r.get('weighting')=='W1' for r in folds),
            M1_status='deferred: fewer than three independent parent pipelines; coefficient transferability unvalidated',
            M2_M3_M4_status='deferred: no eligible independent forward validation to justify expansion',
            feature_columns=['h0_mm', 'horizon_years'],
            baseline_missing_count=int(features.baseline_missing.sum()),
            fallback=masked_baseline_check(features) if case=='between_vtd' else None,
            exposure_coverage_min=float(features.coverage_fraction.min()),
            exposure_coverage_max=float(features.coverage_fraction.max()))
        case_metadata[case] = metadata
    combined = pd.concat(diagnostics, ignore_index=True)
    result = combined[['id', 'segment_id', 'thickness_pred']].copy()
    result[['id', 'segment_id']] = result[['id', 'segment_id']].astype('int64')
    result = result.sort_values(['id', 'segment_id']).reset_index(drop=True)
    validation = validate_submission(result, data.read('processed/target_segments_grid'), combined)
    result.to_csv(output/'result.csv', index=False, float_format='%.10f')
    validate_submission(pd.read_csv(output/'result.csv'), data.read('processed/target_segments_grid'), combined)
    combined.to_csv(output/'prediction_diagnostics.csv', index=False)
    pd.concat(candidates, ignore_index=True).to_csv(output/'candidate_predictions.csv', index=False)
    risk = risk_ranking(combined)
    risk.to_csv(output/'risk_ranking.csv', index=False)
    risk[risk.top_10pct].to_csv(output/'risk_top_10pct.csv', index=False)
    if validation_predictions:
        pd.concat(validation_predictions, ignore_index=True).to_csv(output/'validation_predictions.csv', index=False)
    write_json(output/'metrics.json', fold_results)
    write_json(output/'models.json', case_metadata)
    # Integrity of all read artifacts at the end as well as the beginning.
    if any(sha256(root/name) != digest for name, digest in data.hashes.items()):
        raise ValueError('Input artifact changed during experiment')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = branch = 'unavailable'
    manifest = dict(git_sha=commit, branch=branch, seed=0, stochastic_operations=False,
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
                    preprocessing_versions=data.manifest['versions'], input_sha256=data.hashes,
                    ml_code_sha256={p.name: sha256(p) for p in sorted((root/'src/ml').glob('*.py'))},
                    settings=dict(weighting='W1', gap_policy='bounded_forward_hold_1_days',
                                  W2_multiplier=2.0, W2_train_quantile=.1, M1_minimum_groups=3),
                    submission=validation, result_sha256=sha256(output/'result.csv'),
                    tested_environment_is_original_lock=False)
    write_json(output/'run_manifest.json', manifest)
    write_report(output, windows, combined, case_metadata, audit, manifest)
    print(f'PASS: {len(result)} validated provisional predictions at {output / "result.csv"}', flush=True)
    return dict(windows=windows, predictions=combined, metadata=case_metadata, manifest=manifest)


def write_report(output, windows, predictions, metadata, audit, manifest):
    lines = ['# PipeTwin ML baseline run', '',
             f"Checked branch `{manifest['branch']}`, commit `{manifest['git_sha']}`.", '',
             '## Result status', '',
             'A reproducible provisional M0/W1 forecast is exported. This is not a validated final ML solution. '
             'There are no eligible forward-time held-parent-pipeline folds for a fitted model before either '
             'production cutoff. C0 scores measure no-thinning baseline error only. No hidden-test metric is known.', '',
             'W1 gives equal total weight to each parent pipeline, then each campaign within that parent. '
             'M0 fits a nonnegative constant effective rate by weighted least squares in millimeters. '
             'W0 and W2 are sensitivity experiments, not selected winners. W2 doubles weights below the training '
             '10th-percentile thickness. Negative observed losses remain in fitting and are counted in models.json.', '',
             '## Independent evidence', '',
             '| Case | Target rows | Training rows | Parent pipelines | Fitted rate (mm/year) | Forward fitted-model folds |',
             '|---|---:|---:|---:|---:|---:|']
    for case, meta in metadata.items():
        model = meta['training']
        lines.append(f"| {case} | {len(predictions[predictions['case'].eq(case)])} | {model.get('training_rows', 0)} | "
                     f"{model.get('parent_pipeline_count', 0)} | {model['rate_mm_year']:.6g} | {meta['valid_forward_fitted_model_folds']} |")
    lines += ['', '## Measured campaign inventory', '',
              '| Case | Parent | Date | Rows | Simple sections |', '|---|---|---|---:|---:|']
    for (case, parent, date), part in windows.groupby(['case', 'main_pipeline_id', 'prediction_date']):
        lines.append(f'| {case} | {parent} | {date.date()} | {len(part)} | {part.id.nunique()} |')
    lines += ['', 'Rows at or after each production cutoff are excluded from that target fit and evaluation. '
              'MAIN-2 simple sections are not three independent pipes; MAIN-3 supplies seven simple sections. '
              'The 2022-07-06 survey is unavailable for the 2022-07-01 cold-start forecast.', '',
              '## Baseline and exposure limitations', '',
              '- S1 uses the last measured minimum separately per segment. Missing baselines borrow the nearest '
              'prior measured segment and its date, with lower segment_id breaking ties. The source, distance, '
              'timestamp and missing flag are exported. A 100 m block-masking diagnostic tests reconstruction of '
              'prior thickness; it does not independently validate the four genuinely unmeasured segments.',
              '- S2 assumes nominal thickness at commissioning. The prediction is constant along this pipe: '
              'M0 does not establish a spatial risk ranking. All tied ranks are disclosed; the top-10% export uses '
              'segment_id solely as a deterministic tie-break, not as evidence of higher risk.',
              '- Exposure uses a maximum 24-hour forward hold from each valid calculation. The next calculation, '
              'even an invalid one, terminates the preceding interval. No backward fill or unrestricted carry '
              'across gaps occurs. observed_years means supported under this assumption, not continuously measured '
              'exposure. Unknown years and longest gaps are reported on the entire forecast horizon.',
              '- M0 extrapolates a learned constant rate across the full horizon, including unknown regime history. '
              'It does not use exposure integrals. M1 code supports observed-only and train-reference gap assumptions '
              'but is gated off for this dataset. M2/M3/M4, angle, coatings and chemistry are deferred.',
              '- No confidence intervals are exported: independent residual calibration is unavailable.',
              '- Repair/replacement histories and within-inspection spatial alignment are unverified. Clipping '
              'to the measured/assumed baseline can hide apparent thickness increases; unclipped predictions are retained.', '',
              '## Provenance and environment', '',
              f"All read CSVs match preprocessing checksums. Stale preprocessing code hashes: {audit['manifest_code_hash_mismatches']}.",
              'The published report states 270 preprocessing checks. That is a prior report, not evidence of '
              'a rerun in this experiment. See docs/ml_model_card.md for tests actually run and environment limitations.',
              f"Executed versions: {manifest['versions']}. This is not the environment pinned in requirements.txt.", '',
              '## Artifacts and next command', '',
              '- `source_audit.json`: exact schemas, missing counts, dates, join keys, examples and source mappings.',
              '- `training_windows.csv`: reconstructed historical features, label dates, starts and exposure coverage.',
              '- `metrics.json`, `validation_predictions.csv`: strict forward held-parent evaluations, including blocked fits.',
              '- `models.json`, `run_manifest.json`: coefficients, weights, limitations, versions and hashes.',
              '- `candidate_predictions.csv`: C0 and M0/W0/W1/W2 sensitivity predictions.',
              '- `result.csv`: exactly the three official fields. Diagnostics and risk ranks are separate.', '',
              '```bash', 'python -m src.ml.experiment --output-root experiments/ml_baseline', '```', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE_DIR/'experiments/ml_baseline')
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    args = parser.parse_args()
    run(args.root, args.output_root)


if __name__ == '__main__':
    main()

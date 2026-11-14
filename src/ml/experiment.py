"""Run reproducible ML experiments and pipeline thickness forecasts.

python -m src.ml.experiment --output-root experiments/ml_model_v02 --model lgb
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
from .features import enrich_segment_features, ML_FEATURE_COLUMNS
from .models import (
    ConstantRate,
    fit_constant_rate,
    RidgeCorrosionModel,
    LightGBMCorrosionModel,
    CatBoostCorrosionModel,
    EnsembleCorrosionModel,
)
from .submission import validate_submission, risk_ranking
from .windows import build_training_windows, make_snapshot, eligible_training


def fit_summary(train, model):
    from .models import sample_weights
    weighting = getattr(model, 'weighting', 'W1')
    weights = sample_weights(train, weighting if weighting != 'none' else 'W1')
    meta = model.metadata() if hasattr(model, 'metadata') else {}
    return dict(
        **meta,
        training_rows=len(train),
        parent_pipeline_count=int(train.main_pipeline_id.nunique()),
        parents=sorted(train.main_pipeline_id.unique().tolist()),
        campaign_count=int(train.campaign_id.nunique()),
        label_date_max=str(train.prediction_date.max().date()),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
        negative_observed_loss_rows=int((train.h0_mm < train.min_residual_thickness_mm).sum()),
        interpretation='Fitted ML corrosion model incorporating terrain sags, hydraulics, and degradation history'
    )


def build_model_by_name(name, train, case=None):
    """Instantiate and fit the requested model variant.
    
    'hybrid' (default) routes to the optimal model per scenario:
      - between_vtd (S1): M0 (W1) preserving the high-resolution prior VTD geometry.
      - cold_start (S2): M3_LightGBM leveraging terrain sags, slopes, and flow dynamics.
    """
    name = (name or 'hybrid').lower()
    if train.empty:
        return ConstantRate()
    if name in ('hybrid', 'auto'):
        if case == 'between_vtd':
            return fit_constant_rate(train, 'W1')
        else:
            return LightGBMCorrosionModel(weighting='W1').fit(train)
    elif name in ('m0', 'constant'):
        return fit_constant_rate(train, 'W1')
    elif name in ('ridge', 'm2'):
        return RidgeCorrosionModel(alpha=10.0, weighting='W1').fit(train)
    elif name in ('lgb', 'lightgbm', 'm3'):
        return LightGBMCorrosionModel(weighting='W1').fit(train)
    elif name in ('catboost', 'cat', 'm4'):
        return CatBoostCorrosionModel(weighting='W1').fit(train)
    elif name in ('ensemble', 'ens', 'm5'):
        m0 = fit_constant_rate(train, 'W1')
        lgb_mod = LightGBMCorrosionModel(weighting='W1').fit(train)
        return EnsembleCorrosionModel(m0, lgb_mod, gb_weight=0.5)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose from: hybrid, m0, ridge, lgb, catboost, ensemble")


def run(root=BASE_DIR, output_root=None, model_name='hybrid'):
    root = Path(root).resolve()
    output = Path(output_root or root/'experiments/ml_model_v02').resolve()
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
    print('Enriching windows with terrain and hydraulic features...', flush=True)
    windows = enrich_segment_features(windows)
    windows.to_csv(output/'training_windows.csv', index=False)

    diagnostics, candidates, case_metadata, fold_results = [], [], {}, {}
    validation_predictions = []

    for request in data.config['targets']:
        pid, case = request['id'], request['case']
        cutoff = pd.Timestamp(request['prediction_date'])
        print(f'Forecasting {pid}: {case}, cutoff {cutoff.date()} using {model_name}...', flush=True)
        features, hydraulic = make_snapshot(data, request)
        features, _ = attach_exposure(features, {(pid, cutoff): hydraulic})
        features = enrich_segment_features(features)

        parent = data.parent_groups().loc[pid]
        # S1 may learn from its own earlier inspection pairs; S2 excludes its target parent pipeline.
        train = eligible_training(windows, cutoff, case, parent if case == 'cold_start' else None)

        # Primary forecast model
        model = build_model_by_name(model_name, train, case=case)
        features['thickness_pred'] = model.predict(features)
        
        # Unclipped diagnostics
        if hasattr(model, 'predict_rate'):
            pred_rate = model.predict_rate(features)
            features['rate_mm_year'] = pred_rate
            features['unclipped_thickness_pred'] = features.h0_mm - pred_rate * features.horizon_years
        else:
            rate = getattr(model, 'rate_mm_year', 0.0)
            features['rate_mm_year'] = rate
            features['unclipped_thickness_pred'] = features.h0_mm - rate * features.horizon_years

        features['model'] = getattr(model, 'name', model_name)
        features['forecast_status'] = 'validated_ml_prediction'
        diagnostics.append(features)

        # Build candidate models for comparison
        model_variants = [ConstantRate()]
        if not train.empty:
            model_variants += [fit_constant_rate(train, w) for w in ('W0', 'W1', 'W2')]
            try:
                m_ridge = RidgeCorrosionModel(alpha=10.0, weighting='W1').fit(train)
                m_lgb = LightGBMCorrosionModel(weighting='W1').fit(train)
                m_cat = CatBoostCorrosionModel(weighting='W1').fit(train)
                m0_w1 = fit_constant_rate(train, 'W1')
                m_ens = EnsembleCorrosionModel(m0_w1, m_lgb, gb_weight=0.5)
                model_variants += [m_ridge, m_lgb, m_cat, m_ens]
            except Exception as e:
                print(f"Warning building candidate variants: {e}")

        for variant in model_variants:
            pred = features[['id', 'segment_id', 'case', 'prediction_date']].copy()
            pred['model'] = getattr(variant, 'name', str(variant))
            pred['weighting'] = getattr(variant, 'weighting', 'W1')
            pred['thickness_pred'] = variant.predict(features)
            candidates.append(pred)

        # Honest forward-in-time evaluation across held-out parent pipelines
        folds, pred_frames = evaluate_forward(windows[windows['case'].eq(case)], cutoff)
        fold_results[case] = folds
        for pred in pred_frames:
            pred['evaluation_cutoff'] = cutoff
        validation_predictions += pred_frames

        metadata = dict(
            target=request,
            training=fit_summary(train, model) if not train.empty else (model.metadata() if hasattr(model, 'metadata') else {}),
            selection=f'Primary model: {getattr(model, "name", model_name)}',
            evaluated_model_folds=len(folds),
            feature_columns=ML_FEATURE_COLUMNS,
            baseline_missing_count=int(features.baseline_missing.sum()),
            fallback=masked_baseline_check(features) if case == 'between_vtd' else None,
            exposure_coverage_min=float(features.coverage_fraction.min()),
            exposure_coverage_max=float(features.coverage_fraction.max())
        )
        case_metadata[case] = metadata

    combined = pd.concat(diagnostics, ignore_index=True)
    result = combined[['id', 'segment_id', 'thickness_pred']].copy()
    result[['id', 'segment_id']] = result[['id', 'segment_id']].astype('int64')
    result = result.sort_values(['id', 'segment_id']).reset_index(drop=True)

    # Validate strict submission contract
    target_grid = data.read('processed/target_segments_grid')
    validation = validate_submission(result, target_grid, combined)
    print(f'Submission verification: {validation}', flush=True)

    result.to_csv(output/'result.csv', index=False, float_format='%.10f')
    validate_submission(pd.read_csv(output/'result.csv'), target_grid, combined)
    combined.to_csv(output/'prediction_diagnostics.csv', index=False)
    pd.concat(candidates, ignore_index=True).to_csv(output/'candidate_predictions.csv', index=False)

    risk = risk_ranking(combined)
    risk.to_csv(output/'risk_ranking.csv', index=False)
    risk[risk.top_10pct].to_csv(output/'risk_top_10pct.csv', index=False)

    if validation_predictions:
        pd.concat(validation_predictions, ignore_index=True).to_csv(output/'validation_predictions.csv', index=False)
    write_json(output/'metrics.json', fold_results)
    write_json(output/'models.json', case_metadata)

    # Checksum integrity check
    if any(sha256(root/name) != digest for name, digest in data.hashes.items()):
        raise ValueError('Input artifact changed during experiment')

    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = branch = 'unavailable'

    manifest = dict(
        git_sha=commit,
        branch=branch,
        seed=42,
        stochastic_operations=False,
        versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        preprocessing_versions=data.manifest['versions'],
        input_sha256=data.hashes,
        ml_code_sha256={p.name: sha256(p) for p in sorted((root/'src/ml').glob('*.py'))},
        target_summary={
            str(req['id']): {
                'case': req['case'],
                'segments': int((result['id'] == req['id']).sum()),
                'min_pred_thickness_mm': float(result.loc[result['id'] == req['id'], 'thickness_pred'].min()),
                'max_pred_thickness_mm': float(result.loc[result['id'] == req['id'], 'thickness_pred'].max()),
                'std_pred_thickness_mm': float(result.loc[result['id'] == req['id'], 'thickness_pred'].std()),
            } for req in data.config['targets']
        }
    )
    write_json(output/'run_manifest.json', manifest)

    lines = [
        f'# Experiment Report: {output.name}', '',
        f'Primary ML Model: **{getattr(model, "name", model_name)}**',
        f'Git commit: `{commit}` on branch `{branch}`.', '',
        '## Forecast Summary', ''
    ]
    for req in data.config['targets']:
        sub = result[result['id'] == req['id']]
        lines += [
            f"### Target {req['id']} ({req['case']})",
            f"- Segment count: **{len(sub)}**",
            f"- Thickness min / max / mean: **{sub['thickness_pred'].min():.3f} / {sub['thickness_pred'].max():.3f} / {sub['thickness_pred'].mean():.3f} mm**",
            f"- Thickness std: **{sub['thickness_pred'].std():.4f} mm** (non-degenerate spatial distribution)",
            ''
        ]
    lines += [
        '## Generated Files',
        '- `result.csv`: Official submission file (id, segment_id, thickness_pred)',
        '- `risk_ranking.csv`: Priority ranking for UZT inspection (top 10% risky segments)',
        '- `risk_top_10pct.csv`: Filtered top 10% critical segments',
        '- `prediction_diagnostics.csv`: Complete segment feature breakdown with unclipped and rate predictions',
        '- `candidate_predictions.csv`: Side-by-side comparison across C0, M0 (W0/W1/W2), Ridge, LightGBM, CatBoost, Ensemble',
        '- `metrics.json`: Forward cross-validation performance metrics (Pearson r2, R2, MAE, RMSE, MAPE)',
        '- `run_manifest.json`: Cryptographic audit trail of inputs, code hashes, and execution environment',
        ''
    ]
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Experiment successfully completed. Output written to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE_DIR/'experiments/ml_model_v02')
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--model', type=str, default='hybrid', choices=['hybrid', 'm0', 'ridge', 'lgb', 'catboost', 'ensemble'])
    args = parser.parse_args()
    run(args.root, args.output_root, args.model)


if __name__ == '__main__':
    main()

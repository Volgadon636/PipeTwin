"""Train-only preprocessing and deterministic Ridge correction for S1C.

The learned quantity is ``measured_thickness - baseline`` in millimetres.  It is
an effective correction to a historical minimum, not a physical corrosion rate.
The model is deliberately small: one Ridge with a fixed ``alpha=10`` and a
fixed feature order.  No hyperparameter search is performed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .evaluation import metrics
from .s1_contract import (CONTRACT_VERSION, RIDGE_FEATURES, TARGET_COLUMN)

ARTIFACT_VERSION = 's1c-ridge-v1'
DEFAULT_ALPHA = 10.0
DEFAULT_SOLVER = 'cholesky'
WEIGHTING = 'equal_total_per_simple_pipe_id_normalized_mean_1'


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f'Missing declared feature columns: {missing}')
    result = frame[columns].apply(pd.to_numeric, errors='coerce')
    if np.isinf(result.to_numpy(float)).any():
        raise ValueError('Infinite feature value; only missing values may be imputed')
    return result


class S1Preprocessor:
    """Train-only median imputation, constant/empty exclusion and scaling.

    ``fit`` sees only the training fold.  ``transform`` applies the stored
    statistics and the exact stored feature order; it never refits.
    """

    def __init__(self, feature_order: list[str] | tuple[str, ...] = RIDGE_FEATURES):
        self.feature_order_ = list(feature_order)
        self.empty_columns_: list[str] = []
        self.constant_columns_: list[str] = []
        self.used_features_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.imputer_statistics_: dict[str, float] = {}
        self.scaler_mean_: dict[str, float] = {}
        self.scaler_scale_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> 'S1Preprocessor':
        x = _numeric(frame, self.feature_order_)
        self.empty_columns_ = [c for c in self.feature_order_ if x[c].isna().all()]
        nonempty = [c for c in self.feature_order_ if c not in self.empty_columns_]
        self.constant_columns_ = [
            c for c in nonempty if x[c].nunique(dropna=True) <= 1]
        self.used_features_ = [
            c for c in nonempty if c not in self.constant_columns_]
        if not self.used_features_:
            raise ValueError('No usable features after train-only exclusions')
        # All-empty and constant columns were removed explicitly, so the
        # imputer is never asked to silently drop a feature.
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=False)
        imputed = self.imputer_.fit_transform(x[self.used_features_])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(imputed)
        if not np.isfinite(self.scaler_.scale_).all() or (self.scaler_.scale_ <= 0).any():
            raise ValueError('Nonpositive or nonfinite train scaler scale')
        self.imputer_statistics_ = {
            c: float(v) for c, v in zip(self.used_features_, self.imputer_.statistics_)}
        self.scaler_mean_ = {
            c: float(v) for c, v in zip(self.used_features_, self.scaler_.mean_)}
        self.scaler_scale_ = {
            c: float(v) for c, v in zip(self.used_features_, self.scaler_.scale_)}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError('S1Preprocessor is not fitted')
        x = _numeric(frame, self.feature_order_)
        if not self.used_features_:
            raise RuntimeError('No used features in the fitted preprocessor')
        imputed = self.imputer_.transform(x[self.used_features_])
        scaled = self.scaler_.transform(imputed)
        if not np.isfinite(scaled).all():
            raise ValueError('Nonfinite transformed design matrix')
        return scaled

    def metadata(self) -> dict[str, Any]:
        return {
            'feature_order': list(self.feature_order_),
            'used_features': list(self.used_features_),
            'empty_columns': list(self.empty_columns_),
            'constant_columns': list(self.constant_columns_),
            'dropped_columns': {
                **{c: 'all_missing_on_training_fold' for c in self.empty_columns_},
                **{c: 'constant_on_training_fold' for c in self.constant_columns_},
            },
            'imputer': 'SimpleImputer(strategy=median)',
            'imputer_statistics': dict(self.imputer_statistics_),
            'scaler': 'StandardScaler',
            'scaler_mean': dict(self.scaler_mean_),
            'scaler_scale': dict(self.scaler_scale_),
        }


def equal_simple_pipe_weights(train: pd.DataFrame) -> np.ndarray:
    """Equal total weight per training simple section, normalized to mean 1.

    This is intentionally *not* the parent/campaign weighting used by v2: all
    three pre-cutoff simple sections share one parent pipeline, so parent
    balancing would silently reduce to row/campaign balancing.
    """
    if train.empty:
        raise ValueError('Cannot compute weights on an empty training fold')
    if train.id.isna().any():
        raise ValueError('Training simple section id is missing')
    counts = train.groupby('id').segment_id.transform('size').to_numpy(float)
    weights = 1.0 / counts
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid simple-section weights')
    return weights / weights.mean()


def fit_s1_ridge(train: pd.DataFrame, alpha: float = DEFAULT_ALPHA,
                 feature_order: list[str] | tuple[str, ...] = RIDGE_FEATURES,
                 target: str = 'target_delta_mm',
                 baseline_column: str = 's1b_prediction',
                 weighting: str = WEIGHTING,
                 model_name: str = 'S1C_ridge') -> dict[str, Any]:
    """Fit the full S1C artifact on a training frame.

    The frame must already be filtered to rows with an own prior baseline.
    """
    if not alpha > 0:
        raise ValueError('Ridge alpha must be positive')
    if train.empty:
        raise ValueError('Cannot fit S1C on zero rows')
    if target not in train or not np.isfinite(train[target].to_numpy(float)).all():
        raise ValueError(f'Missing or nonfinite target: {target}')
    if baseline_column not in train:
        raise ValueError(f'Missing baseline column: {baseline_column}')
    if not train.get('fit_eligible', pd.Series(True, index=train.index)).astype(bool).all():
        raise ValueError('S1C fit rows must all have an own prior baseline')
    preprocessor = S1Preprocessor(feature_order).fit(train)
    design = preprocessor.transform(train)
    weights = equal_simple_pipe_weights(train)
    ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver=DEFAULT_SOLVER)
    ridge.fit(design, train[target].to_numpy(float), sample_weight=weights)
    coefficients = [float(v) for v in ridge.coef_]
    metadata = {
        'artifact_version': ARTIFACT_VERSION,
        'contract_version': CONTRACT_VERSION,
        'model_name': model_name,
        'target': target,
        'baseline_column': baseline_column,
        'prediction_rule': 'clip(baseline + ridge_delta, 0, nominal_thickness_mm)',
        'bounds_policy': {'lower_mm': 0.0, 'upper_column': 'nominal_thickness_mm',
                          'upper_is_not_baseline': True},
        'fallback_policy': 'rows without own prior measurement keep S1B and do not enter Ridge',
        'alpha': float(alpha),
        'fit_intercept': True,
        'solver': DEFAULT_SOLVER,
        'weighting': weighting,
        'weighting_detail': 'each simple section has equal total weight; weights normalized to mean 1',
        'feature_order': list(preprocessor.feature_order_),
        'used_features': list(preprocessor.used_features_),
        'dropped_columns': preprocessor.metadata()['dropped_columns'],
        'coefficients': coefficients,
        'intercept': float(ridge.intercept_),
        'coefficient_by_declared_feature': {
            c: (float(coefficients[preprocessor.used_features_.index(c)])
                if c in preprocessor.used_features_ else None)
            for c in preprocessor.feature_order_
        },
        'training_rows': int(len(train)),
        'training_simple_sections': sorted(int(v) for v in train.id.unique()),
        'training_dates': sorted(str(pd.Timestamp(v).date())
                                 for v in train.prediction_date.unique()),
        'training_date_max': str(pd.Timestamp(train.prediction_date.max()).date()),
        'training_parents': sorted(str(v) for v in train.main_pipeline_id.unique()),
        'target_delta_mean_mm': float(train[target].mean()),
        'target_delta_std_mm': float(train[target].std(ddof=0)),
        'weights_min': float(weights.min()),
        'weights_max': float(weights.max()),
        'preprocessing': preprocessor.metadata(),
        'interpretation': ('Learned effective correction to a conservative past minimum; '
                           'not a physical corrosion rate. Both signs are allowed.'),
    }
    if len(coefficients) != len(preprocessor.used_features_):
        raise ValueError('Coefficient/feature length mismatch')
    return {
        'artifact_version': ARTIFACT_VERSION,
        'contract_version': CONTRACT_VERSION,
        'preprocessor': preprocessor,
        'ridge': ridge,
        'metadata': metadata,
    }


def _artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError('Model artifact must be a dictionary')
    if artifact.get('artifact_version') != ARTIFACT_VERSION:
        raise ValueError('Unsupported S1C artifact version')
    if artifact.get('contract_version') != CONTRACT_VERSION:
        raise ValueError('S1C artifact contract version mismatch')
    metadata = artifact.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('S1C artifact metadata is missing')
    return metadata


def predict_s1_ridge(artifact: dict[str, Any], frame: pd.DataFrame,
                     baseline_column: str | None = None) -> pd.DataFrame:
    """Apply a saved artifact; no fit and no training labels are used."""
    metadata = _artifact_metadata(artifact)
    preprocessor: S1Preprocessor = artifact['preprocessor']
    ridge: Ridge = artifact['ridge']
    if not hasattr(ridge, 'coef_'):
        raise ValueError('Ridge model is not fitted')
    baseline_column = baseline_column or metadata['baseline_column']
    required = {'id', 'segment_id', 'has_own_prior_vtd', baseline_column,
                'nominal_thickness_mm'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Prediction frame is missing columns: {sorted(missing)}')
    result = frame.copy()
    apply = result.has_own_prior_vtd.fillna(False).astype(bool)
    baseline = result[baseline_column].astype(float)
    delta = np.zeros(len(result), dtype=float)
    if apply.any():
        design = preprocessor.transform(result.loc[apply])
        delta[apply.to_numpy()] = ridge.predict(design)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite Ridge delta')
    upper = result.nominal_thickness_mm.to_numpy(float)
    unclipped = baseline.to_numpy(float) + delta
    prediction = np.clip(unclipped, 0.0, upper)
    clipped = (unclipped < 0.0) | (unclipped > upper)
    result['ridge_delta_mm'] = delta
    result['thickness_pred_unclipped'] = unclipped
    result['thickness_pred'] = prediction
    result['clipped'] = clipped
    result['ridge_applied'] = apply.to_numpy()
    result['prediction_source'] = np.where(
        apply.to_numpy(), 'ridge_s1c', 's1b_fallback')
    result['prediction_lower_bound_mm'] = 0.0
    result['prediction_upper_bound_mm'] = upper
    result['baseline_column_used'] = baseline_column
    if not np.isfinite(result[['thickness_pred', 'thickness_pred_unclipped']]
                       .to_numpy(float)).all():
        raise ValueError('Nonfinite S1C prediction')
    return result


def save_artifact(artifact: dict[str, Any], path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    metadata_path = path.with_name('model_metadata.json')
    metadata_path.write_text(json.dumps(artifact['metadata'], ensure_ascii=False,
                                        indent=2, sort_keys=True, allow_nan=False) + '\n',
                             encoding='utf-8')
    return artifact['metadata']


def load_artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model artifact not found: {path}')
    artifact = joblib.load(path)
    _artifact_metadata(artifact)
    return artifact


def _metric_record(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    raw = metrics(truth, prediction)
    return {
        'n': int(raw['n']),
        'mae_mm': raw['mae_mm'],
        'rmse_mm': raw['rmse_mm'],
        'bias_mm': raw['bias_mm'],
        'pearson_r': raw['pearson_r'],
        'pearson_squared': raw['pearson_squared'],
        'pearson_status': raw['pearson_status'],
        'pearson_reason': raw['pearson_reason'],
        'r2': raw['r2'],
        'r2_status': raw['r2_status'],
        'r2_reason': raw['r2_reason'],
    }


def metrics_by_group(frame: pd.DataFrame, prediction: str,
                     truth: str = TARGET_COLUMN,
                     group: str = 'id') -> pd.DataFrame:
    records = []
    for key, part in frame.groupby(group, sort=True):
        if part.empty:
            continue
        record = {'group': str(key), **_metric_record(
            part[truth].to_numpy(float), part[prediction].to_numpy(float))}
        record['coverage'] = int(len(part))
        records.append(record)
    return pd.DataFrame(records)


def pooled_metrics(frame: pd.DataFrame, prediction: str,
                   truth: str = TARGET_COLUMN) -> dict[str, Any]:
    record = _metric_record(frame[truth].to_numpy(float),
                            frame[prediction].to_numpy(float))
    record['coverage'] = int(len(frame))
    return record


def macro_metrics(per_group: pd.DataFrame) -> dict[str, Any]:
    result = {'groups': int(len(per_group))}
    for column in ('mae_mm', 'rmse_mm', 'bias_mm', 'pearson_r',
                   'pearson_squared', 'r2'):
        values = pd.to_numeric(per_group[column], errors='coerce').dropna()
        result[column] = None if values.empty else float(values.mean())
        result[column + '_defined_groups'] = int(len(values))
    return result


def evaluate_models_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-fold and pooled metrics on the exact same out-of-fold rows."""
    models = ['S1A', 'S1B', 'S1C', 'S1A_ridge_control']
    columns = {'S1A': 'prediction_s1a', 'S1B': 'prediction_s1b',
               'S1C': 'prediction_s1c', 'S1A_ridge_control': 'prediction_s1a_control'}
    rows = []
    for model, column in columns.items():
        per_group = metrics_by_group(oof, column)
        per_group.insert(0, 'model', model)
        rows.append(per_group)
    per_fold = pd.concat(rows, ignore_index=True)
    pooled = {}
    for model, column in columns.items():
        pooled[model] = {
            'pooled': pooled_metrics(oof, column),
            'macro': macro_metrics(per_fold[per_fold.model.eq(model)]),
            'per_simple_section': per_fold[per_fold.model.eq(model)]
                .drop(columns='model').to_dict(orient='records'),
        }
    return per_fold, pooled

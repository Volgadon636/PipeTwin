"""Shared utilities for the next S1/S2 cycle.

This module deliberately does not modify the verified preprocessing artifacts.
It adds a stricter experiment layer: train-only preprocessing with duplicate
detection, predeclared weights, per-section and within-section metrics, and
artifact/manifest helpers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.storage import sha256, write_json
from .data import RepositoryData
from .evaluation import metrics as base_metrics
from .s1_contract import (S1_CUTOFF, S1_ID, TARGET_COLUMN,
                          build_s1_training_frame, production_frame)

YEAR_SECONDS = 365.25 * 86400
S2_ID = 1250009583
S2_CUTOFF = pd.Timestamp('2022-07-01')
RNG_SEED = 17

HISTORY_FEATURES = [
    'last_mm',
    'age_last_vtd_years',
    'prior_vtd_campaign_count',
    'neighbor_prior_median_mm',
    'neighbor_prior_count',
    'neighbor_missing',
]
GEOMETRY_FEATURES = [
    'slope_m_per_m',
    'abs_slope_m_per_m',
    'local_low_m',
    'relative_position',
]
S1_CORE_FEATURES = HISTORY_FEATURES + GEOMETRY_FEATURES
S1_HISTORY_ONLY_FEATURES = HISTORY_FEATURES

# Local denoising features of the last prior VTD profile.  The target is a
# 10 m segment *minimum* of residual wall thickness, so a single-segment prior
# minimum is noisy; a 3-segment rolling median and rolling minimum recover the
# persistent local thinning pattern (validated in block CV, see
# scratch/metrics_upgrade).  These are inputs-only features, never labels.
S1_LOCAL_PRIOR_FEATURES = ['last_smooth3', 'last_rollmin3']


def add_local_prior_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 3-segment rolling median/min of ``last_mm`` (input-only).

    The window is spatial only: features are computed within a single
    ``(id, prediction_date)`` snapshot so that different campaigns for the
    same segment are never mixed into one window.
    """
    missing = {'id', 'segment_id', 'prediction_date', 'last_mm'} - set(frame.columns)
    if missing:
        raise ValueError(f'Local prior features need columns: {sorted(missing)}')
    result = frame.copy()
    ordered = result.sort_values(['id', 'prediction_date', 'segment_id'])
    group_keys = ['id', 'prediction_date']
    for name, window in [('last_smooth3', 3), ('last_rollmin3', 3)]:
        func = 'median' if 'smooth' in name else 'min'
        ordered[name] = ordered.groupby(group_keys, sort=True)['last_mm'].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).agg(func))
    ordered = ordered.sort_index()
    for column in S1_LOCAL_PRIOR_FEATURES:
        result[column] = ordered[column].to_numpy(float)
    return result


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(['git', *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {'commit': run('rev-parse', 'HEAD'),
                'branch': run('branch', '--show-current'),
                'dirty': bool(run('status', '--porcelain')),
                'tracked_dirty': bool(run('status', '--porcelain', '--untracked-files=no')),
                'status': run('status', '--porcelain').splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {'commit': 'unavailable', 'branch': 'unavailable', 'error': str(error)}


def versions() -> dict[str, str]:
    result = {'python': platform.python_version(), 'pandas': pd.__version__,
              'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
              'joblib': joblib.__version__, 'executable': sys.executable,
              'platform': platform.platform()}
    try:
        import pyarrow
        result['pyarrow'] = pyarrow.__version__
    except Exception:
        pass
    try:
        import catboost
        result['catboost'] = catboost.__version__
    except Exception:
        pass
    try:
        import matplotlib
        result['matplotlib'] = matplotlib.__version__
    except Exception:
        pass
    return result


def code_hashes(root: Path) -> dict[str, str]:
    names = sorted((root / 'src/ml').glob('next_*.py'))
    names += sorted((root / 'src/ml').glob('s1_next.py'))
    names += sorted((root / 'src/ml').glob('s2_next.py'))
    names += sorted((root / 'src/ml').glob('s1_*.py'))
    names.append(root / 'src/ml/submission.py')
    return {str(p.relative_to(root)): sha256(p) for p in names if p.is_file()}


def write_csv(frame: pd.DataFrame, path: Path, float_format: str = '%.10g') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8', lineterminator='\n',
                 float_format=float_format)


def write_manifest(path: Path, *, root: Path, data: RepositoryData, output: Path,
                   stage: str, started: str, extra: dict[str, Any] | None = None,
                   input_paths: Iterable[Path] = ()) -> dict[str, Any]:
    inputs = dict(data.hashes)
    for p in input_paths:
        p = Path(p)
        if p.is_file():
            inputs[str(p)] = sha256(p)
    for name in ('configs/preprocessing.json', 'reports/run_manifest.json'):
        p = data.root / name
        if p.is_file():
            inputs[name] = sha256(p)
    outputs = {p.name: sha256(p) for p in sorted(Path(output).glob('*'))
               if p.is_file() and p.name != 'run_manifest.json'}
    payload = {
        'stage': stage,
        'started_utc': started,
        'finished_utc': utcnow(),
        'root': str(root),
        'data_root': str(data.root),
        'output_root': str(output),
        'git': git_state(root),
        'versions': versions(),
        'input_sha256': inputs,
        'code_sha256': code_hashes(root),
        'outputs_sha256': outputs,
    }
    if extra:
        payload.update(extra)
    write_json(Path(path), json_safe(payload))
    return payload


def load_data(root: Path) -> RepositoryData:
    return RepositoryData(root)


# --------------------------------------------------------------------------- #
# S1 table
# --------------------------------------------------------------------------- #
def s1_training_table(data: RepositoryData) -> pd.DataFrame:
    """Historical S1 rows with last-baseline semantics, labels and geometry."""
    frame = build_s1_training_frame(data)
    frame = frame.copy()
    frame['next_mm'] = frame[TARGET_COLUMN].astype(float)
    frame['last_mm'] = frame['s1a_prediction'].astype(float)
    frame['last_own_mm'] = frame['own_last_thickness_mm'].astype(float)
    frame['dt_years'] = (frame['prediction_date'] - frame['last_vtd_date']).dt.total_seconds() / YEAR_SECONDS
    frame['rate_mm_year'] = np.where(
        frame['has_own_prior_vtd'] & frame['dt_years'].gt(0),
        (frame['last_own_mm'] - frame['next_mm']) / frame['dt_years'], np.nan)
    frame['section_size'] = frame.groupby('id').segment_id.transform('size')
    frame['last_minus_min_mm'] = (frame['last_own_mm']
                                  - frame['prior_vtd_min_thickness_mm'])
    frame['baseline_missing'] = ~frame['has_own_prior_vtd'].astype(bool)
    return add_local_prior_features(frame)


def s1_fit_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table.fit_eligible.astype(bool)].copy()


def s1_production_table(data: RepositoryData) -> pd.DataFrame:
    prod, _ = production_frame(data)
    prod = prod.copy()
    prod['last_mm'] = prod['s1a_prediction'].astype(float)
    prod['last_own_mm'] = prod['own_last_thickness_mm'].astype(float)
    prod['dt_years'] = (prod['prediction_date'] - prod['last_vtd_date']).dt.total_seconds() / YEAR_SECONDS
    prod['section_size'] = prod.groupby('id').segment_id.transform('size')
    prod['baseline_missing'] = ~prod['has_own_prior_vtd'].astype(bool)
    return add_local_prior_features(prod)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f'Missing features: {missing}')
    values = frame[columns].apply(pd.to_numeric, errors='coerce')
    if np.isinf(values.to_numpy(float)).any():
        raise ValueError('Infinite feature value')
    return values


class NextPreprocessor:
    """Train-only imputation/scaling with explicit empty/constant/duplicate drops."""

    def __init__(self, feature_order: list[str]):
        self.feature_order_ = list(feature_order)
        self.empty_columns_: list[str] = []
        self.constant_columns_: list[str] = []
        self.duplicate_columns_: dict[str, dict[str, Any]] = {}
        self.used_features_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.imputer_statistics_: dict[str, float] = {}
        self.scaler_mean_: dict[str, float] = {}
        self.scaler_scale_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> 'NextPreprocessor':
        x = _numeric(frame, self.feature_order_)
        self.empty_columns_ = [c for c in self.feature_order_ if x[c].isna().all()]
        nonempty = [c for c in self.feature_order_ if c not in self.empty_columns_]
        self.constant_columns_ = [c for c in nonempty
                                  if x[c].nunique(dropna=True) <= 1]
        candidates = [c for c in nonempty if c not in self.constant_columns_]
        keep: list[str] = []
        for column in candidates:
            duplicate_of = None
            relation = None
            for earlier in keep:
                both = x[[earlier, column]].dropna()
                if len(both) < 10:
                    continue
                if np.allclose(both[earlier].to_numpy(float),
                               both[column].to_numpy(float), rtol=0, atol=1e-12):
                    duplicate_of, relation = earlier, 'exact_duplicate'
                    break
                a = both[earlier].to_numpy(float)
                b = both[column].to_numpy(float)
                if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                if abs(corr) >= 0.9999:
                    duplicate_of = earlier
                    relation = f'near_duplicate_abs_corr_{abs(corr):.8f}'
                    break
            if duplicate_of is None:
                keep.append(column)
            else:
                self.duplicate_columns_[column] = {
                    'kept': duplicate_of, 'relation': relation}
        self.used_features_ = keep
        if not self.used_features_:
            raise ValueError('No usable features after exclusions')
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=False)
        imputed = self.imputer_.fit_transform(x[self.used_features_])
        self.scaler_ = StandardScaler()
        self.scaler_.fit(imputed)
        if not np.isfinite(self.scaler_.scale_).all() or (self.scaler_.scale_ <= 0).any():
            raise ValueError('Invalid train scaler')
        self.imputer_statistics_ = {
            c: float(v) for c, v in zip(self.used_features_, self.imputer_.statistics_)}
        self.scaler_mean_ = {
            c: float(v) for c, v in zip(self.used_features_, self.scaler_.mean_)}
        self.scaler_scale_ = {
            c: float(v) for c, v in zip(self.used_features_, self.scaler_.scale_)}
        return self

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError('Preprocessor is not fitted')
        x = _numeric(frame, self.feature_order_)
        imputed = self.imputer_.transform(x[self.used_features_])
        scaled = self.scaler_.transform(imputed)
        if not np.isfinite(scaled).all():
            raise ValueError('Nonfinite transformed features')
        return pd.DataFrame(scaled, columns=self.used_features_, index=frame.index)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform_frame(frame).to_numpy(float)

    def metadata(self) -> dict[str, Any]:
        return {
            'feature_order': list(self.feature_order_),
            'used_features': list(self.used_features_),
            'empty_columns': list(self.empty_columns_),
            'constant_columns': list(self.constant_columns_),
            'duplicate_columns': dict(self.duplicate_columns_),
            'dropped_columns': {
                **{c: 'all_missing_on_train' for c in self.empty_columns_},
                **{c: 'constant_on_train' for c in self.constant_columns_},
                **{c: v['relation'] for c, v in self.duplicate_columns_.items()},
            },
            'imputer': 'median',
            'imputer_statistics': dict(self.imputer_statistics_),
            'scaler': 'StandardScaler',
            'scaler_mean': dict(self.scaler_mean_),
            'scaler_scale': dict(self.scaler_scale_),
        }


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def make_weights(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    if frame.empty:
        raise ValueError('No rows for weights')
    counts = frame.groupby('id').segment_id.transform('size').to_numpy(float)
    if scheme == 'row_uniform':
        weights = np.ones(len(frame))
    elif scheme == 'inverse_sqrt_section_size':
        weights = 1.0 / np.sqrt(counts)
    elif scheme == 'inverse_section_size':
        weights = 1.0 / counts
    else:
        raise ValueError(f'Unknown weight scheme: {scheme}')
    weights = weights.astype(float)
    return weights / weights.mean()


def weight_diagnostics(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    weights = np.asarray(weights, float)
    totals = pd.Series(weights).groupby(frame.id.to_numpy()).sum()
    return {
        'mean': float(weights.mean()),
        'min': float(weights.min()),
        'max': float(weights.max()),
        'kish_ess': float(weights.sum() ** 2 / np.sum(weights ** 2)),
        'group_totals': {str(k): float(v) for k, v in totals.items()},
        'rows': int(len(frame)),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metric_record(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    raw = base_metrics(truth, prediction)
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
        'truth_std_mm': float(np.std(truth, ddof=0)),
        'prediction_std_mm': float(np.std(prediction, ddof=0)),
    }


def per_section_metrics(frame: pd.DataFrame, prediction: str,
                        truth: str = 'next_mm', section: str = 'id') -> pd.DataFrame:
    records = []
    for key, part in frame.groupby(section, sort=True):
        record = {'section': int(key), **_metric_record(
            part[truth].to_numpy(float), part[prediction].to_numpy(float))}
        records.append(record)
    return pd.DataFrame(records)


def macro_metrics(per_section: pd.DataFrame) -> dict[str, Any]:
    result = {'sections': int(len(per_section))}
    for column in ('mae_mm', 'rmse_mm', 'bias_mm', 'pearson_r', 'pearson_squared', 'r2'):
        values = pd.to_numeric(per_section[column], errors='coerce').dropna()
        result[column] = None if values.empty else float(values.mean())
        result[column + '_defined_sections'] = int(len(values))
    return result


def within_section_centered_pearson(frame: pd.DataFrame, prediction: str,
                                    truth: str = 'next_mm', section: str = 'id') -> dict[str, Any]:
    ts, ps = [], []
    for _, part in frame.groupby(section, sort=True):
        y = part[truth].to_numpy(float)
        p = part[prediction].to_numpy(float)
        ts.append(y - y.mean())
        ps.append(p - p.mean())
    y = np.concatenate(ts) if ts else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    if len(y) == 0 or np.ptp(y) <= 1e-12 or np.ptp(p) <= 1e-12:
        return {'pearson_r': None, 'pearson_squared': None, 'status': 'undefined_constant'}
    r = float(np.corrcoef(y, p)[0, 1])
    return {'pearson_r': r, 'pearson_squared': r * r, 'status': 'defined'}


def topk_recall(frame: pd.DataFrame, prediction: str, truth: str = 'next_mm',
                section: str = 'id', fraction: float = 0.1) -> dict[str, Any]:
    records, overlaps, ks = [], 0, 0
    for key, part in frame.groupby(section, sort=True):
        part = part.sort_values('segment_id')
        k = int(np.ceil(fraction * len(part)))
        truth_top = set(part.sort_values([truth, 'segment_id']).head(k).segment_id)
        pred_top = set(part.sort_values([prediction, 'segment_id']).head(k).segment_id)
        overlap = len(truth_top & pred_top)
        records.append({'section': int(key), 'n': int(len(part)), 'k': k,
                        'overlap': overlap, 'recall': overlap / k if k else None})
        overlaps += overlap
        ks += k
    return {'fraction': fraction, 'per_section': records,
            'pooled_recall': overlaps / ks if ks else None,
            'macro_recall': float(np.mean([r['recall'] for r in records
                                           if r['recall'] is not None]))
            if records else None}


def evaluate_predictions(frame: pd.DataFrame, prediction: str,
                         truth: str = 'next_mm', section: str = 'id',
                         top_fraction: float = 0.1) -> dict[str, Any]:
    per = per_section_metrics(frame, prediction, truth, section)
    return {
        'pooled': _metric_record(frame[truth].to_numpy(float),
                                 frame[prediction].to_numpy(float)),
        'macro': macro_metrics(per),
        'per_section': per.to_dict(orient='records'),
        'within_section_centered': within_section_centered_pearson(
            frame, prediction, truth, section),
        'topk': topk_recall(frame, prediction, truth, section, top_fraction),
    }


def evaluate_many(frame: pd.DataFrame, predictions: dict[str, str],
                  truth: str = 'next_mm', section: str = 'id') -> dict[str, Any]:
    return {name: evaluate_predictions(frame, column, truth, section)
            for name, column in predictions.items()}


# --------------------------------------------------------------------------- #
# Ridge / CatBoost artifacts
# --------------------------------------------------------------------------- #
def fit_ridge(frame: pd.DataFrame, features: list[str], target: str,
              weights: np.ndarray, alpha: float = 10.0,
              model_name: str = 'ridge', baseline: str | None = None,
              prediction_policy: str = 'delta') -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    y = frame[target].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError(f'Nonfinite target {target}')
    pre = NextPreprocessor(features).fit(frame)
    x = pre.transform(frame)
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver='cholesky')
    model.fit(x, y, sample_weight=weights)
    metadata = {
        'model_name': model_name,
        'model_type': 'Ridge',
        'features': list(features),
        'used_features': list(pre.used_features_),
        'dropped_columns': pre.metadata()['dropped_columns'],
        'preprocessing': pre.metadata(),
        'target': target,
        'baseline_column': baseline,
        'prediction_policy': prediction_policy,
        'alpha': float(alpha),
        'fit_intercept': True,
        'solver': 'cholesky',
        'training_rows': int(len(frame)),
        'training_sections': sorted(int(v) for v in frame.id.unique()),
        'training_dates': sorted(str(pd.Timestamp(v).date())
                                 for v in frame.prediction_date.unique()),
        'intercept': float(model.intercept_),
        'coefficients': [float(v) for v in model.coef_],
        'coefficient_by_feature': dict(zip(pre.used_features_,
                                           [float(v) for v in model.coef_])),
        'target_mean': float(y.mean()),
        'target_std': float(y.std(ddof=0)),
    }
    return {'model_type': 'Ridge', 'preprocessor': pre, 'estimator': model,
            'metadata': metadata}


def predict_ridge(artifact: dict[str, Any], frame: pd.DataFrame,
                  baseline: str | None = None) -> np.ndarray:
    pre: NextPreprocessor = artifact['preprocessor']
    model = artifact['estimator']
    metadata = artifact['metadata']
    baseline = baseline or metadata.get('baseline_column')
    if baseline is None:
        return model.predict(pre.transform(frame))
    if baseline not in frame:
        raise ValueError(f'Missing baseline column: {baseline}')
    delta = model.predict(pre.transform(frame))
    policy = metadata.get('prediction_policy', 'delta')
    raw = frame[baseline].to_numpy(float) + delta
    upper = frame.nominal_thickness_mm.to_numpy(float)
    return np.clip(raw, 0.0, upper)


def fit_catboost(frame: pd.DataFrame, features: list[str], target: str,
                 weights: np.ndarray, model_name: str = 'catboost',
                 baseline: str | None = None,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
    from catboost import CatBoostRegressor
    y = frame[target].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError(f'Nonfinite target {target}')
    pre = NextPreprocessor(features).fit(frame)
    x = pre.transform_frame(frame)
    defaults = dict(iterations=200, depth=2, learning_rate=0.03,
                    l2_leaf_reg=10, loss_function='RMSE', random_seed=RNG_SEED,
                    use_best_model=False, verbose=False,
                    allow_writing_files=False, thread_count=4)
    if params:
        defaults.update(params)
    model = CatBoostRegressor(**defaults)
    model.fit(x, y, sample_weight=weights)
    metadata = {
        'model_name': model_name,
        'model_type': 'CatBoostRegressor',
        'features': list(features),
        'used_features': list(pre.used_features_),
        'dropped_columns': pre.metadata()['dropped_columns'],
        'preprocessing': pre.metadata(),
        'target': target,
        'baseline_column': baseline,
        'prediction_policy': 'delta',
        'params': {k: v for k, v in defaults.items() if k not in ('verbose',)},
        'training_rows': int(len(frame)),
        'training_sections': sorted(int(v) for v in frame.id.unique()),
        'training_dates': sorted(str(pd.Timestamp(v).date())
                                 for v in frame.prediction_date.unique()),
        'feature_importance': dict(zip(pre.used_features_,
                                       [float(v) for v in model.get_feature_importance()])),
    }
    return {'model_type': 'CatBoost', 'preprocessor': pre,
            'estimator': model, 'metadata': metadata}


def predict_catboost(artifact: dict[str, Any], frame: pd.DataFrame,
                     baseline: str | None = None) -> np.ndarray:
    pre: NextPreprocessor = artifact['preprocessor']
    model = artifact['estimator']
    metadata = artifact['metadata']
    baseline = baseline or metadata.get('baseline_column')
    if baseline is None:
        return model.predict(pre.transform_frame(frame))
    delta = model.predict(pre.transform_frame(frame))
    raw = frame[baseline].to_numpy(float) + delta
    return np.clip(raw, 0.0, frame.nominal_thickness_mm.to_numpy(float))


def save_artifact(artifact: dict[str, Any], directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'model_artifact.joblib'
    joblib.dump(artifact, path)
    write_json(directory / 'model_metadata.json', json_safe(artifact['metadata']))
    return {'artifact_path': str(path), 'artifact_sha256': sha256(path),
            'metadata': artifact['metadata']}


def load_artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)

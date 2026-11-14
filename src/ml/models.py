"""Interpretable, controlled corrosion prediction models.

Includes:
- ConstantRate (M0): Deterministic weighted least-squares baseline.
- RegularizedCorrosionRegression: Simple L2-regularized linear model with inspectable feature weights.
- ConstrainedTreeCorrosionModel: Low-capacity decision tree (max_depth <= 3) partitioned into physical regimes.
- PhysicalModel (M1): Exposure-integrated Arrhenius and multiphase physics model.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

from .exposure import integrated_years


def sample_weights(train: pd.DataFrame, regime: str = 'W1') -> np.ndarray:
    if train.empty:
        raise ValueError('Cannot fit without eligible training labels')
    if regime == 'W0':
        weights = np.ones(len(train))
    elif regime in ('W1', 'W2'):
        # Equal parent-pipeline totals, equal campaigns within each parent.
        counts = train.groupby(['main_pipeline_id', 'campaign_id']).segment_id.transform('size')
        campaigns = train.groupby('main_pipeline_id').campaign_id.transform('nunique')
        weights = 1 / (counts.to_numpy(float) * campaigns.to_numpy(float))
        if regime == 'W2':
            threshold = float(train.min_residual_thickness_mm.quantile(.1))
            weights *= np.where(train.min_residual_thickness_mm.le(threshold), 2., 1.)
    else:
        raise ValueError(f'Unknown weighting: {regime}')
    return weights / weights.mean()


@dataclass
class ConstantRate:
    """M0: Empirical constant corrosion rate model."""
    rate_mm_year: float = 0.0
    name: str = 'M0'
    weighting: str = 'W1'

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.rate_mm_year, dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(frame.h0_mm.to_numpy(float) - self.rate_mm_year * frame.horizon_years.to_numpy(float),
                       0.0, frame.h0_mm.to_numpy(float))

    def metadata(self) -> Dict[str, Any]:
        return asdict(self)


def fit_constant_rate(train: pd.DataFrame, weighting: str = 'W1') -> ConstantRate:
    weights = sample_weights(train, weighting)
    years = train.horizon_years.to_numpy(float)
    loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)
    if not np.isfinite(np.r_[years, loss]).all() or (years <= 0).any():
        raise ValueError('Invalid training horizon/label')
    rate = max(0.0, float(np.sum(weights * years * loss) / np.sum(weights * years**2)))
    return ConstantRate(rate, 'M0', weighting)


class RegularizedCorrosionRegression:
    """Simple L2-regularized linear model predicting thinning rate using standardized features."""

    def __init__(self, feature_cols: List[str], alpha: float = 10.0, weighting: str = 'W1', name: str = 'Ridge_Regression'):
        self.feature_cols = feature_cols
        self.alpha = alpha
        self.weighting = weighting
        self.name = name
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)
        self.base_rate_mm_year: float = 0.0
        self.feature_weights: Dict[str, float] = {}

    def fit(self, train: pd.DataFrame):
        weights = sample_weights(train, self.weighting)
        years = train.horizon_years.to_numpy(float)
        loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)
        target_rate = np.clip(loss / years, 0.0, None)
        self.base_rate_mm_year = float(np.average(target_rate, weights=weights))

        X = train[self.feature_cols].fillna(0.0).to_numpy(float)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, target_rate, sample_weight=weights)

        # Store standardized feature weights for transparency
        self.feature_weights = {
            col: float(coef) for col, coef in zip(self.feature_cols, self.model.coef_)
        }
        return self

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.feature_cols].fillna(0.0).to_numpy(float)
        X_scaled = self.scaler.transform(X)
        pred_rate = self.model.predict(X_scaled)
        return np.clip(pred_rate, 0.0, None)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        pred_rate = self.predict_rate(frame)
        h0 = frame.h0_mm.to_numpy(float)
        horizon = frame.horizon_years.to_numpy(float)
        return np.clip(h0 - pred_rate * horizon, 0.0, h0)

    def metadata(self) -> Dict[str, Any]:
        return {
            'model': self.name,
            'name': self.name,
            'alpha': self.alpha,
            'weighting': self.weighting,
            'features': self.feature_cols,
            'base_rate_mm_year': self.base_rate_mm_year,
            'feature_weights': self.feature_weights,
        }


class ConstrainedTreeCorrosionModel:
    """Low-capacity decision tree predicting corrosion rate partitioned by physical regimes."""

    def __init__(self, feature_cols: List[str], max_depth: int = 2, min_samples_leaf: int = 15,
                 weighting: str = 'W1', name: str = 'Constrained_Tree'):
        self.feature_cols = feature_cols
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.weighting = weighting
        self.name = name
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.base_rate_mm_year: float = 0.0
        self.feature_importances: Dict[str, float] = {}

    def fit(self, train: pd.DataFrame):
        weights = sample_weights(train, self.weighting)
        years = train.horizon_years.to_numpy(float)
        loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)
        target_rate = np.clip(loss / years, 0.0, None)
        self.base_rate_mm_year = float(np.average(target_rate, weights=weights))

        X = train[self.feature_cols].fillna(0.0)
        self.model.fit(X, target_rate, sample_weight=weights)

        self.feature_importances = {
            col: float(imp) for col, imp in zip(self.feature_cols, self.model.feature_importances_)
        }
        return self

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.feature_cols].fillna(0.0)
        pred_rate = self.model.predict(X)
        return np.clip(pred_rate, 0.0, None)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        pred_rate = self.predict_rate(frame)
        h0 = frame.h0_mm.to_numpy(float)
        horizon = frame.horizon_years.to_numpy(float)
        return np.clip(h0 - pred_rate * horizon, 0.0, h0)

    def metadata(self) -> Dict[str, Any]:
        return {
            'model': self.name,
            'name': self.name,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'weighting': self.weighting,
            'features': self.feature_cols,
            'base_rate_mm_year': self.base_rate_mm_year,
            'feature_importances': self.feature_importances,
            'leaf_count': int(self.model.get_n_leaves()),
        }


@dataclass
class PhysicalModel:
    rate_mm_year: float
    reference: dict
    coefficients: list
    gap_policy: str
    objective: float

    def predict(self, frame: pd.DataFrame, intervals: List[pd.DataFrame]) -> np.ndarray:
        if len(frame) != len(intervals):
            raise ValueError('Exposure/prediction alignment mismatch')
        exposure = np.array([integrated_years(i, self.reference, np.array(self.coefficients),
                             self.gap_policy, t) for i, t in zip(intervals, frame.horizon_years)])
        return np.clip(frame.h0_mm.to_numpy(float) - self.rate_mm_year * exposure,
                       0.0, frame.h0_mm.to_numpy(float))

    def metadata(self) -> Dict[str, Any]:
        result = asdict(self)
        result['theta_T_K'] = 1000 * self.coefficients[0]
        result['name'] = 'M1'
        return result


def fit_physical_model(train: pd.DataFrame, intervals: List[pd.DataFrame], weighting: str = 'W1',
                       gap_policy: str = 'reference', minimum_groups: int = 2) -> PhysicalModel:
    """Regularized coordinate search, deliberately gated by independent groups."""
    if train.main_pipeline_id.nunique() < minimum_groups:
        raise ValueError(f'M1 needs at least {minimum_groups} independent parent pipelines')
    if len(intervals) != len(train):
        raise ValueError('Exposure/label alignment mismatch')
    observed = pd.concat(intervals, ignore_index=True).drop_duplicates(['id', 'date'])
    if observed.empty:
        raise ValueError('No valid exposure intervals')
    reference = {name: float(observed[name].median())
                 for name in ('temperature_K', 'water_fraction', 'velocity_m_s')}
    reference['velocity_offset_m_s'] = .1  # numerical regularizer; m/s
    weights = sample_weights(train, weighting)
    loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)

    def score(coef):
        exp = np.array([integrated_years(i, reference, coef, gap_policy, t)
                        for i, t in zip(intervals, train.horizon_years)])
        denom = np.sum(weights * exp**2)
        rate = max(0.0, float(np.sum(weights * exp * loss) / denom)) if denom > 0 else 0.0
        error = float(np.average((loss - rate * exp)**2, weights=weights) + .1 * np.sum(coef**2))
        return error, rate

    coef = np.zeros(3)
    best, rate = score(coef)
    for step in (1., .25, .05):
        for _ in range(20):
            improved = False
            for axis in range(3):
                for sign in (-1, 1):
                    candidate = coef.copy()
                    candidate[axis] += sign * step
                    if candidate[0] < 0 or np.max(np.abs(candidate)) > 10:
                        continue
                    objective, candidate_rate = score(candidate)
                    if objective < best - 1e-12:
                        coef, best, rate = candidate, objective, candidate_rate
                        improved = True
            if not improved:
                break
    return PhysicalModel(rate, reference, coef.tolist(), gap_policy, best)


# ---------------------------------------------------------------------------
# Gradient-boosting and ensemble adapters (additive — do not alter M0 / M1)
# ---------------------------------------------------------------------------

class RidgeCorrosionModel(RegularizedCorrosionRegression):
    """M2: L2-regularised linear model using all registered ML features.

    Thin adapter over RegularizedCorrosionRegression so that experiment.py
    can import a stable public name without caring about the internal class.
    Feature columns are resolved at fit-time to those present in the frame.
    """

    def __init__(self, alpha: float = 10.0, weighting: str = 'W1'):
        # Feature list resolved lazily at fit() to avoid circular-import at
        # module load; stored on instance after first fit.
        super().__init__(feature_cols=[], alpha=alpha, weighting=weighting, name='M2_Ridge')

    def fit(self, train: pd.DataFrame):
        from .features import ML_FEATURE_COLUMNS
        self.feature_cols = [c for c in ML_FEATURE_COLUMNS if c in train.columns]
        return super().fit(train)


class LightGBMCorrosionModel:
    """M3: Gradient-boosted trees (LightGBM) predicting per-segment corrosion rate."""

    def __init__(self, weighting: str = 'W1', n_estimators: int = 200,
                 learning_rate: float = 0.05, max_depth: int = 4,
                 min_child_samples: int = 10, name: str = 'M3_LightGBM'):
        self.weighting = weighting
        self.name = name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.base_rate_mm_year: float = 0.0
        self._model = None
        self._feature_cols: List[str] = []

    def fit(self, train: pd.DataFrame):
        import lightgbm as lgb
        from .features import ML_FEATURE_COLUMNS
        weights = sample_weights(train, self.weighting)
        years = train.horizon_years.to_numpy(float)
        loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)
        target_rate = np.clip(loss / years, 0.0, None)
        self.base_rate_mm_year = float(np.average(target_rate, weights=weights))
        self._feature_cols = [c for c in ML_FEATURE_COLUMNS if c in train.columns]
        X = train[self._feature_cols].fillna(0.0).to_numpy(float)
        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, target_rate, sample_weight=weights)
        return self

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(frame))
        X = frame[self._feature_cols].fillna(0.0).to_numpy(float)
        return np.clip(self._model.predict(X), 0.0, None)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        pred_rate = self.predict_rate(frame)
        h0 = frame.h0_mm.to_numpy(float)
        horizon = frame.horizon_years.to_numpy(float)
        return np.clip(h0 - pred_rate * horizon, 0.0, h0)

    def metadata(self) -> Dict[str, Any]:
        return dict(name=self.name, model=self.name, weighting=self.weighting,
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                    max_depth=self.max_depth, min_child_samples=self.min_child_samples,
                    base_rate_mm_year=self.base_rate_mm_year, features=self._feature_cols)


class CatBoostCorrosionModel:
    """M4: Gradient-boosted trees (CatBoost) predicting per-segment corrosion rate."""

    def __init__(self, weighting: str = 'W1', iterations: int = 200,
                 learning_rate: float = 0.05, depth: int = 4,
                 name: str = 'M4_CatBoost'):
        self.weighting = weighting
        self.name = name
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.base_rate_mm_year: float = 0.0
        self._model = None
        self._feature_cols: List[str] = []

    def fit(self, train: pd.DataFrame):
        from catboost import CatBoostRegressor
        from .features import ML_FEATURE_COLUMNS
        weights = sample_weights(train, self.weighting)
        years = train.horizon_years.to_numpy(float)
        loss = (train.h0_mm - train.min_residual_thickness_mm).to_numpy(float)
        target_rate = np.clip(loss / years, 0.0, None)
        self.base_rate_mm_year = float(np.average(target_rate, weights=weights))
        self._feature_cols = [c for c in ML_FEATURE_COLUMNS if c in train.columns]
        X = train[self._feature_cols].fillna(0.0).to_numpy(float)
        self._model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=42,
            verbose=False,
        )
        self._model.fit(X, target_rate, sample_weight=weights)
        return self

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(frame))
        X = frame[self._feature_cols].fillna(0.0).to_numpy(float)
        return np.clip(self._model.predict(X), 0.0, None)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        pred_rate = self.predict_rate(frame)
        h0 = frame.h0_mm.to_numpy(float)
        horizon = frame.horizon_years.to_numpy(float)
        return np.clip(h0 - pred_rate * horizon, 0.0, h0)

    def metadata(self) -> Dict[str, Any]:
        return dict(name=self.name, model=self.name, weighting=self.weighting,
                    iterations=self.iterations, learning_rate=self.learning_rate,
                    depth=self.depth, base_rate_mm_year=self.base_rate_mm_year,
                    features=self._feature_cols)


class EnsembleCorrosionModel:
    """M5: Weighted blend of a base model and a gradient-boosting model.

    Blending is done at the *rate* level so that the physical clip (0 ≤ pred ≤ h0)
    is applied once to the final combined rate, not separately to each member.
    """

    def __init__(self, base_model, gb_model, gb_weight: float = 0.5):
        self.base_model = base_model
        self.gb_model = gb_model
        self.gb_weight = float(gb_weight)
        self.name = 'M5_Ensemble'
        self.weighting = getattr(base_model, 'weighting', 'W1')

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        if hasattr(self.base_model, 'predict_rate'):
            base_rate = self.base_model.predict_rate(frame)
        else:
            base_rate = np.full(len(frame), getattr(self.base_model, 'rate_mm_year', 0.0))
        gb_rate = self.gb_model.predict_rate(frame)
        return np.clip((1.0 - self.gb_weight) * base_rate + self.gb_weight * gb_rate, 0.0, None)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        pred_rate = self.predict_rate(frame)
        h0 = frame.h0_mm.to_numpy(float)
        horizon = frame.horizon_years.to_numpy(float)
        return np.clip(h0 - pred_rate * horizon, 0.0, h0)

    def metadata(self) -> Dict[str, Any]:
        return dict(name=self.name, model=self.name, weighting=self.weighting,
                    gb_weight=self.gb_weight,
                    base_model=getattr(self.base_model, 'name', str(self.base_model)),
                    gb_model=getattr(self.gb_model, 'name', str(self.gb_model)))

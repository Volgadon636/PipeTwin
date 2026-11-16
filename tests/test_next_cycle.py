"""Targeted tests for the next S1/S2 cycle."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.heat_transfer import (HeatTransferParameters, segment_outlet_temperature,
                               validate_parameters)
from src.ml.data import RepositoryData
from src.ml.next_common import (NextPreprocessor, evaluate_predictions, fit_ridge,
                                load_artifact, make_weights, predict_ridge,
                                s1_fit_rows, s1_production_table, s1_training_table,
                                weight_diagnostics)
from src.ml.s2_next import (_predict_with_artifact, prepare_s2_target)
from src.ml.s1_contract import S1_ID, S2_ID
from src.ml.submission import validate_submission

ROOT = BASE_DIR / 'experiments/next_cycle'
DATA_ROOT = ROOT / 'data_generation'
S1_ARTIFACTS = ROOT / 's1/candidates'
S2_ARTIFACTS = ROOT / 's2/candidates'


class PreprocessingAndWeights(unittest.TestCase):
    def test_duplicate_constant_empty_and_order(self):
        features = ['last_mm', 'dup', 'constant', 'empty', 'geometry']
        frame = pd.DataFrame({
            'last_mm': np.linspace(4, 7, 20),
            'dup': np.linspace(4, 7, 20),
            'constant': 1.0,
            'empty': np.nan,
            'geometry': np.sin(np.linspace(0, 3, 20)),
        })
        pre = NextPreprocessor(features).fit(frame)
        self.assertIn('dup', pre.duplicate_columns_)
        self.assertIn('constant', pre.constant_columns_)
        self.assertIn('empty', pre.empty_columns_)
        self.assertEqual(pre.feature_order_, features)
        self.assertEqual(pre.used_features_, ['last_mm', 'geometry'])
        transformed = pre.transform(frame.iloc[[0, 10]])
        self.assertEqual(transformed.shape, (2, 2))
        self.assertTrue(np.isfinite(transformed).all())
        with self.assertRaises(ValueError):
            pre.transform(frame.drop(columns=['geometry']))

    def test_weights_are_train_only_and_group_totals(self):
        if not DATA_ROOT.is_dir():
            self.skipTest('next-cycle data generation is absent')
        data = RepositoryData(DATA_ROOT)
        fit = s1_fit_rows(s1_training_table(data))
        for scheme in ('row_uniform', 'inverse_sqrt_section_size', 'inverse_section_size'):
            weights = make_weights(fit, scheme)
            diag = weight_diagnostics(fit, weights)
            self.assertAlmostEqual(weights.mean(), 1.0)
            self.assertEqual(diag['rows'], len(fit))
            if scheme == 'inverse_section_size':
                totals = list(diag['group_totals'].values())
                np.testing.assert_allclose(totals, totals[0])
        self.assertGreater(weight_diagnostics(
            fit, make_weights(fit, 'inverse_section_size'))['kish_ess'], 0)


class S1Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_ROOT.is_dir():
            raise unittest.SkipTest('next-cycle data generation is absent')
        cls.data = RepositoryData(DATA_ROOT)
        cls.fit = s1_fit_rows(s1_training_table(cls.data))
        cls.prod = s1_production_table(cls.data)

    def test_future_labels_do_not_change_fold_model(self):
        held = int(sorted(self.fit.id.unique())[0])
        features = ['last_mm', 'neighbor_prior_median_mm',
                    'neighbor_prior_count', 'neighbor_missing']
        train = self.fit[~self.fit.id.eq(held)].copy()
        weights = make_weights(train, 'row_uniform')
        a = fit_ridge(train, features, 'target_delta_mm', weights, alpha=10.0,
                      baseline='last_mm', model_name='test')
        poisoned_full = self.fit.copy()
        poisoned_full.loc[poisoned_full.id.eq(held), 'target_delta_mm'] += 1e6
        poisoned_train = poisoned_full[~poisoned_full.id.eq(held)].copy()
        b = fit_ridge(poisoned_train, features, 'target_delta_mm', weights,
                      alpha=10.0, baseline='last_mm', model_name='test')
        self.assertEqual(a['metadata']['coefficients'], b['metadata']['coefficients'])
        self.assertEqual(a['metadata']['intercept'], b['metadata']['intercept'])

    def test_last_based_core_ignores_hidden_min_change(self):
        path = S1_ARTIFACTS / 'ridge_history/model_artifact.joblib'
        if not path.is_file():
            self.skipTest('S1 artifact absent')
        artifact = load_artifact(path)
        changed = self.prod.copy()
        changed['prior_vtd_min_thickness_mm'] = 1.0
        changed['s1b_prediction'] = 1.0
        a = predict_ridge(artifact, self.prod, baseline='last_mm')
        b = predict_ridge(artifact, changed, baseline='last_mm')
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_rate_forecast_uses_actual_dt_and_differs_from_delta(self):
        rate_path = S1_ARTIFACTS / 'rate_signed/model_artifact.joblib'
        geom_path = S1_ARTIFACTS / 'ridge_geometry/model_artifact.joblib'
        if not (rate_path.is_file() and geom_path.is_file()):
            self.skipTest('S1 artifacts absent')
        rate = load_artifact(rate_path)
        geom = load_artifact(geom_path)
        own = self.prod[self.prod.has_own_prior_vtd.astype(bool)].head(20).copy()
        short = own.copy()
        short['dt_years'] = 1.0
        long = own.copy()
        long['dt_years'] = 5.0
        rate_short = np.clip(
            short.last_mm.to_numpy(float)
            - rate['estimator'].predict(rate['preprocessor'].transform(short))
            * short.dt_years.to_numpy(float), 0, short.nominal_thickness_mm)
        rate_long = np.clip(
            long.last_mm.to_numpy(float)
            - rate['estimator'].predict(rate['preprocessor'].transform(long))
            * long.dt_years.to_numpy(float), 0, long.nominal_thickness_mm)
        self.assertFalse(np.allclose(rate_short, rate_long))
        delta_short = predict_ridge(geom, short, baseline='last_mm')
        delta_long = predict_ridge(geom, long, baseline='last_mm')
        np.testing.assert_allclose(delta_short, delta_long, rtol=0, atol=0)
        self.assertTrue((short.dt_years > 0).all())

    def test_block_diagnostic_uses_buffer(self):
        metrics_path = ROOT / 's1/block_cv/metrics.json'
        if not metrics_path.is_file():
            self.skipTest('block CV absent')
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        self.assertEqual(payload['block_size_segments'], 10)
        self.assertEqual(payload['buffer_segments'], 2)
        self.assertTrue(payload['records'])
        self.assertTrue(all(r['buffer_segments'] == 2 for r in payload['records']))

    def test_metrics_constant_is_undefined_and_r2_separate(self):
        frame = pd.DataFrame({
            'id': [1] * 10, 'segment_id': range(10),
            'next_mm': np.linspace(5, 7, 10),
            'constant': 6.0,
        })
        record = evaluate_predictions(frame, 'constant')
        self.assertIsNone(record['within_section_centered']['pearson_r'])
        self.assertIsNotNone(record['pooled']['r2'])
        self.assertIsNone(record['pooled']['pearson_r'])


class S2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_ROOT.is_dir():
            raise unittest.SkipTest('next-cycle data generation is absent')
        cls.data = RepositoryData(DATA_ROOT)
        cls.target, cls.interval_map, cls.meta = prepare_s2_target(cls.data)

    def test_target_has_no_vtd_history(self):
        self.assertEqual(len(self.target), 1475)
        self.assertTrue(self.target.last_vtd_thickness_mm.isna().all())
        self.assertTrue(self.target.prior_vtd_min_thickness_mm.isna().all())
        self.assertEqual(self.meta['target_id'], S2_ID)

    def test_strict_eligibility_is_as_of(self):
        path = ROOT / 's2/eligibility.json'
        if not path.is_file():
            self.skipTest('S2 output absent')
        meta = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(meta['strict_dates'], ['2018-09-23'])
        self.assertNotIn(S2_ID, meta['strict_ids'])
        self.assertTrue(all(d >= '2023-08-12' for d in meta['post_dates']))

    def test_temperature_scenario_sign_and_invalid_flow(self):
        params = HeatTransferParameters()
        valid = segment_outlet_temperature(20.0, 10.0, 0.2, 5.0, params)
        self.assertTrue(valid['valid'])
        self.assertLess(valid['t_out_c'], valid['t_in_c'])
        self.assertLess(valid['delta_temperature_c'], 0)
        bad = segment_outlet_temperature(20.0, 10.0, 0.2, 0.0, params)
        self.assertFalse(bad['valid'])
        self.assertIn('nonpositive_or_missing_mass_flow', bad['issues'])
        issues = validate_parameters(params, 0.0, 5.0)
        self.assertIn('nonpositive_or_nonfinite_inner_diameter', issues)

    def test_artifact_predict_without_labels(self):
        path = S2_ARTIFACTS / 'physics_priors/model_artifact.joblib'
        if not path.is_file():
            self.skipTest('S2 artifact absent')
        wrapper = load_artifact(path)
        pred = _predict_with_artifact(wrapper['artifact'], self.target, self.interval_map)
        self.assertEqual(len(pred), 1475)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= 0).all() and (pred <= 4.5 + 1e-9).all())

        class NoLabelsData:
            config = self.data.config
            def __init__(self, base):
                self.base = base
            @property
            def cleaned(self):
                return self.base.cleaned
            def read(self, name):
                if 'training_targets' in name:
                    raise AssertionError('predict read labels')
                return self.base.read(name)
            def parent_groups(self):
                return self.base.parent_groups()

        target, intervals, meta = prepare_s2_target(NoLabelsData(self.data))
        self.assertEqual(len(target), 1475)
        self.assertEqual(meta['target_id'], S2_ID)

    def test_assembly_fixed_halves(self):
        s1 = ROOT / 's1/candidates/s1b/predictions_s1.csv'
        s2 = ROOT / 's2/candidates/physics_priors/result.csv'
        if not (s1.is_file() and s2.is_file()):
            self.skipTest('candidate outputs absent')
        s1_frame = pd.read_csv(s1)
        result = pd.read_csv(s2)
        self.assertEqual(len(result), 2000)
        self.assertFalse(result.duplicated(['id', 'segment_id']).any())
        fixed = result[result.id.eq(1250022871)][['id', 'segment_id', 'thickness_pred']]
        fixed = fixed.sort_values(['id', 'segment_id']).reset_index(drop=True)
        expected = s1_frame.sort_values(['id', 'segment_id']).reset_index(drop=True)
        np.testing.assert_allclose(fixed.thickness_pred, expected.thickness_pred,
                                   rtol=0, atol=1e-12)


class ArtifactSerialization(unittest.TestCase):
    def test_s1_predict_without_training_labels(self):
        path = S1_ARTIFACTS / 'ridge_geometry/model_artifact.joblib'
        if not path.is_file() or not DATA_ROOT.is_dir():
            self.skipTest('S1 artifact or data absent')
        base = RepositoryData(DATA_ROOT)

        class NoLabelsData:
            config = base.config
            def __init__(self, data):
                self.data = data
            @property
            def cleaned(self):
                return self.data.cleaned
            def read(self, name):
                if 'training_targets' in name:
                    raise AssertionError('S1 predict read labels')
                return self.data.read(name)
            def parent_groups(self):
                return self.data.parent_groups()

        prod = s1_production_table(NoLabelsData(base))
        artifact = load_artifact(path)
        prediction = predict_ridge(artifact, prod, baseline='last_mm')
        self.assertEqual(len(prediction), 525)
        self.assertTrue(np.isfinite(prediction).all())

    def test_s1_save_load_predict_same(self):
        path = S1_ARTIFACTS / 'ridge_geometry/model_artifact.joblib'
        if not path.is_file():
            self.skipTest('S1 artifact absent')
        data = RepositoryData(DATA_ROOT) if DATA_ROOT.is_dir() else None
        if data is None:
            self.skipTest('data absent')
        prod = s1_production_table(data)
        a = load_artifact(path)
        p1 = predict_ridge(a, prod, baseline='last_mm')
        p2 = predict_ridge(load_artifact(path), prod, baseline='last_mm')
        np.testing.assert_allclose(p1, p2, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

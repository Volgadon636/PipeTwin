"""Targeted contracts for S1A/S1B/S1C: as-of rules, leakage, Ridge and bounds."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.ml.data import RepositoryData
from src.ml.s1_contract import (RIDGE_FEATURES, S1_CASE, S1_CUTOFF, S1_ID,
                                S2_ID, TARGET_COLUMN, build_s1_training_frame,
                                combine_with_s2, enrich_s1, load_s2_source,
                                load_v2_s1_source, nominal_frame,
                                production_frame, target_key_grid)
from src.ml.s1_model import (S1Preprocessor, fit_s1_ridge, load_artifact,
                             predict_s1_ridge, save_artifact)
from src.ml.s1_stages import _lopo_fit
from src.ml.submission import validate_submission, export_result
from src.ml.windows import add_baselines, make_snapshot


class RealDataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = RepositoryData()
        cls.grid = target_key_grid(cls.data)
        cls.nominal = nominal_frame(cls.data)
        cls.prod, cls.s1_keys = production_frame(cls.data)
        cls.training = build_s1_training_frame(cls.data)
        cls.vtd = cls.data.read('processed/vtd_by_segment')
        cls.profile = cls.data.read('processed/profile_by_segment')

    def test_target_key_counts_and_no_target_grid_checksum_workaround(self):
        counts = self.grid.groupby('id').size().to_dict()
        self.assertEqual(counts[S1_ID], 525)
        self.assertEqual(counts[S2_ID], 1475)
        self.assertEqual(len(self.prod), 525)

    def test_s1a_last_and_s1b_prior_minimum(self):
        own = self.prod.has_own_prior_vtd
        np.testing.assert_allclose(
            self.prod.loc[own, 's1a_prediction'],
            self.prod.loc[own, 'last_vtd_thickness_mm'])
        np.testing.assert_allclose(
            self.prod.loc[own, 's1b_prediction'],
            self.prod.loc[own, 'prior_vtd_min_thickness_mm'])
        self.assertTrue((self.prod.loc[own, 's1b_prediction']
                         <= self.prod.loc[own, 's1a_prediction'] + 1e-9).all())
        changed = (self.prod.loc[own, 's1b_prediction']
                   - self.prod.loc[own, 's1a_prediction']).abs().gt(1e-12)
        self.assertEqual(int(changed.sum()), 157)
        older = own & self.prod.prior_vtd_min_date.lt(self.prod.last_vtd_date)
        self.assertTrue(older.any())

    def test_s1a_matches_previous_v2_on_same_keys(self):
        v2 = load_v2_s1_source(BASE_DIR / 'experiments/ml_v2/result.csv', self.grid)
        merged = self.prod[['id', 'segment_id', 's1a_prediction']].merge(
            v2, on=['id', 'segment_id'], validate='one_to_one')
        difference = np.abs(merged.s1a_prediction.to_numpy(float)
                            - merged.thickness_pred.to_numpy(float))
        self.assertLess(float(difference.max()), 1e-9)
        self.assertEqual(int((difference > 1e-9).sum()), 0)

    def test_fallback_is_documented_and_stays_in_pipe(self):
        missing = self.prod[~self.prod.has_own_prior_vtd]
        self.assertEqual(sorted(missing.segment_id.tolist()), [17, 247, 401, 455])
        self.assertTrue(missing.baseline_source.eq('nearest_prior_vtd_assumption').all())
        self.assertTrue(missing.fallback_segment_id.notna().all())
        self.assertTrue(missing.fallback_distance_m.le(20).all())
        self.assertTrue(missing.fallback_measurement_date.lt(S1_CUTOFF).all())
        np.testing.assert_allclose(missing.s1a_prediction, missing.s1b_prediction)

    def test_2022_snapshot_excludes_the_2022_answer(self):
        request = {'id': S1_ID, 'prediction_date': '2022-07-06', 'case': S1_CASE}
        cutoff = pd.Timestamp('2022-07-06')
        features, _ = make_snapshot(self.data, request)
        enriched = enrich_s1(features, self.vtd, self.profile, cutoff)
        own = enriched.has_own_prior_vtd
        self.assertTrue(own.any())
        self.assertTrue(enriched.loc[own, 'last_vtd_date'].lt(cutoff).all())
        self.assertTrue(enriched.loc[own, 'prior_vtd_min_date'].lt(cutoff).all())
        np.testing.assert_allclose(
            enriched.loc[own, 's1a_prediction'],
            enriched.loc[own, 's1b_prediction'])
        self.assertEqual(int(own.sum()), 334)

    def test_s1b_ignores_poisoned_rows_at_or_after_cutoff(self):
        request = {'id': S1_ID, 'prediction_date': '2022-07-06', 'case': S1_CASE}
        cutoff = pd.Timestamp('2022-07-06')
        base_features, _ = make_snapshot(self.data, request)
        base = enrich_s1(base_features, self.vtd, self.profile, cutoff)

        class PoisonedData:
            config = self.data.config
            def __init__(self, base):
                self.base = base
            @property
            def cleaned(self):
                return self.base.cleaned
            def read(self, name):
                frame = self.base.read(name)
                if name == 'processed/vtd_by_segment':
                    bad = frame.date.ge(cutoff)
                    frame.loc[bad, 'min_residual_thickness_mm'] = 0.01
                return frame
            def parent_groups(self):
                return self.base.parent_groups()

        poisoned = PoisonedData(self.data)
        changed_features, _ = make_snapshot(poisoned, request)
        changed = enrich_s1(changed_features, poisoned.read('processed/vtd_by_segment'),
                            self.profile, cutoff)
        pd.testing.assert_frame_equal(base, changed)

    def test_production_prediction_ignores_extra_rows_on_cutoff(self):
        class PoisonedData:
            config = self.data.config
            def __init__(self, base):
                self.base = base
            @property
            def cleaned(self):
                return self.base.cleaned
            def read(self, name):
                frame = self.base.read(name)
                if name == 'processed/vtd_by_segment':
                    extra = frame.iloc[[0]].copy()
                    extra['id'] = S1_ID
                    extra['segment_id'] = 0
                    extra['date'] = S1_CUTOFF
                    extra['min_residual_thickness_mm'] = 0.01
                    extra['target_eligible'] = True
                    frame = pd.concat([frame, extra], ignore_index=True)
                return frame
            def parent_groups(self):
                return self.base.parent_groups()

        poisoned = PoisonedData(self.data)
        changed, _ = production_frame(poisoned)
        pd.testing.assert_frame_equal(self.prod, changed)
        artifact_path = BASE_DIR / 'experiments/s1abc/s1c/model_artifact.joblib'
        if artifact_path.is_file():
            artifact = load_artifact(artifact_path)
            np.testing.assert_allclose(
                predict_s1_ridge(artifact, self.prod).thickness_pred,
                predict_s1_ridge(artifact, changed).thickness_pred,
                rtol=0, atol=1e-12)

    def test_minimum_tie_uses_earliest_source_date(self):
        # Add an earlier synthetic campaign with exactly the same minimum for
        # segment 0, then exercise the real build_features_asof tie logic.
        from src.segment_mapper import build_features_asof
        prior_rows = self.vtd[
            self.vtd.id.eq(S1_ID) & self.vtd.date.lt(S1_CUTOFF)
            & self.vtd.segment_id.eq(0) & self.vtd.target_eligible.fillna(False)]
        actual_min = float(prior_rows.min_residual_thickness_mm.min())
        actual_last = float(prior_rows.sort_values('date').min_residual_thickness_mm.iloc[-1])
        extra = pd.DataFrame({
            'id': [S1_ID], 'date': [pd.Timestamp('2017-01-01')],
            'segment_id': [0], 'observation_count': [1],
            'thickness_measurement_count': [1], 'min_residual_thickness_mm': [actual_min],
            'before_commissioning_flag': [False], 'commissioning_unknown_flag': [False],
            'wall_nominal_mismatch_flag': [False], 'above_nominal_flag': [False],
            'above_local_wall_flag': [False], 'nominal_unresolved_flag': [False],
            'heldout_or_future_flag': [False], 'target_eligible': [True],
        })
        history = pd.concat([self.vtd[self.vtd.id.eq(S1_ID)], extra], ignore_index=True)
        request = {'id': S1_ID, 'prediction_date': str(S1_CUTOFF.date()),
                   'case': S1_CASE}
        snapshot = build_features_asof(
            self.data.cleaned,
            self.data.read('processed/segments_grid'),
            self.profile,
            history,
            self.data.read('processed/uzt_by_segment').iloc[:0],
            [request], self.data.config)
        row = snapshot['target_features'].query('segment_id == 0').iloc[0]
        self.assertAlmostEqual(float(row.prior_vtd_min_thickness_mm), actual_min)
        self.assertEqual(pd.Timestamp(row.prior_vtd_min_date), pd.Timestamp('2017-01-01'))
        self.assertAlmostEqual(float(row.last_vtd_thickness_mm), actual_last)
        self.assertEqual(pd.Timestamp(row.last_vtd_date), pd.Timestamp('2022-07-06'))

    def test_history_has_no_two_prior_campaign_test(self):
        self.assertEqual(int(self.training.prior_vtd_campaign_count.max()), 1)
        self.assertEqual(self.training.prediction_date.nunique(), 1)
        self.assertEqual(str(self.training.prediction_date.max().date()), '2022-07-06')
        self.assertEqual(int(self.training.fit_eligible.sum()), 357)

    def test_lopo_models_do_not_change_when_heldout_labels_are_poisoned(self):
        base, base_folds = _lopo_fit(self.training)
        poisoned = self.training.copy()
        held = int(sorted(poisoned.id.unique())[0])
        mask = poisoned.id.eq(held)
        poisoned.loc[mask, TARGET_COLUMN] = 1e6
        poisoned.loc[mask, 'target_delta_mm'] = 1e6
        poisoned.loc[mask, 'target_delta_from_s1a_mm'] = 1e6
        changed, changed_folds = _lopo_fit(poisoned)
        base_fold = next(r for r in base_folds
                         if r['heldout_simple_section_id'] == held)
        changed_fold = next(r for r in changed_folds
                            if r['heldout_simple_section_id'] == held)
        self.assertEqual(base_fold['coefficients'], changed_fold['coefficients'])
        self.assertEqual(base_fold['intercept'], changed_fold['intercept'])
        # Held-out predictions are unchanged: only the metrics would change
        # because the labels changed, not the fold model.
        np.testing.assert_allclose(
            base.loc[mask, 'prediction_s1c'], changed.loc[mask, 'prediction_s1c'],
            rtol=0, atol=1e-12)


class SyntheticPreprocessingAndModel(unittest.TestCase):
    @staticmethod
    def synthetic_training(rows=90, seed=7):
        rng = np.random.default_rng(seed)
        ids = np.repeat([11, 22, 33], rows // 3)
        n = len(ids)
        signal = rng.normal(0, 0.4, n)
        frame = pd.DataFrame({
            'id': ids,
            'segment_id': np.tile(np.arange(rows // 3), 3),
            'prediction_date': pd.Timestamp('2022-07-06'),
            'main_pipeline_id': 'P',
            'nominal_thickness_mm': 8.0,
            'fit_eligible': True,
            'has_own_prior_vtd': True,
            's1a_prediction': 6.0 + signal,
            's1b_prediction': 6.0 + signal,
            'last_vtd_thickness_mm': 6.0 + signal,
            'age_last_vtd_years': 3.0 + rng.normal(0, .1, n),
            'age_min_vtd_years': 3.0 + rng.normal(0, .1, n),
            'prior_vtd_campaign_count': rng.integers(1, 4, n).astype(float),
            'last_minus_min_mm': rng.normal(0, .2, n),
            'neighbor_prior_median_mm': 6.0 + signal + rng.normal(0, .05, n),
            'neighbor_prior_count': rng.integers(1, 5, n).astype(float),
            'neighbor_missing': False,
            'slope_m_per_m': rng.normal(0, .1, n),
            'abs_slope_m_per_m': rng.uniform(0, .2, n),
            'relative_position': rng.uniform(0, 1, n),
            'local_low_m': rng.normal(0, .3, n),
        })
        frame['target_delta_mm'] = 0.45 * signal + rng.normal(0, .05, n)
        frame['target_delta_from_s1a_mm'] = frame.target_delta_mm
        return frame

    def test_train_only_imputation_constant_and_empty_columns(self):
        train = self.synthetic_training(60)
        train.loc[train.index[:3], 'neighbor_prior_median_mm'] = np.nan
        train['prior_vtd_campaign_count'] = 1.0
        train['local_low_m'] = np.nan
        pre = S1Preprocessor(RIDGE_FEATURES).fit(train)
        self.assertIn('prior_vtd_campaign_count', pre.constant_columns_)
        self.assertIn('local_low_m', pre.empty_columns_)
        self.assertNotIn('prior_vtd_campaign_count', pre.used_features_)
        self.assertNotIn('local_low_m', pre.used_features_)
        median = float(train.neighbor_prior_median_mm.median())
        self.assertAlmostEqual(pre.imputer_statistics_['neighbor_prior_median_mm'], median)
        test = train.iloc[[0]].copy()
        test['neighbor_prior_median_mm'] = 1e6
        transformed = pre.transform(test)
        self.assertTrue(np.isfinite(transformed).all())
        # The imputer statistic is still the train median, not the test value.
        self.assertAlmostEqual(pre.imputer_statistics_['neighbor_prior_median_mm'], median)

    def test_ridge_learns_synthetic_correction_and_is_not_baseline(self):
        train = self.synthetic_training(120)
        artifact = fit_s1_ridge(train, alpha=10.0)
        prediction = predict_s1_ridge(artifact, train)
        self.assertTrue(prediction.ridge_applied.all())
        self.assertFalse(np.allclose(prediction.thickness_pred,
                                     prediction.s1b_prediction))
        baseline_mae = float(np.mean(np.abs(
            train.s1b_prediction - (train.s1b_prediction + train.target_delta_mm))))
        model_mae = float(np.mean(np.abs(
            prediction.thickness_pred - (train.s1b_prediction + train.target_delta_mm))))
        self.assertLess(model_mae, baseline_mae)
        self.assertGreater(abs(float(np.corrcoef(
            prediction.ridge_delta_mm, train.target_delta_mm)[0, 1])), 0.5)

    def test_save_load_predict_is_equivalent(self):
        train = self.synthetic_training(90)
        artifact = fit_s1_ridge(train, alpha=10.0)
        before = predict_s1_ridge(artifact, train)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'artifact.joblib'
            save_artifact(artifact, path)
            loaded = load_artifact(path)
            after = predict_s1_ridge(loaded, train)
        np.testing.assert_allclose(before.thickness_pred, after.thickness_pred,
                                   rtol=0, atol=1e-12)


class BoundsAndSubmission(unittest.TestCase):
    @staticmethod
    def bounds():
        return pd.DataFrame({
            'id': [S1_ID, S1_ID], 'segment_id': [0, 1],
            'h0_mm': [4.0, 4.0], 'nominal_thickness_mm': [8.0, 8.0],
            'upper_mm': [8.0, 8.0],
        })

    def test_upper_bound_policy_is_model_specific(self):
        grid = pd.DataFrame({'id': [S1_ID, S1_ID], 'segment_id': [0, 1]})
        result = grid.assign(thickness_pred=[6.0, 3.0])
        bounds = self.bounds()
        with self.assertRaises(ValueError):
            validate_submission(result, grid, bounds, upper_column='h0_mm')
        validation = validate_submission(result, grid, bounds, upper_column='upper_mm')
        self.assertTrue(validation['bounded'])
        self.assertEqual(validation['upper_bound_column'], 'upper_mm')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.csv'
            export_result(result, path, grid, bounds, upper_column='upper_mm')
            written = pd.read_csv(path)
        np.testing.assert_allclose(written.thickness_pred, result.thickness_pred)

    def test_s2_join_is_exact_and_keeps_s2(self):
        data = RepositoryData()
        grid = target_key_grid(data)
        nominal = nominal_frame(data)
        v2 = load_v2_s1_source(BASE_DIR / 'experiments/ml_v2/result.csv', grid)
        s2 = load_s2_source(BASE_DIR / 'experiments/ml_v2/result.csv', grid)
        s1 = v2[['id', 'segment_id', 'thickness_pred']].copy()
        bounds = nominal.copy()
        bounds['h0_mm'] = bounds.nominal_thickness_mm
        bounds['upper_mm'] = bounds.nominal_thickness_mm
        with tempfile.TemporaryDirectory() as tmp:
            combined, validation = combine_with_s2(s1, s2, grid, bounds)
            path = Path(tmp) / 'result.csv'
            combined.to_csv(path, index=False, float_format='%.10f')
            written = pd.read_csv(path)
        self.assertEqual(validation['rows'], 2000)
        verify = written[written.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']]
        verify = verify.sort_values(['id', 'segment_id']).reset_index(drop=True)
        expected = s2.sort_values(['id', 'segment_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(verify, expected, check_exact=True)


class ArtifactInferenceWithoutLabels(unittest.TestCase):
    def test_predict_loads_artifact_without_training_labels(self):
        artifact_path = BASE_DIR / 'experiments/s1abc/s1c/model_artifact.joblib'
        if not artifact_path.is_file():
            self.skipTest('Run S1C train first to create the artifact')
        data = RepositoryData()

        class NoLabelsData:
            config = data.config
            @property
            def cleaned(self):
                return data.cleaned
            def read(self, name):
                if 'training_targets' in name:
                    raise AssertionError('Predict must not read training_targets')
                return data.read(name)
            def parent_groups(self):
                return data.parent_groups()

        wrapper = NoLabelsData()
        prod, expected = production_frame(wrapper)
        artifact = load_artifact(artifact_path)
        with mock.patch.object(S1Preprocessor, 'fit',
                               side_effect=AssertionError('predict called fit')):
            prediction = predict_s1_ridge(artifact, prod)
        self.assertEqual(len(prediction), 525)
        self.assertTrue(np.isfinite(prediction.thickness_pred).all())
        self.assertEqual(int(prediction.ridge_applied.sum()), 521)
        self.assertEqual(int((~prediction.ridge_applied).sum()), 4)


if __name__ == '__main__':
    unittest.main()

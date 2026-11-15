"""Contracts for the v2 physics, spatial and rate-selection components."""
import unittest

import numpy as np
import pandas as pd

from src.ml.physics import (ColdStartPhysics, EnvironmentPriors, effective_exposure,
                            environment_factor, fit_cold_start_physics, sample_weights)
from src.ml.spatial import (FEATURES, fit_spatial_ridge, gate_spatial, leave_one_campaign_out,
                            predict_spatial, spatial_features, standardize_within_pipe)
from src.ml.experiment_v2 import (fit_pair_rate, inspection_pairs, json_safe, s1_rate_decision)


def environment_frame(n=3, temperature=20.0, watercut=50.0, velocity=0.5):
    return pd.DataFrame(dict(id=[1] * n, segment_id=range(n),
                             t_mix_mean=[temperature] * n,
                             watercut_pct_mean=[watercut] * n,
                             v_mix_mean=[velocity] * n))


def interval(n=3, duration=5.0, temperature=293.15, water=0.5, velocity=0.5):
    return pd.DataFrame(dict(duration_years=[duration] * n,
                             temperature_K=[temperature] * n,
                             water_fraction=[water] * n,
                             velocity_m_s=[velocity] * n))


def synthetic_windows(rate=0.2, age=10.0, rows=6):
    nominal = np.full(rows, 8.0)
    frame = pd.DataFrame(dict(
        id=np.arange(rows) % 3 + 1, segment_id=np.zeros(rows, int),
        main_pipeline_id=['A', 'B', 'C'] * (rows // 3), campaign_id=['A:1', 'B:1', 'C:1'] * (rows // 3),
        nominal_thickness_mm=nominal, age_years=np.full(rows, age),
        t_mix_mean=np.full(rows, 20.0), watercut_pct_mean=np.full(rows, 50.0),
        v_mix_mean=np.full(rows, 0.5)))
    frame['min_residual_thickness_mm'] = nominal - rate * age
    return frame


class PhysicsTests(unittest.TestCase):
    def test_environment_factors_have_physical_sign(self):
        reference = dict(temperature_K=293.15, water_fraction=0.5, velocity_m_s=0.5)
        priors = EnvironmentPriors()
        cold = environment_factor(environment_frame(temperature=10.0), reference, priors)
        warm = environment_factor(environment_frame(temperature=30.0), reference, priors)
        dry = environment_factor(environment_frame(watercut=0.1), reference, priors)
        wet = environment_factor(environment_frame(watercut=90.0), reference, priors)
        slow = environment_factor(environment_frame(velocity=0.05), reference, priors)
        fast = environment_factor(environment_frame(velocity=1.0), reference, priors)
        self.assertTrue(np.all(cold < 1.0) and np.all(warm > 1.0))
        self.assertTrue(np.all(dry < wet))
        self.assertTrue(np.all(slow < fast))

    def test_gap_policies_bracket_unknown_history(self):
        frame = environment_frame(3, temperature=20.0, watercut=50.0, velocity=0.5)
        frame['age_years'] = 10.0
        reference = dict(temperature_K=293.15, water_fraction=0.5, velocity_m_s=0.5)
        intervals = [interval(n=1, duration=5.0) for _ in range(3)]
        priors = EnvironmentPriors()
        own = effective_exposure(frame, intervals, reference, priors, 'own_mean')
        ref = effective_exposure(frame, intervals, reference, priors, 'reference')
        observed = effective_exposure(frame, intervals, reference, priors, 'observed_only')
        np.testing.assert_allclose(own, 10.0)
        np.testing.assert_allclose(ref, 10.0)
        np.testing.assert_allclose(observed, 5.0)

    def test_rate_fit_recovers_synthetic_constant_rate(self):
        frame = synthetic_windows(rate=0.2, age=10.0)
        intervals = [interval(n=1, duration=10.0) for _ in range(len(frame))]
        model = fit_cold_start_physics(frame, intervals)
        self.assertAlmostEqual(model.r_ref_mm_year, 0.2, places=10)
        np.testing.assert_allclose(model.predict(frame, intervals),
                                   frame.min_residual_thickness_mm, atol=1e-10)

    def test_rate_is_nonnegative_when_labels_thicken(self):
        frame = synthetic_windows(rate=-0.5, age=10.0)
        intervals = [interval(n=1, duration=10.0) for _ in range(len(frame))]
        self.assertEqual(fit_cold_start_physics(frame, intervals).r_ref_mm_year, 0.0)

    def test_parent_balanced_weights(self):
        frame = synthetic_windows(rows=6)
        weights = sample_weights(frame, 'W1')
        totals = pd.Series(weights).groupby(frame.main_pipeline_id).sum()
        np.testing.assert_allclose(totals, totals.iloc[0])
        self.assertAlmostEqual(weights.mean(), 1.0)


class SpatialTests(unittest.TestCase):
    @staticmethod
    def profile(pipe, n=120, effect=0.8, seed=0):
        rng = np.random.default_rng(seed)
        z = np.cumsum(rng.normal(0, 0.4, n))
        slope = np.gradient(z)
        frame = pd.DataFrame(dict(id=pipe, segment_id=np.arange(n),
                                  start_m=np.arange(n) * 10.0, end_m=np.arange(n) * 10.0 + 10.0,
                                  z_center=z, slope_m_per_m=slope))
        frame['prediction_date'] = pd.Timestamp('2022-01-01')
        frame = standardize_within_pipe(spatial_features(frame))
        frame['min_residual_thickness_mm'] = 6.0 + effect * frame['slope'] + rng.normal(0, 0.05, n)
        return frame

    def test_spatial_features_are_finite_and_contract_correct(self):
        frame = self.profile(1)
        self.assertEqual(list(FEATURES), ['z', 'slope', 'abs_slope', 'position', 'local_low', 'pool_depth'])
        self.assertTrue(np.isfinite(frame[FEATURES].to_numpy(float)).all())

    def test_ridge_recovers_synthetic_spatial_signal_out_of_campaign(self):
        frame = pd.concat([self.profile(pipe, seed=pipe) for pipe in (1, 2, 3)], ignore_index=True)
        model = fit_spatial_ridge(frame, lam=1.0)
        slope_coefficient = model['coefficients'][FEATURES.index('slope')]
        self.assertGreater(slope_coefficient, 0.5)
        cv = leave_one_campaign_out(frame, lam=1.0)
        self.assertGreater(cv['pooled_pearson'], 0.5)

    def test_gate_blocks_negative_evidence(self):
        self.assertTrue(gate_spatial(dict(pooled_pearson=0.1))['usable'])
        self.assertFalse(gate_spatial(dict(pooled_pearson=-0.1))['usable'])
        self.assertFalse(gate_spatial(dict(pooled_pearson=None))['usable'])


class PairAndSerializationTests(unittest.TestCase):
    @staticmethod
    def pairs():
        rows = []
        for pid, d0, d1, ratio, n in [(1, '2018-01-01', '2022-01-01', 10.0, 3),
                                      (2, '2022-01-01', '2023-01-01', 1.0, 3)]:
            for segment in range(n):
                rows.append(dict(id=pid, segment_id=segment, date_prior=pd.Timestamp(d0),
                                 date_next=pd.Timestamp(d1), h_prior=6.0, h_next=6.0,
                                 dt_years=1.0, comparable=ratio <= 1.33))
        return pd.DataFrame(rows)

    def test_post_cutoff_pairs_are_never_used(self):
        decision = s1_rate_decision(self.pairs(), pd.Timestamp('2023-01-01'))
        self.assertEqual(decision['used_pipes'], [1])
        self.assertEqual(decision['post_cutoff_diagnostic']['pipes'], [2])
        self.assertEqual(decision['basis'], 'all_pre_cutoff_pairs_no_comparable_basis')

    def test_inspection_density_uses_observation_counts(self):
        vtd = pd.DataFrame(dict(
            id=[1, 1, 1, 1], date=pd.to_datetime(['2020-01-01', '2020-01-01', '2021-01-01', '2021-01-01']),
            segment_id=[0, 1, 0, 1], min_residual_thickness_mm=[6.0, 6.2, 5.8, 6.0],
            observation_count=[1, 3, 10, 30]))
        pair = inspection_pairs(vtd)
        self.assertAlmostEqual(float(pair.density_ratio.iloc[0]), 10.0)

    def test_pair_rate_is_nonnegative(self):
        rate = fit_pair_rate(self.pairs().assign(h_next=[7.0, 7.0, 7.0, 5.0, 5.0, 5.0]))
        self.assertGreaterEqual(rate, 0.0)

    def test_json_safe_removes_nan(self):
        payload = json_safe(dict(a=np.float64(np.nan), b=np.int64(2), c=[float('inf')]))
        self.assertIsNone(payload['a'])
        self.assertEqual(payload['b'], 2)
        self.assertIsNone(payload['c'][0])


if __name__ == '__main__':
    unittest.main()

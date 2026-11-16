import unittest

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage3 import guarded_smooth, impute_at, outer_train, window_features


class Stage3Contracts(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'id': 7, 'start_m': np.arange(5) * 10., 'end_m': np.arange(1, 6) * 10.,
            'nominal_thickness_mm': 8., 'joint_blend': [6., 4., 6., 5., 7.],
            'aligned_measurement': [True, True, True, False, True],
        })

    def test_neighbor_windows_exclude_own_value_and_missing_donors(self):
        values = [6., 4., 6., np.nan, 7.]
        out = window_features(self.frame, values, [10, 20])
        np.testing.assert_array_equal(out['prior_count_10m'], [1, 2, 1, 2, 0])
        self.assertEqual(out['prior_min_20m'][1], 6.)
        self.assertTrue(np.isnan(out['prior_min_10m'][4]))
        with self.assertRaises(ValueError):
            window_features(self.frame.assign(id=[7, 7, 8, 8, 8]), values, [10])

    def test_guarded_smoothing_keeps_local_troughs_and_fallback(self):
        values, protected = guarded_smooth(self.frame, 10, 0.5)
        self.assertEqual(protected.tolist(), [False, True, False, False, False])
        self.assertEqual(values[1], 4.)
        self.assertEqual(values[3], 5.)
        self.assertLess(values[0], 6.)

    def test_imputer_uses_only_remaining_samples_and_respects_nearest_tie(self):
        config = {'local_radius_m': 30, 'minimum_regression_neighbors': 3}
        x, values = np.arange(5) * 10., np.array([4., 5., np.nan, 7., 8.])
        fitted = impute_at(x, values, 2, config)
        self.assertEqual(fitted['nearest'], 5.)
        self.assertEqual(fitted['interpolation'], 6.)
        self.assertAlmostEqual(fitted['local_linear'], 6.)
        with self.assertRaises(ValueError):
            impute_at(x, values, 3, config)

    def test_outer_training_excludes_held_section_and_cutoff_day(self):
        cutoff = pd.Timestamp('2023-12-01')
        fit = pd.DataFrame({
            'id': [7, 8, 9, 10, 11],
            'prediction_date': pd.to_datetime(['2022-07-06'] * 3 + ['2023-12-01', '2024-01-01']),
            'last_vtd_date': pd.Timestamp('2018-09-23'),
        })
        self.assertEqual(outer_train(fit, 7, cutoff).id.tolist(), [8, 9])
        fit.loc[1, 'last_vtd_date'] = fit.loc[1, 'prediction_date']
        with self.assertRaises(ValueError):
            outer_train(fit, 7, cutoff)


if __name__ == '__main__':
    unittest.main()

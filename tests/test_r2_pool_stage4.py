import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage4 import TEMPERATURE, prior_point_summaries, s2_predictions
from src.ml.s1_contract import S2_ID


class Stage4Contracts(unittest.TestCase):
    def test_prior_point_summary_uses_one_section_and_earlier_campaign(self):
        grid = pd.DataFrame({'id': [7, 7], 'segment_id': [0, 1],
                             'start_m': [0., 10.], 'end_m': [10., 20.]})
        past, cutoff = pd.Timestamp('2018-09-23'), pd.Timestamp('2022-07-06')
        raw = pd.DataFrame({
            'id': [7, 7, 7, 7, 8], 'date': [past, past, past, cutoff, past],
            'target_eligible': [True, True, False, True, True],
            'distance': [1., 12., 1., 1., 1.], 'residual_for_target_mm': [4., 5., 0., 0., 0.],
            'defect_length': [12., 24., 100., 100., 100.], 'defect_width': [3.] * 5,
            'feature_description': ['коррозия'] * 5, 'defect_type': ['внутр.'] * 5,
        })
        support = {'neighbor_radius_m': 20, 'minimum_own_measurements': 1,
                   'minimum_measured_neighbors': 1}
        result = prior_point_summaries(raw, grid, cutoff, past, support)
        np.testing.assert_array_equal(result.prior_direct_mm, [4., 5.])
        np.testing.assert_array_equal(result.prior_measurement_count, [1, 1])
        self.assertTrue(result.high_support.all())
        with self.assertRaises(ValueError):
            prior_point_summaries(raw, grid, cutoff, cutoff, support)
        with self.assertRaises(ValueError):
            prior_point_summaries(raw, grid.assign(id=S2_ID), cutoff, past, support)

    def test_invalid_temperature_profile_uses_the_entire_frozen_profile(self):
        day = pd.Timestamp('2023-08-12')
        a = pd.DataFrame({'id': [7, 7], 'segment_id': [0, 1], 'prediction_date': day,
                          TEMPERATURE: [1., np.nan]})
        b = a.assign(id=8, **{TEMPERATURE: [1., -1.]})
        test = pd.concat([a, b]).assign(next_mm=5.)
        grids = {(7, day): a, (8, day): b}
        models = {'baseline': [4., 6.], 'centered_fluid_temperature': [6., 4.]}
        with patch('src.experiments.r2_pool_stage4._predict_candidate',
                   side_effect=lambda model, full, intervals: {'h_clipped': model}):
            predictions = s2_predictions(test, grids, {}, models, 'baseline')
        np.testing.assert_array_equal(predictions.centered_fluid_temperature, [4., 6., 6., 4.])


if __name__ == '__main__':
    unittest.main()

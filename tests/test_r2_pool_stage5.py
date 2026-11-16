import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage5 import (
    AVERAGE_CONDITIONS, INTEGRATED, predict_relative_loss, thermal_exposure,
)
from src.heat_transfer import HeatTransferParameters


class Stage5Contracts(unittest.TestCase):
    def test_thermal_integral_distinguishes_histories_with_equal_mean_inputs(self):
        cutoff = pd.Timestamp('2023-08-12')
        grid = pd.DataFrame({'prediction_date': cutoff, 'd_inner_assumed_mm': 150.,
                             'horizon_years': 1., 'start_m': [0., 500., 1000.],
                             'end_m': [500., 1000., 1500.], 'length_m': 500.})
        intervals = pd.DataFrame({
            'date': cutoff - pd.to_timedelta([2, 1], unit='day'),
            'interval_end': cutoff - pd.to_timedelta([1, 0], unit='day'),
            'duration_years': [1/365.] * 2,
            't_start_usable': [15., 15.], 'density_liquid_work_usable': [900., 900.],
            'velocity_m_s': [.05, .55],
        })
        params = HeatTransferParameters()
        values, audit = thermal_exposure(grid, intervals, params)
        averaged, _ = thermal_exposure(grid, intervals.assign(velocity_m_s=.30), params)
        self.assertEqual(audit['thermal_intervals'], 2)
        self.assertAlmostEqual(np.average(values[INTEGRATED], weights=grid.length_m), 0.)
        np.testing.assert_allclose(values[AVERAGE_CONDITIONS], averaged[INTEGRATED], atol=1e-14)
        self.assertGreater(np.max(np.abs(values[INTEGRATED] - averaged[INTEGRATED])), 1e-4)
        with self.assertRaises(ValueError):
            thermal_exposure(grid, intervals.assign(date=cutoff), params)

    def test_fractional_loss_keeps_full_grid_center_and_converts_to_mm(self):
        full = pd.DataFrame({'id': [7, 7], 'segment_id': [0, 1], 'length_m': [10., 5.],
                             'start_m': [0., 10.], 'end_m': [10., 15.],
                             'nominal_thickness_mm': [8., 6.]})
        with patch('src.experiments.r2_pool_stage5.predict_ridge',
                   return_value=np.array([.1, .4])):
            result = predict_relative_loss({}, full, np.array([5., 5.]))
        np.testing.assert_allclose(result['h_clipped'], [5.8, 3.8])


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage8_models import (
    ROBUST, SOFT, select_by_inner_r_squared, shape_bounds, shape_calibration,
)


class Stage8Contracts(unittest.TestCase):
    def test_shape_calibration_preserves_prediction_grid_moments(self):
        class NonlinearMap:
            def predict(self, values):
                return np.tanh(values)

        raw = np.array([3., 4., 4.25, 6.])
        lengths = np.array([10., 10., 10., 5.])
        revised = shape_calibration(raw, lengths, 8., NonlinearMap(), .5)
        center = np.average(raw, weights=lengths)
        self.assertAlmostEqual(np.average(revised, weights=lengths), center)
        self.assertAlmostEqual(np.average((revised - center) ** 2, weights=lengths),
                               np.average((raw - center) ** 2, weights=lengths))
        self.assertGreater(np.max(np.abs(revised - raw)), .01)

    def test_soft_and_robust_bounds_keep_segment_nominal(self):
        nominal = np.array([5., 6., 7., 8., 8.])
        raw = np.array([-8., 2.9, 3., 3.1, 18.])
        result = shape_bounds(raw, np.clip(raw, 0., nominal), nominal,
                              {'softness_mm': .15, 'iqr_factor': 1.5})
        for name in (SOFT, ROBUST):
            self.assertTrue(np.isfinite(result[name]).all())
            self.assertTrue(((0 <= result[name]) & (result[name] <= nominal)).all())
        self.assertLess(result[ROBUST][-1], nominal[-1])
        self.assertGreater(result[ROBUST][0], 0.)

    def test_selection_never_ignores_an_undefined_fold_to_create_a_winner(self):
        day = pd.Timestamp('2018-09-23')
        truth = [3., 4., 5., 6.] * 2
        oof = pd.DataFrame({'id': [7] * 4 + [8] * 4, 'prediction_date': day, 'next_mm': truth,
                            'base': [3.1, 5., 4., 6.1] * 2, 'bad': truth[:4] + [4.] * 4})
        selected, result = select_by_inner_r_squared(oof, ['bad', 'base'], 4, 'base')
        self.assertEqual(selected, 'base')
        self.assertIsNone(result['scores']['bad']['mean_r_squared'])


if __name__ == '__main__':
    unittest.main()

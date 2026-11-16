import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'scripts/calibrate_intervals.py'
spec = importlib.util.spec_from_file_location('calibration', path)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class IntervalTests(unittest.TestCase):
    def test_residual_rank_and_insufficient_sample(self):
        self.assertEqual(calibration.residual_radius([1., 2., 3., 4.]), 4.)
        for errors in [[], [1.], [np.nan], [-1.]]:
            with self.assertRaises(ValueError):
                calibration.residual_radius(errors)

    def test_bounds_keep_point_and_clip_at_nominal(self):
        lower, upper = calibration.bounded_interval([.1, 7.9], [8., 8.], .8)
        np.testing.assert_allclose(lower, [0., 7.1])
        np.testing.assert_allclose(upper, [.9, 8.])

    def test_outer_truth_cannot_change_its_own_interval_calibration(self):
        train = pd.DataFrame({'id': np.repeat([1, 2, 3], 5), 'segment_id': list(range(5)) * 3,
                              'prediction_date': [pd.Timestamp('2022-01-01')] * 15,
                              'next_mm': np.linspace(3., 6., 15)})

        def predict(fit, held):
            self.assertTrue(set(fit.id).isdisjoint(set(held.id)))
            return held.assign(prediction_mm=float(fit.next_mm.mean()), nominal_thickness_mm=8.)

        before, first = calibration.nested_intervals(train, predict, .2)
        changed = train.copy()
        changed.loc[changed.id.eq(1), 'next_mm'] += .5
        after, second = calibration.nested_intervals(changed, predict, .2)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(before.loc[before.id.eq(1), ['prediction_mm', 'lower_mm', 'upper_mm']],
                                      after.loc[after.id.eq(1), ['prediction_mm', 'lower_mm', 'upper_mm']])


if __name__ == '__main__':
    unittest.main()

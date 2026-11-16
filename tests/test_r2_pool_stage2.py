import json
import unittest

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage2 import PROTOCOL, history_mix, shrink_rates, train_rates


class Stage2ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(PROTOCOL.read_bytes())['hypotheses']
        self.cutoff = pd.Timestamp('2023-12-01')
        self.features = pd.DataFrame({
            'id': 7, 'segment_id': [0, 1, 2, 3], 'prediction_date': self.cutoff,
            'fold': '7:0', 'joint_min': 5., 'joint_blend': 5.5,
            'aligned_measurement': [True, False, True, True], 'nominal_thickness_mm': 6.,
        })
        for column in self.cfg['H10']['features'][1:]:
            self.features[column] = [0., 1., 2., 3.]

    def test_h09_uses_two_direct_prior_measurements(self):
        history = pd.DataFrame({
            'id': 7, 'segment_id': [0, 1, 2, 3],
            'date': pd.to_datetime(['2018-09-23', '2018-09-23', '2023-12-01', '2018-09-23']),
            'target_eligible': [True, True, True, False],
            'min_residual_thickness_mm': 3.,
        })
        result = history_mix(self.features, history, self.cfg['H09'])
        self.assertEqual(result.prediction.tolist(), [4.5, 5.5, 5.5, 5.5])
        self.assertEqual(result.two_direct_campaigns.tolist(), [True, False, False, False])
        with self.assertRaises(ValueError):
            history_mix(self.features.assign(prediction_date=pd.Timestamp('2022-07-06')),
                        history, self.cfg['H09'])

    def test_h10_excludes_held_section_and_on_or_after_cutoff(self):
        train = pd.DataFrame({
            'id': [8, 8, 8, 7, 8, 8], 'segment_id': range(6),
            'prediction_date': pd.to_datetime(['2022-07-06'] * 4 + ['2023-12-01', '2023-12-02']),
            'last_vtd_date': pd.Timestamp('2018-09-23'),
            'rate_mm_year': [0.1, 0.2, 0.3, 100., 100., 100.],
        })
        for column in self.cfg['H10']['features']:
            train[column] = np.arange(len(train), dtype=float)
        _, median, (low, high), rows = train_rates(train, self.cutoff, 7, self.cfg['H10'])
        self.assertEqual(rows.segment_id.tolist(), [0, 1, 2])
        np.testing.assert_allclose([median, low, high], [0.2, 0.12, 0.28])
        result, keys, _ = shrink_rates(self.features, train, self.cfg['H10'])
        self.assertEqual(keys.segment_id.tolist(), [0, 1, 2])
        self.assertEqual(result.loc[1, 'prediction'], result.loc[1, 'median_control'])
        self.assertTrue(result.prediction.between(0., 6.).all())


if __name__ == '__main__':
    unittest.main()

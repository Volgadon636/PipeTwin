import unittest

import pandas as pd

from src.experiments.r2_pool_stage1 import checked_comparison, scores, summarize


class PoolProtocolTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame({'id': 7, 'segment_id': [0, 1, 2],
                                  'prediction_date': pd.Timestamp('2022-01-01'),
                                  'next_mm': [1., 2., 3.], 'prediction': [1., 2., 3.]})

    def test_keys_and_labels_are_locked(self):
        reversed_rows = self.base.iloc[::-1]
        result = checked_comparison(self.base, reversed_rows)
        self.assertEqual(result.candidate.tolist(), [1., 2., 3.])
        with self.assertRaises(AssertionError):
            checked_comparison(self.base, self.base.assign(next_mm=0))
        with self.assertRaises(AssertionError):
            checked_comparison(self.base, self.base.iloc[:2])
        with self.assertRaises(ValueError):
            checked_comparison(self.base, pd.concat([self.base, self.base.iloc[:1]]))

    def test_sign_and_constant_profiles_keep_distinct_status(self):
        negative = self.base.assign(prediction=[3., 2., 1.])
        constant = self.base.assign(prediction=2.)
        neg = scores(checked_comparison(self.base, negative), 'S1')
        const = scores(checked_comparison(self.base, constant), 'S1')
        self.assertEqual(neg[0]['candidate_pearson_r'], -1.)
        self.assertEqual(neg[0]['delta_pearson_squared'], 0.)
        self.assertEqual(const[0]['candidate_status'], 'constant_prediction')
        self.assertIsNone(summarize(neg + const)['candidate_r2'])

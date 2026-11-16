import unittest

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage6 import demean_training_profile, donor_ranking, s2_descriptor


class Stage6Contracts(unittest.TestCase):
    def test_training_centering_groups_by_section_and_campaign(self):
        frame = pd.DataFrame({
            'id': [7, 7, 7, 7, 8, 8],
            'prediction_date': pd.to_datetime(['2022-01-01'] * 2 + ['2023-01-01'] * 4),
        })
        features = [[10., 1.], [14., 1.], [40., 7.], [46., 9.], [-1., 20.], [3., 22.]]
        np.testing.assert_array_equal(
            demean_training_profile(features, frame),
            [[-2., 0.], [2., 0.], [-3., -1.], [3., 1.], [-2., -1.], [2., 1.]])

    def test_donor_ranking_uses_descriptor_not_model_or_number_of_rows(self):
        date = pd.Timestamp('2018-09-23')
        donors = {(7, date): ({'water_fraction_mean': .1}, 20, object()),
                  (8, date): ({'water_fraction_mean': .9}, 200, object())}
        result = donor_ranking(donors, {'water_fraction_mean': .2}, {'water_fraction_mean': .25})
        self.assertEqual([entry[0][0] for entry in result], [7, 8])

    def test_donor_matching_rejects_cutoff_day_hydraulics(self):
        day = pd.Timestamp('2022-07-01')
        full = pd.DataFrame({'prediction_date': [day], 'hydraulic_source_date_max': [day],
                             'length_m': [10.], 'abs_slope': [.1]})
        with self.assertRaises(ValueError):
            s2_descriptor(full, ['mean_abs_slope'])
        prior = full.assign(hydraulic_source_date_max=day - pd.Timedelta(days=1))
        self.assertAlmostEqual(s2_descriptor(prior, ['mean_abs_slope'])['mean_abs_slope'], .1)


if __name__ == '__main__':
    unittest.main()

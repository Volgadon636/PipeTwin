import unittest

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from src.experiments.r2_pool_stage7 import choose_weight
from src.experiments.r2_pool_stage7_models import (
    fit_gaussian_process, joint_design, pearson_loss_and_gradient, predict_gaussian_process,
)
from sklearn.linear_model import Ridge


class Stage7Contracts(unittest.TestCase):
    def test_squared_pearson_derivative_matches_numerical_gradient(self):
        x = np.array([[1., -.5], [-.5, 2.], [-.5, -1.5]])
        profiles = [(x, np.array([1., 0., -1.]), np.array([.15, -.1, -.05])),
                    (x[::-1], np.array([1., -2., 1.]), np.array([.1, -.15, .05]))]
        point = np.array([.23, -.47])
        _, analytic = pearson_loss_and_gradient(point, profiles)
        numeric = approx_fprime(point, lambda value: pearson_loss_and_gradient(value, profiles)[0])
        np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=1e-7)

    def test_spatial_covariance_zero_variance_matches_paired_ridge(self):
        train = np.array([[-1., 0.], [0., -1.], [1., 1.]])
        test = np.array([[.7, -.1], [.2, -.9]])
        y = np.array([2.3, 1.2, 1.7])
        config = {'length_fraction': .25, 'feature_length_per_sqrt_dimension': 1.,
                  'spatial_variance': 0., 'noise_to_linear_variance': 10.}
        model = fit_gaussian_process(train, np.array([.1, .2, .8]), y, config)
        values = predict_gaussian_process(model, test, np.array([.3, .9]), config)
        np.testing.assert_allclose(values, Ridge(alpha=10.).fit(train, y).predict(test), atol=1e-12)

    def test_multitask_design_preserves_the_s1_only_gram_matrix(self):
        s1 = np.array([[1., -2., .2], [2., 5., -.7], [0., 2., 1.1]])
        s2 = np.array([[3., .5], [4., .2]])
        first, second = joint_design(s1, s2, [0, 2])
        np.testing.assert_allclose(first @ first.T, s1 @ s1.T, atol=1e-12)
        np.testing.assert_allclose(second @ second.T, s2 @ s2.T, atol=1e-12)
        np.testing.assert_allclose(first @ second.T, .5 * s1[:, [0, 2]] @ s2.T, atol=1e-12)

    def test_inner_oof_weight_is_chosen_on_profile_scores(self):
        y = np.array([5., 6., 4., 5.5])
        error = np.array([.3, -.4, -.2, .3])
        data = pd.DataFrame({'id': 8, 'prediction_date': pd.Timestamp('2022-07-06'),
                             'next_mm': y, 'baseline': y + error, 'challenger': y - error})
        config = {'minimum_inner_profile_rows': 4, 'weights': [0., .5, 1.]}
        weight, _ = choose_weight(data, config)
        self.assertEqual(weight, .5)


if __name__ == '__main__':
    unittest.main()

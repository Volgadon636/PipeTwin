"""Temporal and geometry contracts for the optional features."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.ml.incremental import read_config
from src.ml.incremental_features import (
    attach_defect_stats, multiscale_elevation, prior_inspection_stats,
)


class HistoricalDefectStatsTests(unittest.TestCase):
    def test_boundary_missing_and_asof(self):
        prior, cutoff = pd.Timestamp('2018-01-01'), pd.Timestamp('2022-01-01')
        grid = pd.DataFrame({
            'id': [1, 1], 'segment_id': [0, 1], 'start_m': [0, 10], 'end_m': [10, 20],
        })
        observations = pd.DataFrame({
            'id': [1] * 6, 'date': [prior] * 5 + [cutoff],
            'distance': [9, 10, 13, 17, 20.1, 14], 'defect_type': ['corrosion'] * 6,
            'residual_for_target_mm': [5, 4, 6, np.nan, 0.5, 0.9],
            'target_eligible': [True, True, True, False, True, True],
        })
        forecasts = pd.DataFrame({
            'id': [1, 1], 'segment_id': [0, 1], 'last_vtd_date': [prior, prior],
            'prediction_date': [cutoff, cutoff], 'has_own_prior_vtd': [True, True],
            'last_mm': [5, 4],
        }, index=[9, 2])
        stats = prior_inspection_stats(observations, grid, forecasts)
        result = attach_defect_stats(forecasts, stats)
        self.assertEqual(result.index.tolist(), [9, 2])
        self.assertEqual(result.prior_min_mm.tolist(), [5, 4])
        self.assertEqual(result.prior_min_boundary_m.tolist(), [1, 0])
        self.assertTrue(np.isnan(result.loc[9, 'prior_p05_mm']))
        self.assertAlmostEqual(result.loc[2, 'prior_p05_mm'], 4.1)
        self.assertAlmostEqual(result.loc[2, 'prior_defect_count_log'], np.log1p(3))
        self.assertAlmostEqual(result.loc[2, 'prior_measured_count_log'], np.log1p(2))
        self.assertEqual(result.prior_neighbor_min_mm.tolist(), [4, 5])
        with self.assertRaisesRegex(ValueError, 'strictly before'):
            prior_inspection_stats(observations, grid, forecasts.assign(last_vtd_date=cutoff))


class ElevationTests(unittest.TestCase):
    def test_physical_window_edges_and_pipe_boundary(self):
        profile = pd.DataFrame({
            'id': [1, 1, 1, 2], 'segment_id': [0, 1, 2, 0],
            'start_m': [0, 10, 20, 0], 'end_m': [10, 20, 25, 10],
            'z_center': [5, 15, 22.5, 100],
        })
        out = multiscale_elevation(profile.iloc[::-1], [20]).set_index(['id', 'segment_id'])
        self.assertAlmostEqual(out.loc[(1, 0), 'elevation_smooth_20m'], 125 / 15)
        self.assertAlmostEqual(out.loc[(1, 1), 'elevation_smooth_20m'], 14.375)
        self.assertAlmostEqual(out.loc[(1, 1), 'elevation_slope_20m'], 1)
        self.assertEqual(out.loc[(2, 0), 'elevation_smooth_20m'], 100)
        self.assertTrue(np.isnan(out.loc[(2, 0), 'elevation_slope_20m']))

    def test_configuration_defaults_are_off(self):
        path = Path(__file__).resolve().parents[1] / 'configs/incremental.json'
        config = read_config(path)
        for key, value in config.items():
            if '_enable_' in key:
                self.assertIs(value, False)

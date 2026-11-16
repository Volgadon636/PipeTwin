import unittest

import numpy as np
import pandas as pd

from src.grid_builder import build_segments_grid
from src.segment_quantiles import extract_segment_state, prepare_quantile_outputs, run_quantile_ablation


class SegmentQuantileTests(unittest.TestCase):
    def setUp(self):
        self.grid = build_segments_grid(pd.DataFrame({'id': [1], 'length_km': [.055]}))

    def test_measured_precedence_sparse_defects_and_coverage(self):
        vtd = pd.DataFrame({
            'id': 1, 'distance': [9., 10., 11., 18., 24., 48.],
            'distance_aligned': [10., 11., 15., 18., 24., 48.],
            'residual_wall_thickness': [5., 6., np.nan, np.nan, np.nan, 9.],
            'residual_for_target_mm': [5., 6., np.nan, np.nan, np.nan, np.nan],
            'defect_type': ['internal', 'internal', 'internal', 'internal', 'internal', 'internal'],
            'defect_depth': [3., 7., 2., np.nan, np.nan, 2.],
            'defect_depth_pct': [np.nan, np.nan, 50., 50., np.nan, np.nan],
        })
        state = extract_segment_state(vtd, self.grid, 8., 'distance_aligned', (0, 50))
        self.assertEqual(state.segment_id.tolist(), list(range(6)))
        self.assertEqual(state.loc[0, 'rw_q05'], 8)
        self.assertTrue(state.loc[0, 'nominal_fallback'])
        self.assertEqual(state.loc[1, 'defect_count'], 4)
        self.assertEqual(state.loc[1, 'derived_thickness_count'], 2)
        self.assertEqual(state.loc[1, 'rw_min'], 4)
        self.assertAlmostEqual(state.loc[1, 'rw_q05'], 4.15)
        self.assertEqual(state.loc[1, 'rw_mean'], 5.25)
        for index in (2, 4, 5):
            self.assertTrue(np.isnan(state.loc[index, 'rw_q05']))
            self.assertFalse(state.loc[index, 'nominal_fallback'])
        # A singleton uses the one measured thickness; direct data wins over depth.
        singleton = extract_segment_state(vtd.iloc[[1]].drop(columns='residual_for_target_mm'),
                                          self.grid, 8.)
        self.assertEqual(singleton.loc[1, 'rw_q05'], 6.)
        self.assertTrue(extract_segment_state(pd.DataFrame(), self.grid, 8.).nominal_fallback.all())

    def test_ablation_uses_same_keys_and_compares_measured_truth(self):
        observations = pd.DataFrame({'id': 1, 'distance': [1., 11., 21., 41.],
                                     'residual_wall_thickness': [5., 6., 7., 8.]})
        raw = extract_segment_state(observations, self.grid, 8.)
        aligned = extract_segment_state(observations, self.grid, 8., covered_m=(0, 40))
        truth = self.grid[['id', 'segment_id']].assign(
            min_residual_thickness_mm=[5., 6., 7., 8., 8., 1.])
        scores = run_quantile_ablation(truth, {'raw': raw, 'aligned': aligned})
        self.assertTrue(scores[scores.cohort.eq('all_available')].n.eq(5).all())
        self.assertTrue(scores[scores.cohort.eq('measured_only')].n.eq(4).all())
        self.assertTrue(scores.pearson_r.eq(1).all())

    def test_snapshot_excludes_cutoff_day_and_unobserved_cold_start(self):
        date = pd.Timestamp('2022-01-01')
        cutoff = pd.Timestamp('2023-01-01')
        vtd = pd.DataFrame({'id': [1, 1, 1], 'date': [date, cutoff, date - pd.Timedelta(days=1)],
                            'distance': [1., 1., 1.], 'residual_wall_thickness': [6., 2., 1.],
                            'before_commissioning_flag': [False, False, True],
                            'commissioning_unknown_flag': False, 'heldout_or_future_flag': False})
        measured = vtd.assign(segment_id=0, target_eligible=[True, True, False],
                              min_residual_thickness_mm=vtd.residual_wall_thickness)
        cleaned = {'vtd.csv': vtd, 'passport.csv': pd.DataFrame({'id': [1], 'nominal_thickness_mm': [8.]})}
        requests = [{'id': 1, 'prediction_date': str(cutoff.date()), 'case': 'between_vtd'},
                    {'id': 1, 'prediction_date': '2021-01-01', 'case': 'between_vtd'},
                    {'id': 1, 'prediction_date': '2024-01-01', 'case': 'cold_start'}]
        out = prepare_quantile_outputs(cleaned, self.grid, measured, requests, {}, None)
        features = out['target_features_vtd_quantiles']
        self.assertEqual(len(features), 12)
        current = features[features.prediction_date.eq(cutoff)]
        self.assertTrue(current.source_vtd_date.eq(date).all())
        self.assertEqual(current.iloc[0].rw_q05, 6.)
        earlier = features[features.prediction_date.lt(date)]
        self.assertTrue(earlier.source_vtd_date.isna().all())
        self.assertTrue(earlier.rw_q05.isna().all())

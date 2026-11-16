from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.ml.s1_contract import S1_ID
from src.ml.s1_joint_alignment import (
    PRIOR_DATE, heldout_anchors, joint_anchors, map_coordinates,
    predict_aligned, verified_target_anchors,
)


class JointAlignmentTests(unittest.TestCase):
    def test_anchors_match_within_section_and_exclude_both_holdout_coordinates(self):
        date = pd.Timestamp('2023-12-01')
        rows = pd.DataFrame([
            {'id': pid, 'section_number': '10', 'date': day, 'distance': start + rel,
             'relative_distance': str(rel), 'distance_raw': -999}
            for pid, first, second in [(1, 0, 2), (2, 9, 12)]
            for day, start in [(PRIOR_DATE, first), (date, second)]
            for rel in (1, 2)
        ])
        anchors = joint_anchors(rows, PRIOR_DATE, date, [1, 2])
        self.assertEqual(anchors.shift_m.tolist(), [2, 3])
        self.assertTrue(heldout_anchors(anchors, 2, 10).empty)

    def test_maps_before_binning_with_distinct_quantile_and_valid_bounds(self):
        grid = pd.DataFrame({'id': [1, 1], 'segment_id': [0, 1],
                             'start_m': [0, 10], 'end_m': [10, 20]})
        fallback = grid.assign(thickness_pred=[7., 7.], nominal_thickness_mm=8.)
        vtd = pd.DataFrame({'id': [1] * 4, 'date': [PRIOR_DATE] * 4,
                            'distance': [2., 8., 9., 19.],
                            'target_eligible': [True] * 4,
                            'residual_for_target_mm': [5., 4., 6., 3.]})
        anchors = pd.DataFrame({'source_m': [0, 20], 'reference_m': [3, 24]})
        np.testing.assert_allclose(map_coordinates([0, 10, 20], anchors), [3, 13.5, 24])
        result, dropped = predict_aligned(vtd, grid, fallback, anchors)
        self.assertEqual(dropped, 1)
        self.assertEqual(result.joint_min.tolist(), [5., 4.])
        self.assertAlmostEqual(result.joint_p05.iloc[1], 4.1)
        self.assertTrue(result.joint_blend.between(0, 8).all())
        with self.assertRaisesRegex(ValueError, 'bracket'):
            map_coordinates([-1, 20], anchors, outside='reject')

    def test_strict_export_rejects_same_day_anchors(self):
        anchors = pd.DataFrame({'id': [S1_ID, S1_ID], 'source_m': [0, 20],
                                'reference_m': [2, 24], 'source_date': '2022-07-06',
                                'available_date': '2023-12-01', 'evidence': 'survey log'})
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'anchors.csv'
            anchors.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'strictly before'):
                verified_target_anchors(path)
            anchors.assign(available_date='2023-11-30').to_csv(path, index=False)
            self.assertEqual(len(verified_target_anchors(path)), 2)

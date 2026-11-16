"""Geometry and time contracts for the odometer preprocessing path."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.odo_align import OdometerAligner, evaluate_alignment_quality, prepare_odometer_outputs
from src.segment_mapper import map_vtd_to_segments


class OdometerAlignmentTests(unittest.TestCase):
    def test_joint_lengths_reject_a_mismatched_weld_and_preserve_monotonicity(self):
        source = pd.DataFrame({'id': 1, 'distance': [1, 2, 11, 12, 21, 22, 31, 32],
                               'section_number': np.repeat([10, 20, 30, 40], 2),
                               'relative_distance': [1, 2] * 4, 'pipe_length': 10.})
        reference = source.assign(distance=source.distance + np.repeat([2, 3, 4, 5], 2),
                                  pipe_length=np.repeat([10, 11, 10, 10], 2))
        aligner = OdometerAligner().fit(reference, source)
        self.assertEqual(aligner.method_, 'joints')
        self.assertEqual(aligner.weld_counts_, {'source': 4, 'reference': 4,
                                               'common_ids': 4, 'accepted': 3})
        np.testing.assert_allclose(aligner.transform([-1, 0, 10, 30, 40, np.nan]),
                                   [1, 2, 13, 35, 45, np.nan], equal_nan=True)
        with self.assertRaisesRegex(ValueError, 'same section'):
            aligner.fit(reference.assign(id=2), source)

    def test_missing_welds_use_signed_density_shift(self):
        rng = np.random.default_rng(123)
        positions = np.sort(rng.uniform(20, 240, 200))
        before = pd.DataFrame({'distance': positions})
        after = before.assign(distance=before.distance + 3)
        fit = OdometerAligner().fit(after, before)
        self.assertEqual(fit.method_, 'density')
        np.testing.assert_allclose(fit.anchors_.shift_m, 3, atol=0.21)
        sparse = fit.fit(after.iloc[:2], before.iloc[:2])
        self.assertEqual(sparse.method_, 'unresolved')
        np.testing.assert_array_equal(sparse.transform([10, 30]), [10, 30])

    def test_aligned_rows_bin_before_aggregation_and_metrics_keep_common_keys(self):
        grid = pd.DataFrame({'id': 1, 'segment_id': [0, 1, 2],
                             'start_m': [0, 10, 20], 'end_m': [10, 20, 30]})
        frame = pd.DataFrame({
            'id': 1, 'date': pd.Timestamp('2022-01-01'), 'source_row': [1, 2, 3, 4],
            'distance': [7, 8, 17, 28], 'relative_distance': [1, 2, 1, 12],
            'section_number': ['1', '1', '2', '2'], 'pipe_length': 10.,
            'target_eligible': True, 'residual_for_target_mm': [5., 4., 6., 3.],
            'defect_depth_pct': 20, 'before_commissioning_flag': False,
            'commissioning_unknown_flag': False, 'wall_nominal_mismatch_flag': False,
            'nominal_unresolved_flag': False, 'heldout_or_future_flag': False,
        })
        ref = frame.assign(date=pd.Timestamp('2023-01-01'), distance=frame.distance + 3)
        model = OdometerAligner()
        moved = model.fit_transform(ref, frame)
        out = map_vtd_to_segments(moved.loc[moved.distance_aligned.le(30)], grid,
                                  coordinate='distance_aligned')
        self.assertEqual(out.set_index('segment_id').min_residual_thickness_mm.to_dict(),
                         {1: 4., 2: 6.})
        report, profile = evaluate_alignment_quality(model, moved, ref, grid)
        self.assertEqual(report['comparison']['min_mm']['matched_segments'], 2)
        self.assertIsNone(report['comparison']['min_mm']['after']['pearson_r2'])
        self.assertEqual(profile.count_after.tolist(), [0, 2, 1])
        self.assertEqual(frame.distance.tolist(), [7, 8, 17, 28])
        plan = [{'id': 1, 'source_date': '2022-01-01', 'reference_date': '2023-01-01',
                 'feature_cutoff': '2023-01-01'}]
        requests = [{'id': 1, 'prediction_date': '2023-01-01', 'case': 'between_vtd'}]
        # The preprocessing entry receives cleaned reference rows.
        joint_observation = ref.iloc[[2]].assign(
            source_row=5, distance=21., relative_distance=2., target_eligible=False,
            residual_for_target_mm=np.nan)
        cleaned = {'vtd.csv': pd.concat([frame, ref[ref.distance.le(30)], joint_observation])}
        with self.assertRaisesRegex(ValueError, 'strictly before cutoff'):
            prepare_odometer_outputs(cleaned, grid, grid, grid, requests,
                                     {'vtd_odometer_alignment': plan})
        plan[0]['feature_cutoff'] = requests[0]['prediction_date'] = '2023-01-02'
        with patch('src.segment_mapper.build_features_asof',
                   return_value={'target_features': grid}) as build:
            tables, _ = prepare_odometer_outputs(cleaned, grid, grid, grid, requests,
                                                 {'vtd_odometer_alignment': plan})
        history = build.call_args.args[3]
        self.assertEqual(history.groupby('date').segment_id.apply(list).to_dict(),
                         {pd.Timestamp('2022-01-01'): [1, 2], pd.Timestamp('2023-01-01'): [1, 2]})
        self.assertIn('target_features_odo_aligned', tables)

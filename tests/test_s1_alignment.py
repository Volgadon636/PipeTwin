import unittest

import pandas as pd

from src.ml.s1_alignment import PEERS, cross_section_offsets, predict_section


class AlignmentTests(unittest.TestCase):
    def test_offsets_apply_before_binning_and_exclude_future_and_outside_rows(self):
        date = pd.Timestamp('2023-12-01')
        past = pd.Timestamp('2022-07-06')
        grid = pd.DataFrame({'id': [1, 1, 1], 'segment_id': [0, 1, 2],
                             'start_m': [0, 10, 20], 'end_m': [10, 20, 30]})
        vtd = pd.DataFrame({
            'id': [1] * 7, 'distance': [2, 6, 24, 14, 3, 3, 3],
            'date': [past, past, past, past, date, past, past - pd.Timedelta(days=1)],
            'residual_for_target_mm': [5., 4., 2., 3., 1., 1., 1.],
            'target_eligible': [True, True, True, True, True, False, True],
        })
        snapshot = grid.assign(h0_mm=[5., 6., 7.], nominal_thickness_mm=8.,
                               last_vtd_thickness_mm=[5., 6., 7.])
        before = vtd.copy()
        pred, details = predict_section(snapshot, vtd, grid, date, 8)
        self.assertEqual(pred.thickness_pred.tolist(), [5., 4., 3.])
        self.assertEqual(pred.fallback.tolist(), [True, False, False])
        self.assertEqual(details['observations_outside_grid'], 1)
        pd.testing.assert_frame_equal(vtd, before)

    def test_cross_section_calibration_uses_only_the_other_section(self):
        peer = pd.DataFrame({'id': [PEERS[0]] * 4 + [PEERS[1]] * 4,
                             'next_mm': [1, 2, 3, 4] * 2})
        for offset in range(11):
            peer[f'shift_{offset}'] = [4, 3, 2, 1] * 2
        peer.loc[:3, 'shift_8'] = [1, 2, 3, 4]
        peer.loc[4:, 'shift_10'] = [1, 2, 3, 4]
        _, folds = cross_section_offsets(peer)
        self.assertEqual([fold['selected_offset_m'] for fold in folds], [10, 8])

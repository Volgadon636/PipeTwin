"""Stage 2 contracts: full-grid thermal, centering, partial inference."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.heat_transfer import HeatTransferParameters, segment_outlet_temperature
from src.ml.data import RepositoryData
from src.ml.next_common import load_artifact
from src.ml.targeted_s2 import (CENTRED, CANDIDATES, _grid_check,
                                _thermal_features, _predict_candidate,
                                prepare_full_grid, prepare_target_full,
                                predict_saved, predict_selected)

_DATA_ENV = os.environ.get('PIPETWIN_NEXT_DATA_ROOT')
if not _DATA_ENV:
    raise RuntimeError('Set PIPETWIN_NEXT_DATA_ROOT to the verified LF data_generation root')
ROOT = Path(_DATA_ENV)
ARTIFACTS = Path('experiments/targeted_cycle/stage_2_s2_fixes/models')
OUT = Path('experiments/targeted_cycle/stage_2_s2_fixes')


class Stage2ThermalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ROOT.is_dir():
            raise unittest.SkipTest('LF data generation is absent')
        cls.data = RepositoryData(ROOT)
        cls.target, cls.interval = prepare_target_full(cls.data)
        cls.params = HeatTransferParameters()

    def test_label_columns_do_not_change_target_thermal_profile(self):
        a = _thermal_features(self.target, self.params)
        with_labels = self.target.copy()
        with_labels['next_mm'] = 1.0
        b = _thermal_features(with_labels.drop(columns=['next_mm']), self.params)
        pd.testing.assert_frame_equal(
            a.sort_values('segment_id').reset_index(drop=True),
            b.sort_values('segment_id').reset_index(drop=True))

    def test_label_subset_through_training_path_keeps_full_grid_thermal(self):
        base = prepare_full_grid(self.data)
        labels = base['measured'][['id', 'segment_id', 'prediction_date']].copy()
        # Remove every third measured label; full campaign grids must remain.
        remove = labels.iloc[::3]
        keys = [(int(r.id), int(r.segment_id), pd.Timestamp(r.prediction_date))
                for r in labels.itertuples()
                if (int(r.id), int(r.segment_id), pd.Timestamp(r.prediction_date))
                not in set((int(a), int(b), pd.Timestamp(c))
                           for a, b, c in zip(remove.id, remove.segment_id,
                                              remove.prediction_date))]
        reduced = prepare_full_grid(self.data, label_keys=keys)
        self.assertEqual(set(base['full_frames']), set(reduced['full_frames']))
        for key in sorted(base['full_frames']):
            left = base['full_frames'][key].sort_values('segment_id').reset_index(drop=True)
            right = reduced['full_frames'][key].sort_values('segment_id').reset_index(drop=True)
            np.testing.assert_allclose(left.scenario_cooling_c.to_numpy(float),
                                       right.scenario_cooling_c.to_numpy(float),
                                       equal_nan=True)
        common = base['measured'][['id', 'segment_id', 'prediction_date',
                                   'scenario_cooling_c']].merge(
            reduced['measured'][['id', 'segment_id', 'prediction_date',
                                 'scenario_cooling_c']],
            on=['id', 'segment_id', 'prediction_date'], suffixes=('_a', '_b'))
        self.assertGreater(len(common), 0)
        np.testing.assert_allclose(common.scenario_cooling_c_a.to_numpy(float),
                                   common.scenario_cooling_c_b.to_numpy(float),
                                   equal_nan=True)

    def test_full_grid_to_subset_and_local_sum(self):
        full = _thermal_features(self.target, self.params)
        subset_ids = self.target.sort_values('segment_id').segment_id.iloc[:25].tolist()
        subset = full[full.segment_id.isin(subset_ids)].sort_values('segment_id')
        expected = self.target[self.target.segment_id.isin(subset_ids)].sort_values('segment_id')
        np.testing.assert_allclose(subset.scenario_cooling_c.to_numpy(float),
                                   expected.scenario_cooling_c.to_numpy(float),
                                   equal_nan=True)
        ordered = self.target.sort_values('start_m')
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        total = last.t_out_segment_c - first.t_in_segment_c
        self.assertAlmostEqual(float(full.local_delta_t_c.sum()), float(total), places=8)

    def test_gap_and_overlap_detection(self):
        base = pd.DataFrame({
            'id': [1, 1], 'prediction_date': [pd.Timestamp('2020-01-01')] * 2,
            'start_m': [0., 20.], 'end_m': [10., 30.], 'length_m': [10., 10.]})
        self.assertFalse(_grid_check(base)['valid'])
        overlap = base.copy()
        overlap.loc[1, 'start_m'] = 5.
        overlap.loc[1, 'end_m'] = 15.
        self.assertFalse(_grid_check(overlap)['valid'])

    def test_analytic_exponential_and_zero_flow(self):
        params = self.params
        result = segment_outlet_temperature(20.0, 10.0, 0.2, 5.0, params)
        perimeter = np.pi * (0.2 - 2 * params.roughness_m)
        exponent = -(params.overall_heat_transfer_w_m2_k * perimeter * 10.0
                     / (5.0 * params.heat_capacity_j_kg_k))
        expected = params.ground_temperature_c + (20 - params.ground_temperature_c) * np.exp(exponent)
        self.assertAlmostEqual(result['t_out_c'], expected, places=10)
        self.assertFalse(segment_outlet_temperature(20.0, 10.0, 0.2, 0.0, params)['valid'])


class Stage2CenteringAndArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ROOT.is_dir():
            raise unittest.SkipTest('LF data generation is absent')
        cls.data = RepositoryData(ROOT)
        cls.target, cls.interval = prepare_target_full(cls.data)
        cls.imap = {(1250009583, pd.Timestamp('2022-07-01')): cls.interval}
        cls.artifact = load_artifact(
            ARTIFACTS / 'spatial_no_temperature_centered/model_artifact.joblib')

    def test_zero_mean_length_weighted(self):
        pred = _predict_candidate(self.artifact, self.target, self.imap)
        weights = self.target.length_m.to_numpy(float)
        mean = np.sum(weights * pred['g_centered']) / np.sum(weights)
        self.assertLess(abs(mean), 1e-12)
        checks = json.loads((OUT / 'centering_checks.json').read_text(encoding='utf-8'))
        self.assertEqual(checks['spatial_no_temperature_centered']['centered'], True)

    def test_full_and_partial_inference_match(self):
        full = _predict_candidate(self.artifact, self.target, self.imap)
        ids = self.target.sort_values('segment_id').segment_id.iloc[[3, 77, 500]].tolist()
        partial = predict_selected(self.artifact, self.target, self.imap, ids)
        for segment in ids:
            idx = self.target.index[self.target.segment_id.eq(segment)][0]
            self.assertAlmostEqual(float(partial.loc[partial.segment_id.eq(segment),
                                                     'thickness_pred'].iloc[0]),
                                   float(full['h_clipped'][idx]), places=12)

    def test_save_load_predict_in_new_call_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = predict_saved(
                ARTIFACTS / 'spatial_no_temperature_centered/model_artifact.joblib',
                ROOT, out, 'spatial_no_temperature_centered')
            self.assertEqual(summary['rows'], 1475)
            manual = pd.read_csv(ARTIFACTS / 'spatial_no_temperature_centered/predictions_s2.csv')
            pred = pd.read_csv(out / 'spatial_no_temperature_centered/predictions_s2.csv')
            np.testing.assert_allclose(pred.thickness_pred, manual.thickness_pred,
                                       rtol=0, atol=1e-12)

    def test_new_candidate_metadata_and_clipping(self):
        for name in CANDIDATES:
            meta = json.loads((ARTIFACTS / name / 'model_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['centered'], name in CENTRED)
            self.assertEqual(meta['lambda'], 1.0)
            self.assertIn('centering_context', meta)
            pred = pd.read_csv(ARTIFACTS / name / 'predictions_s2.csv')
            self.assertEqual(len(pred), 1475)
            self.assertTrue(pred.thickness_pred.between(0, 4.5 + 1e-9).all())


if __name__ == '__main__':
    unittest.main()

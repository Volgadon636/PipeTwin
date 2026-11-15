"""Stage 6 contracts: freeze, explicit pairs, artifact-only recheck and evaluator."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.evaluate_external import evaluate

OUT = Path('experiments/targeted_cycle/frozen_candidates')


class Stage6FreezeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = json.loads((OUT / 'candidate_freeze.json').read_text(encoding='utf-8'))
        cls.manifest = json.loads((OUT / 'freeze_manifest.json').read_text(encoding='utf-8'))

    def test_candidate_roles_and_hashes(self):
        candidates = self.freeze['candidates']
        self.assertEqual(self.freeze['candidate_count_s1'], 3)
        self.assertEqual(self.freeze['candidate_count_s2'], 3)
        for cid, rec in candidates.items():
            self.assertTrue(rec['roles'])
            self.assertTrue(rec['prediction_sha256'])
            self.assertTrue(rec['source_prediction_sha256'])
            self.assertTrue(rec['input_sha256'])
            self.assertTrue(rec['feature_contract'])
            self.assertTrue(rec['seen_evaluations'])
        self.assertTrue(candidates['s1a']['roles'].__contains__('level_reference'))
        self.assertTrue(candidates['physics_priors']['roles'].__contains__('level_reference'))

    def test_explicit_pairs_and_contracts(self):
        pairs = self.freeze['explicit_combinations']['pairs']
        self.assertEqual(len(pairs), 5)
        self.assertTrue(self.freeze['explicit_combinations']['explicit_only'])
        for pair in pairs:
            path = Path(pair['result_path'])
            frame = pd.read_csv(path)
            self.assertEqual(len(frame), 2000)
            self.assertFalse(frame.duplicated(['id', 'segment_id']).any())
            self.assertTrue(np.isfinite(frame.thickness_pred).all())
            self.assertEqual(int(frame.id.eq(1250022871).sum()), 525)
            self.assertEqual(int(frame.id.eq(1250009583).sum()), 1475)
            self.assertEqual(pair['s1_max_abs_diff_vs_frozen'], 0.0)
            self.assertEqual(pair['s2_max_abs_diff_vs_frozen'], 0.0)
            self.assertEqual(pair['rows'], 2000)
            self.assertIn('S1', pair['constant_diagnostics'])
            self.assertIn('S2', pair['constant_diagnostics'])

    def test_recheck_new_process_and_no_fit(self):
        self.assertTrue(self.freeze['recheck']['new_process'])
        self.assertTrue(self.freeze['recheck']['no_fit'])
        self.assertTrue(all(v == 0.0 for v in self.freeze['recheck']['max_abs_differences'].values()))
        self.assertFalse(self.freeze['main_result_overwritten'])
        self.assertEqual(self.manifest['no_main_result_overwrite'], True)

    def test_external_evaluator_separate_truth(self):
        pair = OUT / 'combinations/s1a__physics_priors/result.csv'
        pred = pd.read_csv(pair)
        truth = pred[['id', 'segment_id', 'thickness_pred']].rename(
            columns={'thickness_pred': 'thickness_measured'})
        truth['date'] = truth.id.map({1250022871: '2023-12-01',
                                      1250009583: '2022-07-01'})
        # Synthetic evaluator mechanics only; not a quality claim.
        truth['thickness_measured'] = truth.thickness_measured + 0.01
        with tempfile.TemporaryDirectory() as tmp:
            truth_path = Path(tmp) / 'truth.csv'
            out = Path(tmp) / 'eval'
            truth.to_csv(truth_path, index=False)
            payload = evaluate(pair, truth_path, out)
            self.assertEqual(payload['rows'], 2000)
            self.assertTrue(payload['keys_match'])
            self.assertIn('S1', payload['metrics'])
            self.assertIn('S2', payload['metrics'])
            self.assertTrue((out / 'external_metrics.json').is_file())
            self.assertTrue((out / 'external_errors.csv').is_file())


if __name__ == '__main__':
    unittest.main()

"""Stage 4 contracts: physical masks, same keys, archive diagnosis and decisions."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_ENV = os.environ.get('PIPETWIN_NEXT_DATA_ROOT')
if not _DATA_ENV:
    raise RuntimeError('Set PIPETWIN_NEXT_DATA_ROOT to the verified LF data_generation root')
DATA_ROOT = Path(_DATA_ENV)
OUT = Path('experiments/targeted_cycle/stage_4_s1_robustness')


class Stage4Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diag = json.loads((OUT / 'stage_4_archive_diagnosis.json').read_text(encoding='utf-8'))
        cls.decision = json.loads((OUT / 'candidate_decisions.json').read_text(encoding='utf-8'))
        cls.schemes = ['200m_50m', '500m_100m']

    def test_archive_difference_is_mask_only(self):
        self.assertEqual(self.diag['s1a_current_vs_archive_max_abs'], 0.0)
        self.assertGreater(self.diag['mask_diff_blocks'], 0)
        self.assertGreater(self.diag['mask_diff_keys_total'], 0)
        self.assertLess(self.diag['archive_mask_refit_ridge_max_abs'], 1e-9)
        self.assertLess(self.diag['archive_mask_refit_catboost_max_abs'], 1e-9)
        self.assertTrue(self.diag['used_for_selection'] is False)

    def test_physical_masks_and_keys(self):
        for scheme in self.schemes:
            base = OUT / scheme
            folds = json.loads((base / 'fold_assignments.json').read_text(encoding='utf-8'))
            oof = pd.read_csv(base / 'oof_predictions.csv')
            keys = pd.read_csv(base / 'fold_training_keys.csv')
            self.assertGreaterEqual(len(folds['folds']), 2)
            self.assertEqual(len(oof), 321)
            self.assertFalse(oof.duplicated(['id', 'segment_id']).any())
            self.assertEqual(set(oof.id), {1250022871})
            for f in folds['folds']:
                test = {(int(a), int(b)) for a, b in f['test_keys']}
                train = keys[keys.fold_block.eq(f['block'])]
                global_keys = set(zip(train[train.train_kind.eq('global')].id,
                                      train[train.train_kind.eq('global')].segment_id))
                local_keys = set(zip(train[train.train_kind.eq('target_only')].id,
                                     train[train.train_kind.eq('target_only')].segment_id))
                self.assertFalse(test & global_keys)
                self.assertFalse(test & local_keys)
                self.assertEqual(f['buffer_m'], folds['buffer_m'])
                self.assertTrue(f['buffer_exclusion_ok'])
                self.assertGreaterEqual(f['min_target_train_center_distance_m'],
                                        folds['buffer_m'] - 1e-8)

    def test_all_models_share_keys_and_are_finite(self):
        for scheme in self.schemes:
            oof = pd.read_csv(OUT / scheme / 'oof_predictions.csv')
            for model in ('s1a', 'train_median', 'affine_global', 'affine_target',
                          'global_ridge_geometry', 'target_ridge_geometry',
                          'catboost_reference'):
                self.assertIn(model, oof.columns)
                self.assertTrue(np.isfinite(oof[model]).all())

    def test_decision_and_best_mae(self):
        self.assertLessEqual(len(self.decision['selected_candidates']), 2)
        self.assertEqual(self.decision['reference'], 's1a')
        self.assertTrue(self.decision['common_promising'])
        self.assertIn(self.decision['best_mae']['model'],
                      ('global_ridge_geometry', 'target_ridge_geometry', 'catboost_reference'))
        for scheme in self.schemes:
            self.assertIn(scheme, self.decision['promising_by_scheme'])

    def test_lopo_not_mixed_into_schemes(self):
        for scheme in self.schemes:
            text = (OUT / scheme / 'summary.json').read_text(encoding='utf-8')
            self.assertIn('evaluated', text)
            self.assertNotIn('LOPO', text)


if __name__ == '__main__':
    unittest.main()

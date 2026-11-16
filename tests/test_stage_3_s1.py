"""Stage 3: physical block folds, same keys, artifacts and artifact-only predict."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.next_common import load_artifact
from src.ml.targeted_s1 import (MODELS, predict_saved)

_DATA_ENV = os.environ.get('PIPETWIN_NEXT_DATA_ROOT')
if not _DATA_ENV:
    raise RuntimeError('Set PIPETWIN_NEXT_DATA_ROOT to the verified LF data_generation root')
DATA_ROOT = Path(_DATA_ENV)
OUT = Path('experiments/targeted_cycle/stage_3_s1_100m_20m')


class Stage3Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folds = json.loads((OUT / 'fold_assignments.json').read_text(encoding='utf-8'))
        cls.oof = pd.read_csv(OUT / 'oof_predictions.csv')
        cls.train_keys = pd.read_csv(OUT / 'fold_training_keys.csv')
        cls.metrics = pd.read_csv(OUT / 'metrics_overall.csv')

    def test_physical_blocks_and_disjoint_train_test(self):
        self.assertEqual(self.folds['block_m'], 100.0)
        self.assertEqual(self.folds['buffer_m'], 20.0)
        self.assertEqual(len(self.folds['folds']), 52)
        fold = self.folds['folds'][0]
        self.assertAlmostEqual(fold['high_m'] - fold['low_m'], 100.0)
        self.assertGreater(len(fold['test_keys']), 0)
        test_keys = set()
        for f in self.folds['folds']:
            self.assertEqual(f['buffer_m'], 20.0)
            test_keys.update((int(a), int(b)) for a, b in f['test_keys'])
        self.assertEqual(len(test_keys), 321)
        for f in self.folds['folds']:
            train = self.train_keys[self.train_keys.fold_block.eq(f['block'])]
            test = {(int(a), int(b)) for a, b in f['test_keys']}
            global_keys = set(zip(train[train.train_kind.eq('global')].id,
                                  train[train.train_kind.eq('global')].segment_id))
            local_keys = set(zip(train[train.train_kind.eq('target_only')].id,
                                 train[train.train_kind.eq('target_only')].segment_id))
            self.assertFalse(test & global_keys)
            self.assertFalse(test & local_keys)

    def test_same_keys_and_finite_predictions(self):
        self.assertEqual(len(self.oof), 321)
        self.assertFalse(self.oof.duplicated(['id', 'segment_id']).any())
        self.assertEqual(set(self.oof.id), {1250022871})
        for model in MODELS:
            self.assertIn(model, self.oof.columns)
            self.assertTrue(np.isfinite(self.oof[model]).all())

    def test_affine_nonnegative_and_catboost_reference(self):
        fold_meta = json.loads((OUT / 'fold_fit_metadata.json').read_text(encoding='utf-8'))
        for rec in fold_meta:
            self.assertGreaterEqual(rec['affine_global']['a'], 0)
            self.assertGreaterEqual(rec['affine_target']['a'], 0)
        artifacts = json.loads((OUT / 'model_artifacts.json').read_text(encoding='utf-8'))
        cat = next(x for x in artifacts if x['model'] == 'catboost_reference')
        self.assertTrue(cat['artifact_path'])
        self.assertEqual(cat['metadata']['model_type'], 'CatBoostRegressor')

    def test_within_block_correlation_counts(self):
        centered = json.loads((OUT / 'within_block_correlation.json').read_text(encoding='utf-8'))
        for model in ('s1a', 'affine_global', 'global_ridge_geometry',
                      'target_ridge_geometry', 'catboost_reference'):
            self.assertEqual(centered[model]['defined_blocks'], 51)
            self.assertEqual(centered[model]['undefined_blocks'], 1)
        self.assertEqual(centered['train_median']['defined_blocks'], 0)

    def test_artifact_only_predict_matches_full_target(self):
        path = OUT / 'models/global_ridge_geometry/model_artifact.joblib'
        if not path.is_file():
            raise AssertionError('Stage 3 artifact is missing')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = predict_saved(path, DATA_ROOT, out, 'global_ridge_geometry')
            self.assertEqual(result['rows'], 525)
            got = pd.read_csv(out / 'global_ridge_geometry/predictions_s1.csv')
            expected = pd.read_csv(OUT / 'target_predictions/global_ridge_geometry/predictions_s1.csv')
            merged = got.merge(expected, on=['id', 'segment_id'], suffixes=('_got', '_exp'))
            self.assertEqual(len(merged), 525)
            np.testing.assert_allclose(merged.thickness_pred_got,
                                       merged.thickness_pred_exp, rtol=0, atol=1e-12)
            artifact = load_artifact(path)
            self.assertEqual(artifact['metadata']['alpha'], 10.0)
            self.assertEqual(artifact['metadata']['baseline_column'], 'last_mm')

    def test_metrics_are_on_same_keys(self):
        self.assertEqual(len(self.metrics), len(MODELS))
        self.assertTrue((self.metrics.n == 321).all())
        self.assertTrue(self.metrics.mae_mm.notna().all())


if __name__ == '__main__':
    unittest.main()

"""Stage 5 temporal-contract checks: actual dt, fallback, bounds and non-selection."""
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
OUT = Path('experiments/targeted_cycle/stage_5_temporal')


class Stage5TemporalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dt = json.loads((OUT / 'dt_audit.json').read_text(encoding='utf-8'))
        cls.proposal = json.loads((OUT / 'proposal.json').read_text(encoding='utf-8'))
        cls.oof = pd.read_csv(OUT / 'oof_temporal_metrics.csv')
        cls.summary = json.loads((OUT / 'production_summary.json').read_text(encoding='utf-8'))
        cls.diag = pd.read_csv(OUT / 'production/global_ridge_geometry_diagnostics.csv')

    def test_actual_dt_not_hardcoded(self):
        self.assertEqual(self.dt['train_rows'], 357)
        self.assertTrue(self.dt['dt_train_common'])
        self.assertAlmostEqual(self.dt['dt_train_years'], 3.783709787817, places=12)
        self.assertEqual(self.dt['production_dt_counts']['1.404517453799'], 508)
        self.assertEqual(self.dt['production_dt_counts']['5.188227241615'], 13)
        self.assertEqual(self.dt['fallback_rows'], 4)

    def test_oof_cannot_identify_time_law(self):
        for base in ('global_ridge_geometry', 'target_ridge_geometry'):
            part = self.oof[self.oof.model_base.eq(base)].set_index('variant')
            self.assertEqual(float(part.loc['correction', 'mae_mm']),
                             float(part.loc['time_scaled', 'mae_mm']))
            self.assertNotEqual(float(part.loc['correction', 'mae_mm']),
                                float(part.loc['conservative_0_25', 'mae_mm']))

    def test_fallback_is_not_ridge_corrected_and_bounds_hold(self):
        fallback = self.diag[~self.diag.has_own_prior_vtd.astype(bool)]
        self.assertEqual(len(fallback), 4)
        for variant in ('correction', 'time_scaled', 'conservative_0_25'):
            np.testing.assert_allclose(fallback[variant], fallback.s1b_prediction,
                                       rtol=0, atol=0)
            self.assertTrue(fallback[variant].between(0, 8).all())
        for variant in ('correction', 'time_scaled', 'conservative_0_25'):
            self.assertTrue(self.diag[variant].between(0, 8).all())
            self.assertEqual(int(self.diag[variant + '_above_nominal'].sum()), 0)

    def test_above_last_is_not_banned(self):
        self.assertGreater(int(self.diag['correction_above_last'].sum()), 0)
        self.assertGreater(int(self.diag['time_scaled_above_last'].sum()), 0)
        self.assertGreater(int(self.diag['conservative_0_25_above_last'].sum()), 0)
        self.assertEqual(int(self.diag['correction_clipped'].sum()), 0)

    def test_raw_and_clipped_columns_are_saved(self):
        for variant in ('correction', 'time_scaled', 'conservative_0_25'):
            self.assertIn(variant + '_raw', self.diag.columns)
            self.assertIn(variant, self.diag.columns)
            self.assertIn(variant + '_clipped', self.diag.columns)

    def test_proposal_is_at_most_two_and_no_winner(self):
        self.assertEqual(self.proposal['reference'], 's1a')
        self.assertLessEqual(len(self.proposal['challengers']), 2)
        self.assertEqual(self.proposal['status'], 'awaiting_user_freeze_choice')
        self.assertIn('no temporal winner', self.proposal['selection_basis'].lower())


if __name__ == '__main__':
    unittest.main()

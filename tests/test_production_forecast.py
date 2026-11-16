import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.ml.production import assemble, new_output
from src.ml.production_models import (
    H31_PARAMETERS, fit_s1_models, fit_s2_models, point_summaries,
    predict_s1_models, predict_s2_models, s1_prediction_frame,
)
from src.ml.production_validation import score_rows
from src.ml.s1_contract import S1_CUTOFF, S1_ID, S2_ID
from src.ml.targeted_s2 import S2_CUTOFF


class ProductionForecastGuards(unittest.TestCase):
    def test_h18_rejects_partial_grid_before_normalization(self):
        grid = pd.DataFrame({'id': [S1_ID, S1_ID], 'segment_id': [0, 1]})
        partial = grid.iloc[:1].assign(prediction_date=S1_CUTOFF)
        with self.assertRaisesRegex(ValueError, 'complete section grid'):
            s1_prediction_frame(partial, pd.DataFrame(), grid)

    def test_h18_fit_and_predict_reject_cold_start(self):
        cold = pd.DataFrame({'id': [S2_ID], 'case': ['cold_start']})
        with self.assertRaisesRegex(ValueError, 'H18 requires S1'):
            fit_s1_models(cold)
        with self.assertRaisesRegex(ValueError, 'only available for S1'):
            predict_s1_models({}, cold)

    def test_failed_export_does_not_publish_or_overwrite_results(self):
        with TemporaryDirectory() as temp:
            target = Path(temp) / 'forecast'
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                with new_output(target) as staging:
                    (staging / 'result.csv').write_text('incomplete', encoding='utf-8')
                    raise RuntimeError('failed validation')
            self.assertFalse(target.exists())
            with new_output(target) as staging:
                (staging / 'result.csv').write_text('complete', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                with new_output(target):
                    self.fail('Existing result must be preserved')
            self.assertEqual((target / 'result.csv').read_text(encoding='utf-8'), 'complete')

    def test_validation_reports_undefined_and_negative_correlations(self):
        short = pd.DataFrame({'id': [1, 1], 'prediction_date': [S1_CUTOFF] * 2,
                              'next_mm': [1., 2.], 'baseline': [1., 2.], 'candidate': [2., 1.]})
        self.assertTrue(all(row['pearson_squared'] is None
                            and row['delta_pearson_squared'] is None
                            for row in score_rows('S1_peer_2023', short)))
        normal = pd.concat([short, short.iloc[:1].assign(next_mm=3., baseline=3., candidate=0.)])
        scores = list(score_rows('S1_peer_2023', normal))
        selected = next(row for row in scores if row['method'] == 'h18_h31')
        self.assertAlmostEqual(selected['pearson_squared'], 1.)
        self.assertAlmostEqual(selected['pearson_r'], -1.)

    def test_h18_features_ignore_future_ineligible_and_other_section_points(self):
        prior = pd.Timestamp('2022-07-06')
        grid = pd.DataFrame({'id': [S1_ID, S1_ID], 'segment_id': [0, 1],
                             'start_m': [0., 10.], 'end_m': [10., 20.]})
        raw = pd.DataFrame({
            'id': [S1_ID, S1_ID, S1_ID, S1_ID, S2_ID],
            'date': [prior, prior, S1_CUTOFF, prior, prior], 'distance': [1., 1., 1., 11., 1.],
            'target_eligible': [True, False, True, True, True],
            'residual_for_target_mm': [4., 1., 2., 5., 1.],
            'defect_length': [1., 99., 99., 2., 99.], 'defect_width': [2.] * 5,
            'feature_description': ['коррозия'] * 5, 'defect_type': ['внутренний'] * 5,
        })
        summaries, _ = point_summaries(raw, grid, S1_CUTOFF, prior)
        self.assertEqual(summaries.prior_direct_mm.tolist(), [4., 5.])
        self.assertEqual(summaries.prior_reported_metal_loss_fraction.tolist(), [1., 1.])
        with self.assertRaises(ValueError):
            point_summaries(raw, grid, S1_CUTOFF, S1_CUTOFF)

    def test_h31_training_rejects_target_inspections_and_cutoff_date(self):
        rows = pd.DataFrame({'id': [S2_ID], 'prediction_date': [S2_CUTOFF - pd.Timedelta(days=1)],
                             'case': ['cold_start']})
        with self.assertRaises(ValueError):
            fit_s2_models(rows, {}, H31_PARAMETERS)
        with self.assertRaises(ValueError):
            fit_s2_models(rows.assign(id=S1_ID, prediction_date=S2_CUTOFF), {}, H31_PARAMETERS)

    def test_h31_rejects_shortened_grid_before_centering(self):
        partial = pd.DataFrame({
            'id': [S2_ID], 'prediction_date': [S2_CUTOFF], 'case': ['cold_start'],
            'segment_id': [0], 'start_m': [0.], 'end_m': [10.], 'length_m': [10.],
            'forecast_grid_rows': [2], 'forecast_grid_extent_m': [20.],
            'last_vtd_thickness_mm': [float('nan')], 'last_uzt_thickness_mm': [float('nan')],
        })
        with self.assertRaisesRegex(ValueError, 'complete section grid'):
            predict_s2_models({}, partial, {})
        with self.assertRaisesRegex(ValueError, 'own inspection'):
            predict_s2_models({}, partial.assign(last_vtd_thickness_mm=1.), {})

    def test_export_disallows_h31_for_s1_and_wrong_cutoff(self):
        s1 = pd.DataFrame({'id': [S1_ID], 'segment_id': [0], 'prediction_date': [S1_CUTOFF],
                           'H18': [5.1], 'baseline': [5.]})
        s2 = pd.DataFrame({'id': [S2_ID], 'segment_id': [0], 'prediction_date': [S2_CUTOFF],
                           'H31': [4.1], 'baseline': [4.]})
        self.assertEqual(assemble(s1, s2, 'H18', 'H31').thickness_pred.tolist(), [5.1, 4.1])
        with self.assertRaises(ValueError):
            assemble(s1, s2, 'H31', 'baseline')
        with self.assertRaises(ValueError):
            assemble(s1, s2.assign(prediction_date=S1_CUTOFF), 'H18', 'H31')


if __name__ == '__main__':
    unittest.main()

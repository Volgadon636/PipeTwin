"""ML contracts, temporal counterexamples, units and real-data integration tests."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, TECH_FILE
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics, evaluate_forward, masked_baseline_check
from src.ml.exposure import exposure_intervals, integrated_years, attach_exposure
from src.ml.models import fit_constant_rate, fit_physical_model, sample_weights
from src.ml.submission import validate_submission, risk_ranking
from src.ml.windows import add_baselines, make_snapshot, build_training_windows, eligible_training


def synthetic_training():
    return pd.DataFrame(dict(id=[1, 2, 3], segment_id=[0, 0, 0],
                             main_pipeline_id=['A', 'B', 'C'], campaign_id=['A:1', 'B:1', 'C:1'],
                             h0_mm=[8., 8., 8.], horizon_years=[1., 2., 3.],
                             min_residual_thickness_mm=[7.8, 7.6, 7.4]))


class ModelTests(unittest.TestCase):
    def test_constant_rate_units_and_no_label_clipping(self):
        train = synthetic_training()
        model = fit_constant_rate(train)
        self.assertAlmostEqual(model.rate_mm_year, .2)
        np.testing.assert_allclose(model.predict(train), train.min_residual_thickness_mm)
        train.min_residual_thickness_mm = [9., 9., 9.]
        self.assertEqual(fit_constant_rate(train).rate_mm_year, 0.)

    def test_parent_balance_not_simple_section_balance(self):
        train = synthetic_training()
        train = pd.concat([train, train.iloc[[0]], train.iloc[[0]]], ignore_index=True)
        weights = sample_weights(train, 'W1')
        totals = pd.Series(weights).groupby(train.main_pipeline_id).sum()
        np.testing.assert_allclose(totals, totals.iloc[0])
        self.assertAlmostEqual(weights.mean(), 1.)

    def test_pearson_square_is_not_r2(self):
        result = metrics([1, 2, 3], [3, 5, 7])
        self.assertAlmostEqual(result['pearson_squared'], 1.)
        self.assertLess(result['r2'], 0.)
        self.assertIsNone(metrics([1, 2, 3], [2, 2, 2])['pearson_squared'])
        self.assertIsNone(metrics([0., 1.], [0., 1.])['mape_pct'])

    def test_cold_constant_ranking_discloses_ties(self):
        frame = pd.DataFrame({'id': [1]*20, 'segment_id': range(20), 'thickness_pred': [4.]*20})
        ranked = risk_ranking(frame)
        self.assertTrue(ranked.prediction_tie_count.eq(20).all())
        self.assertEqual(ranked.top_10pct.sum(), 2)

    def test_submission_rejects_duplicate_extra_missing_and_nonfinite(self):
        grid = pd.DataFrame({'id': [1, 1], 'segment_id': [0, 1]})
        bounds = grid.assign(h0_mm=[5., 6.])
        good = grid.assign(thickness_pred=[4., 5.])
        validate_submission(good.iloc[::-1], grid, bounds)
        bad = [good.iloc[[0, 0]], good.iloc[:1], good.assign(extra=1),
               good.assign(thickness_pred=[np.inf, 4.]), good.assign(thickness_pred=[-1., 4.]),
               good.assign(thickness_pred=[6., 4.]), good.assign(segment_id=[.5, 1]),
               good.assign(thickness_pred=['4', '5'])]
        for frame in bad:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(ValueError):
                validate_submission(frame, grid, bounds)

    def test_exposure_holds_exact_duration_and_preserves_gap(self):
        rows = pd.DataFrame(dict(id=[1, 1, 1],
            date=pd.to_datetime(['2020-01-01 12:00', '2020-01-02 00:00', '2020-01-04 12:00']),
            t_mix_usable=[20., 20., 99.], watercut_pct_usable=[50., 50., 50.],
            v_liquid_true_usable=[1., 1., 1.]))
        intervals, summary = exposure_intervals(rows, '2020-01-01', '2020-01-04')
        self.assertAlmostEqual(intervals.duration_years.sum()*365.25, 1.5)
        self.assertAlmostEqual(summary['coverage_fraction'], .5)
        self.assertAlmostEqual(summary['longest_gap_days'], 1.)
        np.testing.assert_allclose(intervals.temperature_K, [293.15, 293.15])
        np.testing.assert_allclose(intervals.water_fraction, [.5, .5])
        reference = dict(temperature_K=293.15, water_fraction=.5, velocity_m_s=1., velocity_offset_m_s=.1)
        self.assertAlmostEqual(integrated_years(intervals, reference, np.zeros(3), 'observed_only'), 1.5/365.25)
        self.assertAlmostEqual(integrated_years(intervals, reference, np.zeros(3), 'reference', 3/365.25), 3/365.25)

    def test_invalid_row_stops_prior_interval(self):
        rows = pd.DataFrame(dict(id=[1, 1], date=pd.to_datetime(['2020-01-01 00:00', '2020-01-01 06:00']),
                                 t_mix_usable=[20., np.nan], watercut_pct_usable=[50., 50.], v_liquid_true_usable=[1., 1.]))
        _, summary = exposure_intervals(rows, '2020-01-01', '2020-01-03')
        self.assertAlmostEqual(summary['observed_years']*365.25, .25)

    def test_missing_exposure_is_unknown_not_zero_measurement(self):
        rows = pd.DataFrame(dict(id=pd.Series(dtype=int), date=pd.Series(dtype='datetime64[ns]'),
                                 t_mix_usable=[], watercut_pct_usable=[], v_liquid_true_usable=[]))
        _, summary = exposure_intervals(rows, '2020-01-01', '2021-01-01')
        self.assertEqual(summary['observed_years'], 0.)
        self.assertAlmostEqual(summary['unobserved_years'], 366/365.25)
        self.assertEqual(summary['longest_gap_days'], 366)

    def test_physics_fit_recovers_constant_synthetic_rate_and_gates_groups(self):
        train = synthetic_training()
        intervals = [pd.DataFrame(dict(id=[i+1], date=pd.to_datetime(['2019-01-01']),
                        temperature_K=[290.+i*10], water_fraction=[.5], velocity_m_s=[1.],
                        duration_years=[float(i+1)])) for i in range(3)]
        model = fit_physical_model(train, intervals)
        self.assertAlmostEqual(model.rate_mm_year, .2)
        np.testing.assert_allclose(model.predict(train, intervals), train.min_residual_thickness_mm)
        self.assertGreaterEqual(model.metadata()['theta_T_K'], 0.)
        with self.assertRaisesRegex(ValueError, 'independent'):
            fit_physical_model(train.iloc[:1], intervals[:1])  # 1 unique parent < minimum_groups=2


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = RepositoryData()
        cls.windows, cls.histories = build_training_windows(cls.data)

    def test_published_grid_and_features_reused_exactly(self):
        published = self.data.read('processed/target_features')
        for request in self.data.config['targets']:
            features, _ = make_snapshot(self.data, request)
            expected = published[published.id.eq(request['id'])].reset_index(drop=True)
            pd.testing.assert_frame_equal(features[list(expected)].reset_index(drop=True), expected, check_dtype=False)
        self.assertEqual(len(published), 2000)

    def test_cold_training_stops_before_july_2022_inspection(self):
        train = eligible_training(self.windows, '2022-07-01', 'cold_start', 'MAIN-4')
        self.assertEqual(len(train), 372)
        self.assertEqual(set(train.main_pipeline_id), {'MAIN-2'})
        self.assertTrue(train.prediction_date.eq(pd.Timestamp('2018-09-23')).all())
        self.assertFalse(self.windows.id.eq(1250009583).any())

    def test_all_historical_features_precede_their_own_label_dates(self):
        for col in ('hydraulic_source_date_max', 'monitoring_source_date_max', 'last_vtd_date', 'last_uzt_date'):
            self.assertTrue((self.windows[col].isna() | self.windows[col].lt(self.windows.prediction_date)).all())
        cold = self.windows[self.windows['case'].eq('cold_start')]
        self.assertTrue(cold[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].isna().all().all())
        labels = self.data.read('processed/training_targets')
        joined = self.windows.merge(labels, left_on=['id', 'segment_id', 'prediction_date'],
                                    right_on=['id', 'segment_id', 'date'], validate='one_to_one')
        np.testing.assert_allclose(joined.min_residual_thickness_mm_x, joined.min_residual_thickness_mm_y)

    def test_current_data_has_no_independent_forward_fit_before_targets(self):
        # Only *fitted* models (train_rows > 0) are checked.  The unfitted C0
        # constant (rate=0, train_rows=0) is always trivially evaluable and is
        # intentionally excluded from this gate.
        for request in self.data.config['targets']:
            windows = self.windows[self.windows['case'].eq(request['case'])]
            records, _ = evaluate_forward(windows, pd.Timestamp(request['prediction_date']))
            self.assertFalse(any(r['status'] == 'evaluated' and r.get('train_rows', 0) > 0
                                 for r in records))

    def test_future_hydraulics_and_diagnostics_cannot_change_snapshot(self):
        base = self.data

        class PoisonedData:
            config = base.config
            def __init__(self, request):
                self.request = request
            @property
            def cleaned(self):
                frames = base.cleaned
                tech = frames[TECH_FILE]
                future = tech.date.ge(pd.Timestamp(self.request['prediction_date']))
                tech.loc[future, ['q_liquid', 't_mix', 'watercut_pct']] = 1e12
                # This archived flag must not leak target diagnosis into cold age.
                if self.request['case'] == 'cold_start':
                    frames['passport.csv']['history_date_conflict_flag'] = True
                return frames
            def read(self, name):
                frame = base.read(name)
                if name in ('processed/vtd_by_segment', 'processed/uzt_by_segment'):
                    bad = frame.date.ge(pd.Timestamp(self.request['prediction_date']))
                    if self.request['case'] == 'cold_start':
                        bad[:] = True
                    frame.loc[bad, 'min_residual_thickness_mm'] = 1e12
                return frame
            def parent_groups(self):
                return base.parent_groups()

        for request in base.config['targets']:
            expected, rows = make_snapshot(base, request)
            actual, changed_rows = make_snapshot(PoisonedData(request), request)
            pd.testing.assert_frame_equal(expected, actual)
            pd.testing.assert_frame_equal(rows, changed_rows)

    def test_missing_prior_fallback_is_explicit_and_uses_prior_timestamp(self):
        request = self.data.config['targets'][0]
        features, _ = make_snapshot(self.data, request)
        missing = features[features.baseline_missing]
        self.assertEqual(missing.segment_id.tolist(), [17, 247, 401, 455])
        self.assertTrue(missing.baseline_source.eq('nearest_prior_vtd_assumption').all())
        self.assertTrue(missing.start_date.lt(missing.prediction_date).all())
        self.assertTrue(missing.last_vtd_thickness_mm.isna().all())
        self.assertEqual(masked_baseline_check(features)['n'], 521)

    def test_future_labels_cannot_change_production_fitted_rate(self):
        poisoned = self.windows.copy()
        cutoff = pd.Timestamp('2022-07-01')
        poisoned.loc[poisoned.prediction_date.ge(cutoff), 'min_residual_thickness_mm'] = 1e12
        a = fit_constant_rate(eligible_training(self.windows, cutoff, 'cold_start', 'MAIN-4'))
        b = fit_constant_rate(eligible_training(poisoned, cutoff, 'cold_start', 'MAIN-4'))
        self.assertEqual(a.metadata(), b.metadata())

    def test_checksum_failure_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'configs').mkdir()
            (root/'reports').mkdir()
            (root/'processed').mkdir()
            (root/'configs/preprocessing.json').write_text(json.dumps(self.data.config))
            (root/'reports/run_manifest.json').write_text(json.dumps(self.data.manifest))
            (root/'processed/target_segments_grid.csv').write_text('id,segment_id\n1,0\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                RepositoryData(root).read('processed/target_segments_grid')


if __name__ == '__main__':
    unittest.main()

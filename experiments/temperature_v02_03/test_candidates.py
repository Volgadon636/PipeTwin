"""Stage 03 numerical, cohort, leakage and evidence contracts."""
import json
import unittest

import numpy as np
import pandas as pd

from src.ml.temperature import EA_CANDIDATES, temperature_features, temperature_intervals, thermal_factor
from experiments.temperature_v02_03.run import (
    STAGE_ROOT, REVIEW_COLUMNS, audit_folds, candidate_rows, factor_stats, interval_digest, protocol,
)


class CandidateTests(unittest.TestCase):
    def test_preregistered_grid_and_controls_are_separate(self):
        fixed = protocol()
        self.assertEqual(list(fixed['candidates_kJ_mol'].values()), [0, 25, 40, 55, 70])
        self.assertEqual(fixed['gap_policies'], ['observed_only', 'reference'])
        self.assertEqual(fixed['legacy_role'], 'compatibility control only; excluded from candidate grid and selection')
        self.assertNotIn(31.26, EA_CANDIDATES.values())
        self.assertNotIn(73.29, EA_CANDIDATES.values())

    def test_fixed_intervals_agree_with_stage02_for_every_candidate(self):
        rows = pd.DataFrame(dict(id=[1]*3, date=pd.to_datetime(['2020-01-01 12:00','2020-01-02 00:00','2020-01-03 00:00']),
                                 t_mix_usable=[0., np.nan, 40.]))
        intervals, summary = temperature_intervals(rows, '2020-01-01', '2020-01-04')
        before = intervals.copy(deep=True)
        hashes = set()
        for candidate, ea in EA_CANDIDATES.items():
            exposure, stats = candidate_rows(intervals, summary, 3/365.25, ea, candidate)
            expected = temperature_features(rows, '2020-01-01', '2020-01-04', ea_kj_mol=ea, gap_policy='reference')
            for name in ['thermal_exposure_observed_years','thermal_exposure_reference_years','thermal_observed_years',
                         'thermal_unobserved_years','thermal_coverage_fraction','thermal_factor_observed_mean']:
                self.assertEqual(exposure[name], expected[name])
            self.assertEqual(stats['finite_count'], 2)
            hashes.add(exposure['interval_sha256'])
        self.assertEqual(len(hashes), 1)
        pd.testing.assert_frame_equal(intervals, before)

    def test_observation_distribution_distinct_from_duration_weighting(self):
        stats = factor_stats([1., 3.], [.9, .1])
        self.assertEqual(stats['mean_F_observed'], 2.)
        self.assertAlmostEqual(stats['duration_weighted_mean_F'], 1.2)
        self.assertAlmostEqual(stats['p95_F_observed'], 2.9)
        self.assertEqual(stats['min_F_observed'], 1.)
        with self.assertRaises(ValueError):
            factor_stats([1, np.inf], [1, 1])

    def test_no_intervals_stays_unknown(self):
        rows = pd.DataFrame(dict(id=pd.Series(dtype=int), date=pd.Series(dtype='datetime64[ns]'), t_mix_usable=[]))
        intervals, summary = temperature_intervals(rows, '2020-01-01','2020-01-04')
        for name, ea in EA_CANDIDATES.items():
            result, stats = candidate_rows(intervals, summary, 3/365.25, ea, name)
            self.assertEqual(result['thermal_exposure_observed_years'], 0)
            self.assertEqual(result['thermal_exposure_reference_years'], 3/365.25)
            self.assertEqual(result['thermal_coverage_fraction'], 0)
            self.assertIsNone(result['thermal_factor_observed_mean'])
            self.assertIsNone(stats['p95_F_observed'])

    def test_fold_gates_exclude_same_parent_same_day_future_and_cold_target(self):
        frame = pd.DataFrame(dict(
            main_pipeline_id=['A','A','B','B','TARGET','C'],
            prediction_date=pd.to_datetime(['2018-01-01','2019-01-01','2019-01-01','2020-01-01','2017-01-01','2030-01-01']),
            case=['cold_start']*6, campaign_id=['A:1','A:2','B:1','B:2','T:1','C:1'], baseline_missing=[False]*6))
        requests = [dict(id=9, prediction_date='2021-01-01', case='cold_start')]
        folds = audit_folds(frame, requests, pd.Series({9:'TARGET'}))
        self.assertNotIn('TARGET', folds.heldout_parent.tolist())
        self.assertNotIn('C', folds.heldout_parent.tolist())
        first = folds[folds.heldout_parent.eq('A') & folds.test_date.eq(pd.Timestamp('2018-01-01'))].iloc[0]
        self.assertFalse(first.admissible)
        same_parent = folds[folds.heldout_parent.eq('A') & folds.test_date.eq(pd.Timestamp('2019-01-01'))].iloc[0]
        self.assertFalse(same_parent.admissible)
        independent = folds[folds.heldout_parent.eq('B') & folds.test_date.eq(pd.Timestamp('2019-01-01'))].iloc[0]
        self.assertEqual(independent.train_rows, 1)
        self.assertTrue(independent.admissible)
        self.assertLess(independent.train_date_max, independent.test_date)


class CandidateArtifactTests(unittest.TestCase):
    """Audit the published reproducible outputs; run the CLI before discovery."""
    def test_exact_review_schema_and_null_metrics(self):
        frame = pd.read_csv(STAGE_ROOT/'ea_candidate_review.csv')
        self.assertEqual(list(frame), REVIEW_COLUMNS)
        self.assertEqual(len(frame), 10)
        self.assertEqual(set(frame.candidate_id), set(EA_CANDIDATES))
        self.assertTrue(frame[['pearson_squared','mae_mm','rmse_mm','bias_mm']].isna().all().all())
        self.assertTrue(frame.evidence_status.eq('diagnostic_only').all())
        self.assertTrue(frame.validation_fold_count.eq(0).all())
        self.assertTrue(frame.decision.eq('retain_for_testing').all())
        payload = json.loads((STAGE_ROOT/'ea_candidate_review.json').read_text())
        self.assertTrue(all(r['pearson_squared'] is None for r in payload))

    def test_every_candidate_uses_identical_support(self):
        frame = pd.read_csv(STAGE_ROOT/'candidate_exposures.csv')
        for _, group in frame.groupby('window_id'):
            self.assertEqual(set(group.thermal_candidate_id), set(EA_CANDIDATES))
            for column in ['interval_sha256','thermal_observed_years','thermal_unobserved_years','thermal_coverage_fraction']:
                self.assertEqual(group[column].nunique(), 1)
            null = group[group.thermal_candidate_id.eq('EA00_null')].iloc[0]
            self.assertEqual(null.thermal_exposure_observed_years, null.thermal_observed_years)
            self.assertEqual(null.thermal_exposure_reference_years, null.horizon_years)
        for row in frame.itertuples():
            self.assertAlmostEqual(row.thermal_exposure_reference_years,
                                   row.thermal_exposure_observed_years+row.thermal_unobserved_years)

    def test_historical_cohorts_and_production_keys(self):
        maps = pd.read_csv(STAGE_ROOT/'window_segments.csv')
        production = maps[maps.dataset.eq('production')]
        self.assertEqual(len(production), 2000)
        self.assertFalse(production.duplicated(['id','segment_id']).any())
        self.assertEqual(production.groupby('id').size().to_dict(), {1250009583:1475, 1250022871:525})
        historical = maps[maps.dataset.eq('historical_eligible')]
        self.assertEqual(len(historical), 729)
        for request in protocol()['targets']:
            sub = historical[historical['case'].eq(request['case'])]
            self.assertTrue(pd.to_datetime(sub.prediction_date).lt(pd.Timestamp(request['prediction_date'])).all())
        self.assertFalse(historical.id.eq(1250009583).any())
        folds = pd.read_csv(STAGE_ROOT/'validation_fold_audit.csv')
        self.assertEqual(len(folds), 2)
        self.assertFalse(folds.admissible.any())

    def test_sensitivity_and_legacy_control(self):
        frame = pd.read_csv(STAGE_ROOT/'sensitivity_temperatures.csv')
        self.assertTrue(frame.loc[frame.temperature_C.eq(20),'factor'].eq(1).all())
        for row in frame.itertuples():
            self.assertAlmostEqual(row.factor, float(thermal_factor(row.temperature_K,row.E_a_kJ_mol)))
        legacy = json.loads((STAGE_ROOT/'legacy_compatibility.json').read_text())
        self.assertFalse(legacy['candidate'])
        self.assertEqual(legacy['source_type'],'legacy_placeholder')
        self.assertLess(legacy['max_absolute_difference'],1e-12)


if __name__ == '__main__':
    unittest.main()

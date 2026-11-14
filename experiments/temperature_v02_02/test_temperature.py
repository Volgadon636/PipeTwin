"""Stage 02 contracts; discover directly here or through tests/test_temperature.py."""
import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.data import RepositoryData
from src.ml.exposure import exposure_intervals
from src.ml.features import REGISTRY, ML_FEATURE_COLUMNS, _compute_arrhenius_factor
from src.ml.temperature import (
    EA_CANDIDATES, R_J_MOL_K, REFERENCE_TEMPERATURE_K, THERMAL_FEATURE_COLUMNS,
    TemperatureFeatureRegistry, attach_temperature_features, temperature_features,
    temperature_intervals, thermal_factor,
)
from src.ml.windows import YEAR_SECONDS, build_training_windows, make_snapshot


def history(dates=(), temperatures=()):
    return pd.DataFrame({'id': pd.Series([1] * len(dates), dtype=int),
                         'date': pd.to_datetime(list(dates), format='mixed'),
                         't_mix_usable': pd.Series(temperatures, dtype=float)})


class TemperatureTests(unittest.TestCase):
    def features(self, rows, start='2020-01-01', end='2020-01-04', ea=40, policy='reference'):
        return temperature_features(rows, start, end, ea_kj_mol=ea, gap_policy=policy)

    def test_kelvin_reference_and_all_candidate_units(self):
        rows = history(['2020-01-01'], [20])
        intervals, _ = temperature_intervals(rows, '2020-01-01', '2020-01-02')
        self.assertEqual(intervals.temperature_K.iloc[0], 293.15)
        for candidate, ea in EA_CANDIDATES.items():
            with self.subTest(candidate=candidate):
                self.assertEqual(thermal_factor(REFERENCE_TEMPERATURE_K, ea), 1)
                result = self.features(rows, ea=ea)
                self.assertEqual(result['thermal_theta_K'], ea*1000/R_J_MOL_K)
                self.assertEqual(result['thermal_candidate_id'], candidate)
        self.assertNotIn(2500*R_J_MOL_K/1000, EA_CANDIDATES.values())

    def test_null_candidate_exact_identities_with_gaps(self):
        rows = history(['2020-01-01 12:00', '2020-01-02 00:00'], [0, 44.52])
        result = self.features(rows, ea=0)
        self.assertEqual(result['thermal_exposure_observed_years'], result['thermal_observed_years'])
        self.assertEqual(result['thermal_exposure_reference_years'], 3/365.25)
        self.assertEqual(result['thermal_factor_observed_mean'], 1)
        np.testing.assert_array_equal(thermal_factor([273.15, 293.15, 317.67], 0), [1, 1, 1])

    def test_monotonic_temperature_and_energy_above_reference(self):
        for ea in [25, 40, 55, 70]:
            factors = thermal_factor([253.15, 273.15, 293.15, 317.67], ea)
            self.assertTrue(np.isfinite(factors).all())
            self.assertTrue((np.diff(factors) > 0).all())
        self.assertTrue((np.diff([thermal_factor(317.67, ea) for ea in EA_CANDIDATES.values()]) > 0).all())

    def test_integrates_factors_before_averaging_temperature(self):
        rows = history(['2020-01-01', '2020-01-02'], [0, 40])
        result = self.features(rows)
        expected = (thermal_factor(273.15, 40) + thermal_factor(313.15, 40)) / 365.25
        self.assertAlmostEqual(result['thermal_exposure_observed_years'], expected)
        self.assertNotAlmostEqual(result['thermal_factor_observed_mean'], 1.0)

    def test_start_end_clipping_invalid_next_and_no_future(self):
        rows = history(['2019-12-31 18:00', '2020-01-01 06:00', '2020-01-03 18:00',
                        '2020-01-04', '2020-01-05'], [20, np.nan, 40, 1e12, 1e12])
        intervals, summary = temperature_intervals(rows, '2020-01-01', '2020-01-04')
        np.testing.assert_allclose(intervals.duration_years*365.25, [.25, .25])
        self.assertEqual(intervals.interval_start.iloc[0], pd.Timestamp('2020-01-01'))
        self.assertEqual(intervals.interval_end.iloc[-1], pd.Timestamp('2020-01-04'))
        self.assertAlmostEqual(summary['coverage_fraction'], .5/3)
        self.assertEqual(summary['longest_gap_days'], 2.5)
        self.assertTrue(intervals.interval_start.iloc[1] >= intervals.interval_end.iloc[0])
        self.assertEqual(self.features(rows), self.features(rows.iloc[:3]))

    def test_hold_cap_leading_trailing_gaps_and_no_backward_fill(self):
        rows = history(['2019-12-20', '2020-01-02 12:00'], [20, 20])
        intervals, summary = temperature_intervals(rows, '2020-01-01', '2020-01-04', .5)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(summary['observed_years']*365.25, .5)
        self.assertEqual(summary['longest_gap_days'], 1.5)
        self.assertAlmostEqual(summary['observed_years']+summary['unobserved_years'], 3/365.25)

    def test_adjacent_unsorted_intraday_intervals_never_overlap(self):
        rows = history(['2020-01-02', '2020-01-01 06:00', '2020-01-01'], [20, 30, 0])
        intervals, summary = temperature_intervals(rows, '2020-01-01', '2020-01-03')
        self.assertEqual(summary['coverage_fraction'], 1)
        self.assertEqual(summary['longest_gap_days'], 0)
        self.assertEqual(summary['unobserved_years'], 0)
        self.assertTrue((intervals.interval_start.iloc[1:].to_numpy() >=
                         intervals.interval_end.iloc[:-1].to_numpy()).all())

    def test_temperature_available_without_water_or_velocity(self):
        rows = history(['2020-01-01'], [20])
        self.assertEqual(len(temperature_intervals(rows, '2020-01-01', '2020-01-02')[0]), 1)
        for water, velocity in [(np.nan, 1), (50, np.nan), (101, -1)]:
            joint_rows = rows.assign(watercut_pct_usable=water, v_liquid_true_usable=velocity)
            self.assertTrue(exposure_intervals(joint_rows, '2020-01-01', '2020-01-02')[0].empty)
            self.assertEqual(self.features(rows), self.features(joint_rows))

    def test_no_support_has_zero_integral_null_mean_and_explicit_assumption(self):
        for rows in [history(), history(['2020-01-02'], [np.nan]),
                     history(['2020-01-05'], [20])]:
            result = self.features(rows)
            self.assertEqual(result['thermal_exposure_observed_years'], 0)
            self.assertEqual(result['thermal_coverage_fraction'], 0)
            self.assertTrue(np.isnan(result['thermal_factor_observed_mean']))
            self.assertTrue(result['thermal_history_missing'])
            self.assertTrue(result['thermal_history_incomplete'])
            self.assertEqual(result['thermal_unobserved_years'], 3/365.25)
            self.assertEqual(result['thermal_exposure_reference_years'], 3/365.25)
            self.assertEqual(result['thermal_longest_gap_days'], 3)
            self.assertTrue(np.isnan(self.features(rows, policy='observed_only')['thermal_exposure_reference_years']))

    def test_invalid_physical_temperatures_stop_hold_but_zero_celsius_valid(self):
        for invalid in [np.nan, np.inf, -np.inf, -273.15, -300]:
            rows = history(['2020-01-01', '2020-01-01 06:00', '2020-01-02'], [20, invalid, 0])
            intervals, summary = temperature_intervals(rows, '2020-01-01', '2020-01-03')
            self.assertEqual(len(intervals), 2)
            self.assertAlmostEqual(summary['observed_years']*365.25, 1.25)
            self.assertEqual(intervals.temperature_K.iloc[-1], 273.15)

    def test_bad_horizons_duplicates_and_unknown_parameters_fail(self):
        rows = history(['2020-01-01', '2020-01-01'], [20, 30])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.features(rows)
        with self.assertRaisesRegex(ValueError, 'one known pipe'):
            self.features(rows.assign(id=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'Missing hydraulic timestamp'):
            self.features(history([None], [20]))
        for start, end, cap in [('2020-01-02', '2020-01-01', 1), (None, '2020-01-02', 1),
                                ('2020-01-01', '2020-01-02', 0), ('2020-01-01', '2020-01-02', np.inf)]:
            with self.assertRaises(ValueError):
                temperature_intervals(history(), start, end, cap)
        with self.assertRaises(ValueError):
            self.features(history(), ea=20.79)
        with self.assertRaises(ValueError):
            self.features(history(), policy='implicit_fill')
        for temperature, ea in [(0, 40), (np.inf, 40), (293.15, -1), (293.15, np.nan),
                                (1e10, 1e10), (1e-12, 70)]:
            with self.assertRaises(ValueError):
                thermal_factor(temperature, ea)

    def test_opt_in_registry_keeps_default_inputs_and_indices(self):
        default_names = list(ML_FEATURE_COLUMNS)
        cutoff = pd.Timestamp('2020-01-04')
        windows = pd.DataFrame(dict(id=[1, 1], segment_id=[8, 3], main_pipeline_id=['A', 'A'],
                                    prediction_date=[cutoff]*2, case=['cold_start']*2,
                                    start_date=pd.to_datetime(['2020-01-01', '2020-01-02'])), index=[7, 7])
        histories = {(1, cutoff): history(['2020-01-01'], [20])}
        registry = TemperatureFeatureRegistry(histories, ea_kj_mol=40, gap_policy='reference')
        enriched, reports = registry.apply(windows, ['thermal_exposure_observed_years'])
        pd.testing.assert_frame_equal(enriched[list(windows)], windows)
        self.assertEqual(enriched.thermal_history_missing.tolist(), [False, True])
        self.assertEqual(len(reports), len(THERMAL_FEATURE_COLUMNS))
        self.assertEqual(next(r.coverage for r in reports if r.feature_name == 'thermal_factor_observed_mean'), .5)
        self.assertEqual(ML_FEATURE_COLUMNS, default_names)
        self.assertEqual(REGISTRY.list_features(), default_names)
        self.assertFalse(set(default_names) & set(THERMAL_FEATURE_COLUMNS))
        empty, _ = registry.apply(windows.iloc[:0])
        self.assertEqual(list(empty), list(enriched))
        self.assertTrue(empty.empty)

    def test_window_history_identity_and_qa_cutoff_checked(self):
        cutoff = pd.Timestamp('2020-01-04')
        window = pd.DataFrame(dict(id=[1], start_date=[pd.Timestamp('2020-01-01')], prediction_date=[cutoff]))
        for rows in [history(['2020-01-01'], [20]).assign(id=2),
                     history(['2020-01-01'], [20]).assign(prediction_date=pd.Timestamp('2021-01-01'))]:
            with self.assertRaises(ValueError):
                attach_temperature_features(window, {(1, cutoff): rows}, ea_kj_mol=40, gap_policy='reference')
        with self.assertRaises(KeyError):
            attach_temperature_features(window, {}, ea_kj_mol=40, gap_policy='reference')

    def test_legacy_factor_and_baseline_files_unchanged(self):
        frame = pd.DataFrame(dict(id=[1]*3, segment_id=[0, 1, 2], t_mix_mean=[0., 20., 44.52]))
        expected = np.exp(2500*(1/293.15 - 1/(frame.t_mix_mean+273.15)))
        np.testing.assert_array_equal(_compute_arrhenius_factor(frame), expected)
        np.testing.assert_allclose(thermal_factor(frame.t_mix_mean+273.15, 2500*R_J_MOL_K/1000), expected,
                                   rtol=1e-15, atol=0)
        for name in ['ml_baseline', 'm0_baseline_reference']:
            path = Path(__file__).resolve().parents[1] / name / 'result.csv'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             'ef1c4e4e33217a5c821a94739f43288942f99e351c05ee454305eae9e3644394')
            result = pd.read_csv(path)
            self.assertEqual(list(result), ['id', 'segment_id', 'thickness_pred'])
            self.assertEqual(len(result), 2000)


class TemperatureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = RepositoryData()

    def test_production_asof_windows_and_joint_compatibility(self):
        counts = []
        for request in self.data.config['targets']:
            windows, rows = make_snapshot(self.data, request)
            cutoff = pd.Timestamp(request['prediction_date'])
            original_rows = rows.copy(deep=True)
            histories = {(request['id'], cutoff): rows}
            enriched, _ = attach_temperature_features(windows, histories, ea_kj_mol=40, gap_policy='reference')
            pd.testing.assert_frame_equal(windows, enriched[list(windows)])
            pd.testing.assert_frame_equal(rows, original_rows)
            counts.append(len(enriched))
            for start in windows.start_date.unique():
                ti, ts = temperature_intervals(rows, start, cutoff)
                ji, js = exposure_intervals(rows, start, cutoff)
                self.assertTrue(ti.date.lt(cutoff).all())
                self.assertGreaterEqual(ts['observed_years'], js['observed_years'])
                shared = ti[ti.date.isin(ji.date)].reset_index(drop=True)
                cols = ['date', 'interval_start', 'interval_end', 'duration_years', 'temperature_K']
                pd.testing.assert_frame_equal(shared[cols], ji[cols].reset_index(drop=True), check_dtype=False)
                poison = rows.iloc[:1].copy()
                poison['date'] = cutoff
                poison['t_mix_usable'] = 1e12
                pd.testing.assert_frame_equal(ti, temperature_intervals(pd.concat([rows, poison, poison]), start, cutoff)[0])
        self.assertEqual(sorted(counts), [525, 1475])

    def test_historical_windows_use_own_qa_and_cutoffs(self):
        windows, histories = build_training_windows(self.data)
        frame, _ = attach_temperature_features(windows, histories, ea_kj_mol=0, gap_policy='reference')
        np.testing.assert_array_equal(frame.thermal_exposure_observed_years, frame.thermal_observed_years)
        np.testing.assert_array_equal(frame.thermal_exposure_reference_years, windows.horizon_years)
        self.assertFalse(frame.id.eq(1250009583).any())
        self.assertTrue(frame.thermal_coverage_fraction.between(0, 1).all())
        for (pid, cutoff), rows in histories.items():
            self.assertTrue(rows.date.lt(cutoff).all())
            self.assertTrue(rows.prediction_date.eq(cutoff).all())


if __name__ == '__main__':
    unittest.main()

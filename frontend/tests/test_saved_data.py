"""Verify the dashboard against the selected release and tested intervals."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / 'frontend'
DATA = FRONTEND / 'public/data'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class SavedDataTests(unittest.TestCase):
    def test_five_files_and_provenance_have_one_release(self):
        documents = [read_json(DATA / (name + '.json')) for name in
                     ['metrics', 'feature_importance', 'top_risk_segments', 'validation', 'profile']]
        self.assertEqual({doc['schema_version'] for doc in documents}, {2})
        self.assertEqual(len({doc['release_id'] for doc in documents}), 1)
        provenance = documents[0]['provenance']
        for path, expected in provenance['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)
        self.assertFalse(provenance['models_trained'])
        self.assertFalse(provenance['metrics_recomputed'])

    def test_metrics_are_saved_values_with_separate_cases(self):
        exported = read_json(DATA / 'metrics.json')
        saved = read_csv(ROOT / 'production/h18_h31_validation/metrics.csv')
        for row in exported['comparison']:
            source = next(item for item in saved if item['scheme'] == row['cohort']
                          and item['scope'] == 'pooled_replay_only'
                          and item['method'] == ('h18_h31' if row['final'] else 'baseline'))
            for field in ['mae_mm', 'rmse_mm', 'bias_mm', 'pearson_r', 'pearson_squared', 'r2']:
                self.assertEqual(row['metrics'][field], float(source[field]))
        self.assertEqual({row['case']: row['model'] for row in exported['headlines']}, {'S1': 'H18', 'S2': 'H31'})
        self.assertNotIn('headline', exported)
        self.assertTrue(all(row['label'] != 'H31' or row['cohort'].startswith('S2') for row in exported['comparison']))

    def test_every_profile_value_matches_result_and_interval_sources(self):
        result = {(row['id'], int(row['segment_id'])): row for row in
                  read_csv(ROOT / 'production/h18_h31_verified/result.csv')}
        intervals = {(row['id'], int(row['segment_id'])): row for row in
                     read_csv(FRONTEND / 'calibration/intervals/target_intervals.csv')}
        count = 0
        for pipe in read_json(DATA / 'profile.json')['pipes']:
            self.assertEqual(pipe['segment_count'], len(pipe['points']))
            for point in pipe['points']:
                key = pipe['pipe_id'], point['segment_id']
                self.assertEqual(point['prediction_mm'], float(result[key]['thickness_pred']))
                self.assertEqual(point['lower_mm'], float(intervals[key]['lower_mm']))
                self.assertEqual(point['upper_mm'], float(intervals[key]['upper_mm']))
                self.assertLessEqual(point['lower_mm'], point['prediction_mm'])
                self.assertLessEqual(point['prediction_mm'], point['upper_mm'])
                if pipe['case'] == 'cold_start':
                    self.assertIsNone(point['actual_mm'])
                    self.assertIsNone(point['actual_date'])
                if point['actual_date']:
                    self.assertLess(point['actual_date'], pipe['prediction_date'])
                count += 1
        self.assertEqual(count, 2000)

    def test_s1_shows_only_direct_2022_measurements(self):
        pipe = next(pipe for pipe in read_json(DATA / 'profile.json')['pipes'] if pipe['case'] == 'between_vtd')
        observations = {int(row['segment_id']): row for row in
                        read_csv(ROOT / 'scratch/odo_alignment_verified/processed/vtd_by_segment.csv')
                        if row['id'] == pipe['pipe_id'] and row['date'][:10] == '2022-07-06'
                        and row['target_eligible'] == 'True'}
        self.assertEqual(len(observations), 508)
        self.assertEqual(pipe['baseline_missing_count'], 17)
        for point in pipe['points']:
            measured = observations.get(point['segment_id'])
            self.assertEqual(point['actual_mm'], float(measured['min_residual_thickness_mm']) if measured else None)

    def test_risk_membership_and_values_match_the_profile(self):
        source = read_csv(ROOT / 'reports/final_top10_segments.csv')
        exported = read_json(DATA / 'top_risk_segments.json')['rows']
        self.assertEqual({(row['id'], int(row['segment_id'])) for row in source},
                         {(row['pipe_id'], row['segment_id']) for row in exported})
        self.assertEqual(len(exported), 201)
        profile = {(pipe['pipe_id'], point['segment_id']): point
                   for pipe in read_json(DATA / 'profile.json')['pipes'] for point in pipe['points']}
        for row in exported:
            point = profile[row['pipe_id'], row['segment_id']]
            self.assertTrue(point['is_risk'])
            for key in ['prediction_mm', 'lower_mm', 'upper_mm']:
                self.assertEqual(row[key], point[key])

    def test_validation_keeps_negative_and_undefined_profiles(self):
        groups = read_json(DATA / 'validation.json')['groups']
        for group in groups:
            self.assertEqual(group['metrics']['n'], len(group['rows']))
        s2 = next(group for group in groups if group['id'] == 'S2_retro_2023')
        self.assertEqual(sum(profile['metrics']['pearson_r'] is not None and profile['metrics']['pearson_r'] < 0
                             for profile in s2['profiles']), 2)
        self.assertTrue(any(profile['metrics']['pearson_squared'] is None for profile in s2['profiles']))
        self.assertFalse(read_json(DATA / 'feature_importance.json')['available'])

    def test_interval_coverage_is_measured_on_outer_holdouts(self):
        summary = read_json(FRONTEND / 'calibration/intervals/summary.json')
        observed = read_csv(FRONTEND / 'calibration/intervals/nested_validation.csv')
        for case, outcome in summary['cases'].items():
            rows = [row for row in observed if row['case'] == case]
            covered = sum(float(row['lower_mm']) <= float(row['next_mm']) <= float(row['upper_mm']) for row in rows)
            self.assertEqual(covered / len(rows), outcome['nested_validation']['coverage'])
            self.assertTrue(all(row['prediction_date'][:10] < outcome['cutoff'] for row in rows))
            self.assertTrue(all(row['id'] != '1250009583' for row in rows))
            for fold in summary['folds'][case]:
                self.assertNotIn(fold['held_id'], fold['fit_ids'])
                self.assertNotIn(fold['held_id'], fold['calibration_ids'])
        self.assertLess(summary['cases']['S1']['nested_validation']['coverage'], .8)


if __name__ == '__main__':
    unittest.main()

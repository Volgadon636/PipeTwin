"""Verify that the website publishes saved results without changing their values."""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'frontend/public/data'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class SavedDataTests(unittest.TestCase):
    def test_metrics_are_exact_saved_values(self):
        exported = read_json(DATA / 'metrics.json')
        source = read_json(ROOT / 'experiments/ml_v2/metrics.json')
        for row in exported['comparison']:
            key = 'cold_start_pooled' if row['cohort'] == 'cold_start' else 'between_vtd_pooled'
            original = next(item for item in source[key] if item['model'] == row['id'])
            for field in ['mae_mm', 'rmse_mm', 'bias_mm']:
                self.assertEqual(row['metrics'][field], original[field])
            self.assertEqual(row['metrics']['pearson_squared'], original['pooled_within_pipe_pearson_squared'])
        self.assertFalse(exported['provenance']['models_trained'])
        self.assertFalse(exported['provenance']['metrics_recomputed'])

    def test_profile_uses_saved_predictions_and_real_measurements(self):
        source = {(row['id'], int(row['segment_id'])): row for row in read_csv(ROOT / 'experiments/ml_v2/prediction_diagnostics.csv')}
        count = 0
        for pipe in read_json(DATA / 'profile.json')['pipes']:
            self.assertEqual(pipe['segment_count'], len(pipe['points']))
            for point in pipe['points']:
                original = source[(pipe['pipe_id'], point['segment_id'])]
                self.assertAlmostEqual(point['prediction_mm'], float(original['thickness_pred']), places=12)
                self.assertAlmostEqual(point['p10_mm'], float(original['thickness_p10']), places=12)
                self.assertLessEqual(point['p10_mm'], point['p90_mm'])
                if pipe['case'] == 'cold_start' or original['has_previous_vtd'] != 'True':
                    self.assertIsNone(point['actual_mm'])
                count += 1
        self.assertEqual(count, len(source))

    def test_risk_membership_is_copied(self):
        source = read_csv(ROOT / 'experiments/ml_v2/risk_top_10pct.csv')
        exported = read_json(DATA / 'top_risk_segments.json')['rows']
        self.assertEqual({(row['id'], int(row['segment_id'])) for row in source},
                         {(row['pipe_id'], row['segment_id']) for row in exported})
        self.assertEqual(len(source), len(exported))

    def test_missing_evidence_stays_missing(self):
        importance = read_json(DATA / 'feature_importance.json')
        self.assertFalse(importance['available'])
        self.assertEqual(importance['models'], [])
        baseline = read_json(ROOT / 'experiments/ml_baseline/metrics.json')
        for group in read_json(DATA / 'validation.json')['groups']:
            original = next(row for row in baseline[group['id']] if row.get('status') == 'evaluated' and row.get('subset') == 'all')
            self.assertEqual(group['metrics']['r2'], original['r2'])
            self.assertEqual(group['metrics']['n'], len(group['rows']))
            self.assertIsNone(group['metrics']['coverage'])
            self.assertTrue(all(row['p10_mm'] is None and row['p90_mm'] is None for row in group['rows']))


if __name__ == '__main__':
    unittest.main()

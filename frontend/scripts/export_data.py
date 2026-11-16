"""Export the verified H18/H31 release and tested intervals to five browser JSONs.

No model fitting, prediction, metric computation or unpickling occurs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / 'frontend'
INTERVALS = FRONTEND / 'calibration/intervals'
CASES = {'S1': {'id': '1250022871', 'date': '2023-12-01', 'count': 525, 'model': 'H18', 'case': 'between_vtd'},
         'S2': {'id': '1250009583', 'date': '2022-07-01', 'count': 1475, 'model': 'H31', 'case': 'cold_start'}}
SCHEMES = [
    ('S1_peer_2023', 'S1 · 2023', 'S1', 'joint_blend',
     'Ретроспектива 2022→2023: 52 сегмента двух связанных участков. Геометрия соседних труб '
     'на дату ответа; одна родительская труба. Сводная корреляция не является скрытым конкурсным баллом.'),
    ('S2_retro_2023', 'S2 · 2023', 'S2', 'spatial_no_temperature_centered',
     'Перенос на первое обследование 2023 года: 94 сегмента семи связанных участков. '
     'Модель обучена до 01.07.2022. Есть отрицательные корреляции и малые группы.'),
    ('S2_LOSO_pooled', 'S2 · 2018', 'S2', 'basin_ranks',
     'Исключение участков кампании 2018 года: 372 сегмента, фиксированные признаки basin_ranks. '
     'H31 уступает базе на всех трёх участках. Это дополнительная схема, не итоговый прогноз S1.'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches_saved_hash(path, expected):
    """Git may change text CSV line endings; accept only that byte transformation."""
    content = Path(path).read_bytes()
    return (hashlib.sha256(content).hexdigest() == expected
            or hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest() == expected)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric(value):
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite source value')
    return result


def unique(rows, columns):
    result = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in result:
            raise ValueError(f'Duplicate keys: {key}')
        result[key] = row
    return result


def metric(row):
    result = {name: numeric(row.get(name)) for name in [
        'mae_mm', 'rmse_mm', 'bias_mm', 'r2', 'pearson_r', 'pearson_squared',
        'coverage', 'mean_interval_width_mm']}
    return {'n': int(row['n']), **result}


def build(data_root=ROOT / 'scratch/odo_alignment_verified', intervals_root=INTERVALS):
    data_root, intervals_root = Path(data_root), Path(intervals_root)
    state_path = ROOT / 'experiments/r2_pool/workflow_state.json'
    selection = read_json(state_path)['integration']
    if selection['methods'] != {'S1': 'H18', 'S2': 'H31'}:
        raise ValueError('The frontend export expects the selected H18/S1 + H31/S2 release')
    forecast = ROOT / selection['forecast']
    forecast_manifest_path = forecast.parent / 'manifest.json'
    models = ROOT / selection['models']
    validation = ROOT / selection['validation']
    manifest = read_json(forecast_manifest_path)
    model_manifest = read_json(models / 'manifest.json')
    interval = read_json(intervals_root / 'summary.json')
    prep_manifest = read_json(data_root / 'reports/run_manifest.json')
    source_paths = {state_path, forecast, forecast_manifest_path, models / 'manifest.json',
                    validation / 'metrics.csv', validation / 'manifest.json',
                    data_root / 'reports/run_manifest.json', intervals_root / 'summary.json',
                    intervals_root / 'target_intervals.csv', intervals_root / 'nested_validation.csv',
                    ROOT / 'reports/final_top10_segments.csv'}
    for stem in ['processed/segments_grid', 'processed/target_features', 'processed/vtd_by_segment']:
        path = data_root / (stem + '.csv')
        if sha(path) != prep_manifest['outputs'][stem]['csv_sha256']:
            raise ValueError(f'Preprocessing input checksum mismatch: {stem}')
        source_paths.add(path)
    if (not matches_saved_hash(forecast, manifest['outputs_sha256']['result.csv'])
            or sha(forecast) != interval['result_sha256']
            or manifest['bundle_sha256'] != model_manifest['bundle_sha256']
            or manifest['methods'] != selection['methods']
            or sha(data_root / 'reports/run_manifest.json') != interval['preprocessing_manifest_sha256']):
        raise ValueError('Model, predictions or interval calibration belong to different releases')
    for name, digest in interval['output_sha256'].items():
        if sha(intervals_root / name) != digest:
            raise ValueError(f'Interval artifact checksum mismatch: {name}')
    for case, spec in CASES.items():
        if any(date >= spec['date'] for date in interval['cases'][case]['calibration_dates']):
            raise ValueError('Interval calibration contains future labels')
    for name in ['predictions_s1.csv', 'predictions_s2.csv']:
        source_paths.add(forecast.parent / name)
    for scheme, *_ in SCHEMES:
        source_paths.add(validation / scheme / 'validation_predictions.csv')
    before = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in sorted(source_paths)}
    release = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    common = {'schema_version': 2, 'release_id': release}
    predictions = unique(read_csv(forecast), ['id', 'segment_id'])
    grid = unique(read_csv(data_root / 'processed/segments_grid.csv'), ['id', 'segment_id'])
    features = unique(read_csv(data_root / 'processed/target_features.csv'), ['id', 'segment_id'])
    bounds = unique(read_csv(intervals_root / 'target_intervals.csv'), ['id', 'segment_id'])
    if set(predictions) != set(features) or set(bounds) != set(predictions):
        raise ValueError('Prediction, feature and interval keys differ')
    risk_source = read_csv(ROOT / 'reports/final_top10_segments.csv')
    risk_keys = set(unique(risk_source, ['id', 'segment_id']))
    risk_rows, pipes = [], []
    vtd = read_csv(data_root / 'processed/vtd_by_segment.csv')
    for case, spec in CASES.items():
        pid = spec['id']
        diagnostic = unique(read_csv(forecast.parent / ('predictions_s1.csv' if case == 'S1' else 'predictions_s2.csv')),
                            ['id', 'segment_id'])
        own = [key for key in predictions if key[0] == pid]
        if len(own) != spec['count'] or sorted(int(key[1]) for key in own) != list(range(spec['count'])):
            raise ValueError('Missing, unexpected or noncontiguous target keys')
        ranked = sorted(own, key=lambda key: (numeric(predictions[key]['thickness_pred']), int(key[1])))
        if {key for key in risk_keys if key[0] == pid} != set(ranked[:math.ceil(len(own) * .1)]):
            raise ValueError('Saved risk ranking is stale')
        source_date = manifest['S1']['prior_source_date'] if case == 'S1' else None
        actual = {(row['id'], row['segment_id']): row for row in vtd
                  if row['id'] == pid and case == 'S1' and row['date'][:10] == source_date
                  and row['target_eligible'] == 'True'}
        points = []
        for key in sorted(own, key=lambda key: int(key[1])):
            pred = numeric(predictions[key]['thickness_pred'])
            original = diagnostic[key]
            if not math.isclose(pred, numeric(original[spec['model']]), abs_tol=5e-8, rel_tol=0.):
                raise ValueError('Saved production CSV and diagnostic predictions disagree')
            radius = bounds[key]
            if not math.isclose(pred, numeric(radius['thickness_pred']), abs_tol=1e-9, rel_tol=0.):
                raise ValueError('Intervals have different point predictions')
            lower, upper = numeric(radius['lower_mm']), numeric(radius['upper_mm'])
            if not 0 <= lower <= pred <= upper <= numeric(features[key]['nominal_thickness_mm']):
                raise ValueError('Invalid interval bounds')
            measured = actual.get(key)
            points.append({'segment_id': int(key[1]), 'position_m': numeric(grid[key]['start_m']),
                           'end_m': numeric(grid[key]['end_m']), 'prediction_mm': pred,
                           'lower_mm': lower, 'upper_mm': upper,
                           'actual_mm': numeric(measured['min_residual_thickness_mm']) if measured else None,
                           'actual_date': source_date if measured else None, 'is_risk': key in risk_keys})
        summaries = interval['cases'][case]
        pipes.append({'pipe_id': pid, 'label': case + ' · ' + spec['model'], 'model': spec['model'],
                      'case': spec['case'], 'prediction_date': spec['date'],
                      'length_m': max(point['end_m'] for point in points), 'segment_count': len(points),
                      'risk_count': len([point for point in points if point['is_risk']]),
                      'baseline_missing_count': len([point for point in points if point['actual_mm'] is None]),
                      'interval_summary': {'nominal_coverage': interval['nominal_coverage'],
                                           **summaries['nested_validation'],
                                           'radius_mm': summaries['calibration_radius_mm']}, 'points': points})
        for row in risk_source:
            key = row['id'], row['segment_id']
            if row['id'] != pid:
                continue
            if numeric(row['thickness_pred']) != numeric(predictions[key]['thickness_pred']):
                raise ValueError('Saved risk table has stale predictions')
            risk_rows.append({'pipe_id': pid, 'segment_id': int(row['segment_id']),
                              'position_m': numeric(row['start_m']),
                              'prediction_mm': numeric(row['thickness_pred']),
                              'lower_mm': numeric(bounds[key]['lower_mm']),
                              'upper_mm': numeric(bounds[key]['upper_mm']),
                              'rank': int(row['priority_rank']), 'risk_level': 'inspection_priority',
                              'criterion': 'Минимальная прогнозная толщина; при равенстве меньший segment_id'})
    scores = read_csv(validation / 'metrics.csv')
    groups, comparisons, validation_groups, headlines = [], [], [], []
    for scheme, label, case, base, description in SCHEMES:
        groups.append({'id': scheme, 'label': label, 'description': description})
        candidates = {row['method']: row for row in scores
                      if row['scheme'] == scheme and row['scope'] == 'pooled_replay_only'}
        selected = candidates['h18_h31']
        for method, source in [('baseline', candidates['baseline']), (CASES[case]['model'], selected)]:
            comparisons.append({'id': scheme + ':' + method, 'cohort': scheme,
                                'label': base if method == 'baseline' else method,
                                'final': method != 'baseline', 'metrics': metric(source)})
        if scheme != 'S2_LOSO_pooled':
            headlines.append({'case': case, 'model': CASES[case]['model'], 'scheme': scheme,
                              'metrics': metric(selected)})
        rows = read_csv(validation / scheme / 'validation_predictions.csv')
        if len(rows) != int(selected['n']):
            raise ValueError('Saved validation metric and row counts disagree')
        validation_groups.append({'id': scheme, 'label': label, 'model': CASES[case]['model'],
                                  'description': description, 'metrics': metric(selected),
                                  'has_intervals': False,
                                  'profiles': [{'pipe_id': row['id'], 'metrics': metric(row)} for row in scores
                                               if row['scheme'] == scheme and row['scope'] == 'profile'
                                               and row['method'] == 'h18_h31'],
                                  'rows': [{'pipe_id': row['id'], 'segment_id': int(row['segment_id']),
                                            'position_m': numeric(grid[row['id'], row['segment_id']]['start_m']),
                                            'date': row['prediction_date'][:10], 'actual_mm': numeric(row['next_mm']),
                                            'prediction_mm': numeric(row['candidate']),
                                            'lower_mm': None, 'upper_mm': None} for row in rows]})
    nested = read_csv(intervals_root / 'nested_validation.csv')
    for case, spec in CASES.items():
        summary = interval['cases'][case]
        rows = [row for row in nested if row['case'] == case]
        validation_groups.append({'id': case + '_intervals', 'label': case + ' · интервалы',
                                  'model': spec['model'], 'has_intervals': True,
                                  'description': 'Вложенное исключение простых участков: внешний участок не '
                                  'участвует ни в fit, ни во внутренней калибровке. Только даты до cutoff. '
                                  'Три участка одной родительской трубы; эмпирическое покрытие не гарантирует '
                                  '80 % на целевом прогнозе. Для S1 оно ниже заявленного уровня.',
                                  'metrics': metric(summary['nested_validation']),
                                  'profiles': [{'pipe_id': str(profile['id']), 'metrics': metric(profile)}
                                               for profile in summary['by_profile']],
                                  'rows': [{'pipe_id': row['id'], 'segment_id': int(row['segment_id']),
                                            'position_m': numeric(grid[row['id'], row['segment_id']]['start_m']),
                                            'date': row['prediction_date'][:10], 'actual_mm': numeric(row['next_mm']),
                                            'prediction_mm': numeric(row['prediction_mm']),
                                            'lower_mm': numeric(row['lower_mm']),
                                            'upper_mm': numeric(row['upper_mm'])} for row in rows]})
    level = f"{interval['nominal_coverage'] * 100:.0f}"
    measured_coverage = ', '.join(f"{case} {summary['nested_validation']['coverage'] * 100:.1f} %".replace('.', ',')
                                  for case, summary in interval['cases'].items())
    below = ', '.join(case for case, summary in interval['cases'].items()
                      if summary['nested_validation']['coverage'] < interval['nominal_coverage'])
    note = (f'Эмпирические {level}%-интервалы по абсолютным ошибкам исторических OOF-прогнозов. '
            f'Покрытие проверено вложенным исключением участков: {measured_coverage}. '
            'Это не P10/P90 квантильной модели и не гарантия покрытия на скрытом ВТД. '
            + (f'Для {below} заявленный уровень не подтверждён. ' if below else '')
            + 'Все участки калибровки связаны.')
    outputs = {
        'metrics': {**common, 'final_model': 'H18 / H31', 'model_status': 'H18 для S1; H31 только для S2. '
                    'Исследовательская оценка; скрытые ответы недоступны.',
                    'segment_count': len(predictions), 'total_length_m': sum(pipe['length_m'] for pipe in pipes),
                    'segment_size_m': 10, 'headlines': headlines, 'comparison_groups': groups,
                    'comparison_note': 'Сравнение только внутри одной схемы и одинаковых ключей. '
                    'Pearson² — квадрат корреляции, R² — отдельная метрика. '
                    'S1 и S2 не объединяются в общий балл. S2 на проверке 2018 года ухудшается.',
                    'comparison': comparisons,
                    'provenance': {'source_sha256': before, 'models_trained': False,
                                   'metrics_recomputed': False, 'intervals_tested_separately': True,
                                   'result_sha256': sha(forecast), 'bundle_sha256': manifest['bundle_sha256']}},
        'profile': {**common, 'actual_note': 'Прошлое ВТД S1 от ' + manifest['S1']['prior_source_date']
                    + '; только прямые измерения. 17 пропусков не заполнены фиктивным фактом. '
                    'У S2 собственная диагностика отсутствует.', 'pipes': pipes},
        'top_risk_segments': {**common, 'fraction': .1, 'criterion': 'Нижние 10 % прогнозов внутри каждой трубы',
                              'interval_note': note, 'rows': risk_rows},
        'validation': {**common, 'nominal_coverage': interval['nominal_coverage'],
                       'interval_method': note, 'groups': validation_groups},
        'feature_importance': {**common, 'available': False, 'method': None, 'models': [],
                               'description': 'SHAP и числовая важность H18/H31 не рассчитаны. '
                               'Список показывает использованные признаки, а не причинную силу влияния.',
                               'physical_model': 'H18 / H31', 'source': str(models.relative_to(ROOT)).replace('\\', '/') + '/manifest.json',
                               'known_factors': ['S1 · собственные прошлые толщины и соседние замеры',
                                                 'S1 · длина и ширина прошлых дефектов',
                                                 'S1 · распределение толщин и описание потери металла',
                                                 'S1 / S2 · положение и рельеф',
                                                 'S2 · возраст, температура, обводнённость и скорость']},
    }
    for path, expected in before.items():
        if sha(ROOT / path) != expected:
            raise ValueError('A source changed during export: ' + path)
    return outputs


def export(output=FRONTEND / 'public/data', data_root=ROOT / 'scratch/odo_alignment_verified', intervals_root=INTERVALS):
    output = Path(output).resolve()
    if not output.is_relative_to(FRONTEND.resolve()):
        raise ValueError('Export output must be inside frontend')
    outputs = build(data_root, intervals_root)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.export-', dir=output.parent) as temp:
        for name, document in outputs.items():
            path = Path(temp) / (name + '.json')
            path.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        for name in outputs:
            (Path(temp) / (name + '.json')).replace(output / (name + '.json'))
    print('Exported H18/H31 release:', outputs['metrics']['release_id'])
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=FRONTEND / 'public/data')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'scratch/odo_alignment_verified')
    parser.add_argument('--intervals-root', type=Path, default=INTERVALS)
    args = parser.parse_args()
    export(args.output, args.data_root, args.intervals_root)


if __name__ == '__main__':
    main()

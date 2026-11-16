"""CLI: python -m src.data_prep. Sources stay immutable; only verified builds publish."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'src'

import numpy as np
import pandas as pd
import pyarrow

from .cache import cache_key, code_snapshot
from .config import BASE_DIR, DATA_DIR, TECH_FILE
from .data_cleaner import QualityLog, clean_datasets
from .format_detector import read_and_standardize_all
from .grid_builder import build_segments_grid
from .leakage import audit_feature_table, audit_monitoring_use
from .lineage import build_lineage
from .odo_align import prepare_odometer_outputs
from .quality import build_quality_bundle
from .schema_registry import build_feature_inventory, feature_inventory_summary
from .segment_mapper import (map_vtd_to_segments, map_uzt_to_segments,
                             map_profile_to_segments, build_features_asof)
from .segment_quantiles import prepare_quantile_outputs
from .source_registry import CONTRACTS, DISPLAY_NAME, discover_sources
from .storage import (publish_tree, save_table, sha256, write_json,
                      write_spreadsheet_safe_csv)
from .validator import validate_and_summarize


def _git_info(root):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True)
        return {'revision': revision, 'working_tree_dirty': bool(status.strip()),
                'dirty_status_sha256': hashlib.sha256(status.encode('utf-8')).hexdigest(),
                'dirty_status_excerpt': status.splitlines()[:20]}
    except Exception as exc:
        return {'revision': None, 'working_tree_dirty': None, 'error': str(exc)}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode('utf-8')).hexdigest()


def required_source_types(requests):
    required = {'passport', 'profile', 'hydraulic_regime'}
    if any(r['case'] == 'between_vtd' for r in requests):
        required.add('vtd')
    return required


def _source_inventory_payload(discovery, source_summary):
    payload = []
    for item in discovery.inventory():
        source_type = item['source_type']
        contract = CONTRACTS.get(source_type)
        summary_match = next((info for info in source_summary.values()
                              if info.get('source_filename') == Path(item['path']).name), {})
        payload.append({**item, 'display_name': DISPLAY_NAME.get(source_type, source_type),
                        'required_fields': list(contract.required_fields) if contract else [],
                        'optional_fields': list(contract.optional_fields) if contract else [],
                        'record_count': summary_match.get('record_count'),
                        'record_count_summary': summary_match.get('record_count')})
    for source_type in discovery.missing_optional:
        payload.append({'source_type': source_type, 'status': 'missing_optional',
                        'display_name': DISPLAY_NAME.get(source_type, source_type),
                        'limitations': ['Optional source absent; missingness is explicit, no values invented.']})
    return {'version': 'source-inventory-v2', 'sources': payload,
            'warnings': discovery.warnings,
            'policy': 'Schema fingerprint decides meaning; filenames are hints. Conflicting copies fail closed.'}


def _schema_mapping_payload(discovery, source_summary, quality_bundle):
    sources = {}
    for filename, info in source_summary.items():
        source_type = info.get('source_type')
        contract = CONTRACTS.get(source_type)
        sources[filename] = {
            'source_type': source_type,
            'source_filename': info.get('source_filename'),
            'sha256': info.get('sha256'),
            'schema_version': contract.schema_version if contract else None,
            'adapter_version': contract.adapter_version if contract else None,
            'required_fields': list(contract.required_fields) if contract else [],
            'optional_fields': list(contract.optional_fields) if contract else [],
            'column_mapping': info.get('column_mapping', {}),
            'raw_headers': info.get('raw_headers', []),
            'duplicate_headers': info.get('duplicate_headers', []),
            'unknown_columns': quality_bundle['schema_drift'].get(filename, {}).get('unknown_columns', []),
            'missing_by_column': info.get('missing_by_column', {}),
            'status': info.get('status', 'parsed'),
        }
    return {'version': 'schema-mapping-v2', 'sources': sources,
            'contracts': {k: {'schema_version': v.schema_version,
                              'adapter_version': v.adapter_version,
                              'required_fields': list(v.required_fields),
                              'optional_fields': list(v.optional_fields)}
                          for k, v in CONTRACTS.items()}}


def _template_payload(discovery, target_grid):
    candidate = discovery.template_validation or {'status': 'no_candidate'}
    result = dict(candidate)
    result['derived_grid'] = {
        'status': 'derived_target_grid_validated',
        'rows': int(len(target_grid)),
        'unique_keys': bool(not target_grid.duplicated(['id', 'segment_id']).any()),
        'ids': sorted(int(x) for x in target_grid.id.unique()),
        'counts_by_id': {str(int(k)): int(v) for k, v in target_grid.groupby('id').size().items()},
    }
    if candidate.get('status') == 'authoritative_schema_match':
        template = pd.read_csv(candidate['path'], dtype={'id': 'int64', 'segment_id': 'int64'},
                               usecols=['id', 'segment_id'])
        actual = pd.MultiIndex.from_frame(template[['id', 'segment_id']])
        expected = pd.MultiIndex.from_frame(target_grid[['id', 'segment_id']])
        result['key_set_comparison'] = {
            'template_rows': int(len(actual)),
            'derived_rows': int(len(expected)),
            'template_minus_derived': int(len(actual.difference(expected))),
            'derived_minus_template': int(len(expected.difference(actual))),
            'exact': bool(len(actual) == len(expected) and not len(actual.difference(expected))),
            'validation_grid': 'authoritative_template',
        }
    else:
        result['key_set_comparison'] = {
            'validation_grid': 'derived_target_grid',
            'detail': ('No CSV with exact id,segment_id,thickness_pred schema was found. '
                       'The supplied sample_result.csv is not treated as a template by filename alone.'),
        }
    return result


def _validate_source_stability(discovery, source_summary):
    paths = list(discovery.by_legacy_key.values())
    for candidate in discovery.selected.values():
        paths.append(candidate.path)
    seen = set()
    for path in paths:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        expected = next((info.get('sha256') for info in source_summary.values()
                         if info.get('source_filename') == path.name), None)
        if expected is not None and sha256(path) != expected:
            raise RuntimeError(f'Source changed during execution: {path}')


def quality_report(cleaned, quarantine, tables, issues, validation, extra=None):
    extra = extra or {}
    lines = ['# Отчёт проверки подготовки данных', '',
             'Это подготовка данных, не обученная модель и не файл прогноза.', '',
             f"Проверки структуры и содержимого: {validation['checks_passed']} успешно.",
             f"Полная сетка: {validation['all_grid_rows']}; строки целевых признаков: {validation['target_rows']}.", '',
             '| Источник | Сохранено строк |', '|---|---:|']
    for name, df in cleaned.items():
        lines.append(f'| {name} | {len(df)} |')
    lines += ['', '| Карантин | Строк |', '|---|---:|']
    for name, df in quarantine.items():
        lines.append(f'| {name} | {len(df)} |')
    source_inventory = extra.get('source_inventory', {})
    if source_inventory:
        lines += ['', '## Источники: роль, хэш и схема', '',
                  '| Путь | Тип | Строк | SHA-256 (12) | Schema | Ограничение |', '|---|---|---:|---|---|---|']
        for item in source_inventory.get('sources', []):
            lines.append(f"| {Path(item.get('path','')).name} | {item.get('source_type')} | "
                         f"{item.get('record_count','')} | {str(item.get('sha256'))[:12]} | "
                         f"{item.get('schema_version','')} | {', '.join(item.get('limitations', []))} |")
    if extra.get('row_reconciliation'):
        lines += ['', '## Сверка строк (непересекающиеся disposition)', '',
                  '| Источник | Вход | Оставлено | Дубли | Карантин | Malformed | OK |', '|---|---:|---:|---:|---:|---:|---|']
        for row in extra['row_reconciliation']:
            lines.append(f"| {row['source_file']} | {row['input_records']} | {row['retained_records']} | "
                         f"{row['duplicate_records']} | {row['quarantined_records']} | {row['malformed_records']} | "
                         f"{row['disjoint_accounting_ok']} |")
    if extra.get('unit_resolution'):
        unit = extra['unit_resolution']
        lines += ['', '## Единицы', '', '| Правило | Поле | Строк | Статус |', '|---|---|---:|---|']
        for row in unit.get('conversions', []):
            lines.append(f"| {row['rule_id']} | {row['field']} | {row['rows']} | {row['status']} |")
        lines += ['', 'Неразрешённые единицы: ' + ', '.join(
            f"{x['rule_id']}={x['field']} ({x['status']})" for x in unit.get('unresolved', [])[:20])]
    if extra.get('monitoring_join'):
        m = extra['monitoring_join']
        lines += ['', '## Мониторинг/физико-химия', '',
                  f"Строк: {m.get('rows')}; неизвестный ID: {m.get('unknown_id_rows')}; "
                  f"конфликт направления: {m.get('direction_conflict_rows')}; "
                  f"допустимо к присоединению: {m.get('mapping_eligible_rows')}; "
                  f"наблюдений: {m.get('observation_eligible_rows')}."]
        lines.append('Политика: ' + m.get('policy', ''))
    if extra.get('template_validation'):
        t = extra['template_validation']
        lines += ['', '## Шаблон результата', '',
                  f"Статус: {t.get('status')}; grid: {t.get('key_set_comparison', {}).get('validation_grid')}; "
                  f"exact={t.get('key_set_comparison', {}).get('exact')}."]
    if extra.get('leakage_audit'):
        l = extra['leakage_audit']
        lines += ['', '## Утечка/сроки', '', f"Passed: {l.get('passed')}; failures: {l.get('failures')}"]
    if extra.get('feature_inventory_summary'):
        s = extra['feature_inventory_summary']
        lines += ['', '## Инвентарь полей', '', f"Rows: {s.get('rows')}; by status: {s.get('by_status')}"]
    lines += ['', '## Остаточные ограничения', '',
              '- УЗТ выше номинала сохранены, но исключены из признаков до подтверждения методики.',
              '- ВТД до ввода сохранены как наблюдения; такие метки не допускаются в training_targets.',
              '- Спорные связи физико-химии не исправлены догадкой; переноса медиан между трубами нет.',
              '- Пороговые значения гидравлики — параметры проверки качества, не пределы безопасной эксплуатации.',
              '- Устойчивые три календарных дня подозрительные значения сохранены с флагом. Это не подтверждение их физической достоверности.',
              '- Отсечение строго перед 00:00 даты прогноза; запись самого дня T не используется.',
              '- Статические диаметр, номинал, ввод и профиль принимаются из паспорта/профиля. История замен не предоставлена.',
              '- Поздние диагностические сведения паспорта и расчётная pit_corrosion_intensity не включаются в признаки.',
              '- Архивные таблицы содержат всю историю. Для каждого обучающего фолда требуется отдельный вызов build_features_asof.',
              '- Температурный сценарий не находится в target_features и не является измерением.',
              '- Метрики модели и result.csv строятся на этапе моделирования; нулевые независимые forward-фолды не считаются успешной валидацией.', '',
              '## Покрытие признаков по целевым датам', '',
              '| ID | Дата | Сегментов | Дней режима | Последняя дата режима | Сегментов с прошлым ВТД | Записей физико-химии |', '|---|---|---:|---:|---|---:|---:|']
    for (pid, date), group in tables['target_features'].groupby(['id','prediction_date']):
        row = group.iloc[0]
        lines.append(f'| {pid} | {date.date()} | {len(group)} | {row.hydraulic_days_count} | {row.hydraulic_source_date_max} | {int(group.has_previous_vtd.sum())} | {row.monitoring_records_count} |')
    lines += ['', '## Срабатывания правил (не число независимых ошибок)', '', '| Правило | Строк |', '|---|---:|']
    for rule, count in issues.rule.value_counts().sort_index().items():
        lines.append(f'| {rule} | {count} |')
    lines += ['', 'Полные примеры и действия: processed/quality_issues.csv, processed/rejected_records.* и processed/hydraulic_quality.csv.',
              'Машиночитаемые отчёты: reports/source_inventory.json, reports/schema_mapping.json, '
              'reports/feature_inventory.csv, reports/leakage_audit.json, reports/run_manifest.json.', '']
    return '\n'.join(lines)


def run_pipeline(data_dir=DATA_DIR, output_root=BASE_DIR, config_path=None, requests=None,
                 source_overrides=None, cache_dir=None, odometer_alignment_plan=None,
                 vtd_quantiles=False):
    started = time.perf_counter()
    tracemalloc.start()
    stage_times = {}
    def stage(name, since):
        now = time.perf_counter()
        stage_times[name] = round(now - since, 4)
        return now

    config_path = Path(config_path or BASE_DIR/'configs/preprocessing.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if odometer_alignment_plan is not None:
        config['vtd_odometer_alignment'] = json.loads(
            Path(odometer_alignment_plan).read_text(encoding='utf-8'))
    if vtd_quantiles:
        config['vtd_quantiles'] = True
    if config['cutoff_policy'] != 'strictly_before_prediction_date' or config['persistent_days'] < 1:
        raise ValueError('Unsupported cutoff or persistence configuration')
    requests = config['targets'] if requests is None else requests
    keys = [(int(r['id']), str(r['prediction_date'])) for r in requests]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate prediction requests')
    data_dir, output_root = Path(data_dir).resolve(), Path(output_root).resolve()
    overrides = source_overrides or config.get('source_overrides')
    print('1/6 Classifying sources by schema, then parsing loss-aware...', flush=True)
    t = time.perf_counter()
    discovery = discover_sources(data_dir, required_types=required_source_types(requests),
                                 optional_types=('uzt', 'corrosion_monitoring'),
                                 source_overrides=overrides)
    raw, source_summary, parse_issues = read_and_standardize_all(
        data_dir, discovery, locale=config.get('date_locale'),
        decimal_by_source=config.get('decimal_by_source'))
    log = QualityLog(parse_issues)
    cleaned, quarantine = clean_datasets(raw, config, log)
    t = stage('parse_clean', t)
    print('2/6 Constructing complete grids and observed thickness histories...', flush=True)
    grid = build_segments_grid(cleaned['passport.csv'])
    uzt = map_uzt_to_segments(cleaned['uzt.csv'], grid)
    profile = map_profile_to_segments(cleaned['profile.csv'], grid)
    alignment_tables, alignment_report = ({}, None)
    if config.get('vtd_odometer_alignment'):
        alignment_tables, alignment_report = prepare_odometer_outputs(
            cleaned, grid, profile, uzt, requests, config)
    vtd = map_vtd_to_segments(cleaned['vtd.csv'], grid)
    quantile_tables = (prepare_quantile_outputs(cleaned, grid, vtd, requests,
                                                alignment_tables, alignment_report)
                       if config.get('vtd_quantiles') else {})
    t = stage('grid_and_measurements', t)
    print('3/6 Building cutoff-specific features and calendar-day filters...', flush=True)
    tables = build_features_asof(cleaned, grid, profile, vtd, uzt, requests, config)
    tables.update({'segments_grid': grid, 'vtd_by_segment': vtd, 'uzt_by_segment': uzt,
                   'profile_by_segment': profile, 'training_targets': vtd[vtd.target_eligible].copy()})
    target_grid = tables['target_features'][['id','segment_id','start_m','end_m','length_m','prediction_date','case']].copy()
    tables['target_segments_grid'] = target_grid
    issues = log.frame()
    tables['quality_issues'] = issues
    t = stage('features', t)
    print('4/6 Building inventories, lineage, quality and leakage audits...', flush=True)
    quality = build_quality_bundle(cleaned, source_summary, quarantine, issues, grid, tables, discovery)
    quality['unit_resolution']['config_version'] = config.get('unit_policy_version', 'unit-policy-v1')
    feature_inventory = build_feature_inventory(cleaned, source_summary, tables['target_features'])
    tables['rejected_records'] = quality['rejected_records']
    template_validation = _template_payload(discovery, target_grid)
    leakage = audit_feature_table(tables['target_features'], target_ids=[int(r['id']) for r in requests],
                                  cold_start_id=next((int(r['id']) for r in requests
                                                      if r['case'] == 'cold_start'), None))
    leakage['monitoring_use'] = audit_monitoring_use(tables['target_features'], quality['monitoring_join'])
    leakage['passed'] = bool(leakage.get('passed') and leakage['monitoring_use'].get('passed'))
    lineage_report, lineage_keys = build_lineage(cleaned, source_summary, tables, config, grid)
    source_inventory = _source_inventory_payload(discovery, source_summary)
    schema_mapping = _schema_mapping_payload(discovery, source_summary, quality)
    t = stage('audits', t)
    print('5/6 Validating measurements, boundaries, cutoffs and coverage...', flush=True)
    summary, validation = validate_and_summarize(grid, tables, cleaned, requests, template_validation)
    tables['pipes_summary'] = summary
    _validate_source_stability(discovery, source_summary)
    t = stage('validation', t)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prep-build-', dir=output_root) as temp:
        stage_dir = Path(temp)
        code_hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        selected_paths = {key: str(path) for key, path in discovery.by_legacy_key.items()}
        key = cache_key(selected_paths, config, requests, schema_version='schema-v2',
                        rule_version='rules-v2', code_hashes=code_snapshot(BASE_DIR))
        manifest = {
            'run_id': key[:20],
            'status': 'validated_then_published',
            'inputs': source_summary,
            'source_snapshot': source_inventory,
            'source_hashes': {name: info.get('sha256') for name, info in source_summary.items()},
            'config': config,
            'config_hash': _json_hash(config),
            'schema_hash': _json_hash(schema_mapping.get('contracts', {})),
            'prediction_requests': requests,
            'scenario_cutoffs': {str(r['id']): str(r['prediction_date']) for r in requests},
            'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                         'numpy': np.__version__, 'pyarrow': pyarrow.__version__},
            'code_sha256': code_hashes,
            'git': _git_info(BASE_DIR),
            'source_snapshot_policy': 'one consistent source hash snapshot per run; changed files fail the run',
            'row_accounting': quality['row_reconciliation'],
            'quality_summary': {
                'schema_drift_unknown_columns': {k: v.get('unknown_column_count', 0)
                                                 for k, v in quality['schema_drift'].items()},
                'monitoring_join': quality['monitoring_join'],
                'feature_inventory': feature_inventory_summary(feature_inventory),
                'template_status': template_validation.get('status'),
                'leakage_passed': leakage.get('passed'),
            },
            'rule_versions': {'schema': 'schema-v2', 'rules': 'rules-v2',
                              'cutoff': config['cutoff_policy'],
                              'vtd_nominal_tolerance_mm': config['vtd_thickness_tolerance_mm'],
                              'persistence_days': config['persistent_days']},
            'cache_key': key,
            'cache_reused': False,
            'warnings': list(discovery.warnings),
            'tests': {'status': 'not_run_in_this_command',
                      'note': 'Run unittest separately; the manifest does not claim unrun tests.'},
            'outputs': {},
        }
        print('6/6 Writing, round-trip checking and failure-safe publishing...', flush=True)
        t_write = time.perf_counter()
        for name, df in cleaned.items():
            relative = Path('data/cleaned')/Path(name).stem
            manifest['outputs'][relative.as_posix()] = save_table(df, stage_dir/relative)
        for name, df in {**tables, **quarantine, **alignment_tables, **quantile_tables}.items():
            relative = Path('processed')/name
            manifest['outputs'][relative.as_posix()] = save_table(df, stage_dir/relative)
        manifest['outputs']['reports/feature_inventory'] = save_table(
            feature_inventory, stage_dir/'reports/feature_inventory')
        manifest['outputs']['processed/lineage_keys'] = save_table(
            lineage_keys, stage_dir/'processed/lineage_keys')
        config_stage = stage_dir/'configs/preprocessing.json'
        config_stage.parent.mkdir(parents=True, exist_ok=True)
        config_stage.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n',
                                encoding='utf-8')
        write_json(stage_dir/'processed/step1_column_mappings.json',
                   {k: v['column_mapping'] for k, v in source_summary.items()})
        audit = {name: {'input_rows': len(raw[name]), 'cleaned_rows': len(df),
                        'flag_counts': {c: int(df[c].sum()) for c in df
                                        if c.endswith('_flag') and pd.api.types.is_bool_dtype(df[c])}}
                 for name, df in cleaned.items()}
        write_json(stage_dir/'processed/step2_cleaning_audit.json', audit)
        write_json(stage_dir/'reports/source_inventory.json', source_inventory)
        write_json(stage_dir/'reports/schema_mapping.json', schema_mapping)
        write_json(stage_dir/'reports/unit_resolution.json', quality['unit_resolution'])
        write_json(stage_dir/'reports/row_reconciliation.json',
                   {'sources': quality['row_reconciliation']})
        write_json(stage_dir/'reports/join_audit.json', quality['monitoring_join'])
        write_json(stage_dir/'reports/date_coverage.json', quality['date_coverage'])
        write_json(stage_dir/'reports/spatial_coverage.json', quality['spatial_coverage'])
        write_json(stage_dir/'reports/vtd_campaigns.json', quality['vtd_campaigns'])
        if alignment_report is not None:
            write_json(stage_dir/'reports/odometer_alignment.json', alignment_report)
        write_json(stage_dir/'reports/feature_coverage.json', quality['feature_coverage'])
        write_json(stage_dir/'reports/persistence_filter.json', quality['persistence_filter'])
        write_json(stage_dir/'reports/leakage_audit.json', leakage)
        write_json(stage_dir/'reports/template_validation.json', template_validation)
        write_json(stage_dir/'reports/lineage.json', lineage_report)
        write_json(stage_dir/'reports/data_quality.json', {
            'source_inventory': source_inventory,
            'schema_drift': quality['schema_drift'],
            'row_reconciliation': quality['row_reconciliation'],
            'date_coverage': quality['date_coverage'],
            'spatial_coverage': quality['spatial_coverage'],
            'vtd_campaigns': quality['vtd_campaigns'],
            'feature_coverage': quality['feature_coverage'],
            'persistence_filter': quality['persistence_filter'],
            'feature_inventory_summary': feature_inventory_summary(feature_inventory),
        })
        write_spreadsheet_safe_csv(issues, stage_dir/'reports/spreadsheet_safe_quality_issues.csv')
        validation['csv_parquet_pairs_verified'] = len(manifest['outputs'])
        validation['template_validation'] = template_validation.get('status')
        write_json(stage_dir/'reports/validation.json', validation)
        (stage_dir/'reports/data_quality.md').write_text(
            quality_report(cleaned, quarantine, tables, issues, validation, {
                'source_inventory': source_inventory,
                'row_reconciliation': quality['row_reconciliation'],
                'unit_resolution': quality['unit_resolution'],
                'monitoring_join': quality['monitoring_join'],
                'template_validation': template_validation,
                'leakage_audit': leakage,
                'feature_inventory_summary': feature_inventory_summary(feature_inventory),
            }), encoding='utf-8')
        stage_times['write_and_roundtrip'] = round(time.perf_counter() - t_write, 4)
        # Artifact hashes include every published file; manifest is deliberately last.
        artifact_hashes = {}
        for path in sorted(stage_dir.rglob('*')):
            if path.is_file():
                artifact_hashes[path.relative_to(stage_dir).as_posix()] = sha256(path)
        manifest['artifact_hashes'] = artifact_hashes
        manifest['runtime_seconds'] = {**stage_times,
                                       'total_before_publish': round(time.perf_counter() - started, 4)}
        manifest['peak_traced_memory_mb'] = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 3)
        manifest['warnings'].append(f'peak_traced_memory_mb={manifest["peak_traced_memory_mb"]}')
        write_json(stage_dir/'reports/run_manifest.json', manifest)
        publish_tree(stage_dir, output_root)
    tracemalloc.stop()
    if cache_dir is not None:
        from .cache import RunCache
        RunCache(cache_dir).record(key, {'output_root': str(output_root),
                                         'manifest': 'reports/run_manifest.json'})
    took = round(time.perf_counter() - started, 4)
    print(f"PASS: {validation['checks_passed']} checks, {validation['csv_parquet_pairs_verified']} CSV/Parquet pairs; "
          f"{len(target_grid)} target segments; {took}s.", flush=True)
    return {'cleaned': cleaned, 'quarantine': quarantine, 'grid': grid,
            'tables': {**tables, **alignment_tables, **quantile_tables},
            'validation': validation, 'source_summary': source_summary,
            'source_inventory': source_inventory, 'feature_inventory': feature_inventory,
            'quality': quality, 'leakage': leakage, 'lineage': lineage_report,
            'template_validation': template_validation, 'manifest': manifest}


def inspect_sources(data_dir=DATA_DIR):
    discovery = discover_sources(data_dir, required_types=('passport', 'profile'),
                                 optional_types=('vtd', 'uzt', 'corrosion_monitoring', 'hydraulic_regime'))
    return {'sources': discovery.inventory(), 'warnings': discovery.warnings,
            'template_validation': discovery.template_validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=BASE_DIR)
    parser.add_argument('--config', type=Path, default=BASE_DIR/'configs/preprocessing.json')
    parser.add_argument('--as-of', help='Historical snapshot cutoff YYYY-MM-DD; excludes that day')
    parser.add_argument('--ids', type=int, nargs='+')
    parser.add_argument('--case', choices=['between_vtd','cold_start'], default='between_vtd')
    parser.add_argument('--inspect-only', action='store_true', help='Print source classification without building')
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--odometer-alignment-plan', type=Path, help='JSON list of explicit VTD run pairs')
    parser.add_argument('--vtd-quantiles', action='store_true', help='Publish VTD segment states and ablation')
    args = parser.parse_args()
    if args.inspect_only:
        print(json.dumps(inspect_sources(args.data_dir), ensure_ascii=False, indent=2, default=str))
        return
    if bool(args.as_of) != bool(args.ids):
        parser.error('--as-of and --ids must be supplied together')
    requests = None if not args.as_of else [dict(id=i, prediction_date=args.as_of, case=args.case) for i in args.ids]
    run_pipeline(args.data_dir, args.output_root, args.config, requests, cache_dir=args.cache_dir,
                 odometer_alignment_plan=args.odometer_alignment_plan,
                 vtd_quantiles=args.vtd_quantiles)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()

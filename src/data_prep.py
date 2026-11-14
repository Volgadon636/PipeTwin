"""CLI: python -m src.data_prep. Sources stay immutable; only verified builds publish."""
import argparse
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'src'

import numpy as np
import pandas as pd
import pyarrow

from .config import BASE_DIR, DATA_DIR, TECH_FILE
from .data_cleaner import QualityLog, clean_datasets
from .format_detector import read_and_standardize_all, resolve_sources
from .grid_builder import build_segments_grid
from .segment_mapper import (map_vtd_to_segments, map_uzt_to_segments,
                             map_profile_to_segments, build_features_asof)
from .storage import write_json, save_table, sha256, publish_file
from .validator import validate_and_summarize


def quality_report(cleaned, quarantine, tables, issues, validation):
    lines = ['# Отчёт проверки подготовки данных', '',
             'Это подготовка данных, не обученная модель и не файл прогноза.', '',
             f"Проверки структуры и содержимого: {validation['checks_passed']} успешно.",
             f"Полная сетка: {validation['all_grid_rows']}; строки целевых признаков: {validation['target_rows']}.", '',
             '| Источник | Сохранено строк |', '|---|---:|']
    for name, df in cleaned.items():
        lines.append(f'| {name} | {len(df)} |')
    lines += ['', '| Карантин координат | Строк |', '|---|---:|']
    for name, df in quarantine.items():
        lines.append(f'| {name} | {len(df)} |')
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
              '- Метрики модели и result.csv появятся на этапе моделирования.', '',
              '## Покрытие признаков по целевым датам', '',
              '| ID | Дата | Сегментов | Дней режима | Последняя дата режима | Сегментов с прошлым ВТД | Записей физико-химии |', '|---|---|---:|---:|---|---:|---:|']
    for (pid, date), group in tables['target_features'].groupby(['id','prediction_date']):
        row = group.iloc[0]
        lines.append(f'| {pid} | {date.date()} | {len(group)} | {row.hydraulic_days_count} | {row.hydraulic_source_date_max} | {int(group.has_previous_vtd.sum())} | {row.monitoring_records_count} |')
    lines += ['', '## Срабатывания правил (не число независимых ошибок)', '', '| Правило | Строк |', '|---|---:|']
    for rule, count in issues.rule.value_counts().sort_index().items():
        lines.append(f'| {rule} | {count} |')
    lines += ['', 'Полные примеры и действия: processed/quality_issues.csv и processed/hydraulic_quality.csv.',
              'Контрольные суммы, версии и проверка CSV/Parquet: reports/run_manifest.json.', '']
    return '\n'.join(lines)


def run_pipeline(data_dir=DATA_DIR, output_root=BASE_DIR, config_path=None, requests=None):
    config_path = Path(config_path or BASE_DIR/'configs/preprocessing.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['cutoff_policy'] != 'strictly_before_prediction_date' or config['persistent_days'] < 1:
        raise ValueError('Unsupported cutoff or persistence configuration')
    requests = config['targets'] if requests is None else requests
    keys = [(int(r['id']), str(r['prediction_date'])) for r in requests]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate prediction requests')
    data_dir, output_root = Path(data_dir).resolve(), Path(output_root).resolve()
    print('1/5 Reading and checking source schemas...', flush=True)
    raw, source_summary, parse_issues = read_and_standardize_all(data_dir)
    log = QualityLog(parse_issues)
    cleaned, quarantine = clean_datasets(raw, config, log)
    print('2/5 Constructing complete grids and observed thickness histories...', flush=True)
    grid = build_segments_grid(cleaned['passport.csv'])
    vtd = map_vtd_to_segments(cleaned['vtd.csv'], grid)
    uzt = map_uzt_to_segments(cleaned['uzt.csv'], grid)
    profile = map_profile_to_segments(cleaned['profile.csv'], grid)
    print('3/5 Building cutoff-specific features and calendar-day filters...', flush=True)
    tables = build_features_asof(cleaned, grid, profile, vtd, uzt, requests, config)
    tables.update({'segments_grid':grid, 'vtd_by_segment':vtd, 'uzt_by_segment':uzt,
                   'profile_by_segment':profile, 'training_targets':vtd[vtd.target_eligible].copy()})
    target_grid = tables['target_features'][['id','segment_id','start_m','end_m','length_m','prediction_date','case']].copy()
    tables['target_segments_grid'] = target_grid
    issues = log.frame()
    tables['quality_issues'] = issues
    print('4/5 Validating measurements, boundaries, cutoffs and coverage...', flush=True)
    summary, validation = validate_and_summarize(grid, tables, cleaned, requests)
    tables['pipes_summary'] = summary
    for name, path in resolve_sources(data_dir).items():
        if sha256(path) != source_summary[name]['sha256']:
            raise RuntimeError(f'Source changed during execution: {path}')

    # Build all files and check both serializations before replacing any published output.
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prep-build-', dir=output_root) as temp:
        stage = Path(temp)
        manifest = {'inputs':source_summary, 'config':config, 'prediction_requests':requests,
                    'versions':{'python':platform.python_version(), 'pandas':pd.__version__,
                                'numpy':np.__version__, 'pyarrow':pyarrow.__version__},
                    'code_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                    'outputs':{}}
        print('5/5 Writing and verifying every CSV/Parquet pair...', flush=True)
        for name, df in cleaned.items():
            relative = Path('data/cleaned')/Path(name).stem
            manifest['outputs'][relative.as_posix()] = save_table(df, stage/relative)
        for name, df in {**tables, **quarantine}.items():
            relative = Path('processed')/name
            manifest['outputs'][relative.as_posix()] = save_table(df, stage/relative)
        write_json(stage/'processed/step1_column_mappings.json', {k:v['column_mapping'] for k,v in source_summary.items()})
        audit = {name:{'input_rows':len(raw[name]), 'cleaned_rows':len(df),
                       'flag_counts':{c:int(df[c].sum()) for c in df if c.endswith('_flag') and pd.api.types.is_bool_dtype(df[c])}}
                 for name,df in cleaned.items()}
        write_json(stage/'processed/step2_cleaning_audit.json', audit)
        validation['csv_parquet_pairs_verified'] = len(manifest['outputs'])
        write_json(stage/'reports/validation.json', validation)
        write_json(stage/'reports/run_manifest.json', manifest)
        (stage/'reports/data_quality.md').write_text(quality_report(cleaned,quarantine,tables,issues,validation),encoding='utf-8')
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                dest = output_root/path.relative_to(stage)
                if not dest.resolve().is_relative_to(output_root):
                    raise ValueError(f'Output escapes destination root: {dest}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                publish_file(path, dest)
    print(f"PASS: {validation['checks_passed']} checks, {validation['csv_parquet_pairs_verified']} CSV/Parquet pairs; {len(target_grid)} target segments.", flush=True)
    return {'cleaned':cleaned, 'quarantine':quarantine, 'grid':grid, 'tables':tables,
            'validation':validation, 'source_summary':source_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=BASE_DIR)
    parser.add_argument('--config', type=Path, default=BASE_DIR/'configs/preprocessing.json')
    parser.add_argument('--as-of', help='Historical snapshot cutoff YYYY-MM-DD; excludes that day')
    parser.add_argument('--ids', type=int, nargs='+')
    parser.add_argument('--case', choices=['between_vtd','cold_start'], default='between_vtd')
    args = parser.parse_args()
    if bool(args.as_of) != bool(args.ids):
        parser.error('--as-of and --ids must be supplied together')
    requests = None if not args.as_of else [dict(id=i,prediction_date=args.as_of,case=args.case) for i in args.ids]
    run_pipeline(args.data_dir, args.output_root, args.config, requests)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()

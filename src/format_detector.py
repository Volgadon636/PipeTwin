"""Explicit source schemas, original filename aliases and loss-aware parsing.

Backwards-compatible helpers (``detect_file_format``, ``build_column_mapping``,
``robust_parse_dates``, ``parse_numeric``, ``resolve_sources``) remain.  New code
uses :mod:`src.parsing` and :mod:`src.source_registry` for source semantics.
"""
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (FILE_ALIASES, CANONICAL_COL_MAPPINGS, CYRILLIC_TO_LATIN,
                     NUMERIC_COLUMNS, DATE_COLUMNS, REQUIRED_COLUMNS)
from .parsing import (ParseError, detect_dialect, parse_date_series,
                      parse_identifier_series, parse_numeric_series,
                      parse_number_text, read_raw_table)
from .source_registry import DiscoveryResult, LEGACY_KEY


def to_snake_case(text):
    return re.sub(r'_+', '_', ''.join(CYRILLIC_TO_LATIN.get(c, c if c.isalnum() else '_')
                                    for c in text.strip().lower())).strip('_') or 'col'


def detect_file_format(path):
    dialect = detect_dialect(path)
    return dialect.encoding, dialect.delimiter


def build_column_mapping(filename, columns):
    canonical = CANONICAL_COL_MAPPINGS[filename]
    mapping, used = {}, set()
    for original in columns:
        name = canonical.get(original, to_snake_case(original))
        if name in used:
            if original in canonical:
                raise ValueError(f'{filename}: canonical column collision: {original} -> {name}')
            # Archival fields such as impact strength at +20/-20 must both survive.
            name += '_' + hashlib.sha256(original.encode('utf-8')).hexdigest()[:8]
        mapping[original] = name
        used.add(name)
    return mapping


def robust_parse_dates(series):
    # Dates are parsed per value, never inferred from the majority of a column.
    return parse_date_series(series.astype('string').str.strip())


def parse_numeric(series):
    return parse_numeric_series(series.astype('string'))[0]


def resolve_sources(data_dir):
    """Legacy filename-only resolver kept for existing callers and tests.

    The production pipeline uses :func:`src.source_registry.discover_sources`.
    """
    resolved = {}
    for name, aliases in FILE_ALIASES.items():
        found = [Path(data_dir) / alias for alias in aliases if (Path(data_dir) / alias).is_file()]
        if not found:
            raise FileNotFoundError(f'Missing source {name}; accepted names: {aliases}')
        # Reject ambiguous copies, allowing identical files with different line endings.
        content = [p.read_bytes().replace(b'\r\n', b'\n') for p in found]
        if any(c != content[0] for c in content[1:]):
            raise ValueError(f'Conflicting aliases for {name}: {found}')
        resolved[name] = found[0]
    return resolved


def _empty_canonical_frame(name):
    """Create a typed empty frame for an optional source that was not supplied."""
    fields = []
    for raw, canonical in CANONICAL_COL_MAPPINGS[name].items():
        if canonical not in fields:
            fields.append(canonical)
    for field in REQUIRED_COLUMNS[name]:
        if field not in fields:
            fields.append(field)
    frame = pd.DataFrame({field: pd.Series(dtype='string') for field in fields})
    for field in DATE_COLUMNS[name]:
        frame[field] = pd.Series(dtype='datetime64[ns]')
    for field in NUMERIC_COLUMNS[name]:
        if field == 'id':
            frame[field] = pd.Series(dtype='Int64')
        else:
            frame[field] = pd.Series(dtype='float64')
    frame.insert(0, 'source_row', pd.Series(dtype='int64'))
    return frame


def _resolve_input_paths(data_dir, sources):
    if sources is None:
        return resolve_sources(data_dir), None
    if isinstance(sources, DiscoveryResult):
        discovery = sources
    elif isinstance(sources, dict):
        discovery = None
        # A mapping may use legacy keys or semantic source names.
        paths = {}
        for key, value in sources.items():
            legacy = LEGACY_KEY.get(key, key)
            paths[legacy] = Path(value)
    else:
        raise TypeError('sources must be a DiscoveryResult, a mapping, or None')
    if discovery is not None:
        paths = dict(discovery.by_legacy_key)
    return paths, discovery


def read_and_standardize_all(data_dir, sources=None, *, locale=None,
                             decimal_by_source=None):
    """Read supplied sources with explicit schemas and row-level provenance.

    Missing optional sources become typed empty frames when a ``DiscoveryResult``
    is passed.  Missing required sources still fail.  Raw source files are never
    modified; parse failures/censored values are recorded and masked, not guessed.
    """
    datasets, summary, issues = {}, {}, []
    paths, discovery = _resolve_input_paths(data_dir, sources)
    decimal_by_source = decimal_by_source or {}
    for name in CANONICAL_COL_MAPPINGS:
        path = paths.get(name)
        if path is None:
            # Only optional sources may be absent; the dynamic caller decides which
            # ones are required. A completely absent input here is explicit.
            frame = _empty_canonical_frame(name)
            datasets[name] = frame
            summary[name] = {
                'source_filename': None, 'sha256': None, 'rows': 0,
                'columns': len([c for c in frame.columns if c not in ('source_row',)]),
                'encoding': None, 'delimiter': None, 'column_mapping': {},
                'missing_by_column': {}, 'status': 'missing_optional',
                'record_count': 0, 'malformed_records': 0,
            }
            continue
        path = Path(path)
        table = read_raw_table(path)
        mapping = build_column_mapping(name, table.unique_headers)
        # Preserve raw duplicate headers as distinct archival columns and report
        # them before pandas can silently mangle names.
        for duplicate in table.duplicate_headers:
            issues.append({'file': name, 'source_row': 1, 'field': duplicate,
                           'rule': 'duplicate_raw_header', 'action': 'rename_archival',
                           'original_value': duplicate})
        for index in table.empty_headers:
            issues.append({'file': name, 'source_row': 1, 'field': f'column_{index + 1}',
                           'rule': 'empty_raw_header', 'action': 'rename_archival',
                           'original_value': ''})
        for bad in table.malformed:
            issues.append({'file': name, 'source_row': int(bad.get('record_number', 0)) + 1,
                           'field': '', 'rule': bad.get('error_code', 'malformed_record'),
                           'action': 'quarantine_record',
                           'original_value': bad.get('original_content', '')})
        raw = pd.DataFrame(table.rows, columns=table.unique_headers, dtype='string')
        df = raw.rename(columns=mapping).copy()
        missing = set(REQUIRED_COLUMNS[name]) - set(df.columns)
        if missing:
            if path is not None and summary.get(name, {}).get('status') == 'missing_optional':
                pass
            raise ValueError(f'{path.name}: missing required columns {sorted(missing)}')
        df.insert(0, 'source_row', np.asarray(table.record_numbers, dtype='int64') + 1)
        df.insert(1, 'source_record_number', np.asarray(table.record_numbers, dtype='int64'))
        df.insert(2, 'source_physical_line', np.asarray(table.physical_lines, dtype='int64'))

        # Identifier parsing is exact and happens before any float conversion.
        id_issues, id_leading = [], pd.Series(False, index=df.index)
        if 'id' in df.columns:
            df['id_source_text'] = df['id'].astype('string')
            parsed_id, id_issues, id_leading = parse_identifier_series(df['id'])
            df['id'] = parsed_id
            df['id_leading_zero_flag'] = id_leading
            for item in id_issues:
                if item['state'] in ('absent', 'not_applicable'):
                    continue
                issues.append({'file': name, 'source_row': int(df.at[item['index'], 'source_row']),
                               'field': 'id', 'rule': item['reason'], 'action': 'reject_record',
                               'original_value': item['original_value']})
            absent_markers = {'', '#n/a', 'n/a', 'na', 'nan', 'null', 'none', 'нет данных', 'нет',
                              '-', '—', '–', 'не применимо', 'not applicable'}
            invalid = (df['id'].isna() & df['id_source_text'].notna()
                       & ~df['id_source_text'].str.strip().str.lower().isin(absent_markers))
            if invalid.any():
                examples = df.loc[invalid, ['source_row', 'id_source_text']].head(5).to_dict('records')
                raise ValueError(f'{name}: invalid/non-integer IDs; examples: {examples}')

        for col in NUMERIC_COLUMNS[name]:
            if col == 'id' or col not in df.columns:
                continue
            before = df[col].copy()
            parsed, numeric_issues = parse_numeric_series(
                before, decimal=decimal_by_source.get(name),
                thousands=None)
            if numeric_issues:
                df[col + '_source_text'] = before
                for item in numeric_issues:
                    issues.append({
                        'file': name, 'source_row': int(df.at[item['index'], 'source_row']),
                        'field': col, 'rule': item['reason'] if item['state'] == 'parse_failed' else item['state'],
                        'action': ('set_missing' if item['state'] == 'parse_failed' else
                                   ('keep_missing_censored' if item['state'] == 'censored' else 'keep_not_applicable')),
                        'original_value': item['original_value']})
            df[col] = parsed

        for col in DATE_COLUMNS[name]:
            if col not in df.columns:
                continue
            before = df[col].copy()
            parsed, date_only, tz_offset, date_issues = parse_date_series(
                before, locale=locale, with_metadata=True)
            if date_issues:
                df[col + '_source_text'] = before
                for item in date_issues:
                    if item['state'] in ('absent', 'not_applicable'):
                        continue
                    issues.append({
                        'file': name, 'source_row': int(df.at[item['index'], 'source_row']),
                        'field': col, 'rule': item['reason'],
                        'action': ('set_missing' if item['state'] in ('parse_failed', 'ambiguous_date_locale') else 'keep_missing'),
                        'original_value': item['original_value']})
            df[col] = parsed
            # Preserve whether the source was date-only and any declared offset.
            df[col + '_date_only'] = date_only
            df[col + '_tz_offset'] = tz_offset

        if 'id' in df.columns:
            df['id'] = df['id'].astype('Int64')
        datasets[name] = df
        summary[name] = {
            'source_filename': path.name,
            'source_type': None if discovery is None else next(
                (st for st, cand in discovery.selected.items() if cand.path == path), None),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows': len(df), 'columns': len(table.unique_headers) + 3,
            'encoding': table.encoding, 'delimiter': table.delimiter,
            'raw_headers': table.raw_headers,
            'duplicate_headers': table.duplicate_headers,
            'empty_headers': table.empty_headers,
            'record_count': table.record_count,
            'malformed_records': len(table.malformed),
            'parse_issue_records': len({i['source_row'] for i in issues if i['file'] == name}),
            'id_leading_zero_count': int(id_leading.sum()),
            'column_mapping': mapping,
            'missing_by_column': {c: int(df[c].isna().sum()) for c in df},
            'status': 'parsed',
        }
    return datasets, summary, issues

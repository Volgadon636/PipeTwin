"""Explicit source schemas, original filename aliases and loss-aware parsing."""
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (FILE_ALIASES, CANONICAL_COL_MAPPINGS, CYRILLIC_TO_LATIN,
                     NUMERIC_COLUMNS, DATE_COLUMNS, REQUIRED_COLUMNS)


def to_snake_case(text):
    return re.sub(r'_+', '_', ''.join(CYRILLIC_TO_LATIN.get(c, c if c.isalnum() else '_')
                                    for c in text.strip().lower())).strip('_') or 'col'


def detect_file_format(path):
    blob = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'cp1251'):
        try:
            sample = blob.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f'Unsupported encoding: {path}')
    try:
        sep = csv.Sniffer().sniff(sample[:65536], delimiters=',;\t').delimiter
    except csv.Error:
        sep = ',' if sample.splitlines()[0].count(',') > sample.splitlines()[0].count(';') else ';'
    return encoding, sep


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
    s = series.astype('string').str.strip()
    parsed = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    iso = s.str.match(r'^\d{4}-\d{2}-\d{2}', na=False)
    ru = s.str.match(r'^\d{2}\.\d{2}\.\d{4}', na=False)
    parsed.loc[iso] = pd.to_datetime(s[iso], format='ISO8601', errors='coerce').astype('datetime64[ns]')
    parsed.loc[ru] = pd.to_datetime(s[ru], format='mixed', dayfirst=True, errors='coerce').astype('datetime64[ns]')
    return parsed


def parse_numeric(series):
    s = series.astype('string').str.strip().str.replace('\u00a0', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


def resolve_sources(data_dir):
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


def read_and_standardize_all(data_dir):
    datasets, summary, issues = {}, {}, []
    for name, path in resolve_sources(data_dir).items():
        encoding, sep = detect_file_format(path)
        raw = pd.read_csv(path, encoding=encoding, sep=sep, dtype='string',
                          keep_default_na=False)
        mapping = build_column_mapping(name, raw.columns)
        df = raw.rename(columns=mapping).copy()
        # Literal absence markers only; a physical zero is never a global NA marker.
        df = df.replace(['', '#N/A', 'NaN', 'nan', 'NULL'], pd.NA)
        missing = set(REQUIRED_COLUMNS[name]) - set(df.columns)
        if missing:
            raise ValueError(f'{path.name}: missing required columns {sorted(missing)}')
        df.insert(0, 'source_row', np.arange(2, len(df) + 2))
        for col in NUMERIC_COLUMNS[name] + DATE_COLUMNS[name]:
            before = df[col].copy()
            parsed = robust_parse_dates(before) if col in DATE_COLUMNS[name] else parse_numeric(before)
            bad = before.notna() & parsed.isna()
            if bad.any():
                df[col + '_source_text'] = before
                for index in df.index[bad]:
                    issues.append({'file': name, 'source_row': int(df.at[index, 'source_row']),
                                   'field': col, 'rule': 'parse_failed', 'action': 'set_missing',
                                   'original_value': str(before[index])})
            df[col] = parsed
        invalid_id = df.id.notna() & ((df.id % 1 != 0) | (df.id <= 0))
        if invalid_id.any():
            raise ValueError(f'{name}: non-integer or nonpositive IDs')
        df['id'] = df.id.astype('Int64')
        datasets[name] = df
        summary[name] = {
            'source_filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows': len(df), 'columns': len(raw.columns), 'encoding': encoding, 'delimiter': sep,
            'column_mapping': mapping,
            'missing_by_column': {c: int(df[c].isna().sum()) for c in df},
        }
    return datasets, summary, issues

"""
Format detection, character encoding sniffer, date parser, and column standardizer.
"""

import re
import csv
from pathlib import Path
from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np

from .config import CYRILLIC_TO_LATIN, CANONICAL_COL_MAPPINGS


def to_snake_case(text: str) -> str:
    """
    Converts arbitrary string into clean Latin snake_case.
    Transliterates Cyrillic characters, strips special symbols, and merges underscores.
    """
    text = text.strip()
    res = []
    for ch in text.lower():
        if ch in CYRILLIC_TO_LATIN:
            res.append(CYRILLIC_TO_LATIN[ch])
        elif ch.isalnum():
            res.append(ch)
        else:
            res.append('_')
    s = ''.join(res)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "col"


def detect_file_format(filepath: Path) -> Tuple[str, str, str]:
    """
    Autodetects file encoding, field delimiter, and decimal separator.
    Returns:
        (encoding, delimiter, decimal_separator)
    """
    # 1. Detect encoding
    detected_enc = 'utf-8'
    for enc in ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251', 'iso-8859-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                sample = f.read(65536)
                detected_enc = enc
                break
        except UnicodeDecodeError:
            continue

    # 2. Detect delimiter
    try:
        lines = [line for line in sample.splitlines()[:10] if line.strip()]
        sample_text = '\n'.join(lines)
        dialect = csv.Sniffer().sniff(sample_text, delimiters=',;\t|')
        delim = dialect.delimiter
    except Exception:
        counts = {d: sample[:2000].count(d) for d in [',', ';', '\t', '|']}
        delim = max(counts, key=counts.get)

    # 3. Detect decimal separator
    dec = '.'
    try:
        sample_df = pd.read_csv(filepath, encoding=detected_enc, sep=delim, nrows=50)
        for col in sample_df.columns:
            if sample_df[col].dtype == object or str(sample_df[col].dtype) in ('object', 'str', 'string'):
                vals = sample_df[col].dropna().astype(str)
                if any(re.match(r'^-?\d+,\d+$', s.strip()) for s in vals):
                    dec = ','
                    break
    except Exception:
        pass

    return detected_enc, delim, dec


def build_column_mapping(filename: str, original_cols: List[str]) -> Dict[str, str]:
    """
    Builds a column rename mapping dictionary:
    assigns canonical names to key attributes and automated snake_case to the rest.
    Guarantees unique column names without collisions.
    """
    canon = CANONICAL_COL_MAPPINGS.get(filename, {})
    mapping = {}
    used_names = set()

    for col in original_cols:
        if col in canon:
            new_name = canon[col]
        else:
            new_name = to_snake_case(col)

        base_name = new_name
        counter = 1
        while new_name in used_names:
            new_name = f"{base_name}_{counter}"
            counter += 1

        used_names.add(new_name)
        mapping[col] = new_name

    return mapping


def robust_parse_dates(series: pd.Series) -> pd.Series:
    """
    Robust datetime parser supporting ISO (YYYY-MM-DD) and Russian format (DD.MM.YYYY).
    """
    sample = series.dropna().astype(str).head(100)
    if len(sample) == 0:
        return pd.to_datetime(series, errors='coerce')
    if sample.str.match(r'^\d{4}-\d{2}-\d{2}').sum() > len(sample) * 0.5:
        return pd.to_datetime(series, format='ISO8601', errors='coerce')
    elif sample.str.match(r'^\d{2}\.\d{2}\.\d{4}').sum() > len(sample) * 0.5:
        return pd.to_datetime(series, format='%d.%m.%Y', errors='coerce')
    else:
        return pd.to_datetime(series, format='mixed', errors='coerce')


def read_and_standardize_all(data_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Loads all raw data files, standardizes column names, converts datatypes,
    and returns DataFrames along with metadata summary.
    """
    files = [
        'passport.csv',
        'profile.csv',
        'uzt.csv',
        'vtd.csv',
        'regime.csv',
        'tech_mode_for_embedings_with_length_with_temperatures_ids16.csv'
    ]

    dfs: Dict[str, pd.DataFrame] = {}
    summary: Dict[str, Any] = {}

    for fn in files:
        filepath = data_dir / fn
        if not filepath.exists():
            continue

        enc, delim, dec = detect_file_format(filepath)

        df_raw = pd.read_csv(
            filepath,
            encoding=enc,
            sep=delim,
            decimal=dec,
            low_memory=False
        )

        col_mapping = build_column_mapping(fn, df_raw.columns.tolist())
        df = df_raw.rename(columns=col_mapping).copy()

        # Locate pipeline ID column
        id_col = 'id' if 'id' in df.columns else None
        if id_col is None:
            for c in df.columns:
                if 'id' in c:
                    id_col = c
                    break

        # Standardize ID column to integer
        if id_col and id_col in df.columns:
            df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
            unique_ids = [int(x) for x in df[id_col].dropna().unique()]
            unique_ids.sort()
        else:
            unique_ids = []

        # Standardize date columns to datetime
        date_cols_found = []
        date_ranges = {}
        for col in df.columns:
            if any(k in col.lower() for k in ['year', 'god', 'srok', 'let']):
                continue

            if any(k in col.lower() for k in ['date', 'data', 'vvod']):
                if df[col].dtype.kind in ('O', 'U', 'S') or str(df[col].dtype) in ('object', 'str', 'string'):
                    try:
                        parsed = robust_parse_dates(df[col])
                        if parsed.notnull().sum() > 0:
                            df[col] = parsed
                            date_cols_found.append(col)
                            d_min = parsed.min().strftime('%Y-%m-%d') if pd.notnull(parsed.min()) else 'None'
                            d_max = parsed.max().strftime('%Y-%m-%d') if pd.notnull(parsed.max()) else 'None'
                            date_ranges[col] = f"{d_min} ... {d_max}"
                    except Exception:
                        pass

        # Standardize numerical text columns to float
        for col in df.columns:
            if col not in date_cols_found and col != id_col:
                if df[col].dtype.kind in ('O', 'U', 'S') or str(df[col].dtype) in ('object', 'str', 'string'):
                    sample_vals = df[col].dropna()
                    if len(sample_vals) > 0 and sample_vals.astype(str).str.replace('.', '', regex=False).str.replace('-', '', regex=False).str.isdigit().sum() > len(sample_vals) * 0.9:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

        duplicates_count = int(df_raw.duplicated().sum())
        null_counts = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}

        summary[fn] = {
            'encoding': enc,
            'delimiter': delim,
            'decimal': dec,
            'rows_count': len(df),
            'cols_count': len(df.columns),
            'duplicates_count': duplicates_count,
            'unique_pipe_ids': unique_ids,
            'pipe_ids_count': len(unique_ids),
            'date_ranges': date_ranges,
            'null_cols_count': len(null_counts),
            'null_sample': {k: null_counts[k] for k in list(null_counts.keys())[:8]},
            'column_mapping': col_mapping,
        }

        dfs[fn] = df

    return dfs, summary

"""Deterministic persistence and semantic CSV/Parquet round-trip checks."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False, default=str)+'\n', encoding='utf-8')


def validate_roundtrip(expected, actual, name):
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        raise ValueError(f'{name}: shape/columns changed after serialization')
    for col in expected:
        left, right = expected[col], actual[col]
        if pd.api.types.is_datetime64_any_dtype(left):
            parsed = pd.to_datetime(right, format='ISO8601', errors='coerce')
            if not np.array_equal(left.astype('datetime64[ns]').astype('int64'), parsed.astype('datetime64[ns]').astype('int64')):
                raise ValueError(f'{name}: date values changed in {col}')
        elif pd.api.types.is_bool_dtype(left):
            values = right.astype('string').str.lower().map({'true':True, 'false':False})
            if not np.array_equal(left.to_numpy(), values.to_numpy()):
                raise ValueError(f'{name}: boolean values changed in {col}')
        elif pd.api.types.is_integer_dtype(left):
            from decimal import Decimal
            for a, b in zip(left, right):
                if pd.isna(a) and (pd.isna(b) or b == ''):
                    continue
                if pd.isna(a) or pd.isna(b) or Decimal(str(a)) != Decimal(str(b)):
                    raise ValueError(f'{name}: integer values changed in {col}')
        elif pd.api.types.is_numeric_dtype(left):
            a = left.to_numpy(dtype=float, na_value=np.nan)
            b = pd.to_numeric(right, errors='raise').to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(a,b,equal_nan=True,atol=1e-10,rtol=1e-12):
                raise ValueError(f'{name}: numeric values changed in {col}')
        else:
            a = left.astype('string').fillna('')
            b = right.astype('string').fillna('')
            if not np.array_equal(a.to_numpy(),b.to_numpy()):
                raise ValueError(f'{name}: text values changed in {col}')


def save_table(df, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    for col in df:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype('datetime64[ns]')
    csv_path, parquet_path = stem.with_suffix('.csv'), stem.with_suffix('.parquet')
    df.to_csv(csv_path, index=False, encoding='utf-8', lineterminator='\n',
              float_format='%.15g')
    df.to_parquet(parquet_path, index=False)
    # Explicit string read preserves archival categories such as 'NA' or '001'.
    csv_copy = pd.read_csv(csv_path, dtype='string', keep_default_na=False, na_values=[''])
    validate_roundtrip(df, csv_copy, csv_path.name)
    parquet_copy = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(df, parquet_copy, check_dtype=True)
    return {'rows':len(df), 'columns':len(df.columns), 'csv_sha256':sha256(csv_path),
            'parquet_sha256':sha256(parquet_path), 'csv_parquet_verified':True,
            'dtypes':{c:str(df[c].dtype) for c in df}}


def sha256(path):
    data = Path(path).read_bytes()
    if str(path).lower().endswith(('.csv', '.json', '.md', '.txt')):
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def publish_file(source, destination):
    """Prefer atomic replacement; Windows readers may deny rename but allow writing.

    Keep a byte-for-byte rollback copy until a fallback overwrite is verified.
    A genuine write lock still raises an actionable error without claiming success.
    """
    source, destination = Path(source), Path(destination)
    expected = sha256(source)
    try:
        source.replace(destination)
    except PermissionError:
        previous = destination.read_bytes() if destination.exists() else None
        try:
            shutil.copyfile(source, destination)
            if sha256(destination) != expected:
                raise OSError(f'Copy verification failed: {destination}')
        except OSError as error:
            if previous is not None:
                try:
                    destination.write_bytes(previous)
                except OSError:
                    pass
            raise PermissionError(f'Cannot update {destination}. Close the application holding this file and retry.') from error
    if sha256(destination) != expected:
        raise OSError(f'Published checksum mismatch: {destination}')

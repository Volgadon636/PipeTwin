"""Read-only audit of the next-cycle data generation and S1ABC reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .next_common import json_safe, utcnow
from .s1_contract import S1_ID, S2_ID, target_key_grid


def run_audit(root: Path = BASE_DIR, data_root: Path | None = None,
              output_root: Path | None = None) -> dict:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    output = Path(output_root or root / 'experiments/next_cycle/audit').resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = RepositoryData(data_root)
    grid = target_key_grid(data)
    manifest = json.loads((data_root / 'reports/run_manifest.json').read_text(encoding='utf-8'))
    hashes = {}
    for name, entry in manifest['outputs'].items():
        path = data_root / (name + '.csv')
        actual = sha256(path)
        hashes[name] = {'rows': entry['rows'], 'csv_sha256_ok': actual == entry['csv_sha256'],
                        'actual': actual, 'expected': entry['csv_sha256']}
    old = root / 'experiments/s1abc'
    repro = root / 'experiments/next_cycle/repro_s1abc'
    reproduction = {}
    for stage in ('s1a', 's1b', 's1c'):
        a = old / stage / 'result.csv'
        b = repro / stage / 'result.csv'
        if a.is_file() and b.is_file():
            x = pd.read_csv(a)
            y = pd.read_csv(b)
            merged = x.merge(y, on=['id', 'segment_id'], suffixes=('_old', '_new'))
            reproduction[stage] = {
                'rows': int(len(merged)),
                'max_abs_thickness_diff': float(np.abs(
                    merged.thickness_pred_old - merged.thickness_pred_new).max()),
            }
    payload = {
        'generated_utc': utcnow(),
        'data_root': str(data_root),
        'manifest_sha256': sha256(data_root / 'reports/run_manifest.json'),
        'keys': {str(k): int(v) for k, v in grid.groupby('id').size().items()},
        'manifest_outputs': hashes,
        'all_manifest_csv_hashes_ok': bool(all(v['csv_sha256_ok'] for v in hashes.values())),
        's1abc_reproduction': reproduction,
        'target_reference': {'s1_id': S1_ID, 's2_id': S2_ID},
        'warning': ('This audit is read-only; it does not rebuild data. Run src.data_prep '
                    'to create a new generation.'),
    }
    write_json(output / 'audit_recheck.json', json_safe(payload))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path,
                        default=BASE_DIR / 'experiments/next_cycle/data_generation')
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/next_cycle/audit')
    args = parser.parse_args(argv)
    run_audit(args.root, args.data_root, args.output_root)


if __name__ == '__main__':
    main()

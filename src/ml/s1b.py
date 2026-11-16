"""CLI for S1B (minimum of all prior VTD, no training)."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BASE_DIR

from .s1_stages import run_s1b


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/s1abc/s1b')
    parser.add_argument('--s2-source', type=Path,
                        default=BASE_DIR / 'experiments/ml_v2/result.csv')
    args = parser.parse_args(argv)
    run_s1b(args.root, args.output_root, args.s2_source)


if __name__ == '__main__':
    main()

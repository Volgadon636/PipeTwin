"""Sequential S1A -> S1B -> S1C train -> S1C predict."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BASE_DIR

from .s1_stages import run_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/s1abc')
    parser.add_argument('--s2-source', type=Path,
                        default=BASE_DIR / 'experiments/ml_v2/result.csv')
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args(argv)
    run_all(args.root, args.output_root, args.s2_source, args.alpha)


if __name__ == '__main__':
    main()

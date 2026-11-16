"""Regenerate MAE/R² comparison tables from saved S1C OOF predictions.

This command reads historical labels only for evaluation; it does not fit or
change any model, and it does not alter S2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BASE_DIR

from .s1_stages import write_metric_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/s1abc')
    args = parser.parse_args(argv)
    write_metric_artifacts(args.root, args.output_root)


if __name__ == '__main__':
    main()

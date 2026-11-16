"""CLI for S1 next-cycle train and artifact-only predict."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BASE_DIR
from .s1_next import run_s1, run_s1_predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'predict'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, default=BASE_DIR)
        p.add_argument('--data-root', type=Path,
                       default=BASE_DIR / 'experiments/next_cycle/data_generation')
        p.add_argument('--output-root', type=Path,
                       default=BASE_DIR / ('experiments/next_cycle/s1' if name == 'train'
                                           else 'experiments/next_cycle/s1_predict'))
        if name == 'predict':
            p.add_argument('--candidate', default='ridge_geometry')
            p.add_argument('--artifact', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == 'train':
        run_s1(args.root, args.data_root, args.output_root)
    else:
        run_s1_predict(args.root, args.data_root, args.output_root,
                       args.candidate, args.artifact)


if __name__ == '__main__':
    main()

"""CLI for S1C: ``train`` fits and saves an artifact; ``predict`` loads it only."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BASE_DIR

from .s1_stages import run_s1c_predict, run_s1c_train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    train = sub.add_parser('train', help='fit Ridge, run LOPO diagnostics and save artifact')
    train.add_argument('--root', type=Path, default=BASE_DIR)
    train.add_argument('--output-root', type=Path,
                       default=BASE_DIR / 'experiments/s1abc/s1c')
    train.add_argument('--alpha', type=float, default=10.0)

    predict = sub.add_parser('predict', help='load a saved artifact and write predictions')
    predict.add_argument('--root', type=Path, default=BASE_DIR)
    predict.add_argument('--output-root', type=Path,
                         default=BASE_DIR / 'experiments/s1abc/s1c')
    predict.add_argument('--artifact', type=Path, default=None)
    predict.add_argument('--s2-source', type=Path,
                         default=BASE_DIR / 'experiments/ml_v2/result.csv')

    args = parser.parse_args(argv)
    if args.command == 'train':
        run_s1c_train(args.root, args.output_root, args.alpha)
    else:
        run_s1c_predict(args.root, args.output_root, args.artifact, args.s2_source)


if __name__ == '__main__':
    main()

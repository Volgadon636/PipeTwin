"""Sequential next-cycle: S1 train, S2 train, explicit comparison/assembly."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BASE_DIR
from .next_compare import run_compare
from .s1_next import run_s1
from .s2_next import run_s2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path,
                        default=BASE_DIR / 'experiments/next_cycle/data_generation')
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/next_cycle')
    parser.add_argument('--s1-candidate', default='s1a')
    parser.add_argument('--s2-candidate', default='physics_priors')
    args = parser.parse_args(argv)
    run_s1(args.root, args.data_root, args.output_root / 's1')
    run_s2(args.root, args.data_root, args.output_root / 's2')
    run_compare(args.root, args.output_root, args.s1_candidate, args.s2_candidate)


if __name__ == '__main__':
    main()

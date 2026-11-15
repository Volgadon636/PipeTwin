"""Recompute Stage 5 std/difference tables from saved CSVs without fit."""
from __future__ import annotations

import argparse
from pathlib import Path

from .targeted_s1_temporal import recalc_stage5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path('experiments/targeted_cycle/stage_5_temporal'))
    args = parser.parse_args(argv)
    recalc_stage5(args.output_root)


if __name__ == '__main__':
    main()

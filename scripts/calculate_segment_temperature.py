"""Build an isolated heat-loss scenario artifact; never mutate the verified grid.

The old version wrote ``delta_t_c`` into ``processed/target_segments_grid.csv``.
That mixed an illustrative scenario into the authoritative geometry.  This version
writes only ``processed/temperature_scenarios.*`` with parameter rows and
validity/reason flags.  Run it explicitly when a temperature scenario is wanted.

    python scripts/calculate_segment_temperature.py --output-root scratch/temperature
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BASE_DIR
from src.heat_transfer import (DEFAULT_PARAMETERS, HeatTransferParameters, build_scenario,
                               sensitivity_parameters)
from src.storage import save_table, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=BASE_DIR / "processed/target_features.csv")
    parser.add_argument("--output-root", type=Path, default=BASE_DIR)
    parser.add_argument("--sensitivity", action="store_true",
                        help="also write the declared 27-point parameter scenario grid")
    return parser.parse_args()


def main():
    args = parse_args()
    features = pd.read_csv(args.features, dtype="string", keep_default_na=False, na_values=[""])
    for column in ("prediction_date",):
        if column in features:
            features[column] = pd.to_datetime(features[column], format="ISO8601")
    numeric = ["segment_id", "start_m", "end_m", "length_m", "diameter_mm",
               "nominal_thickness_mm", "d_inner_assumed_mm", "t_start_mean",
               "density_liquid_work_mean", "v_liquid_true_mean"]
    for column in numeric:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    frames = [build_scenario(features, HeatTransferParameters(), "BASE_DECLARED")]
    if args.sensitivity:
        for index, params in enumerate(sensitivity_parameters(), start=1):
            frames.append(build_scenario(features, params, f"SENS_{index:02d}"))
    scenarios = pd.concat(frames, ignore_index=True)
    output = Path(args.output_root)
    outputs = save_table(scenarios, output / "processed/temperature_scenarios")
    report = {
        "status": "scenario_only",
        "source_features": str(args.features),
        "scenario_rows": int(len(scenarios)),
        "base_parameters": DEFAULT_PARAMETERS,
        "sensitivity_points": len(sensitivity_parameters()) if args.sensitivity else 0,
        "valid_rows": int(scenarios.valid.fillna(False).sum()) if "valid" in scenarios else 0,
        "invalid_rows": int((~scenarios.valid.fillna(False)).sum()) if "valid" in scenarios else 0,
        "invalid_reasons": (scenarios.loc[~scenarios.valid.fillna(False), "reason"]
                            .value_counts().to_dict() if "reason" in scenarios else {}),
        "warning": "Parameter bands are scenario assumptions, not calibrated confidence intervals.",
        "grid_mutated": False,
    }
    write_json(output / "reports/temperature_scenario_report.json", report)
    write_json(output / "reports/temperature_scenario_manifest.json",
               {"output": outputs, "report": report})
    print(f"Wrote {len(scenarios)} scenario rows to {output / 'processed/temperature_scenarios.csv'}")
    print(report["warning"])


if __name__ == "__main__":
    main()

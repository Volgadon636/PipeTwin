# Stage 02 — temperature-only exposure

Implemented on `model_v02`, HEAD `8770be9693de299f0b9b4c27db1ad7ae9ce517af`.
This is feature engineering only: no temperature model is fitted or selected.

## API

```python
from src.ml.temperature import attach_temperature_features

# Use make_snapshot / build_training_windows histories, keyed by (id, cutoff).
# Their t_mix_usable values already have cutoff-specific hydraulic QA.
features, reports = attach_temperature_features(
    windows, histories,
    ea_kj_mol=40,              # EA40_mid: provisional team hypothesis
    gap_policy='reference',   # explicitly assume F=1 during unknown history
    max_hold_days=1.0,
)
```

`temperature_intervals(rows, start, end)` provides the separate T-only interval
path. `temperature_features(...)` returns all features and metadata for one
window. `TemperatureFeatureRegistry` uses the existing `FeatureDescriptor` and
`FeatureReport` contracts, but is opt-in: it does not modify `REGISTRY`,
`ML_FEATURE_COLUMNS`, or any existing model. The registry includes seven numeric
exposure/coverage features and two missingness flags. Ea, theta, candidate ID,
gap/hold policy, and profile source are attached metadata, not learnable inputs.
All outputs of the feature API retain coverage and missingness even when the
caller selects a single feature. A registry instance binds one immutable history
snapshot; create a new instance after changing histories.

Allowed energies are 0, 25, 40, 55, 70 kJ/mol. There is no implicit energy or gap
policy default. `observed_only` leaves `thermal_exposure_reference_years` null;
`reference` computes the explicitly assumed reference-gap integral. Both report
zero observed integral, zero coverage and a null observed mean for no support.
An absent history mapping key raises an error; an explicitly empty history table
represents no observations. No mean, spatial temperature, 4°C or 20°C fallback is
used as a measurement.

`R=8.314462618 J/(mol·K)`, `T_ref=293.15 K`, `theta=1000*Ea/R`,
`F=exp(theta*(1/T_ref-1/T))`, `T=t_mix_usable+273.15`.
Durations use the existing `YEAR_SECONDS=86400*365.25`. The low-level factor
function rejects nonfinite/nonpositive Kelvin and numerical overflow/underflow;
it does not clip factors. The interval builder excludes nonfinite or nonpositive
Kelvin values but keeps their timestamps as hold boundaries. Valid 0°C survives.

## Reproduction commands

Run from the repository root. Install only into the existing `.venv`:

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-ml.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s experiments/temperature_v02_02 -p test_temperature.py -v
.venv/bin/python -m experiments.temperature_v02_02.compute
.venv/bin/python -m src.ml.experiment --model m0 --output-root experiments/temperature_v02_02/baseline_control
.venv/bin/python -m experiments.temperature_v02_02.verify
```

`compute` writes only inside this stage directory. `verify` compares all 104
protected file hashes, compares the regenerated M0 predictions with both saved
baselines, reruns feature generation in a temporary stage directory, and checks
eight deterministic output checksums. It never rewrites the saved predictions.
`protected_before.json` is the pre-implementation byte inventory; do not regenerate
it to make a failed preservation check pass. `dependencies.txt` records installed
versions. Execution evidence and caveats are in `qa_report.md` and `qa_results.json`.

## Deliverables

- `production_features.csv`: 2,000 segment rows for provisional EA40_mid.
- `historical_features.csv`: 1,080 historical rows with their own QA cutoffs.
- `coverage_by_window.csv`: 19 actual exposure-window groups, including three
  production groups; retains parent ID, pipe ID, scenario, start and cutoff.
- `temperature_intervals.csv`: supporting interval rows per window, without
  expanding them per segment or treating segments as independent observations.
- `example_rows.csv` / `.json`: 30 deterministic examples covering all five
  candidates, both gap policies, measured support, invalid records and no history.
- `feature_catalog.json` / `feature_audit.json`: formulas, provenance, units,
  parameter metadata, missingness and per-column coverage reports.
- `run_manifest.json`, `verification.json`, `spec_sources.json`: input/code/output
  checksums, reproducibility results and supplied specification provenance.
- `test_temperature.py`: stage tests, also included by `tests/test_temperature.py`.
- `baseline_control/`: isolated execution of the existing M0 CLI for reproduction.

CSV missing values are empty fields; the JSON examples use `null`. The numeric
registry audit's coverage means non-null finite column values, not time coverage:
use `thermal_coverage_fraction` for supported exposure time. Source-row/day counts
in the interval audit refer to observations contributing positive duration within
the window, including capped pre-start observations when applicable.

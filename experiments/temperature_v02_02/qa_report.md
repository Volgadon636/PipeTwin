# Stage 02 QA and handoff

## Checkout and specification

- Checked HEAD: `8770be9693de299f0b9b4c27db1ad7ae9ce517af` (Stage 01 complete).
- Initially detached at `origin/model_v02`; attached to local `model_v02` with
  `git switch -c model_v02`. No commit, push, merge or official-result edit.
- Initial `git status --short` was:

```text
 M .gitignore
 D agent/PipeTwin_GitHub_ML_Start_Prompt_EN.md
 D agent/PipeTwin_ML_Technical_Specification_EN.md
?? .agent/
?? .mcp.json
?? .windsurfrules
?? AGENTS.md
```

These existing user changes were preserved. The English technical specification
named by the prompt is absent from the package. The supplied Russian v1.1 technical
specification was read in full; its exact exposure definitions and candidate names
were used. `spec_sources.json` records the document hashes. Stage 00/01 prompts
were context; their completed tasks were not implemented again.

Inspected: all three supplied stage prompts, the RU technical spec, AGENTS.md,
Stage 00 audit/manifest, Stage 01 audit/JSON, README, requirements files,
`src/ml/exposure.py`, `features.py`, `windows.py`, `data.py`, `experiment.py`,
`src/temporal.py`, `tests/test_ml.py`, and preprocessing test imports. The data API
read and hash-verified the published CSV/configuration inputs listed in the run
manifest. The code-review-graph MCP index matched HEAD; its search, impact and
test queries narrowed the legacy joint/registry dependencies. Final graph review
also found the new tests. Change detection cannot fully characterize untracked
additions, so source inspection and executed tests are the acceptance evidence.

## Work performed

Added `src/ml/temperature.py` with independent, bounded T-only intervals and an
opt-in feature registry. Existing `src/ml/exposure.py`, `src/ml/features.py`,
models, preprocessing and tests were not edited. Added a test-discovery shim and
the stage computation, verification, tests, documentation and evidence here.

The interval path uses only `t_mix_usable` after cutoff-specific hydraulic QA.
It rejects duplicate pre-cutoff timestamps, excludes records on/after the cutoff,
clips to the forecast window and hold cap, and terminates a hold at the next record
even when invalid. Missing W/v do not remove temperature support. It preserves
gaps and 0°C values that survived upstream QA. Neither spatial snapshot means nor
fallback temperatures feed this calculation. All seven specified exposure and
coverage columns, Ea/theta/candidate metadata, and explicit missingness flags are
exported. Reference-gap filling requires an explicit policy and is labelled as an
assumption. The fixed reference temperature is recorded in configuration metadata.

EA40_mid is used only for the deterministic real-data feature artifact. All five
candidate formulas are covered by examples/tests. No physical coefficient has
been validated, selected or fitted. Registry parameter metadata is not added to
the legacy model feature list.

## Executed verification

| Check | Actual result | Evidence |
|---|---|---|
| Initial full suite / M0 attempt | Failed: virtual environment lacked numpy/pandas | `environment_initial_tests.txt`, `environment_initial_cli.txt` |
| Dependency installation | Declared requirements installed in `.venv`; no requirement files changed | `dependencies.txt` |
| Current ML entrypoint import | Passed | `import_check.txt` ends `IMPORT_OK` |
| Full unittest suite | **63 tests passed**, 25.776 s | `tests.txt` |
| Isolated stage suite | **16 tests passed**, 3.352 s | `temperature_tests.txt` |
| Existing preprocessing integration | **270 checks, 22 CSV/Parquet pairs, 2,000 target segments** | `tests.txt`; tests use a temporary output root |
| Feature CLI | 1,080 historical rows, 2,000 production rows, 19 window groups | `compute.txt`, `run_manifest.json` |
| Existing M0 CLI | Exit 0; 2,000 unique, finite, bounded predictions | `baseline_control.txt` |
| Saved M0 reproduction | Byte-for-byte equal to both archived result.csv files | `verification.json` |
| Protected inputs and legacy code | **104 files checked, zero changed** | `protected_before.json`, `verification.json` |
| Determinism | Independent generation reproduced **all 8 output hashes** | `verify.txt`, `verification.json` |

Shared prediction SHA-256:
`ef1c4e4e33217a5c821a94739f43288942f99e351c05ee454305eae9e3644394`.
Official schema remains `id,segment_id,thickness_pred`, 525 + 1,475 rows.

The tests verify unit conversion, the five candidate theta conversions, exact
EA00 identities, reference-temperature identity, temperature/energy monotonicity,
integration before averaging, intraday/start/end clipping, no backward fill,
invalid-next stopping, duplicate rejection, finite physical inputs, real-data
as-of isolation, historical cutoffs, missingness, explicit gap policy, empty
outputs, repeated dataframe indices, and legacy factor/schema preservation.
Joint interval boundaries and durations match exactly for shared real-data records.

## Actual bounded-hold coverage

The horizon starts at each segment's permitted baseline date, not always at
commissioning. These are the three production groups with a one-day hold cap:

| Pipe / case | Start → cutoff | Segments | Supported years | T coverage | Longest unsupported gap, days |
|---|---|---:|---:|---:|---:|
| 1250022871 / between_vtd | 2018-09-23 → 2023-12-01 | 14 | 4.316044 | 83.1892% | 72.013102 |
| 1250022871 / between_vtd | 2022-07-06 → 2023-12-01 | 511 | 1.283946 | 91.4155% | 35.297569 |
| 1250009583 / cold_start | 1993-06-20 → 2022-07-01 | 1,475 | 3.863407 | **13.3086%** | 7857.474178 |

The cold-start joint coverage is 13.2992%; its T-only gain is one supported day
from a record with missing velocity. The earlier Stage 01 estimate of ~26%
described an observation span, not the sum of bounded valid holds. Do not use it
as measured/support coverage. The exact cold-start unobserved fraction is ~86.69%,
including gaps within the observation era. These missing years are unknown, not
zero corrosion. T-only section features cannot distinguish segments sharing the
same pipe and time window; segment counts do not add independent validation groups.

## Limitations and next-stage gate

**GO for Stage 03** candidate sensitivity using these audited features. Stage 02
has no unresolved implementation blocker. The full suite passes and the legacy
M0 prediction bytes are preserved.

The current machine lacks native `libomp.dylib`; the existing experiment CLI
warned and skipped optional LightGBM candidate variants. M0 and Stage 02 do not
require that library. Full hybrid/LightGBM execution has not been validated here.
The old Stage 00 report's missing-import and stale-test findings describe an
earlier SHA; current import/tests now succeed. Its report was left unchanged.

The existing M0 CLI also retains its own legacy validation/status wording in
`baseline_control/`; that reproduction output is not evidence of a validated
temperature model. No new prediction or validation claim is made in this stage.
Ea remains a team hypothesis, sensor location interpretation remains unresolved,
and the large missing history cannot be inferred from the available observations.
Stage 03 must compare all five predeclared candidates on identical histories and
must not choose a winner without adequate independent validation.

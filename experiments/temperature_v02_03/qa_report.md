# Stage 03 QA and artifact inventory

Branch: `model_v02`. Checked HEAD: `8770be9693de299f0b9b4c27db1ad7ae9ce517af`.
The existing user edits and uncommitted Stage 02 additions were preserved. Initial
`git status --short` showed `.gitignore` modified, the two `agent/` prompt files
deleted, and `.agent/`, `.mcp.json`, `.windsurfrules`, `AGENTS.md`,
`experiments/temperature_v02_02/`, `src/ml/temperature.py` and
`tests/test_temperature.py` untracked. No commits, merges or pushes were made.

The Stage 03 prompt was read in full. The English technical spec remains absent;
the supplied Russian v1.1 specification was read during Stage 02 and its exact
Section 8 contract was checked again here. Other inspected files: Stage 02
handoff, run manifest, source and computation script; `src/ml/evaluation.py`,
`models.py` (weighting/constant-rate implementation), `windows.py`, and graph
source/test context. The original Okafor/Brown/Nesic paper was verified through
the authors' institutional PDF, with source details in `literature_context.json`.

The code-review-graph MCP index matches HEAD. Search and caller/test context
identified the temperature, eligibility and validation functions. Change/flow
analysis was also run, but Git diff analysis does not fully describe untracked
additions; executed source-level tests and preservation hashes are the evidence.

## Changes

- Added only `experiments/temperature_v02_03/` and
  `tests/test_temperature_candidates.py` (standard test discovery).
- No existing implementation or test was edited; Stage 02 APIs are reused.
- `preregistration.json` fixes candidates, constants, scenarios, hold cap,
  gap assumptions, observation weighting, review scope and decision rules.
- `run.py` is the executable study; `verify.py` checks preserved bytes and
  deterministic regeneration; `test_candidates.py` contains nine new tests.

## Results and checks

- Full command: `.venv/bin/python -m unittest discover -s tests -v`.
- Actual result: **72 tests passed in 20.789 seconds**; see `tests.txt`.
- Existing preprocessing integration also passed **270 checks, 22 CSV/Parquet
  pairs, 2,000 segments**, using its temporary output root.
- Candidate run: **11 windows, 55 exposure rows, 10 exact-schema review rows,
  3,000 pooled observations, zero admissible folds, zero fitted models**.
- Same interval checksum, horizon and support for every candidate in each window.
- Every required prediction metric is null; no fabricated validation score.
- Legacy 2500 K scalar-factor comparison: maximum absolute difference **0.0**.
- The graph was rendered and visually inspected; axes, units and legends are
  legible. PNG and SVG are both provided. First-run fontconfig cache warnings did
  not prevent rendering; the command completed successfully.
- `verification.json` records preservation of **152 pre-existing files** and
  exact regeneration of **13 data/graph artifacts**, including PNG and SVG.
- Official baseline prediction hash remains
  `ef1c4e4e33217a5c821a94739f43288942f99e351c05ee454305eae9e3644394`.

The new tests check fixed candidates, separate controls, exact Stage 02 agreement,
unchanged intervals, observation vs duration weighting, empty histories, valid
independent-fold detection, exclusion of same-parent/same-day/future labels,
review schema/null metrics, identical support hashes, all 2,000 production keys,
729 eligible historical keys, factor sensitivities and legacy compatibility.

## Deliverables

| File | Purpose |
|---|---|
| `ea_candidate_review.csv` / `.json` | Exact Section 8 fields, five hypotheses × two gap assumptions |
| `methods.md` | Full numeric grid, scope, aggregation, findings, identifiability and decisions |
| `candidate_exposures.csv` | Both exposure integrals with all support/missingness metadata per candidate/window |
| `factor_statistics.csv` | Min/mean/p95/max, duration-weighted mean, finite/overflow/underflow counts |
| `sensitivity_temperatures.csv` | Factors at 20, 40 and 58.16°C |
| `candidate_sensitivity.png` / `.svg` | Exportable graph of factors and gap-assumption sensitivity |
| `intervals.csv` | Frozen bounded interval support and temperatures |
| `window_segments.csv` | Auditable mapping to segment/scenario/cutoff keys |
| `validation_fold_audit.csv` | Explicit failed fold opportunities and training counts |
| `eligible_training_summary.json` | Target-specific permitted parent/campaign counts |
| `temperature_confounding.csv` | Pairwise association diagnostics with sample counts |
| `legacy_compatibility.json` | Placeholder control, excluded from the grid |
| `literature_context.json` | External values and verified primary citation, context only |
| `preregistration.json` | Fixed study design |
| `run_manifest.json` | Commands, SHA, versions, input/code/output checksums |
| `protected_before.json`, `verification.json` | Preservation and deterministic reproduction evidence |
| `run.txt`, `tests.txt`, `verify.txt` | Actual execution output |

Run commands are in `methods.md`. No Stage 03 implementation blocker remains.
English-spec absence and the lack of independent validation are disclosed.
Stage 04 may proceed as an exploratory training-only model; selecting a validated
coefficient or replacing the official baseline remains unsupported.

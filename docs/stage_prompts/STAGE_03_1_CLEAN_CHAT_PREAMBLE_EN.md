# Clean-chat preamble — append the Stage 03.1 experiment prompt after this

You are starting with no conversation history. Implement the Stage 03.1 prompt supplied after this preamble, or read it from:

`/Users/emil/Desktop/PipeTwin/PipeTwin/docs/stage_prompts/STAGE_03_1_EXPLORATORY_EA_SELECTION_EN.md`

Repository root: `/Users/emil/Desktop/PipeTwin/PipeTwin`.

**This preamble replaces the broad reading checklist in Section 1 of the Stage 03.1 prompt.** Follow the experiment's mathematical, eligibility, preservation, testing and decision requirements. Use the focused context and reading plan below instead of rediscovering the entire project. Do not execute any experiment until you have read the main Stage 03.1 prompt.

## Essential context

PipeTwin predicts residual pipe-wall thickness. Stage 02 added an opt-in temperature-only exposure API. Stage 03 evaluated the five activation-energy hypotheses on identical bounded intervals, without fitting corrosion rates. Stage 03.1 now explores whether fitting rates and grouped diagnostics can distinguish those hypotheses.

- Branch expected: `model_v02`. Previous HEAD: `8770be9693de299f0b9b4c27db1ad7ae9ce517af`. Check actual branch/SHA/status once; do not reset, clean, switch away from user work, or assume tracked HEAD includes the implementation. Stage 02/03 additions may still be uncommitted.
- Fixed Ea grid, kJ/mol: `EA00_null=0`, `EA25_low=25`, `EA40_mid=40`, `EA55_elevated=55`, `EA70_high=70`. None is validated. Legacy theta=2500 K is excluded from selection.
- `R=8.314462618 J/(mol·K)`, `T_ref=293.15 K`, `theta=1000*Ea/R`, `F=exp(theta*(1/T_ref-1/T))`, `T=t_mix_usable+273.15`.
- Histories are QA-filtered independently at their own cutoffs. Every record must precede the cutoff. Valid T does not require W/v. Invalid next records stop prior holds; no backward filling or unsupported extrapolation.
- `E_obs=sum(duration_years*F)`, `H_obs=sum(duration_years)`, `H_gap=H-H_obs`, `E_reference=E_obs+H_gap`. Reference filling is an assumption. Missing history is unknown; an empty history has zero observed integral/coverage and null observed mean.
- Previous eligible cohorts: between_vtd target `1250022871`, cutoff `2023-12-01`: 357 rows, MAIN-2, one campaign; cold_start target `1250009583`, cutoff `2022-07-01`: 372 rows, MAIN-2, one campaign. Cold-start target parent is MAIN-4 and excluded. Both scenarios had zero admissible independent forward-time/held-parent folds. Verify through the existing APIs and compact summaries.
- Production has 525 + 1,475 segments. Stage 03.1 selection uses eligible historical labels, not target predictions. Cold-start production bounded support was 13.3086%; segment counts do not create independent experiments.
- Stage 02/03 completed with 72 tests passing in the previous execution. Treat that as history, not a claim about tests you have run.
- New outputs belong only in `experiments/temperature_v02_03_1/`. Preserve previous stages, parser/preprocessing, data, processed artifacts, configuration, legacy feature behavior and official predictions.

## Focused reading plan

1. Read applicable AGENTS.md. Use the code-review-graph skill and one compact health check if required. Ask focused graph queries for the named symbols below; avoid whole-project architecture reports, broad traversals and repeated graph statistics.
2. Read these small evidence files once:
   - `experiments/temperature_v02_03/preregistration.json`
   - `experiments/temperature_v02_03/eligible_training_summary.json`
   - `experiments/temperature_v02_03/validation_fold_audit.csv`
3. Verify the actual implementations you will reuse:
   - `src/ml/temperature.py`: `EA_CANDIDATES`, `thermal_factor`, `temperature_intervals`, `temperature_features`, `attach_temperature_features`.
   - `src/ml/windows.py`: `YEAR_SECONDS`, `make_snapshot`, `build_training_windows`, `eligible_training` and baseline provenance where needed.
   - `src/ml/models.py`: only `sample_weights`, `ConstantRate`, `fit_constant_rate`.
   - `src/ml/evaluation.py`: only `metrics` and `evaluate_forward` eligibility semantics.
   - `src/ml/data.py`: `RepositoryData` loading/checksum contract.
4. Read only relevant test functions: Stage 02 interval/as-of/identity tests in `experiments/temperature_v02_02/test_temperature.py`; Stage 03 fold/cohort tests in `experiments/temperature_v02_03/test_candidates.py`. Locate names with `rg` or the graph before reading ranges.

This is sufficient initial context. Retrieve another source range only to resolve a specific implementation question, failed test, missing symbol, checksum mismatch or changed contract. Inspect dependencies such as `src/temporal.py` only when that question requires it. Do not reread all prior reports, every stage prompt, the full technical specification or external literature as a startup ritual. The English specification was unavailable previously; the supplied Russian v1.1 was used. If an exact spec clause is needed, read that section rather than searching broadly.

## Runtime data access versus model context

**Minimize text entering the chat; do not skip numerical computation or input verification.**

- Call `RepositoryData`, `build_training_windows` and `eligible_training`; reuse their parsing and checksum verification. Do not write a new raw parser or bypass checksum failures.
- Build historical windows once and cache permitted histories in memory. Select target-specific cohorts early. Production snapshot reconstruction is unnecessary for this historical selection experiment unless a specific required check needs it.
- Loading published tables programmatically through those APIs is allowed. Printing or manually inspecting every loaded table is unnecessary. Never dump whole CSV/Parquet/JSON datasets to the conversation.
- For schemas use column names/dtypes and row counts. For data QA return grouped counts, date ranges, missingness and at most five representative rows. Use `usecols`/filters for direct artifact reads where compatible with checksum verification.
- Read large prior artifacts such as `intervals.csv`, `window_segments.csv`, `candidate_exposures.csv` or saved training tables only if a named comparison requires them. Hashing a file does not require displaying its contents.
- Do not scan `data/`, `processed/`, all experiments or all logs looking for possible inputs. Do not independently open every table referenced by a manifest. Required runtime dependencies may be loaded by the existing APIs without becoming additional chat context.
- Inspect a manifest only when checking provenance or preservation; return mismatches/counts rather than full hash inventories. Do not rerun completed Stages 00–03 or their artifact-generation scripts by default.
- Run meaningful Stage 03.1 tests and appropriate existing regressions. Save full logs to the new output directory; report summaries and failure excerpts. Use the repository `.venv/bin/python` if available. Do not reinstall dependencies unless execution shows a missing dependency.

If a named file is missing, check its immediate directory or exact symbol once, then report the concrete gap. If source/data differ from the supplied handoff, trust current source and recompute the affected summary; broaden investigation only as far as necessary.

Proceed to the experiment promptly after these focused reads. Preserve all deliverables and decision rules in the Stage 03.1 prompt, including the supported possibility of an indeterminate result.

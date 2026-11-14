# Stage 03.1 — Explore whether the data can support a provisional E_a choice

## Copy this prompt into the coding assistant

You are the physics analyst and experiment engineer for PipeTwin. Run an exploratory activation-energy selection experiment after Stages 02 and 03. Your objective is to determine whether the permitted data meaningfully distinguish the predeclared E_a candidates, and, only if they do, recommend a provisional development value for Stage 04.

Complete the experiment and produce a decision. `indeterminate` is a valid scientific result, but support it with executed fits, identifiability diagnostics and explicit reasons. Do not stop merely because independent validation is unavailable. Conversely, do not force a numerical winner from a flat objective or an unstable ranking.

This stage authorizes isolated diagnostic rate fits and internal diagnostic predictions needed to study E_a. It does not authorize production forecast generation, baseline replacement, or a claim that an effective material constant has been identified.

## 1. Read the actual checkout and previous evidence

Work in the existing `model_v02` checkout. Record actual branch, exact HEAD SHA and `git status --short` before edits. Preserve existing user changes and uncommitted Stage 02/03 work; commit SHA alone does not describe that implementation, so also record source and input hashes.

Follow applicable AGENTS.md instructions, including graph-first code exploration followed by source verification. Read:

- The supplied technical specification v1.1 in full. If the English copy is absent, use the available Russian copy and explicitly record that substitution.
- `experiments/temperature_v02_02/README.md`, `qa_report.md`, `run_manifest.json` and the actual temperature implementation/tests.
- `experiments/temperature_v02_03/preregistration.json`, `methods.md`, `qa_report.md`, `ea_candidate_review.csv`, `candidate_exposures.csv`, `window_segments.csv`, `validation_fold_audit.csv` and `eligible_training_summary.json`.
- Actual eligibility, W1 weighting, evaluation and baseline implementation in `src/ml/windows.py`, `models.py`, `evaluation.py`, `exposure.py`, and `features.py`.
- The Stage 04 prompt, to keep this study focused on parameter selection evidence and provide a usable handoff.

Previous execution found 357 eligible between-VTD rows and 372 eligible cold-start rows, each from one parent (MAIN-2) and one campaign. Neither scenario had an admissible held-parent/forward-time fold. Cold-start production support was only 13.3086%. Verify these facts against the current checkout; do not assume they are immutable.

## 2. Freeze the exploratory protocol before fitting

Create `experiments/temperature_v02_03_1/preregistration.json` before examining candidate fit losses. Preserve previous stage outputs. Record constants, grid, cohorts, weighting, scenarios, folds, missing-data treatment, metrics and decision rules. Changes after seeing results require an explicit versioned amendment and justification; never tune a rule to obtain a preferred candidate.

Use exactly these five team hypotheses:

| candidate_id | E_a_kJ_mol | Reason |
|---|---:|---|
| EA00_null | 0 | Negative control: duration with no thermal response |
| EA25_low | 25 | Low effective sensitivity |
| EA40_mid | 40 | Provisional engineering hypothesis, with no privileged selection status |
| EA55_elevated | 55 | Elevated sensitivity |
| EA70_high | 70 | High-sensitivity stress test |

Use `R=8.314462618 J/(mol·K)`, `T_ref=293.15 K`, `theta_K=1000*E_a_kJ_mol/R`, and full numeric precision. Exclude the legacy `theta=2500 K` from selection. The literature values 31.26 and 73.29 kJ/mol are context only. Do not add a continuous optimizer, new grid points or chemistry coefficients in this stage.

Use a one-day bounded hold as the primary policy. Preregister 0.5-day and 0.25-day holds as robustness diagnostics; never extend support beyond one day. For every hold use both:

- `observed_only`: an incomplete-history exposure descriptor.
- `reference`: explicitly assume F=1 during unsupported history.

These are alternative assumptions, not independent replications. Rank candidates separately within each assumption and scenario. Do not average their raw errors into a single score.

## 3. Enforce target-specific eligibility

Build cohorts using the existing as-of and eligibility APIs:

- between_vtd: target `1250022871`, cutoff `2023-12-01`; all fitting/selection labels must precede the cutoff.
- cold_start: target `1250009583`, cutoff `2022-07-01`; exclude its entire parent pipeline and prohibited target diagnostics from training and selection.

Keep these production scenarios separate. A later campaign allowed for one scenario cannot enter the earlier scenario's selection pool. Preserve `main_pipeline_id`, `campaign_id`, `id`, `segment_id`, `case`, `start_date` and `prediction_date`.

Use measured labels exactly as the existing eligibility logic permits. Keep negative observed thinning in the fitting diagnostic and flag it; do not clip observed loss or silently delete contradictory readings. Record baseline provenance and missingness exclusions.

Temperature histories must use cutoff-specific `t_mix_usable`, Kelvin conversion and Stage 02 bounds. Valid T must not depend on W/v availability. No spatial snapshot means, 4°C soil fallback or 20°C fill may become a measured temporal history. Freeze intervals once per window/hold and reuse them for every candidate; publish interval checksums.

## 4. Fit the reference rate and profile the objective

For each case, hold, gap policy and candidate separately, fit one nonnegative reference rate using existing W1 weights:

`loss_i = h0_mm_i - min_residual_thickness_mm_i`

`r_ref = max(0, sum(w_i * E_i * loss_i) / sum(w_i * E_i**2))`

`predicted_loss_i = r_ref * E_i`

`h_pred_i = clip(h0_mm_i - predicted_loss_i, 0, h0_mm_i)`

Require finite values and a positive denominator. For zero-exposure records, retain their missingness and residual contribution; a zero integral is not evidence of zero corrosion. If no exposure permits a fit, report unavailable. Use identical eligible rows across candidates within a comparison, with all exclusions enumerated.

Report for every candidate:

- r_ref in mm/year; denominator; counts of rows, parents, campaigns, sections and distinct exposure histories.
- Unclipped W1 weighted SSE and weighted RMSE in mm for the fitting objective; clipped thickness MAE/RMSE/bias in mm as separate diagnostics.
- Pearson² only when defined, null for constant predictions. Do not substitute R².
- Relative objective change against EA00_null and against the within-comparison minimum.
- Observed support, unknown share, reference-gap contribution, and negative-observed-loss counts.

Fit a separate r_ref per case. Do not impose a common rate across cases to manufacture identification. Do not fit a free rate per section as the primary selection model, because that can absorb the thermal effect. Such a fit may only be an explicitly labelled structural non-identifiability demonstration.

Verify that EA00 with reference gaps reproduces the existing M0 W1 rate and predictions on identical eligible rows. EA00 observed-only uses H_obs and is not expected to equal the full-horizon M0.

## 5. Test whether E_a is distinguishable from the rate scale

Perform these diagnostics on eligible training histories, without using production targets to choose a value:

1. Count unique thermal trajectories, distinguishing parent/campaign/section/window from segment rows sharing an exposure. Explain dependence between overlapping histories.
2. Compare candidate exposure vectors after normalization by their W1 weighted norm. Report cosine similarity and proportionality residuals relative to EA00 and EA40. Nearly proportional vectors allow r_ref to absorb a change in E_a.
3. Plot profiled fit loss versus E_a alongside fitted r_ref. Report the compensation between exposure and rate, and the set of practically tied candidates.
4. Compute the weighted two-column sensitivity matrix for parameters `(log(r_ref), E_a_kJ_mol)` at positive-rate fits. Derive the exposure derivative analytically from the fixed interval formula. Report normalized-column similarity, singular values and numerical rank, with units/scaling documented. At r_ref=0, report this diagnostic as degenerate rather than dividing by zero.
5. Show training-history associations between temperature, elapsed time, watercut and velocity using paired valid observations and explicit counts. Missing W/v must not remove records from temperature exposure. Distinguish descriptive association from causal evidence.
6. Quantify how much candidate differences survive the reference-gap contribution. Sparse historical support may let the gap assumption determine the apparent ranking.

Do not interpret small residual differences, matrix condition numbers or many segment rows as independent proof of a physical activation energy.

## 6. Assess transfer and robustness without inventing validation

First audit admissible independent held-parent/forward-time folds using strict label cutoffs and exclusion of the held parent. If any exist, fit r_ref using only that fold's training data, evaluate all fixed candidates on the same held-out rows, and report the proper metrics. Any candidate selection on these folds makes their winning score a selection diagnostic; do not call it an unbiased final performance estimate without an untouched or outer test set.

If no such folds exist, continue the exploration with within-parent diagnostics:

- Use leave-one-simple-section-out within the permitted training campaign when at least three distinct sections can form nonempty calibration/test sets. Hold out all segments/windows of a section together; fit r_ref on the remaining sections only. Recompute W1 weights on each calibration set.
- Explicitly label these `within_parent_same_campaign_diagnostic`. Same-campaign labels are allowed only inside this exploratory diagnostic and only before the production cutoff. This does not establish temporal or parent-pipeline generalization.
- Do not use random segment splits. Do not split identical trajectories into nominally independent groups. Report repeated/overlapping trajectories and shared operational histories across folds.
- If too few sections exist, report the diagnostic unavailable. Do not create pseudo-folds to satisfy a count.
- Compare paired held-out errors with equal section contribution. Publish per-section results and rankings, not only a pooled row-weighted score.

Repeat the preregistered hold/gap alternatives. Report changes in rank and candidate exposure shape. Leave-one-section perturbations are stability diagnostics, not confidence intervals. Avoid segment bootstraps and nominal population confidence intervals with one parent/campaign. Report unavailable uncertainty honestly.

## 7. Make a transparent decision

Publish one of these outcomes for each case and, if supported, a shared recommendation:

- `provisional_development_choice`: one fixed-grid candidate is consistently distinguishable under the rules below. It remains an effective modelling hypothesis.
- `candidate_set_retained`: more than one candidate remains practically tied or sensitivity-dependent.
- `indeterminate`: available support, exposure diversity or transfer evidence cannot distinguish the candidates.

Preregister these exploratory decision rules; they are engineering tolerances, not statistical significance tests:

1. Treat candidates within 1% of the minimum W1 profiled SSE as a fit tie. Handle a zero minimum explicitly; do not divide by zero.
2. A lower training SSE alone cannot select a candidate. Require an available grouped transfer diagnostic or admissible independent validation.
3. For an internal provisional choice, require at least three usable section holdouts, a unique best mean held-out MAE at the primary one-day hold under both gap policies, and no practically meaningful reversal across the two shorter holds. Treat held-out MAE differences below `max(0.01 mm, 1% of the comparator MAE)` as ties. Publish exceptions and opposing folds.
4. Require the proposed candidate to beat EA00 beyond that same tolerance and have numerically distinguishable exposure shape. If EA00 ties, retain the no-temperature control and report the additional thermal response unsupported.
5. A shared E_a recommendation requires support in both cases. Conflicting per-case choices justify retaining a set or reporting indeterminate; do not pool cases to hide disagreement.
6. Do not claim that the above rules provide independent physical validation. If the decision depends on one section, gap filling, a rank-deficient sensitivity matrix, or effectively proportional exposure vectors, report the fragility and withhold a unique recommendation.

If no unique choice is justified, name the retained candidate set and the specific missing evidence. EA40 may remain a continuity setting for software development only if clearly separated from the experiment's scientific decision. Never present continuity as an experimentally selected winner.

## 8. Deliverables and tests

Write new outputs under `experiments/temperature_v02_03_1/`:

- `preregistration.json`, executable reproduction command and source.
- `cohort_audit.csv`, `fold_audit.csv`, interval identity/checksum evidence.
- `candidate_fits.csv`, `profile_objective.csv`, `exposure_shape_diagnostics.csv`, `sensitivity_matrix_diagnostics.csv`.
- `diagnostic_predictions.csv` containing only permitted historical fitting/held-out rows, with split provenance and honest evidence status.
- `grouped_transfer_metrics.csv`, `robustness_summary.csv`, and any unavailable-result reasons.
- `decision.json`: outcome per case, selected value or null, retained candidates, decision-rule results, limitations and Stage 04 handoff.
- `methods.md`, a readable chart of objective/rate compensation, and a candidate-by-policy stability table.
- Run manifest, input/source/output hashes, protected-file verification, actual test output and QA report.

Add meaningful tests for analytic derivatives, synthetic constant-rate recovery, proportional-exposure non-identifiability, known heterogeneous synthetic temperature histories, zero-exposure/zero-rate handling, EA00 reference identity with M0, grouped split isolation, strict target cutoffs, and refusal to choose from ties or training loss alone. Synthetic recovery verifies implementation; it does not validate the real-data E_a.

Preserve parser/preprocessing, original CSV/Parquet, `processed/`, `configs/preprocessing.json`, existing baseline predictions, original `arrhenius_thermal_factor`, existing tests and Stage 00–03 artifacts. Keep exact field names and aliases. No push, merge, existing official-result edits or production promotion.

## Final response

Return checked branch/SHA; inspected files; executed commands and actual test results; a table for all five E_a values with fitted rates and honest evidence status; the decision for each case and shared recommendation; deliverable links; unresolved limitations; and a go/no-go for Stage 04.

If the outcome is indeterminate, explicitly state what the experiment ruled out, which candidates remain compatible, why a unique choice is unsupported, and what prospective measurements or independent campaigns would resolve it. Do not fabricate a winner or report an unexecuted test as passed.

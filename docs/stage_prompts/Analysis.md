# PipeTwin — Deep VTD Consistency and ML Training Label Reliability Investigation

## 1. Role and Repository

You are a senior pipeline integrity engineer, corrosion researcher, in-line inspection (ILI/VTD) specialist, and machine learning data quality analyst.

You are investigating historical VTD measurements for the **Nedra Digital: PipeTwin** project.

**Repository:** https://github.com/Volgadon636/PipeTwin

**MANDATORY WORKING BRANCH: `model_v02_exp2`**

Before starting:

1. Fetch the latest changes from the remote repository.
2. Check out `model_v02_exp2`.
3. Verify the current branch name and commit hash.
4. Inspect the actual code, datasets, configurations, and documentation in this branch.
5. Use this branch as the sole authoritative source for the current implementation.

**Do not use `main`, `model_v02`, or any other branch as a substitute.**

If `model_v02_exp2` is unavailable, report the problem rather than silently switching branches.

Do not rely on previous versions of the project when determining the current schema, preprocessing logic, or ML target definitions.

## 2. Main Research Objective

Conduct a comprehensive investigation of anomalies in historical VTD measurements, specifically:

- Minimum remaining wall thickness increasing between consecutive inspections.
- Measured remaining wall thickness exceeding passport nominal thickness.
- Changes in the location of the deepest defect between inspections.
- Systematic patterns suggesting measurement inconsistency, incorrect defect matching, coordinate shifts, or other data quality problems.

The investigation must cover **every relevant anomaly** in the available historical dataset.

The goal is not simply to identify problematic measurements, but to determine **whether and how they should be used for ML training**.

Your final recommendations must distinguish between:

- Reliable observations.
- Plausible but uncertain observations.
- Likely inspection or preprocessing inconsistencies.
- Unverifiable observations.
- Records unsuitable for particular training targets.

Do not assume that every positive thickness change represents measurement error.

Do not assume corrosion physically reverses.

Do not invent missing inspection accuracy values, repair histories, or other explanations.

## 3. Initial Repository Audit

Before performing the analysis, inspect the relevant implementation in `model_v02_exp2`.

Find and understand:

- Raw and cleaned VTD data.
- Pipeline passport data.
- The 10-meter segment mapping logic.
- Historical VTD aggregation procedures.
- Training target construction.
- Historical inspection pairing.
- Existing quality flags and anomaly filters.
- ML baseline and experimental model data requirements.
- Available tests and reports.

Start by looking for relevant modules in `src/`, `src/ml/`, `processed/`, `data/`, `configs/`, `docs/`, and `reports/`, but discover actual paths rather than assuming they exist.

Relevant raw field names may include:

- `id`
- `date`
- `Distance`
- `Distance_adjusted`
- `Angle`
- `Residual Wall Thickness`
- `Defect Depth in %`
- `Defect Type`

Inspect the actual field names in the current branch.

Preserve existing field names exactly. Document all mappings between raw and normalized columns.

Determine whether VTD measurements use local or global distance coordinates.

Investigate whether the angular coordinate `Angle` has a documented orientation and whether angular positions are comparable between inspections.

Do not modify existing preprocessing or training logic during the investigation.

## 4. Reconstruct Historical VTD Inspection Pairs

Identify every pipeline section with at least two valid historical VTD inspections.

For consecutive inspections at dates \(t_1\) and \(t_2\), where \(t_2>t_1\), obtain the minimum remaining wall thickness of each 10-meter segment:

\[
h_{k,1}=h_{\min,k}(t_1)
\]

\[
h_{k,2}=h_{\min,k}(t_2)
\]

Calculate:

\[
\boxed{\Delta h_k=h_{k,2}-h_{k,1}}
\]

Interpretation:

- \(\Delta h_k>0\): measured minimum thickness increased.
- \(\Delta h_k=0\): measured minimum thickness remained unchanged.
- \(\Delta h_k<0\): measured minimum thickness decreased.

Also calculate:

\[
\boxed{
r_k^{obs}=\frac{h_{k,1}-h_{k,2}}{t_2-t_1}
}
\]

Express elapsed time in years and apparent corrosion rate in mm/year.

Distinguish measured apparent rates from actual physical corrosion rates.

For every pair, preserve identifiers, inspection dates, segment boundaries, measurement references, nominal thickness, inspection quality flags, and both minimum-thickness values.

Report exactly how many valid pairs can be constructed and why other pairs cannot be used.

Do not treat missing measurements as zero.

## 5. Analyze Every Positive Thickness Change

Find every pair satisfying:

\[
\Delta h_k>0.
\]

For each anomaly, determine:

### 5.1. Magnitude

Calculate:

\[
\Delta h_k=h_{k,2}-h_{k,1}
\]

and:

\[
\delta_k=
100\frac{h_{k,2}-h_{k,1}}{h_{k,1}}
\]

when \(h_{k,1}>0\).

Report:

- Absolute increase in mm.
- Relative increase in percent.
- Apparent negative corrosion rate in mm/year.
- Magnitude relative to nominal wall thickness.
- Duration between inspections.

Produce distributions including mean, median, standard deviation, percentiles, and extreme values.

### 5.2. Spatial and temporal concentration

Group anomalies by:

- Pipeline identifier.
- Simple-section identifier.
- Inspection campaign.
- Inspection date pair.
- 10-meter segment index.
- Position along the pipeline.
- Defect category where available.
- Nominal thickness and pipe geometry.

Determine whether anomalies are randomly distributed or concentrated within particular sections, campaigns, or spatial regions.

Investigate contiguous clusters of anomalous segments.

Pay particular attention to inspection-wide systematic offsets.

### 5.3. Local comparison

For every anomalous segment, compare thickness changes in neighboring 10-meter segments.

Determine whether the increase is:

- An isolated local phenomenon.
- Part of a cluster.
- A systematic change across a complete section.
- Associated with a sudden change in reference thickness or geometry.

Do not assume neighboring segments are statistically independent.

## 6. Analyze Measurements Above Passport Nominal Thickness

Compare measured remaining wall thickness with the applicable nominal wall thickness from the passport.

Calculate:

\[
\boxed{
e_{nom}=h_{measured}-h_{nominal}
}
\]

Flag all observations where:

\[
e_{nom}>0.
\]

Investigate both:

1. Individual VTD measurements exceeding nominal thickness.
2. Aggregated segment minimum thickness exceeding nominal thickness.

Keep these categories separate.

Where available, compare the measurement with the local reference wall thickness recorded during VTD, independently of the passport value.

Investigate:

- Section-specific inconsistencies.
- Differences between passport and VTD reference thickness.
- Possible manufacturing tolerances.
- Potential pipe replacement or repair, if documented.
- Inconsistent nominal thickness assignment.
- Unit conversion problems.
- Incorrect joins between pipeline sections.
- Possible changes in inspection calibration.

Do not automatically classify thickness above passport nominal as physically impossible.

Do not invent manufacturing tolerances.

Quantify exceedance magnitudes and identify systematic patterns.

## 7. Match the Deepest Defects Between Inspections

This is a critical part of the investigation.

For every 10-meter segment with an increasing minimum thickness:

1. Identify the defect responsible for the minimum thickness at \(t_1\).
2. Identify the defect responsible for the minimum thickness at \(t_2\).
3. Recover their source VTD records.
4. Compare their spatial locations.
5. Evaluate whether they plausibly represent the same physical defect.

Compare:

- Longitudinal position \(x\).
- Circumferential position \(\theta\).
- Remaining thickness.
- Defect depth.
- Defect length and width, if available.
- Defect type.
- Relevant pipe or weld identifiers.
- Local reference thickness.

Calculate longitudinal displacement:

\[
\boxed{\Delta x=|x_2-x_1|}
\]

and circular angular displacement:

\[
\boxed{
\Delta\theta=
\min(
|\theta_2-\theta_1|,
360^\circ-|\theta_2-\theta_1|
)
}
\]

Use degrees only when the source values are confirmed to be in degrees.

**Do not assume that the deepest defect in a segment at \(t_1\) must remain the deepest at \(t_2\).**

Distinguish:

- Same-defect match supported by coordinates and geometry.
- Possible match with uncertainty.
- Different defects responsible for the two minima.
- No suitable matching defect.
- Insufficient information.

If appropriate, perform one-to-one candidate defect matching using longitudinal position, circular angular distance, dimensions, defect type, and available inspection landmarks.

Do not choose arbitrary matching tolerances without documenting sensitivity.

Avoid circular reasoning: a physically implausible thickness change is not sufficient evidence by itself to declare two records unmatched.

Account for changes in spatial registration between inspection campaigns.

Where defensible, test whether a systematic longitudinal or angular alignment correction improves matching.

Do not alter source coordinates without preserving their original values.

## 8. Investigate Possible Diagnostic Inconsistencies

Analyze whether observed anomalies can be associated with recognizable data-quality patterns.

Potential explanations to investigate include:

- Instrument measurement uncertainty.
- Inspection calibration differences.
- Changed defect detection thresholds.
- Different defects being detected.
- Incorrect spatial registration.
- Segment boundary effects.
- Incorrect passport thickness.
- Incorrect VTD reference thickness.
- Unit conversion inconsistencies.
- Duplicate or inconsistent records.
- Different interpretations of defect depth.
- Repairs or replacements, where documented.
- Changes in inspection resolution or technology, where documented.

For each proposed explanation, classify the evidence as:

**Confirmed:** directly supported by repository records or documentation.

**Supported hypothesis:** consistent with multiple observations but not directly verified.

**Possible:** physically or technically plausible but not established.

**Unsupported:** no sufficient evidence.

Do not claim that an anomaly proves a particular diagnostic failure.

Where the data permit, quantify how much of the anomalous population each explanation could account for.

## 9. Statistical Investigation

Provide a comprehensive quantitative analysis.

At minimum, investigate:

- Frequency of positive thickness changes.
- Distribution of increase magnitudes.
- Distribution of apparent negative corrosion rates.
- Differences between inspection campaigns.
- Differences between pipeline sections.
- Spatial clustering of anomalies.
- Relationship between anomalies and defect location changes.
- Relationship between anomalies and passport thickness exceedances.
- Relationship between anomalies and available quality flags.
- Sensitivity to defect-matching tolerances.
- Sensitivity to segment boundary alignment.
- Sensitivity to measurement uncertainty assumptions.

Where statistical inference is attempted, account for the dependence between neighboring segments and between measurements from the same inspection campaign.

Avoid treating thousands of segments from one pipe as thousands of independent experimental replicates.

Use visualizations where they provide clear analytical value.

## 10. Determine ML Training Reliability

Develop a proposed observation reliability framework.

Suggested categories:

### A — High reliability

Measurements and spatial matching are consistent, with no significant unresolved data-quality issue.

### B — Usable with caution

Minor inconsistencies exist but the observation remains interpretable.

### C — Low reliability

Substantial uncertainty concerning defect matching, inspection comparability, or reference thickness.

### D — Unsuitable for the specified training target

The observation cannot support the intended physical interpretation without unjustified assumptions.

These are proposed analytical categories, not existing repository labels.

Define transparent criteria for each category based on observed evidence.

For every observation, assign a proposed category and list the reasons.

Distinguish reliability for different training objectives:

- Direct prediction of remaining thickness.
- Prediction of change between VTD inspections.
- Estimation of nonnegative physical corrosion rate.
- Defect-level growth estimation.
- Spatial defect-risk modeling.

An observation unsuitable for fitting corrosion growth may still be useful for learning measured remaining thickness.

Do not automatically exclude every pair with positive thickness change.

Do not silently replace negative observed corrosion rates with zero.

## 11. Compare Candidate Data-Handling Strategies

Evaluate several approaches:

**Strategy 1 — Retain all observations**

Train or evaluate using all eligible records, retaining anomaly flags.

**Strategy 2 — Reliability-based weighting**

Assign weights based on evidence-supported reliability scores.

**Strategy 3 — High-confidence subset**

Use only observations satisfying predefined reliability requirements.

**Strategy 4 — Robust regression**

Retain uncertain observations but reduce the influence of extreme residuals.

**Strategy 5 — Separate targets**

Treat measured remaining thickness and physically constrained corrosion loss as distinct prediction targets.

**Strategy 6 — Inspection-level calibration or offsets**

Investigate whether systematic campaign-level discrepancies can be estimated without future-data leakage.

For each strategy explain:

- Physical justification.
- Statistical consequences.
- Data requirements.
- Risk of introducing bias.
- Risk of reducing training sample size.
- Expected compatibility with the current PipeTwin ML architecture.
- Whether implementation is recommended.

If sufficient valid evaluation data exist, compare strategies experimentally on appropriately separated historical data.

If they do not, provide a reasoned recommendation and explicitly state that superiority has not been empirically established.

**Do not optimize anomaly rules using hidden or future target inspections.**

## 12. Required Deliverables

Create investigation outputs in a separate research directory, without overwriting existing processed datasets.

Use meaningful filenames, for example:

- `vtd_pair_audit.csv`
- `positive_thickness_changes.csv`
- `above_nominal_thickness.csv`
- `deepest_defect_matching.csv`
- `inspection_campaign_comparison.csv`
- `vtd_label_reliability.csv`
- `vtd_anomaly_analysis.md`
- `ml_training_recommendations.md`

These are proposed output names; adapt them to existing repository conventions.

Each report must be reproducible from the inspected `model_v02_exp2` commit.

The main research report must contain:

1. Executive summary.
2. Dataset and inspection coverage.
3. Historical pair construction methodology.
4. Statistical analysis of positive thickness changes.
5. Passport nominal-thickness exceedance analysis.
6. Defect matching and spatial displacement analysis.
7. Possible inspection inconsistency mechanisms.
8. Investigation of section-level and campaign-level patterns.
9. Proposed reliability classification.
10. Comparison of ML data-handling strategies.
11. Recommended training rules for `between_vtd`.
12. Recommended training rules for `cold_start`.
13. Remaining uncertainties.
14. Priority recommendations for the ML team.

For every finding, provide actual record counts, section identifiers, inspection dates, and numerical evidence.

Do not fabricate results.

## 13. Reproducibility and Validation

Provide a reproducible analysis script or notebook using the repository's existing conventions.

It must:

- Read available repository datasets.
- Preserve source field names.
- Use deterministic analysis steps.
- Keep original records intact.
- Validate units and types.
- Handle missing data explicitly.
- Preserve provenance and record identifiers.
- Avoid arbitrary silent filtering.
- Produce the required reports.

Include tests for:

- Correct signs of thickness changes.
- Time interval calculations.
- Circular angular distance.
- Missing measurement handling.
- Nominal-thickness comparison.
- Defect matching across segment boundaries.
- No duplicate counting of inspection pairs.
- Temporal data leakage prevention.

Run the analysis and tests where the environment permits.

Distinguish completed executions from proposed procedures.

## 14. Final Decision Required

Your conclusion must directly answer:

**Which historical VTD observations should be used for ML training, under what conditions, and why?**

Provide separate recommendations for:

1. Training corrosion-rate models.
2. Training direct remaining-thickness models.
3. Training models that use previous VTD history.
4. Training cold-start models without previous VTD measurements.

Also identify which anomalies should be preserved as useful evidence of measurement uncertainty rather than discarded.

The final output should provide **actionable, evidence-based guidance** that the PipeTwin ML developers can implement without independently repeating the entire investigation.

## 15. Execution Instructions

Work directly from the latest accessible commit of **`model_v02_exp2`**.

Begin by verifying the branch, commit hash, current repository structure, and data schemas.

Then perform the complete analysis rather than merely describing how it could be done.

Do not ask for confirmation between stages unless an essential dependency is missing.

Do not modify production preprocessing or ML training behavior without explicit authorization.

Do not open a pull request or merge changes unless explicitly requested.

Finish with a concise summary of confirmed findings, unresolved uncertainties, generated reports, and recommended next actions for the ML team.

**The primary goal is to establish reliable training-label rules based on actual inspection evidence, not to force all historical VTD observations into a physically monotonic corrosion model.**
# PipeTwin: comprehensive parser, mentor follow-up, and evaluation prompt

Prepared 15 November 2026. Copy everything from **BEGIN PROMPT** through **END PROMPT** into your coding agent. This is an implementation brief, not a claim that the proposed features are implemented.

## BEGIN PROMPT

You are the senior data engineer and scientific software engineer for **PipeTwin_project**. Build a reliable, reproducible ingestion and preprocessing system for pipeline wall-thickness forecasting. Extend the existing repository rather than replacing working components. Incorporate the available mentor discussions into traceable requirements and maintain a repeatable follow-up workflow for subsequent calls.

Your responsibility includes parsing, source identification, schema normalization, unit validation, spatial and temporal alignment, feature generation, quality reporting, and interfaces to the existing forecasting and evaluation code. A parser alone does not predict thickness: keep ingestion, feature construction, model fitting, and evaluation as separate stages with explicit contracts.

First produce a concrete implementation plan and then implement it in the authorized workspace. Continue through verification and documentation. Ask only about material ambiguities that cannot be resolved from evidence; continue independent work while those ambiguities remain open. Do not silently choose an interpretation that changes the scientific target or permits information leakage.

### 1. Project objective and non-negotiable outputs

The target is the **minimum residual wall thickness, in millimetres, for each 10 m segment on its forecast date**. It is not mean thickness, defect depth, corrosion rate, or a score without physical units.

Current production requests, subject to verification against current configuration:

| Pipeline ID | Forecast date | Scenario | Previously reported segment count |
|---|---|---|---:|
| 1250022871 | 2023-12-01 | Between VTD inspections | 525 |
| 1250009583 | 2022-07-01 | Cold start from commissioning | 1,475 |

The current total is 2,000 target segments. Recompute it from verified lengths and segmentation rules; do not manufacture rows to satisfy a hard-coded count when inputs change. Diagnose any disagreement before publishing.

The prediction export is exactly:

```csv
id,segment_id,thickness_pred
```

One unique row per requested segment, numeric thickness in mm, decimal point, no index column, missing predictions, infinity, duplicate keys, or unrelated pipes. If an authoritative `sample_result.csv` is available, validate its schema and compare the complete key set. Its name alone does not prove it is a submission template. If no template exists, explicitly report validation against the derived grid instead.

Produce a separate UZT inspection-priority table containing pipeline ID, forecast date, segment ID, start/end distance, predicted minimum thickness, nominal thickness, relative thinning, rank, evidence quality, and the reason for priority. Keep it out of the three-column submission.

### 2. Begin with a repository and evidence audit

Read applicable AGENTS.md instructions and inspect the current working tree before editing. Preserve existing user changes. In the snapshot used to prepare this prompt, `processed/target_segments_grid.csv` was modified and `scripts/` was untracked; those observations may have changed. Do not reset, overwrite, or clean them.

Read and reconcile at least:

- `README.md`.
- `Резюме встречи с менторами.md`.
- `docs/cleaning_rules.md`, `docs/verification.md`.
- `agent/PipeTwin_ML_Technical_Specification_EN.md` and `agent/PipeTwin_GitHub_ML_Start_Prompt_EN.md`.
- `docs/ml_model_card.md`, `docs/model_v2.md`.
- `configs/preprocessing.json`, `src/config.py`.
- `src/format_detector.py`, `src/data_cleaner.py`, `src/grid_builder.py`, `src/segment_mapper.py`, `src/temporal.py`, `src/storage.py`, `src/validator.py`, `src/data_prep.py`.
- Existing tests and `src/ml/`, especially evaluation, exposure, audit, and submission validation.
- Current input headers, quality reports, mapping reports, run manifests, and model reports.
- `scripts/calculate_segment_temperature.py`, if still present.
- Relevant case specifications and mentor material available in the repository, supplied attachments, or explicitly connected project sources. Search document contents, not just names.

Use the correct encoding when reading Russian text; console mojibake is not evidence that the underlying file is corrupt. Do not rewrite a source merely to fix its display in a terminal.

Build an evidence inventory: exact path/source reference, content hash, document type, stated meeting date, participants if supplied, coverage, and limitations. The current mentor summary does not identify the meeting date or participant list. Preserve those unknowns. A summary is secondary evidence, not a verbatim transcript of every call.

If a file disappears or changes during discovery, refresh its metadata, record the change, and use one consistent source snapshot per run. Do not claim to have processed a missing file. Do not assume a `START_DATA` folder exists.

Deliver an initial gap analysis distinguishing existing functionality, missing functionality, implementation defects, unresolved domain questions, and previously reported results that require recomputation. Do not treat an old report as a fresh execution result.

### 3. Resolve file meaning independently of filenames

There is an important legacy naming conflict:

| Semantic source | Actual/possible filename | Recognition evidence |
|---|---|---|
| Passport | `passport.csv`, `pot_south.csv` | Pipeline identifier, dimensions, length, commissioning and material fields |
| Route profile | `profile.csv`, `altitudes_south.csv` | `id`, `total_distance`, `z`; possibly coordinates |
| VTD observations | `vtd.csv`, `vtd_south_train.csv` | `Distance`, `Residual Wall Thickness`, inspection date and defect fields |
| UZT measurements | `uzt.csv`, `uzt_south.csv` | Control point, distance, date, angle, residual thickness |
| Technology/hydraulic regime | `tech_mode_for_embedings_with_length_with_temperatures_ids16.csv`, download suffix variants such as `(2)`, or a new `regime.csv` | `simple_section_id`, `calc_date`, `q`, `q_oil`, `q_gas`, `watercut`, pressures and temperatures |
| Chemistry/corrosion monitoring | `repcorr_south.csv`, `repcorr_south(1).csv`, `corrosion_monitoring.csv`, or legacy `data/regime.csv` | `Simple Section ID`, sample installation/removal dates, UKK fields, pH and chemistry |
| Submission template | `sample_result.csv`, only if its contents match | `id,segment_id,thickness_pred` |

**In the existing `src/config.py`, `regime.csv` is an alias for RepCorr monitoring, and Techmod is a separate input.** Conceptually Techmod supplies the daily regime in the case-description table. Renaming Techmod to `data/regime.csv` without updating the loader would break that contract.

Introduce stable internal source types such as `passport`, `profile`, `vtd`, `uzt`, `hydraulic_regime`, `corrosion_monitoring`, and `submission_template`. Select explicit adapters using schema fingerprints and configured aliases. Report conflicting filename and schema evidence. Never choose between two plausible source types silently.

Support old and intended file layouts through adapters during migration. Preserve existing public APIs and output names where feasible; document deprecations and provide a versioned compatibility map for changes. Identical duplicate copies may be deduplicated with provenance. Conflicting copies must require explicit source selection; never silently concatenate them or prefer the newest-looking name.

Distinguish required sources by command/scenario. Passport is required for the grid; VTD history is required for the between-inspection scenario; cold start must not require target-pipe VTD/UZT features. Missing optional monitoring must produce a usable run with missingness flags, not fabricated chemistry. Define and report the policy for missing profile or regime inputs. Validation must not require a hidden test file.

### 4. Loss-aware parser and versioned schema contracts

Implement a schema registry containing source type, schema version, required and optional fields, aliases, data types, expected units, semantic descriptions, nullable rules, checks, and feature eligibility.

For every file:

1. Record path, SHA-256, byte size, encoding, delimiter, schema fingerprint, adapter version, record count, and source role.
2. Support UTF-8/BOM and Windows-1251; comma, semicolon, and tab delimiters; quoted delimiters, embedded newlines, whitespace, and decimal commas. Test ambiguous delimiter/decimal combinations instead of guessing from one character count.
3. Parse identifiers without floating-point round trips. Accept integer-like legacy `1250013653.0` only after exact validation; preserve original text and leading zeros when semantically meaningful. Reject fractional/nonfinite/invalid identifiers.
4. Parse ISO and unambiguous Russian day-first dates per value. Preserve timestamp precision, declared timezone, and whether a value is date-only. Ambiguous slash dates require a declared locale or quarantine. Do not infer a timezone from the machine running the parser.
5. Handle nonbreaking spaces and confirmed thousands separators. Distinguish absent values, parse failures, not-applicable values, censored values, and genuine zero. Do not turn all zeros or strings such as `<0.1` into numeric zero.
6. Detect duplicate raw headers and canonical-name collisions before pandas can obscure them. Preserve distinct archival fields. Block ambiguous collisions in analytical fields.
7. Preserve unrecognized columns in the archival layer and list them in the schema-drift report. Never silently drop fields just because they are not yet used by the model.
8. Handle malformed records with source record number, original content or safe reference, and error code. A CSV record number is not always a physical line number because quoted records can span lines.
9. Keep raw values immutable. Store normalized values, usable values/masks, transformation IDs, and issue references separately. CSV exports must not become an Excel-formula execution path; use a separate spreadsheet-safe text export if necessary without changing the analytical source values.
10. Make retries and repeated runs idempotent. Do not use an LLM to invent cell values or perform nondeterministic numeric parsing. An LLM may propose schema aliases or extract mentor statements, but accepted mappings and transformations must be explicit, versioned, and testable.

For every rejected record or masked value provide an actionable reason. Report row reconciliation with disjoint disposition counts; multiple quality flags on one record must not be counted as multiple dropped records.

### 5. Complete field and feature coverage

Maintain a feature inventory for **every field in the case-description table and every additional supplied field**. “Use all metrics” means account for every metric, evaluate its eligibility, and use justified inputs; it does not mean feed unavailable, leaking, ambiguous, or redundant fields into a model.

For each canonical field record: source column, source type, scope/grain, raw and normalized unit, temporal availability, missing percentage globally and per target/fold, transformation, feature status, exclusion reason, and mentor question/decision ID where relevant.

Allowed statuses include `measured`, `source_calculated`, `derived_verified`, `assumed_scenario`, `available_but_excluded`, `unresolved_semantics`, and `not_provided`. Keep record-level validity separate from field-level status.

#### 5.1 Passport, profile, VTD, and UZT

| Required concept | Canonical representation and handling |
|---|---|
| Pipeline ID and hierarchy | `id`, separate parent/direction IDs; explicit crosswalks rather than name guessing |
| External diameter | `diameter_outer_mm`; maintain legacy alias where needed |
| Nominal wall | `nominal_thickness_mm`; separate from local VTD wall and rejection threshold |
| Steel and internal coating | Material/coating categories; unknown is not equivalent to absent |
| Commissioning date and length | Date with provenance, `length_m`; distinguish accounting/network lengths |
| Route coordinate and elevation | `distance_m`, `elevation_m`; preserve original coordinate datum if known |
| Inspection date | Campaign ID and date/time with source availability |
| VTD coordinate | Use `Distance`; preserve but do not substitute `Distance_raw` |
| Circumferential position | Original angle and unit/convention; normalize only after confirmation |
| Defect length/width/depth/type | mm, percent of wall where confirmed, categorical type; keep raw values |
| Residual wall thickness | Measured residual mm, quality and target eligibility; primary label source |
| UZT control point | Control point/measurement IDs, pipe-relative distance, date, angle, thickness |

Additional passport fields, including repair/maintenance history, external protection and rejection thickness, require an as-of availability audit. Present-day summaries must not become historical features merely because the passport is treated as static.

#### 5.2 Techmod mappings

| Raw field | Canonical meaning / action |
|---|---|
| `simple_section_id` | `id`, subject to identity validation |
| `calc_date` | Regime calculation timestamp; distinguish calculation time from observation/availability if documented |
| `p_start_fact`, `p_end_fact` | Reported factual inlet/outlet pressure; keep separate from calculated pressure |
| `p_start`, `p_end` | Calculated/source pressure; units and gauge/absolute basis require confirmation |
| `q` | `q_liquid`; verify volume/time and operating/standard conditions |
| `q_oil`, `q_gas` | Oil/gas rates with phase-specific unit and condition metadata |
| `watercut` | `watercut_pct`; convert to fraction only in an explicitly named derived field |
| `gas_content_rate` | Preserve; do not equate with `q_gas` without its definition |
| `v_liquid_true`, `v_mix` | Separate liquid and mixture velocities; verify meanings and units |
| `flow_regime_real` | Source flow-regime category; do not overwrite with a Reynolds classification |
| `fill_coefficient` | Preserve and verify whether fraction or percent and what phase it describes |
| `density_liquid_work`, `density_oil` | Separate densities; units and reference conditions required |
| `viscosity_liquid_work` | Verify dynamic versus kinematic viscosity and units before calculations |
| `length` | Source hydraulic section length; reconcile with passport length |
| `d_mm` | Misleading name in observed data: values such as 0.219; convert only under a passport-checked, documented rule |
| `t_start`, `t_end` | Inlet/outlet temperature; source and validity tracked independently |
| `t_mix` | Preserve separately; its physical meaning must be verified before substituting for inlet/outlet temperature |
| `pit_corrosion_intensity` | Archive and exclude from predictors until provenance and leakage audit pass |

Do not infer hydraulic segment boundaries from row order or assume a `simple_section_id` equals a 10 m segment ID. The current target identifiers refer to complete evaluated pipes. If hydraulic sections subdivide them, require an explicit section-to-pipe interval map.

#### 5.3 RepCorr monitoring and chemistry mappings

| Raw field/group | Handling |
|---|---|
| `Simple Section ID`, protection direction/names | Identity checks and conflicts; no automatic join by similar names |
| `UKK Number`, `UKK Name`, UKK distance | Monitoring station identity and point location |
| Installation/removal dates, `Exposure Time, days` | Preserve declared and calculated exposure durations; flag mismatches |
| `Qzh`, `H2O`, `Fluid Density` | Monitoring-associated liquid rate, watercut, density; not silent replacements for daily Techmod |
| Average/maximum gravimetric rates, ER rate | Separate rate fields, method and exposure interval; eligibility/leakage audit before feature use |
| `Sample Type`, `Sample Metal Grade`, `Measurement Method` | Method metadata relevant to comparability |
| `Corrosion Inhibitor Grade`, `Protective Effect`, `OSI in Water Phase` | Distinct properties; none automatically equals inhibitor dose |
| `pH` | pH with observation time and measurement provenance |
| `CO2 in Water Phase`, `O2`, `H2S` | Separate dissolved constituents; unit verification required |
| Hardness, `Ca2+`, `Mg2+`, `Cl-`, `SO4-`, `HCO3-` | Preserve individually; normalize units only with evidence |
| Total mineralization, total Fe | Concentration fields with units and timestamps |
| `SVB Cell Content` | Preserve biological/microbiological measurement with verified definition and units |
| `Mechanical Impurities` | Candidate for КВЧ only after confirming the abbreviation, measurement method and units |
| Flow phase state, water content/state at 0/50/100/150/200 | Preserve source labels; establish what those levels mean before deriving features |

Distinguish dissolved CO2 concentration from gas-phase CO2 fraction and partial pressure. Do not convert among them without the missing physical inputs and a justified method. Do not infer inhibitor dosage from its name, protection effect, or residual concentration without an explicit model and evidence.

If pH, CO2, КВЧ, inhibitor dose, or Reynolds number cannot be obtained reliably for a target/date, emit null plus a reason and coverage indicator. Never borrow a later observation or unrelated pipe's median to make the table look complete.

### 6. Spatial alignment and observed thickness

Use one authoritative grid per verified pipeline version:

```text
n_segments = ceil(length_m / 10)
start_m(k) = 10*k
end_m(k) = min(10*(k+1), length_m)
segment_length_m(k) = end_m(k) - start_m(k)
```

For interior points use `segment_id = floor(distance_m / 10)`. Current implementation assigns a point exactly at pipe length to the final segment; preserve this documented convention until an authoritative template or organizer clarification changes it. Do not create an extra endpoint segment. Exclude negative/out-of-range coordinates with reasons and explicitly configured numerical tolerance. Do not clip obviously invalid coordinates into the grid.

Interpolate route elevation over actual distance, with duplicates and conflicting heights handled explicitly. Use piecewise-linear, length-weighted elevation aggregates and boundary-based slope over the actual segment length. Report gaps, endpoint extrapolation, abrupt steps, and coverage. Do not smooth away a real elevation discontinuity by default.

Aggregate VTD by pipe, campaign/date, and 10 m segment. Compute the minimum over **eligible measured residual thickness values**, preserving observation count, valid count, rejected count and detection coverage. Missing records do not mean intact wall. Minima from sparse versus dense inspection campaigns have different ascertainment and require comparability diagnostics.

Apply the existing passport-nominal ceiling rule at observation level before taking minima: above-nominal measurements beyond the configured tolerance are retained for audit but excluded from labels. The current tolerance is 0.02 mm; treat it as a repository policy, not a universal instrument specification. Keep local VTD wall comparison separate. Do not clamp an invalid observation to nominal and then use it as measured truth.

The documented 2018-09-23 fraction-to-percent defect-depth conversion is source/campaign-specific. Confirm it on the current export; do not multiply every value below one by 100. Missing residual thickness must not automatically be reconstructed from defect depth for the target. If an explicit reconstruction is later approved, mark it as derived and evaluate it separately from measured labels.

Retain pre-commissioning observations but exclude them from eligible training targets under the current rule. Flag suspected repairs/replacements and chronology conflicts; do not invent repair dates or reinterpret the commissioning date to fit observations.

Do not spread a defect's minimum across every overlapped segment without an approved interpretation of defect start/centre and length. Do not assume angle 180 degrees is the physical bottom until instrument orientation is established.

UZT remains a separate measurement source. Investigate differences in point identity, wall reference, location and method before merging it with VTD. Previously reported all-above-nominal UZT measurements must remain excluded under the current policy unless supported evidence changes that policy.

### 7. Temporal alignment and leakage prevention

Preserve the current conservative cutoff: `timestamp < prediction_date at 00:00` in the declared data time convention. Same-day observations are excluded unless a reviewed requirement explicitly permits them with known observation times. In particular, do not include the 2023-12-01 VTD campaign merely because its calendar date equals the forecast date.

Keep `event_time`, exposure start/end, `available_at` where known, extraction time, and the forecast cutoff separate. For monitoring, a measurement made from a removed coupon is not available before removal; if reporting delay is unknown, label any removal-date availability proxy. A coupon spanning beyond cutoff cannot be used by prorating its final measured rate backwards.

Apply time eligibility **before** anomaly persistence decisions, temporal imputation, aggregation, reference fitting, or feature selection. A future third day must never validate an earlier two-day anomaly in a historical forecast.

Preserve hard physical checks separately from soft outlier filters:

- Persistent soft anomalies may remain after at least three distinct consecutive observed calendar days before cutoff.
- Multiple rows on one date count as one day; missing dates break persistence.
- Persistence does not make negative density or another physically impossible value valid.
- Preserve raw/source values, quality flags, and per-feature usable masks.
- State and test any dependency masking, such as masking derived oil rates or velocities associated with an invalid liquid-rate spike.

Use daily medians for repeated calculations when supported by the existing policy. Keep within-day variability/counts available. Clearly distinguish an average across observed days from a duration-weighted historical exposure integral. Do not forward-fill years of missing data from one daily record.

For each feature window record expected/observed days, usable days, coverage fraction, longest gap, latest event/availability time, and age of last observation. Define the denominator from the actual requested interval, constrained by commissioning and scenario start where appropriate.

Build features independently for each requested forecast date and each validation fold through `build_features_asof` or a compatible API. Never reuse production-cutoff features in earlier historical folds.

Cold start must exclude target-pipe VTD/UZT-derived inputs even if present in the archive. Training labels from other eligible pipes are governed by each fold's temporal and grouping rules. For between-VTD forecasts, only eligible earlier diagnostic history may be used.

Add an automated leakage audit covering source times, label-derived fields, static passport diagnostic summaries, global imputation, scaling/feature selection, and shared parent pipelines. Assert that adding future rows cannot alter features for an earlier cutoff.

### 8. Join policy and transfer of sparse chemistry

Join by validated entity, location, and time, with declared cardinality and unmatched counts. Test against accidental many-to-many multiplication.

A matching numeric ID alone is insufficient when direction/name metadata conflicts. Maintain an evidence-backed crosswalk with validity intervals and source references for changed IDs. Fuzzy matching may propose candidates for review but must not establish production identity automatically.

Do not propagate a point measurement across the whole pipe or across dates without an explicit scope/validity policy. Where approved, distinguish `point_observation`, `pipe_level_assumption`, and `transferred_estimate` in outputs.

If transfer from another pipe is explored, make it an optional model experiment: use only training-fold data, physically justified similarity criteria, target-environment distance checks, uncertainty/sensitivity, and an ablation against no transfer. It is not a parser repair. Never disguise transferred chemistry as measured chemistry.

Previous reports recorded 412 unresolved monitoring IDs, 83 direction conflicts and zero usable monitoring records for both production targets. Recompute these counts. If they persist, report the resulting missing chemistry honestly.

### 9. Derived physics and temperature features

Derived features require verified dimensions, eligible input timestamps, explicit formulas, dependency tracking, quality flags, and a distinction between source measurement and model assumption.

Candidates include water fraction, inlet/outlet pressure difference, elevation drop, slope, local low-point proxies, pipe age, time since previous VTD, hydraulic variability and coverage, Reynolds number, and temperature gradients. Feature selection must still be validated; more features are not automatically better.

For liquid Reynolds number:

```text
Re = rho_liquid * v_liquid * D_internal / mu_dynamic
or Re = v_liquid * D_internal / nu_kinematic
```

Use consistent SI units and positive diameter/viscosity. Do not mix dynamic and kinematic viscosity. `D_internal = D_outer - 2*h_nominal` is an approximation unless internal diameter is supplied, and its assumptions must be recorded. Multiphase flow-regime interpretation must not be reduced to single-phase Reynolds thresholds without qualification. A missing viscosity means Reynolds is unavailable, not zero.

Temperature output must distinguish:

1. **Whole-pipe/section endpoint change:** `delta_temperature_c = t_end - t_start` when both endpoints refer to the same eligible section/time.
2. **Interpolated segment change:** explicitly assumed interpolation between measured/calculated endpoints; segment length and section boundaries matter.
3. **Physics-scenario segment change:** a heat-transfer calculation using declared parameters, not a measurement.

The existing `scripts/calculate_segment_temperature.py` uses illustrative assumptions including ground temperature 5 C, overall heat-transfer coefficient 1 W/(m2 K), heat capacity 3,500 J/(kg K), and roughness 0.00005 m. It writes `delta_t_c` into the target grid. These are not verified thermal properties or observed 10 m temperatures. Audit its dimensional and physical choices, including heat-transfer perimeter, liquid/mixture mass flow, steady-state assumptions and averaging. Do not silently certify its outputs as physically calibrated.

If retaining a steady-state scenario, use a documented formulation such as:

```text
dT/dx = -U*P/(mass_flow*cp) * (T - T_ground)
T_out = T_ground + (T_in - T_ground)*exp(-U*P*segment_length/(mass_flow*cp))
delta_temperature_c = T_out - T_in
```

Define the heat-transfer perimeter and U on the same area basis. Confirm the effective mass flow and cp before calling it a mixture model. Zero/unknown flow does not support this steady-flow solution. Use actual segment length and propagate outlet temperature into the next segment. At changing properties, solve piecewise with continuity. Negative temperature change means cooling under this convention.

Put assumed scenarios in separate artifacts with parameter sets and sensitivity ranges. Keep verified geometric grids immutable. Prefer calculating from eligible daily states before aggregating nonlinear quantities; explain the approximation if only aggregate data exist. Do not turn arbitrary assumptions into spatial signal solely to improve ranking or correlation.

Sediment effects can remain excluded according to the current mentor summary. Preserve supplied sediment-related observations archivally, but do not add a speculative sediment-protection model by default.

### 10. Mentor-call ingestion and follow-up workflow

Build a repeatable evidence-to-requirement workflow for **all supplied or accessible project calls and notes**. Do not claim access to private Telegram history, recordings, or meetings that have not been supplied. Discover available relevant connectors only where needed. If some calls are missing, list the coverage gap and continue with the available record.

Accept text/Markdown transcripts and notes initially; add documented adapters for PDF/DOCX if such inputs are available. For supplied audio, use an available transcription capability and retain timestamp/uncertainty information; never fabricate a transcript. Store original notes separately from extracted structured records.

For each call/source extract:

- Stable source ID and content hash; stated date/time, participants, and source locator.
- Exact supporting excerpt and page/line/timestamp when available.
- Decisions, proposed ideas, assumptions, open questions, action items, data requests, and corrections as distinct record types.
- Technical topic, affected fields/pipelines/dates, confidence in transcription, explicit attribution or unknown speaker.
- Status, owner only if supplied/assigned, due date only if stated, dependencies, and a link to implementation/tests.

Use stable IDs such as `DEC-0001`, `Q-0001`, `ACT-0001`, and `REQ-0001`. IDs must survive re-ingestion. Corrected source text creates a revision; it must not erase the previous interpretation.

Maintain a requirement matrix:

```text
requirement_id, statement, status, source_id, source_locator,
evidence_type, affected_component, implementation_reference,
test_reference, unresolved_question_ids, supersedes_id
```

Statuses should separate `proposed`, `confirmed`, `conflicting`, `implemented`, `verified`, `blocked`, and `superseded`. “Implemented” requires actual code; “verified” requires observed test evidence. A mentor suggestion is not automatically a confirmed instruction.

Resolve priority by explicit authority and scope: formal submission/evaluation requirements govern the submission contract; explicit organizer clarifications may amend them; confirmed mentor decisions guide domain interpretation; team implementation choices must be labelled as such. Later comments do not automatically override a formal contract. Flag unresolved conflicts, preserve both sources, and keep the existing conservative behavior until resolved.

After ingesting a new call, generate:

1. A concise call summary and newly confirmed decisions.
2. Differences from existing requirements, including superseded decisions and contradictions.
3. An action list with known owners/dates and blocking dependencies.
4. A next-call agenda ordered by impact on correctness, evaluation, then improvement.
5. Draft follow-up questions, each tied to concrete evidence and the decision it would unlock.
6. An implementation impact report: affected schema/config, artifacts, models and tests; required reruns and comparisons.

Example question format: “The current monitoring join rejects N rows because the numeric ID agrees but direction differs. Can you provide the historical ID/direction crosswalk, including effective dates? Until then these records are excluded.” Replace N with the freshly measured count; do not reuse old numbers as current facts.

Record the existing confirmed topics: residual thickness target; `Distance` rather than raw distance; three-day persistence; Techmod versus RepCorr meaning; target IDs/dates; r-squared distinction; sediment exclusion; UZT-priority output; and observation-level nominal-wall filtering from the cleaning document. Preserve provenance strength: a cleaning document's reference to a second meeting is not proof that the second meeting transcript is available.

Open questions to carry forward unless answered by newer evidence:

- Authoritative submission grid, endpoint rule and incomplete final segment.
- Units and physical basis of pressure, viscosity, rates, `d_mm`, `t_mix`, КВЧ, CO2 and inhibitor measurements.
- Whether same-day 2023-12-01 diagnostics are allowed; do not change cutoff before resolution.
- Historical ID/direction mapping for RepCorr and permitted spatial/temporal transfer.
- Cause/methodology of UZT values above nominal and differing VTD wall references.
- Campaign detection completeness and interpretation of apparent thickening.
- Train/test role of 1250009583 and permitted cold-start calibration evidence.
- Measured ground temperature/thermal properties, or whether a temperature scenario is desired at all.
- Critical thickness/ranking policy; whether passport rejection thickness is applicable at the forecast date.
- Repair/replacement history and how it resets segment history.

This workflow prepares local follow-up drafts. Do not send messages, contact mentors, join calls, create invitations or promise ongoing monitoring without explicit authorization and available capabilities. New supplied notes can be processed on the next invocation; unattended execution requires an explicitly configured automation.

### 11. Model integration and complete metric definitions

Reuse existing model interfaces and baselines. Do not replace the scientific model as an incidental parser refactor. First demonstrate unchanged results for unchanged accepted inputs/policies, then evaluate intentional changes separately.

Implement or verify all requested metrics on aligned, eligible held-out observations. Let `y` be measured residual thickness and `p` the prediction:

| Metric | Definition | Required interpretation |
|---|---|---|
| Pearson r | `corr(y,p)` | Include signed correlation to expose inverse rankings |
| Pearson r squared | `corr(y,p)^2` | Organizer metric; undefined for constant vectors or too few observations |
| R squared | `1 - sum((y-p)^2)/sum((y-mean(y))^2)` | Distinct from Pearson squared; can be negative; undefined for constant truth |
| MAE | `mean(abs(p-y))` | mm |
| RMSE | `sqrt(mean((p-y)^2))` | mm |
| MAPE | `100*mean(abs(p-y)/abs(y))` | Percent; explicitly define zero/near-zero policy and effective sample count |
| Bias | `mean(p-y)` | mm; positive means overestimated remaining wall |

Preserve and document the existing evaluator's behavior when changing it. It currently suppresses MAPE if any truth value is at or below 0.1 mm. Do not silently replace this with a different sample-selection policy. If adding an eligible-subset MAPE, label it separately, give excluded counts, and keep the threshold configurable. Do not use epsilon to hide division by zero. Undefined scores must be JSON null with a reason, not 0, 1, infinity or fabricated precision.

Report metrics separately by scenario, pipe, independent fold and forecast-horizon bucket. Include model/version, training and validation time ranges, sample count, number of independent pipes/parents/campaigns, coverage, exclusions, and baseline comparison. Predefine horizon buckets from available data and document empty ones. Separate macro averages from pooled scores; pooled correlations across unlike pipes can be misleading.

Validation priorities:

1. Forward-time forecasts with all processing refitted/recomputed within the fold.
2. Held-out pipe or parent pipeline to reduce shared-network dependence.
3. Campaign-level holdouts where supported.
4. Spatial block masking only as an interpolation diagnostic, not future-corrosion validation.

No random neighboring-segment split as the main validation evidence. Do not present zero eligible forward folds as successful validation. Current reports describe a lack of eligible independent forward fitted-model folds; recheck, then state the limitation if it remains.

Compare at least nominal-wall cold-start baseline, last-eligible-VTD persistence where available, and existing nonnegative constant-rate/physics-informed models as applicable. Distinguish fitted parameters from engineering priors. Do not optimize hyperparameters or feature inclusion using hidden or production-date target labels.

Where truth supports ranking evaluation, report top-10-percent overlap/recall with exact truth eligibility, selection denominator and tie policy. Use within-pipe ranking by predicted minimum thickness as an explicit baseline; optionally report relative thinning. Define `ceil(0.10*N)` or another approved rule. Tied constant predictions do not establish spatial discrimination; deterministic tie breaking is for reproducibility only. Do not call a segment critically unsafe without an applicable threshold.

Include observed-vs-predicted plots, residuals, and thickness-versus-distance profiles when evaluations exist. Bootstrap uncertainty only at an appropriate independent group level with sufficient groups. Parameter-sensitivity bands are scenario bands, not calibrated confidence intervals. Hidden-test scores remain unavailable until organizers provide them.

### 12. Quality metrics, lineage and run outputs

Publish machine-readable and human-readable quality reports covering:

- Source role, row/column counts, schema drift, parsing success/failure and raw-value preservation.
- Unit resolution and conversion counts; per-field reasons for exclusion.
- Exact/conflicting duplicates and key uniqueness.
- Spatial coverage, endpoint cases, out-of-pipe coordinates and profile gaps.
- Date coverage, invalid dates, pre-commissioning records, same-day/future exclusions.
- Monitoring identity conflicts, unmatched IDs, join cardinality and expansion checks.
- Measured/derived/assumed/missing feature coverage per pipe, scenario and cutoff.
- VTD nominal-wall violations, valid minima, UZT exclusions and campaign comparability.
- Three-day filter decisions, persistent anomalies, hard-invalid values and daily coverage.
- Thermal/Reynolds eligibility and missing dependency reasons.
- Number of labels, independent groups, eligible validation folds and undefined metrics.
- Runtime and peak memory for each stage, measured on the recorded environment.

Maintain aggregate-to-source lineage without copying enormous row lists into each output cell. A feature value should resolve through a lineage ID to source hashes, contributing record IDs, temporal window, rule/config version and transformations. Retain traceability for minima, medians and interpolated profile/thermal features.

Write outputs into a versioned staging directory and publish only after required validation succeeds. Preserve the prior successful run on failure. Use atomic manifest/pointer publication or a documented equivalent; independent file replacements alone must not leave a mixed-version dataset. Write the manifest last.

The manifest must include source hashes, schema/config hashes, Git revision and dirty-state fingerprint, dependency/runtime versions, seed where applicable, source snapshot, scenario cutoffs, row accounting, artifact hashes, rule/decision versions, test results, runtime, warnings and final status. A Git commit alone is insufficient when working files are modified.

Suggested artifacts, adapted to existing names rather than duplicated blindly:

```text
normalized/<source>.(csv|parquet)
processed/segments_grid.*
processed/target_segments_grid.*
processed/vtd_by_segment.*
processed/uzt_by_segment.*
processed/profile_by_segment.*
processed/hydraulic_asof.*
processed/monitoring_mapped.*
processed/target_features.*
processed/training_targets.*
processed/quality_issues.*
processed/<source>_rejected.*
processed/temperature_scenarios.*          # only if explicitly enabled
reports/source_inventory.json
reports/schema_mapping.json
reports/feature_inventory.csv
reports/data_quality.md
reports/validation.json
reports/lineage.*
reports/run_manifest.json
docs/mentor/requirements.*
docs/mentor/decisions.*
docs/mentor/open_questions.*
docs/mentor/follow_up.md
experiments/<run>/metrics.json
experiments/<run>/validation_predictions.csv
experiments/<run>/risk_ranking.csv
experiments/<run>/result.csv
```

CSV/Parquet must agree on keys and values within declared numeric precision. Validate nulls, dates and integer identifiers across round trips. Keep diagnostic columns out of the submission.

### 13. Implementation phases and checkpoints

**Phase A — Evidence and contract.** Inventory actual files/calls, capture source hashes and user changes, map schemas, reconcile requirements, document blockers. Deliver source/feature inventories and an implementation plan identifying exact existing functions to extend.

**Phase B — Adapters and compatibility.** Introduce semantic source IDs, versioned schemas, optional-field handling, deterministic parsing, old/new layout support and row-level provenance. Demonstrate that the legacy chemistry `regime.csv` and technology `regime.csv` cannot be confused.

**Phase C — Quality and alignment.** Implement unit policies, explicit joins, geometry, measurement eligibility, quarantine and daily temporal processing. Retain raw values and demonstrate row reconciliation.

**Phase D — As-of features.** Build cutoff-specific tables, coverage indicators and leakage checks. Add only supportable derived fields. Isolate temperature scenarios and unresolved chemistry. Keep scenario-specific diagnostic access rules enforced in code.

**Phase E — Mentor follow-up.** Add deterministic source ingestion and structured decision/action extraction with evidence. Generate requirement changes and follow-up drafts. Demonstrate idempotency on the existing notes and a synthetic new-note fixture; never label a synthetic call as real.

**Phase F — Evaluation and exports.** Integrate with current experiments, verify all metric definitions and failure cases, compare baselines and generate schema-valid outputs where inputs permit. Do not make parser success depend on inventing a validation fold.

**Phase G — Verification and handoff.** Run relevant existing and new tests, an isolated full-data run, compatibility comparisons and publication-failure checks. Update README and migration notes. Deliver limitations, blockers, commands and artifact links.

Keep architecture appropriate for the existing Python project: local Pandas/NumPy/Parquet and small explicit modules are adequate unless profiling proves otherwise. Do not introduce distributed infrastructure, a database server, mandatory cloud calls, or a large framework simply to describe the parser as powerful.

Preserve `python -m src.data_prep` and `build_features_asof` compatibility where possible. Add documented commands/options for inspect-only, validate-only, build, mentor-note ingestion, and evaluation if needed. Proposed commands in the plan are not existing functionality; document final commands only after testing them.

Cache only by complete dependency hashes, including cutoff, scenario, schema, rules and configuration. Never reuse an artifact across cutoffs merely because the raw filename is unchanged. Measure a baseline runtime/memory and set justified performance targets from actual input size and hardware. Test scaling on clearly synthetic fixtures without representing them as scientific data.

### 14. Required tests and acceptance criteria

Add meaningful behavioral tests, reusing existing coverage. At minimum verify:

1. UTF-8/CP1251, BOM, quoted commas/newlines, decimal commas, malformed records and ambiguous dates.
2. Exact preservation of identifiers and correct handling of integer-like legacy values.
3. Source classification independent of filename, including both meanings of `regime.csv`, numbered downloads, identical copies and conflicting copies.
4. Missing required versus optional fields, unknown fields and duplicate/colliding headers.
5. Unit conversion only under an approved rule; unknown units stay unresolved.
6. Grid lengths below/equal/above multiples of 10; coordinates at 0, 10, L, below 0 and above L; short final segment.
7. `Distance` wins over `Distance_raw`; irregularly sampled profile slopes use actual spacing.
8. Observed minimum is computed after eligibility filtering; absent/all-rejected observations remain missing, not nominal.
9. Campaign-specific depth conversion, nominal-wall tolerance, pre-commissioning exclusion and UZT ambiguity.
10. Duplicate rows do not inflate daily persistence or independent observation claims.
11. Three consecutive dates pass the soft-persistence rule; duplicate dates/gaps do not; impossible values remain unusable.
12. Adding future observations or a future third anomaly day cannot change earlier-cutoff features.
13. Cold-start features remain unchanged if target VTD/UZT records are added or altered.
14. Monitoring spanning beyond cutoff, conflicting identities and many-to-many joins are blocked from analytical use.
15. Training-fold-only fitting for imputation, transfer, scaling and feature/model selection.
16. Reynolds dimensional checks and missing/zero viscosity; thermal continuity, zero-flow rejection, shortened segment and explicit assumption flags.
17. Metric hand calculations: perfect prediction, inverse ranking, constant prediction, constant truth, negative R-squared, empty samples, one observation, zero/near-zero truth and nonfinite arrays. Perfect inverse ranking has Pearson squared 1 but signed r -1; demonstrate why it must not be presented as useful ranking.
18. Mentor re-ingestion produces no duplicate decisions; unknown speaker/date stays unknown; a conflicting new decision stays flagged; supersession preserves history; drafts are not sent.
19. One source-row change invalidates affected caches and lineage; unchanged reruns give identical semantic outputs except declared run metadata.
20. CSV/Parquet round-trip agreement, immutable raw checksums and failure-safe publication.
21. Submission column order, exact target key set, uniqueness, finite values, bounds policy and deterministic sorting; no silent fabrication of missing predictions.
22. Old and new file layouts produce equivalent semantic results for equivalent input bytes, with documented intentional differences only.

For the current archive, previous reports listed 16 passport rows, 20,180 profile rows, 48,040 VTD rows, 67 UZT rows, 495 monitoring rows and 39,052 hydraulic rows; 4,668 full-grid segments; 2,000 target segments; and 251 quarantined VTD coordinates. These are regression checkpoints to investigate, not invariants to force. Updated source hashes can legitimately change them.

A complete result requires every specification field to have an explicit inventory status, every applied rule to have provenance, every rejected/masked value to be auditable, all required tests to pass, and all unresolved scientific inputs to remain visible. Do not claim full physics coverage, calibrated confidence, or independent forecasting accuracy when the evidence does not support it.

### 15. Final handoff format

Return a concise outcome followed by:

1. Implemented components and preserved interfaces, with exact file links.
2. Tested run commands, runtime/dependency requirements and input-layout examples.
3. Source and metric coverage: measured, derived, scenario-only, excluded and unavailable.
4. Quality/leakage findings and observed test results; distinguish tests run from tests only proposed.
5. Output locations and the compatibility/migration explanation for `regime.csv`.
6. Mentor decisions implemented, decisions still blocked, and prioritized follow-up drafts.
7. Model metrics only where evaluation actually ran, with scenario/fold support and limitations.
8. Remaining questions that change correctness or evaluation, each with the concrete evidence needed to resolve it.

Do not stop after a generic architecture proposal. Complete the authorized implementation and verification, while keeping unavailable scientific information unavailable and preserving the user's work.

## END PROMPT

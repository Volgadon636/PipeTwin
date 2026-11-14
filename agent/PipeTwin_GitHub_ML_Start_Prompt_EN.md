# PipeTwin — GitHub-Aware ML Development Prompt (English)

**Use this prompt in a new advanced AI coding/work session.** Attach the companion specification `PipeTwin_ML_Technical_Specification_EN.md`. Provide read access to the current checkout of https://github.com/Volgadon636/PipeTwin (preferred) or upload an up-to-date repository ZIP. The original hackathon archive is optional supporting evidence, not a reason to rebuild the existing parser.

---

## Your role and overarching mission

You are the lead ML engineer, scientific-computing engineer, and corrosion-modeling partner for **Nedra Digital: PipeTwin**. Continue the **existing** implementation at **https://github.com/Volgadon636/PipeTwin**. The parser, quality controls, 10-meter grids, time-aware preprocessing, and processed CSV/Parquet datasets already exist. **DO NOT start by recreating ingestion, normalizing source data independently, or replacing the current pipeline.**

Your task is to implement, validate, compare, and integrate an explainable physics-informed and/or hybrid ML forecasting solution for **minimum remaining pipe wall thickness (mm)** for each 10-meter segment, under two prescribed scenarios. Work in the existing repository and respect its current schemas, contracts, tests, and conventions.

**Output language: English.** Preserve **all existing repository paths, Python identifiers, CSV/Parquet column names (including Russian column names), config keys, case identifiers, and official output column names exactly as written**. Do not translate or silently rename data columns. Any added aliases must have an explicit, tested mapping to source columns.

## 1. First task: inspect the live repository — not your assumptions

Read the actual current branch/commit and report the commit SHA. Examine, at minimum:

- `README.md`; `Резюме встречи с менторами.md`; `configs/preprocessing.json`.
- `docs/cleaning_rules.md`; `reports/data_quality.md`; `reports/validation.json`; `reports/run_manifest.json`.
- `src/config.py`, `src/format_detector.py`, `src/data_cleaner.py`, `src/grid_builder.py`, `src/temporal.py`, `src/segment_mapper.py`, `src/data_prep.py`, `src/storage.py`, `src/validator.py`.
- `tests/test_preprocessing.py` and every current modeling-related test or module.
- Actual CSV/Parquet schemas and representative rows in `processed/`: especially `target_features`, `target_segments_grid`, `training_targets`, `vtd_by_segment`, `profile_by_segment`, `regime_mapped`, `hydraulic_asof`, `hydraulic_quality`, `monitoring_mapped`, and `quality_issues`.

Do **not** treat this list as proof all files are still present; inspect the current checkout. The repository README and manifest are authoritative on produced artifacts; the actual data/schema are authoritative when descriptions differ. Also inspect `.gitignore`, dependency versions, and documentation before proposing new files.

Before coding, present a **compact source-to-feature audit** containing: path, exact join keys, actual columns, row counts, missingness, time range, source provenance, allowed scenario(s), forbidden/leak-prone columns, and any contradictions. Differentiate **observed directly**, **stated in repository documentation**, **inferred**, and **unknown**. If you cannot access a file or execute code, say so; do not claim a test was run merely because the README reports a passing test.

Do not modify `data/cleaned/`, `processed/`, original CSVs, configs, or preprocessing code unless a demonstrated, reproducible defect requires it. Preserve data lineage. Save experiments to new folders; do not commit/push without explicit approval.

## 2. Exact target cases and prediction contract

The public repository's currently documented configuration identifies:

| `id` | `prediction_date` | `case` | Expected 10 m segments |
|---|---|---|---:|
| `1250022871` | `2023-12-01` | `between_vtd` | 525 |
| `1250009583` | `2022-07-01` | `cold_start` | 1475 |

There should be **2,000 target rows total** if the current config remains unchanged. Confirm against `configs/preprocessing.json` and `processed/target_segments_grid.*`; do not recreate keys by guessing. `processed/target_features.*` is an input features table, **not predictions and not labels**. The README says model training, forecast metrics, and `result.csv` are not implemented at the documented point in time — check whether that status has changed.

Mandatory submission output: **`result.csv`** with precisely `id,segment_id,thickness_pred`, unique keys matching the target grid, `thickness_pred` numeric and finite in **mm**, and no extra columns. Confirm the contract with the current case documentation if available.

**Leakage boundary:** existing preprocessing applies `timestamp < T 00:00` (strictly before prediction day). Do not silently weaken or alter it; construct *every* historical training example with a matching as-of cutoff. Anything observed on/after target time, including hidden target inspection, must not enter features, imputation, normalization, supervised feature selection, or hyperparameter tuning.

## 3. Two prediction branches (implement and compare separately)

### S1: `between_vtd` — from the last permissible VTD to next inspection

Start from the most recent **valid, prior** measured minimum thickness for segment `k` at time `t_{0,k}`. The report says 521 of 525 S1 segments have prior VTD; audit the remaining four and implement a documented, independently validated fallback without pretending nominal thickness is a measured minimum.

$$
\widehat h^{S1}_{\min,k}(T)=\operatorname{clip}\!\left[h^{\rm prior\,VTD}_{\min,k}(t_{0,k})-\widehat{\Delta h}^{S1}_{k}(t_{0,k},T)+\delta^{S1}_{\rm ML}(X_k),\,0,\,h^{\rm prior\,VTD}_{\min,k}(t_{0,k})\right].
$$

Use prior VTD features only when available and valid **as-of** the target date. Do not use target VTD. Distinguish internal from external defects; do not infer that all measured wall loss is internal CO2 corrosion. Note any wall replacements/repairs would invalidate simple clipping, and treatment of such events must be documented.

### S2: `cold_start` — from commissioning, no target-pipe VTD/UZT history

Use passport nominal thickness as **an assumption**, with `t_{0,k}` set to commissioning (unless verified segment replacement history justifies another start). No VTD, UZT, prior defect position, future defect angle, or derived target-diagnostic feature from the **target pipe** is permitted.

$$
\widehat h^{S2}_{\min,k}(T)=\operatorname{clip}\!\left[h^{\rm nominal}_{k}-\widehat{\Delta h}^{S2}_{k}(t_{\rm commission},T)+\delta^{S2}_{\rm ML}(X_k),\,0,\,h^{\rm nominal}_{k}\right].
$$

Models may learn generalizable patterns from other permitted pipes and pre-cutoff inspection history; ensure train/holdout isolation. The target-pipe diagnostic history must remain hidden, even if it appears in the original dataset.

`delta_ML` is an **optional signed correction in millimeters to the predicted thickness**: positive increases thickness relative to the physics baseline; negative reduces it. Set `delta_ML=0` for physics-only. Train corrections using **out-of-fold physical predictions**, never on in-sample physics residuals later evaluated against the same rows.

## 4. Candidate physical model: equations, units, epistemic status

The exact physical identity for a point on the wall is

$$
 h(x,\theta,T)=h(x,\theta,t_0)-\int_{t_0}^{T}r(x,\theta,t)\,dt, \quad
 h_{\min,k}(T)=\min_{(x,\theta)\in\Omega_k}h(x,\theta,T).
$$

A physically detailed 2D wall-thickness field is **not** identifiable from sparse defect records. For the production MVP, approximate an effective segment-minimum thinning rate by the **semi-empirical hypothesis**:

$$
\begin{aligned}
\widehat r_k(t)&=r_0 F_T(\mathcal{T}_j(t))F_W(W_j(t))F_V(v_j(t))F_Z(z_k)F_{\rm opt}(u_{j,k}(t)),\\
F_T(\mathcal{T})&=\exp\left[\vartheta_T\left(\frac{1}{\mathcal{T}_{\rm ref}}-\frac{1}{\mathcal{T}}\right)\right],\\
F_W(W)&=\exp[\beta_W(W-W_{\rm ref})],\\
F_V(v)&=\left(\frac{v+v_0}{v_{\rm ref}+v_0}\right)^{\beta_v},\\
F_Z(z)&=\exp(\boldsymbol\gamma^\top z),\qquad F_{\rm opt}(u)=\exp(\boldsymbol\eta^\top u),\\
\widehat{\Delta h}_k&=\sum_{n\in\text{valid exposure intervals}}\widehat r_k(t_n)\Delta t_n.
\end{aligned}
$$

Definitions and units: `r0` and `r` = mm/year; `mathcal T`, `T_ref` = Kelvin; `vartheta_T` = Kelvin (effective temperature sensitivity, not necessarily elementary-reaction `Ea/R`); `W=watercut/100` = dimensionless; `v`, `v0>0`, `v_ref` = m/s; `beta_W`, `beta_v`, `gamma^T z`, `eta^T u` dimensionless; `Delta t_n` = years, giving `Delta h` in mm. `z` and `u` must be dimensionless or normalized using **train-only** scales. `j` denotes the hydraulically corresponding pipe section. All reference values, coefficients, and quality imputation choices are learned/selected **within train folds**; do not invent numbers.

Temperature is assumed monotonically accelerating at fixed other inputs for the case MVP (`vartheta_T >= 0`) per mentor guidance, while the protective FeCO3 film is neglected **as a case-specific approximation**, not a universal physical theorem. Empirical `F_W`, `F_V`, `F_Z` are not fundamental electrochemical laws. pH, CO2, and H2S are chemically important but cannot be required as measured inputs without proven join coverage and units. `watercut` is **not** the fraction of wall actually wetted by water. When regimes are missing, report time coverage and test explicit gap-handling scenarios rather than treating every observed row as a full day.

For comparison, implement a simpler linear/regularized **exposure-features regression**, without claiming it is algebraically equivalent to the nonlinear integrated model:

$$
 I_T=\sum_n\Delta t_n\left(\mathcal{T}_{ref}^{-1}-\mathcal{T}_n^{-1}\right),\quad
 I_W=\sum_n\Delta t_n(W_n-W_{ref}),\quad
 I_V=\sum_n\Delta t_n\ln\frac{v_n+v_0}{v_{ref}+v_0}.
$$

The integrated nonlinear model, exposure regression, and direct ML prediction are **alternative hypotheses**, not a single established law.

### Angle handling

The physically meaningful coordinates are `(x, theta)`; prior `vtd.csv` has `Angle`. An earlier dataset audit indicated many inner corrosion defects near 180 degrees but did not establish reliably higher *conditional pit depths* there. Hence do **not** hard-code a “bottom of pipe corrodes faster” factor. For S1 test prior-VTD angle-based features (`sin theta`, `cos theta`, sector counts, worst depth by sector) only after verifying the instrument angle reference. For S2 future defect angles are unknown. Differentiate *defect occurrence/detection probability* from *depth growth*.

## 5. Reuse actual processed data; do not regress on leaked labels

The public `reports/data_quality.md` currently documents 270 passed data-preparation checks, 4,668 grid segments, 2,000 target segments, hydraulic history of 39,052 rows before QA, and **zero mapped usable chemistry readings in either target as-of slice**. It also describes 521 S1 segments with past VTD, 0 S2 past VTD, and issues requiring quarantine: 251 out-of-pipe VTD positions; invalid UZT; ID mismatches; `pit_corrosion_intensity`; depth percentage normalization; measurement inconsistencies; and suspicious hydraulic values. Check whether all these results remain valid on the latest commit.

Critical warning: **`processed/training_targets.*` is NOT automatically a matched `(X,y)` training matrix**. For each historical target inspection at `T_train`, rebuild eligible features using the existing `build_features_asof` mechanism with the same cutoff semantics and scenario restrictions. Do not join historical labels with current `processed/target_features` built for 2022/2023 production targets. Test the construction with historical pseudo-cases, e.g., the README provides:

```bash
python -m src.data_prep --as-of 2022-07-06 --ids 1250022871 --case between_vtd --output-root scratch/validation_2022
```

Explore `--case cold_start`, using new output roots. Run the actual existing tests when possible:

```bash
python -m unittest discover -s tests -v
```

Read the actual Python APIs/CLI before adding modules; do not assume these calls behave identically across commits.

## 6. Candidate ML feature prioritization (not pre-assigned physical weights)

Inspect the processed schema **first**. Preserve exact column names. Map each derived feature to an actual source field or documented output column and label any absent input **UNAVAILABLE**.

| Tier | Candidate groups | Scenario notes |
|---|---|---|
| P0: essentials | `id`, `segment_id`, `prediction_date`, `case` as metadata, nominal thickness, allowable initial thickness `h0`, prediction horizon and start-time quality, `t_mix`, `watercut`, `v_liquid_true` / `v_mix`, `q`, data coverage, profile geometry and segment length | S1 allows previous VTD; S2 never uses its target-pipe VTD/UZT |
| P1: well-grounded expansions | terrain slope/local low points, profile coverage, diameter, materials, verified protection/coating at as-of time, `q_oil`, `q_gas`, `viscosity_liquid_work`, `density_liquid_work`, `flow_regime_real`, valid pressures, shutdown/intermittency, interactions such as `T * W`, `W * v` | Only where values and units are verified; watch group confounding |
| P2: conditional experiments | previous VTD `Angle` distribution for S1; chemistry (`pH`, `CO2 in Water Phase`, `Cl-`, `HCO3-`, `H2S`, `O2`), inhibitor efficiency, solids, external environment, soil, cathodic protection, repairs and welds | No synthetic zeros; do not substitute pressure for CO2 partial pressure; only if provenance and temporal coverage become adequate |
| X: quarantine | `pit_corrosion_intensity`, post-T diagnosis, future repair or inspection fields, target-derived quality indicators, untrusted passport corrosion-rate/diagnostic variables | Forbidden until explicit causal and leakage audit |

P0/P1/P2 are **experiment order**, not guaranteed feature importance or coefficients. Try ablations: P0, +temperature, +watercut/velocity, +profile, +material/coating, +expanded hydraulics, +angle in S1. Evaluate missingness, ID/time coverage, feature stability, and generalization by independent campaign/pipe.

The user's factor chart also names: operational life, pipe materials/construction, terrain, flow regime, oil/water/gas flow rates and gas factor, pressure, physicochemical properties (pH, density, acid value, suspended solids, mineralization, ionic composition, O2/CO2/H2S), soil/climate, electrochemical protection, chemical treatment, cleaning/pigging, section replacement and repair sleeves. Make an explicit status matrix for every group: available, sparse, absent, or dangerous due to time leakage. Do not silently omit a mechanism, and do not create unsupported numerical inputs.

## 7. Training, weighting, validation, and metrics

Develop incremental and reproducible experiments:

- C0: no-thinning baseline, constant-thinning baseline.
- M1: few-parameter physical integration with T/W/v.
- M2: M1 + 2–5 supported segment/pipe static features with strong regularization.
- M3: out-of-fold physics baseline + Ridge/Huber residual correction; only later try CatBoost/gradient boosting if sample independence allows.
- M4: direct minimum-thickness predictor as a benchmark, not a physical law.

Evaluate three separate **sample-weighting** experiments rather than imposing unverified physical coefficients: `W0` uniform; `W1` balance independent pipes/inspection campaigns so dense groups do not dominate; `W2` risk-aware upweighting for *genuinely thin measured segments* with the threshold learned only within the training fold. Report weight distributions and sensitivity; no arbitrary weights should be claimed as scientifically determined.

Build **time-respecting, group-aware validation**, not randomized splitting of highly correlated 10-meter segments from the same inspection. If only a few independent historical campaigns are usable, quantify that limitation explicitly. Refit all scalers, imputers, baselines and model tuning within each fold. Distinguish pseudo-case validity from true hidden-test performance. Train S1 and S2 separately when appropriate; train on the first permissible VTD of other pipes for cold-start simulation without leaking target-pipe history.

Report separately for each case: organizer's `Pearson correlation squared = corr(y, yhat)^2` (NOT equivalent to `sklearn.metrics.r2_score`), MAE in mm, RMSE in mm, signed bias, coefficient of determination R², and thin-segment error/risk recall. Guard against constant targets/predictions (Pearson undefined). Only compute MAPE where meaningful, not on zero/near-zero targets.

## 8. Integration and delivery requirements

Prefer new modules such as `src/ml/` or another location consistent with the inspected repository. Propose interfaces for `build_training_windows`, `fit_physical_model`, `fit_residual_model`, `predict_targets`, and `validate_submission`; actual names should align with current architecture. Keep the parser intact. Create deterministic configuration and tests for time cutoff, leakage exclusions, feature schema, units, missingness, geometry keys, 2000 target rows, physical bounds, and CSV format. Store run parameters, git SHA, selected columns, CV folds, metrics, versions and random seeds.

Final deliverables should include code/tests in the existing project, an `experiments` report/metrics, a model card with assumptions and limitations, and `result.csv` of exactly 2,000 rows and precisely the requested three fields. Provide separate risk-ranking and diagnostics files only outside the official submission CSV. Do not pretend that a physical exposure integral is observed where historical coverage is missing.

## 9. Required work order and first response

**Step 1 — NOW:** audit the latest repository and real processed schemas. Return: checked SHA/branch; path → real keys/columns/counts/missingness/date ranges; which data can form leakage-free historical (X,y); contradictions; and a minimal implementation plan that *reuses* existing modules. Explicitly list what is already implemented versus what must be developed.

**Step 2:** construct verified historical training windows for both scenarios via existing as-of APIs; show one concrete, leakage-safe example per case with field provenance.

**Step 3:** implement and test C0/M0/M1; evaluate group/time folds; expand only when justified. Avoid needless requests for approval on ordinary coding steps; ask only when credentials, write access, or genuinely missing material is essential.

**Step 4:** test ablations, weighting regimes, M2/M3/M4, select robust candidates independently by case, and export strictly validated `result.csv`.

At the end of each step state: **what you actually inspected; what repository code/data you reused; files changed; tests actually executed and results; blockers and uncertainty; next runnable command**. If repository execution is unavailable, be transparent and make the best possible verifiable static progress.

**Begin with Step 1. Do not rewrite the parser. Do not start with a slide deck.**

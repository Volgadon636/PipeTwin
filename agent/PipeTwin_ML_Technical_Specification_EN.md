# PipeTwin — Physics-Informed Corrosion Model and ML Technical Handoff

**Version:** 3.0 (GitHub-aware, English)  
**Date:** 2026-11-14  
**Project:** [Volgadon636/PipeTwin](https://github.com/Volgadon636/PipeTwin)  
**Status:** Proposed research/engineering specification. The mathematical structures are testable hypotheses, **not** a trained or industrially validated corrosion law.  
**Intended readers:** ML engineers, corrosion researchers, scientific-computing developers, mentors, and an advanced AI coding agent.

> **Executive contract.** Forecast **minimum remaining wall thickness in millimeters** (`thickness_pred`) for every 10 m segment of two specified pipelines at their respective prediction dates. There are **two different prediction scenarios**, one with a previous in-line inspection and one without any target-pipe inspection history. The existing repository has already implemented parsing, cleaning, quality controls, 10 m segmentation, temporal cutoffs, and published processed CSV/Parquet datasets. **Continue from those artifacts. Do not rebuild the parser.**
>
> **Names rule:** This document's prose is English. Exact source columns, canonicalized columns, Python symbols, file paths, configuration keys, and official output column names remain unchanged, including Russian source names. A proposed model feature (e.g. `arrhenius_exposure`) is *not* automatically a column already present in `processed/`.

## 0. Current GitHub implementation: source of truth

Repository: **https://github.com/Volgadon636/PipeTwin** — public `main` as inspected through its GitHub web pages on 2026-11-14. The README, config, data-quality report, parts of the Python source and published artifact names were reviewed. **A local checkout was not executed in this document-writing environment, and large published CSV contents were not downloaded here.** Recheck commit SHA and all assumptions against the latest checkout before developing.

### 0.1. Resolved competition cases

The current `configs/preprocessing.json` explicitly declares:

| Exact `id` | Exact `prediction_date` | Exact `case` | Expected 10 m segments |
|---|---|---|---:|
| `1250022871` | `2023-12-01` | `between_vtd` | **525** |
| `1250009583` | `2022-07-01` | `cold_start` | **1,475** |
| **Total** | | | **2,000** |

The repository's `reports/data_quality.md` reports: 270 passed preprocessing checks, a full grid of 4,668 segments, 2,000 target-feature rows, 521/525 target S1 segments with previous admissible VTD, 0 previous VTD in S2, and **0 mapped chemistry-monitoring records for either target as-of slice**. The README states that model training, evaluation metrics and `result.csv` are **not yet implemented** at its reviewed revision.

These counts are **published preprocessing results**, not independently rerun computations. New commits can change them.

### 0.2. Existing code and artifacts (reuse, do not recreate)

| Component | Role and integration expectation |
|---|---|
| `README.md`; `Резюме встречи с менторами.md` | Latest project instructions, mentor clarifications, scenario/output interpretation |
| `configs/preprocessing.json` | `targets`, `cutoff_policy`, `persistent_days`, data-QA thresholds |
| `src/config.py` | Source/canonical field mappings, `STATIC_FEATURES`, `HYDRAULIC_FEATURES`, type contracts |
| `src/format_detector.py`, `src/data_cleaner.py` | Source reading, normalization and row-level quality flags |
| `src/grid_builder.py`, `src/segment_mapper.py` | Existing 10 m grid, VTD/UZT/profile mapping and `build_features_asof` |
| `src/temporal.py` | `hydraulic_asof`, `hydraulic_summary`, strict temporal cutoff and 3-consecutive-calendar-day policy |
| `src/storage.py`, `src/validator.py`, `src/data_prep.py` | Persisted artifacts, data checks and pipeline entry point |
| `tests/test_preprocessing.py` | Existing tests; preserve and run before/after new ML work |
| `processed/segments_grid.*` | Grid for all 16 identifiers |
| `processed/target_segments_grid.*` | **The target submission key grid** |
| `processed/target_features.*` | Features at the two target prediction dates; **not labels** |
| `processed/training_targets.*` | Historical eligible measured minima; **not by itself a leakage-free `(X,y)` training set** |
| `processed/vtd_by_segment.*`, `processed/uzt_by_segment.*` | Inspection and measurement histories with quality flags |
| `processed/profile_by_segment.*` | Profile-derived segment geometry |
| `processed/hydraulic_asof.*`, `processed/regime_mapped.*` | Time-filtered hydraulic rows and summaries for current targets |
| `processed/monitoring_mapped.*` | As-of available water chemistry and monitoring coverage |
| `processed/hydraulic_quality.*`, `processed/quality_issues.*` | Anomaly decisions, provenance, rejected records |
| `reports/data_quality.md`, `reports/validation.json`, `reports/run_manifest.json` | Quality report, checksums, validation and provenance; not a model benchmark |

**Authoritative actual names from reviewed Python:** `src/config.py` defines `STATIC_FEATURES = ['id', 'diameter_mm', 'nominal_thickness_mm', 'commissioning_date']`. Thus steel grade, coatings, soil, inhibitor use and weld protection **are not automatically in current `target_features`** just because they exist somewhere in the large `passport.csv`. Adding them requires a controlled, as-of-safe enrichment that reuses the existing cleaned sources, not a parallel parser.

`HYDRAULIC_FEATURES` currently lists: `p_start`, `p_end`, `q_liquid`, `q_oil`, `watercut_pct`, `v_liquid_true`, `v_mix`, `t_mix`, `t_start`, `t_end`, `density_liquid_work`, `density_oil`. In `src/temporal.py`, `hydraulic_summary` creates `<field>_mean`, `<field>_std`, `<field>_valid_days`, plus `hydraulic_records_count`, `hydraulic_days_count`, `hydraulic_source_date_min`, `hydraulic_source_date_max`, `hydraulic_coverage_fraction`, `hydraulic_max_gap_days`. Those are **averages of daily medians**, not integrated exposure since commissioning or previous VTD. Preserve that distinction.

`src/segment_mapper.py` defines actual historical/provenance fields such as `min_residual_thickness_mm`, `target_eligible`, `last_vtd_thickness_mm`, `last_vtd_date`, `has_previous_vtd`, `years_since_vtd`, `last_uzt_thickness_mm`, `age_years`, `z_center`, `z_mean`, `z_min`, `z_max`, `slope_m_per_m`, `profile_extrapolated_flag`, `max_profile_gap_m`, `monitoring_records_count`, `ph_median`, `co2_water_median`, and `total_mineralization_median`. Their **actual presence, nonnull coverage and join keys in each output file must still be checked.**

**Raw-to-canonical name warning:** In the older hackathon archive the hydrodynamic source was referred to as `sample_result.csv`, but GitHub loads `tech_mode_for_embedings_with_length_with_temperatures_ids16.csv` (same source class, subject to checksum verification). Raw `q` may map to canonical `q_liquid`, raw `watercut` to `watercut_pct`, raw VTD `Angle` to `angle`, and raw `Distance` to canonical `distance`. **Do not overwrite these existing names or assume raw and processed schemas are identical.**

### 0.3. Date and data provenance rules

- The current repository applies the conservative rule `timestamp < T 00:00`; records on the prediction day are not used because inspection time is unknown.
- Preprocessing uses the *current cutoff first*, then applies soft anomaly persistence: at least **3 distinct consecutive calendar days**; duplicate calculations on one day cannot fake three days.
- Hydraulic daily medians are computed before simple statistics over observed days. Gaps are **not** automatically filled; coverage must be explicit.
- The previous PDF implied two evaluated pipes, while the raw passport has 16 `id` entries; the current mentor summary resolves the actual two target IDs. Treat target `id` in the current config as the authoritative submission identifier; do not reconstruct targets from ambiguous directions.
- For historical folds, **rerun `build_features_asof` at each historical target date**. Joining `processed/target_features` (which was computed for the production dates) to older labels is a timestamp mismatch and potential information leakage.

### 0.4. Published and observed quality issues

| Issue | Current status / precaution |
|---|---|
| VTD 2018 `Defect Depth in %` | Values on `2018-09-23` represent fractions in the source and are normalized; source raw values preserved |
| VTD coordinate | Repository uses `Distance` (not `Distance Raw`); 251 out-of-pipe VTD positions quarantined |
| UZT | All 67 observations exceed passport nominal thickness; retained but excluded from model features pending methodology confirmation |
| VTD before commissioning | Retained for audit, excluded from valid training targets |
| `regime.csv` chemistry join | 412 rows with unresolved ID, 83 direct matching IDs with direction conflict; no forced join to neighboring pipes |
| Hydraulic raw `d_mm` | Original values look like meters (`0.159`, `0.219`); preprocessing converts to `diameter_mm` under a checked rule |
| Pressure, `t_mix`, density and flow | Suspect values flagged; units and physical meaning still require verification |
| `pit_corrosion_intensity` | Excluded pending derivation/provenance and leakage audit |
| Late passport diagnostic fields / corrosion-rate estimates | Do not use if derived from later inspections or unknown as-of timing |
| Repairs / replacements | No reliable complete time history in the existing as-of features; cannot infer a known intervention from a later condition |

## 1. Scientific definitions and target geometry

Let `x` be the longitudinal coordinate in meters from the verified start of an evaluated pipe, `theta` its circumferential angle (radians modulo `2π`), `t` physical time, and `h(x,theta,t)` local remaining metal thickness (mm).

**Fundamental definitions:**

$$
 r(x,\theta,t)=-\frac{\partial h(x,\theta,t)}{\partial t}\quad[\mathrm{mm/year}],
 \qquad h(x,\theta,T)=h(x,\theta,t_0)-\int_{t_0}^{T}r(x,\theta,t)\,dt.
\tag{1}
$$

For segment `k` and verified pipe length `L`:

$$
 \Omega_k=\{(x,\theta):10k\le x<\min(10(k+1),L),\;0\le\theta<2\pi\},
 \quad h_{\min,k}(T)=\min_{(x,\theta)\in\Omega_k}h(x,\theta,T).
\tag{2}
$$

The physical minimum is **not the average**, and the minimum of `h0 - integral(r)` generally **does not equal** the previous segment minimum minus the integral of the maximum instantaneous rate. Defect locations and the identity of the worst point can change. In-line inspection only observes sampled/detected features, so a minimum among available VTD records is a **measurement proxy**, not a guaranteed continuous-surface minimum.

`theta` can be useful in describing defect occurrence, water-wetting and defect shape. However, the instrument's zero-angle convention and alignment across inspections must be verified before interpreting `theta≈180°` as the pipe bottom. Prior audit found concentration of reported internal defects near this sector but did **not** demonstrate consistently greater depth *conditional on a defect existing*. Hence do **not** hard-code a positive bottom-of-pipe depth-growth multiplier without validation. S1 may use past inspection angular features; S2 has no known location of future defects.

At a whole-segment level, define the **effective minimum-thickness loss rate**:

$$
 r^{\rm eff}_k(t)=-\frac{d h_{\min,k}(t)}{dt},\qquad h_{\min,k}(T)=h_{\min,k}(t_0)-\int_{t_0}^{T}r^{\rm eff}_k(t)dt.
\tag{3}
$$

This is a useful *definition* on intervals where minimum thickness evolves regularly; its parametrization below is an empirical **approximation** to a potentially moving extreme and may be noisy under VTD detection changes. The model must not claim it computes the true growth of each pit.

## 2. Fundamental electrochemical and transport background

The primary anodic dissolution pathway for CO2-associated carbon-steel corrosion is `Fe -> Fe2+ + 2e-`. At free-corrosion potential, anodic and cathodic partial currents balance. Faraday's relation for **equivalent penetration** is:

$$
 r_F[\mathrm{mm/year}]=\frac{M_{Fe}\,i_{corr}}{n F\rho_{Fe}}\,(3.15576\cdot10^{10}),\qquad n=2.
\tag{4}
$$

`M_Fe` = molar mass of Fe (kg/mol); `i_corr` = corrosion current density (A/m²); `n` = electron count per dissolved Fe atom (dimensionless); `F≈96485` C/mol = Faraday constant; `rho_Fe` = metal density (kg/m³). The numerical factor converts m/s to mm/year for a 365.25-day year. **No reliable `i_corr` measurements exist in the CSV, so equation (4) explains the physics but is not directly executable as the MVP predictor.** Do not add the Faraday-derived rate on top of another rate that already represents the same electrochemical dissolution.

At fixed chemical conditions, reaction constants may be temperature dependent via Arrhenius:

$$
 k(\mathcal T)=A\exp\left(-\frac{E_a}{R\mathcal T}\right),
 \qquad \frac{k(\mathcal T)}{k(\mathcal T_{ref})}=\exp\left[\frac{E_a}{R}\left(\frac{1}{\mathcal T_{ref}}-\frac{1}{\mathcal T}\right)\right].
\tag{5}
$$

`k` = process-specific kinetic constant; `A` = same units as `k`; `Ea` = activation energy (J/mol); `R≈8.314` J/(mol K); `mathcal T` = Kelvin. Using this relation for the **net effective minimum-thickness loss** is a model hypothesis, not an exact derivation from elementary kinetics.

Other relevant physics, **not all directly identifiable in this archive**:

$$
 c_{CO_2}=k_H p_{CO_2}\quad(\text{specified Henry-law convention}), \qquad Re=\frac{\rho vD_h}{\mu}, \quad \tau_w=\frac{f_D\rho v^2}{8},\quad J=k_m(C_b-C_s).
\tag{6}
$$

`c_CO2` = dissolved CO2 concentration (mol/m³); `k_H` = Henry coefficient in this convention (mol/(m³ Pa)); `p_CO2` = CO2 partial pressure (Pa), **not** pipeline total pressure; `Re` dimensionless Reynolds number; `rho` fluid density (kg/m³); `v` velocity (m/s); `Dh` hydraulic diameter (m); `mu` dynamic viscosity (Pa·s); `tau_w` single-phase-model wall shear stress (Pa); `f_D` Darcy friction factor (dimensionless); `J` molar transfer flux (mol/(m²·s)); `km` mass-transfer coefficient (m/s); `Cb,Cs` bulk and surface concentrations (mol/m³). The single-phase equations for `Re` and `tau_w` must **not** be presented as exact three-phase oil/water/gas hydrodynamics. Mark any computed surrogate `approx_1phase`.

A FeCO3 protective film may inhibit or change corrosion, but the mentors' current approximation is that film protection can be neglected for this particular data regime (working temperature below 80°C). That does **not** mean FeCO3 cannot form below 80°C; film formation depends on pH, Fe2+, supersaturation, time and other conditions.

## 3. Candidate physical model: executable MVP hypothesis

### 3.1. Core effective thinning rate

A convenient, interpretable but **semi-empirical** model is:

$$
\boxed{\begin{aligned}
\widehat r_k(t;\Theta)&=r_0 F_T(\mathcal T_j(t))F_W(W_j(t))F_V(v_j(t))F_Z(z_k)F_{opt}(u_{j,k}(t)),\\
F_T(\mathcal T)&=\exp\left[\vartheta_T\left(\frac{1}{\mathcal T_{ref}}-\frac{1}{\mathcal T}\right)\right],\\
F_W(W)&=\exp[\beta_W(W-W_{ref})],\\
F_V(v)&=\left(\frac{v+v_0}{v_{ref}+v_0}\right)^{\beta_v},\\
F_Z(z_k)&=\exp(\boldsymbol\gamma^\top z_k),\qquad F_{opt}(u)=\exp(\boldsymbol\eta^\top u).
\end{aligned}}
\tag{7}
$$

| Symbol | Definition and unit | Source or fitting rule |
|---|---|---|
| `r0` | Base **effective minimum-thickness loss rate**, mm/year, positive | Fit from allowed training cases, with regularization; not set arbitrarily |
| `mathcal T`, `T_ref` | Mixture temperature and reference temperature (K) | `t_mix` (canonical `t_mix`) after data QA; `T_ref` selected from train only |
| `vartheta_T` | Effective temperature sensitivity, K | Fitted; can be constrained ≥0 to test mentor monotonicity assumption |
| `W`, `W_ref` | Watercut as fraction, dimensionless, `W=watercut_pct/100` | From canonical `watercut_pct`, if valid; reference from train only |
| `beta_W` | Watercut-response coefficient, dimensionless | Fitted; sign not imposed without evidence |
| `v`, `v0`, `v_ref` | Velocity proxy, regularizer >0, train-reference velocity; m/s | Compare `v_liquid_true` against `v_mix` as alternative proxies; do not combine blindly |
| `beta_v` | Velocity-response exponent, dimensionless | Fitted; no universal sign |
| `z_k`, `gamma` | **Normalized dimensionless** spatial/passport vector and coefficients | Profile geometry and supported static features; regularize strongly |
| `u`, `eta` | Optional normalized auxiliary feature vector and coefficients | **Omit in MVP** if unavailable/unverified |
| `j` | Hydraulic section from which the regime was computed | Existing source mapping; segment features often share the same section-level regime |
| `Theta` | All trainable coefficients/reference choices | Fit and select strictly within each train fold |

Every factor multiplying `r0` is dimensionless, so `r_k` has units mm/year. All temperatures in the reciprocal are **Kelvin**, not degrees Celsius. Time in integration is **years**. Choose `v0` as a transparent numerical regularizer, not a discovered physical flow threshold. The watercut multiplier does not imply zero corrosion at `W=0`; its validity is restricted to observed operating regimes and a different wettability model would be needed to enforce physical dry-wall behavior.

**MVP variants:** `M1 = r0*F_T*F_W*F_V` with `F_Z=F_opt=1`; `M2 = M1*F_Z` with only 2–5 supported static/spatial predictors and strong regularization; `M3 = M2 + an ML residual correction to the predicted thickness`. The fitted `r0` may implicitly absorb unmeasured chemistry/external mechanisms; do not interpret it as a universally transportable pure CO2 rate.

### 3.2. Discrete time integration and missing operational history

For trusted, case-specific, pre-cutoff exposure intervals `I_n=[t_n,t_{n+1})` with duration `Delta t_n` **in years**:

$$
\boxed{\widehat{\Delta h}^{\rm phys}_k(t_0,T)=\sum_{n\subset[t_0,T)}\widehat r_k(t_n;\Theta)\,\Delta t_n}\quad[\mathrm{mm}].
\tag{8}
$$

This approximation presumes piecewise representative rates. **A row in `hydraulic_asof.csv` is not necessarily an entire day, and a missing day is not an observed zero.** `hydraulic_summary` computes daily medians/summary statistics but does not calculate this integral. Implement a new, auditable interval builder that *uses* current as-of and QA fields; do not silently interpolate across gaps or extend observed conditions back to commissioning. Evaluate separately: (a) observed-only cumulative exposure + unknown-duration flags; (b) last-known-regime sensitivity (if defensible and train-only); (c) train-derived seasonal/reference regime for gaps. Report `observed_years`, `unobserved_years`, `coverage_fraction`, and `longest_gap_days` on the **actual exposure horizon**, not just between first and last recorded hydraulic day.

### 3.3. Simpler exposure regression (alternative, not equivalent)

From the same valid intervals calculate

$$
I_T=\sum_n\Delta t_n\left(\frac1{\mathcal T_{ref}}-\frac1{\mathcal T_n}\right),\qquad
I_W=\sum_n\Delta t_n(W_n-W_{ref}),\qquad
I_v=\sum_n\Delta t_n\log\frac{v_n+v_0}{v_{ref}+v_0}.
\tag{9}
$$

`I_T` has units years/K; `I_W`, `I_v` years. These are **derived exposure features**, not independent physical laws. Train a small Ridge/Huber/ElasticNet on these plus horizon, nominal/prior thickness, geometry, coverage. It is a different statistical approximation to the integrated nonlinear (7)–(8): generally `E[exp(Z)] ≠ exp(E[Z])`.

### 3.4. Spatially detailed research alternative (not MVP)

A mechanistic picture may separate local water wetting and local surface behavior:

$$
r_{\rm internal}(x,\theta,t)=r_{\rm wet-env}(j,t)\,w_{\rm wet}(x,\theta,t)\,g_{\rm local}(x,\theta,t),\quad 0\le w_{\rm wet}\le1.
\tag{10}
$$

`w_wet` = fraction of time/contact surface is water-wetted, **not `watercut`**. `g_local` = unknown dimensionless factor due to local conditions, not known from angle alone. Distinct external corrosion and purely mechanical erosion could enter a detailed total wall-thinning balance *only where mechanisms and rates are independently identifiable, and without counting erosion-corrosion synergy twice*. The present defect records do not provide a continuous `(x,theta)` field or complete multi-phase flow distribution. Thus this spatial form is a **research extension**, not a direct replacement for (7).

## 4. Two final forecast equations (the two hackathon scenarios)

Both outputs are *minimum-thickness estimates*. Clipping below assumes no undocumented pipe replacement/repair between baseline and target and can hide measurement inconsistency; keep an unclipped diagnostic and quality flag.

### S1 — `between_vtd`: previous valid VTD known

Let `t0,k` be the latest allowed prior inspection date for segment `k`; `h0,k = last_vtd_thickness_mm` after data QA. A missing previous minimum **must not be silently replaced with the nominal value**.

**Physics only:**

$$
\boxed{\widehat h^{S1,\mathrm{phys}}_{\min,k}(T)=\operatorname{clip}_{[0,h_{0,k}]}\left(h_{0,k}-\sum_{n:t_{0,k}\le t_n<T}\widehat r_k(t_n;\Theta_{S1})\Delta t_n\right).}
\tag{11}
$$

**Hybrid physics + ML residual:**

$$
\boxed{\widehat h^{S1,\mathrm{hybrid}}_{\min,k}(T)=\operatorname{clip}_{[0,h_{0,k}]}\left(\widehat h^{S1,\mathrm{phys,unclipped}}_{\min,k}(T)+g_{S1}(X^{S1}_k)\right).}
\tag{12}
$$

`g_S1` is a **signed predicted error in thickness (mm)** trained on `observed_thickness - out_of_fold_physics_prediction`; positive corrections increase predicted remaining thickness. It is **not** another physical corrosion-rate term. Use *out-of-fold* physics predictions when fitting this stage. S1 features may include historical `last_vtd_date`, `last_vtd_thickness_mm`, `has_previous_vtd`, `years_since_vtd`, past depth/type/count/density, prior-angle aggregates, nearby prior defects, material/profile, and pre-cutoff hydraulic exposures. Never use target-date VTD, results-derived extrema, or subsequent repairs.

**Missing baseline for four documented S1 target segments:** test separate fallback models built from strictly previous neighboring valid measurements and/or nominal thickness plus an explicit `baseline_missing` flag and uncertainty. A nominal fallback is a *prior assumption*, not a measured `h0` and its output should be evaluated independently.

### S2 — `cold_start`: no prior VTD/UZT on target pipe

Let `t0` be the verified commissioning date and `h0,k = nominal_thickness_mm` (explicit initial-condition assumption).

**Physics only:**

$$
\boxed{\widehat h^{S2,\mathrm{phys}}_{\min,k}(T)=\operatorname{clip}_{[0,h^{\rm nom}_k]}\left(h^{\rm nom}_k-\sum_{n:t_{\rm commission}\le t_n<T}\widehat r_k(t_n;\Theta_{S2})\Delta t_n\right).}
\tag{13}
$$

**Hybrid physics + ML residual:**

$$
\boxed{\widehat h^{S2,\mathrm{hybrid}}_{\min,k}(T)=\operatorname{clip}_{[0,h^{\rm nom}_k]}\left(\widehat h^{S2,\mathrm{phys,unclipped}}_{\min,k}(T)+g_{S2}(X^{S2}_k)\right).}
\tag{14}
$$

S2 features: commissioning age, nominal wall, diameter, profile, verified as-of regime/exposure and QA; additional passport material/protection only if enrichment is demonstrably dated. **No diagnostic features from target pipe `1250009583`**, no past or future target VTD/UZT, no angle of an unknown future pit. If commissioning predates hydraulic history by years, the unknown portion of equation (13) must be disclosed and represented with uncertainty/scenario sensitivity, not invented measurements.

**S1 and S2 are different statistical tasks.** Their baseline observation errors, horizons and allowed predictors differ. Train and validate separately at least at the experiment/evaluation level. Shared physically meaningful coefficients may be considered only with valid cross-case transfer and scenario-specific baselines.

### Optional blending

$$
\widehat h^{\rm blend}=(1-\lambda)\widehat h^{\rm phys}+\lambda\widehat h^{\rm hybrid},\qquad 0\le\lambda\le1.
\tag{15}
$$

`lambda` is an **empirical ensemble tuning weight**, not a physical parameter. Compare a small grid such as 0, 0.25, 0.5, 0.75, 1 **inside validation folds**; avoid claiming one value is physically mandated.

## 5. Comprehensive physical-factor inventory and evidence tiers

**Interpretation of priority**: `P0` = first-pass highly available/important for this dataset; `P1` = potentially useful after QA or inexpensive enrichment; `P2` = sparse/high-uncertainty experimental; `P3` = physically possible but not representable numerically today; `X` = forbidden/quarantined due to suspected leakage or unknown provenance. **Priority is not causal importance.** Data availability and training generalization, not a factor diagram alone, decide what enters ML.

### 5.1. Pipeline construction, age, geometry, soil and external exposure

| Physical factor | Actual source/processed status | Plausible role, proposed feature | Tier |
|---|---|---|---|
| Commissioning date / operating years | `passport.csv` -> `commissioning_date`, `age_years` in processed target features | horizon/baseline corrosion exposure; validate chronology | **P0** |
| Nominal wall and diameter | processed `nominal_thickness_mm`, `diameter_mm`; raw passport | baseline and `D/t`, possible wall stress/geometry; verify OD/ID semantics | **P0** |
| Pipe length, 10 m coordinate | `processed/target_segments_grid.*`, `segments_grid.*` | `segment_id`, actual segment length, normalized longitudinal position, edge distance | **P0** |
| Route elevation, inclines, low points | `profile.csv`; processed `z_center`, `z_min`, `z_max`, `slope_m_per_m` | water pooling proxy, slope-change, local minimum, neighborhood features | **P0–P1** |
| Steel grade, material and manufacturing type | raw `passport.csv` has material/construction information; **NOT in current `STATIC_FEATURES`** | categorical as-of feature only after controlled enrichment and enough cross-pipe examples | P1–P2 |
| Inner lining / anticorrosion coating | raw `passport.csv`; not current processed static whitelist | material-contact mitigation; unknown distinct from uncoated | P1–P2 |
| Outer coating and weld-joint insulation | raw passport, date/change status unclear | external deterioration, weld-related risks, as-of provenance required | P2 |
| Pipe joints, bends, elbows, curvature, fittings | profile/possibly passport; no verified 10 m bend table | local turbulence/wear; only if geospatially and temporally validated | P2–P3 |
| Soil type, moisture, resistivity, aggressive soil | raw passport may give soil class, not complete geochemical time series | external corrosion; do not assume watercut captures this | P2–P3 |
| Climate region, groundwater, precipitation, temperature cycles | screenshot/mentor factor inventory; reliable local time-series absent | external corrosion hypotheses, not automatic numeric fields | **P3** |
| Cathodic/electrochemical protection (`ЭХЗ`) | mentions in passport/mentor material, verified pipe-level potential/current record absent | external protection effect; binary known-state only with time validity | P2–P3 |
| Buried versus exposed construction / contact environment | passport if verified | corrosion context, group-confounding risk | P2 |

### 5.2. Operating regime, hydrodynamics and fluid mechanics

| Physical factor | Actual source/processed status | Plausible role / feature | Tier |
|---|---|---|---|
| Working temperature along pipe | raw `t_mix`; processed `t_mix_mean`, `_std`, `_valid_days`, as-of rows | Arrhenius exposure, min/mean/max/P90, warm-operation fraction | **P0** |
| Watercut / water phase availability | raw `watercut`, canonical `watercut_pct` | water fraction, T×W and wet-time proxies, not wetting itself | **P0** |
| Liquid speed | `v_liquid_true`, `v_mix` | velocity exposure, stasis/intermittency, two alternative velocity formulations | **P0** |
| Liquid flow rate | raw `q`, canonical `q_liquid` | throughput, shutdowns, spikes; check units and flow sign | **P0** |
| Oil and gas rates / gas factor | `q_oil`, `q_gas`, `gas_content_rate`; **not all are in current summary whitelist** | multi-phase exposure, ratio proxies; no gas mole fractions inferred | P1 |
| Flow regime (reported) | raw `flow_regime_real`, canonical `flow_regime` in cleaned dataset | regime category only; not proven oil-water-gas stratified/slugs regime | P1 |
| Viscosity | `viscosity_liquid_work`, in cleaned data; not a summary field currently | conditional Reynolds/shear proxy and stratification | P1 |
| Density | `density_liquid_work`, `density_oil`; `density_liquid_work_mean` summary exists | hydrostatic / multi-phase separation proxy | P1 |
| Pressure and differential pressure | `p_start`, `p_end`; summary as-of exists; units unresolved | flow proxy once units verified; never equal to `p_CO2` | P1 after QA |
| Fill factor | `fill_coefficient`, cleaned data; not current summary whitelist | experimental wetted-section/flow proxy, meaning needs verification | P2 |
| Laminar/turbulent Reynolds, wall shear | not directly measured; conditional surrogate from verified `rho`,`mu`,`v`,`D` | `Re_approx_1phase`, `tau_approx_1phase`; do not claim true multiphase | P2 |
| Oil-water wetting pattern, slugs, bubbles, emulsions | no complete phase-distribution instrument record | main mechanism of angle-specific internal corrosion but unresolved | P3 |
| Sand/solid particles, erosional impacts | sparse `regime.Mechanical Impurities` and no particle size/velocity history | erosion/erosion-corrosion synergy; not unconditional additive term | P2–P3 |
| Shutdowns, starts, stagnant water | derive from `q_liquid`, velocity and timestamps after QA | down-time, start count, gap-aware intervals | **P0–P1** |

### 5.3. Physicochemical (FHC), inhibitor and microbiological factors

| Physical factor | Actual source/processed status | Role and restriction | Tier |
|---|---|---|---|
| Water-phase pH | `regime.pH` raw -> canonical `ph`, rare; zero usable mapped target records | kinetics, carbonate equilibrium, possible film effects; no blind join | P2–P3 |
| Dissolved CO2 | raw `CO2 in Water Phase` -> canonical `co2_water`, sparse | chemistry proxy *if units/time/ID verified*; not `p_CO2` | P2–P3 |
| CO2 partial pressure / gas mole fraction | no trustworthy composition histories; total pressure not substitute | necessary for detailed CO2 correlations, unavailable in MVP | **P3** |
| H2S concentration | sparse raw monitoring | sulfide corrosion; insufficient continuous target data | **P3** |
| Dissolved oxygen O2 | extremely sparse raw monitoring | oxygen cathodic corrosion, unavailable on targets | **P3** |
| Ionic composition / chloride, bicarbonate, sulfate | raw `Cl-`, `HCO3-`, `SO4-` and other ions, sparse | electrolyte conductivity, corrosion chemistry and scaling | P2–P3 |
| Total mineralization, ionic strength, conductivity | raw `Total Mineralization` -> canonical `total_mineralization` | aggressiveness proxy, subject to chemistry-ID QA | P2–P3 |
| Water density, acid value (`КВЧ`), alkalinity | mentor chart; some raw `Fluid Density`; acid value not verified continuously | pH buffering, phase behavior, corrosion | P2–P3 |
| Ca/Mg/hardness, Fe2+ and saturation | raw `Ca2+`,`Mg2+`,`Hardness`,`Total Fe` sparse; `Total Fe != Fe2+` | possible scale/FeCO3 conditions; not direct protective film thickness | P2–P3 |
| FeCO3 protective film thickness, porosity, continuity | no measured film thickness; mentors allow ignoring in MVP | omit initially; reassess if empirical evidence indicates nonmonotone T effect | **P3** |
| Corrosion inhibitors and chemical protection | passport inhibitor status; raw `Protective Effect` scarce; actual dosing timeline absent | categorical only if as-of, never invent inhibition percentage | P1–P3 |
| Internal cleaning / pigging (`КППОУ`) | listed in mentor factor diagram, no verified date-resolved event log | deposition removal, film damage and changing water contact | **P3** |
| Suspended solids / mechanical impurities (`СВБ` as indicated) | limited/raw monitoring, abbreviations need definition | deposits, underdeposit corrosion, erosional effects | P2–P3 |
| Bacteria / sulfate-reducing bacteria / MIC | very rare monitoring such as `SVB Cell Content` | relevant mechanism, not data-identifiable | **P3** |
| Organic acids, hydrocarbon composition, surfactants | no continuous verified source | can alter wetting and corrosion kinetics; latent factors | **P3** |
| Electrochemical potential/current, Tafel slopes | no direct polarization/current data | Faraday/Butler–Volmer are physical theory, not numeric ML fields | **P3** |

### 5.4. Inspections, localized defects, interventions and target leakage

| Factor | Actual source/processed status | Use and restriction | Tier |
|---|---|---|---|
| Past minimum thickness | `last_vtd_thickness_mm` and flags in as-of S1 | primary S1 baseline, never S2 target-pipe feature | **P0 S1 / X S2** |
| Past defect length/width/depth/type | raw `vtd.csv` contains `Defect Length`, `Defect Width`, `Defect Type`, `Residual Wall Thickness`; cleaned canonical fields need check | historical S1 only, after date and quality checks | P1 S1 / X S2 |
| Previous defect count/density, severity and neighbor risk | derive from past, admissible VTD and segment neighborhood | S1 ML localization; horizon/coverage matched | P1 S1 / X S2 |
| Circumferential angle | raw `Angle`, canonical `angle`; origin and repeat-inspection alignment unverified | S1 prior-angle sector aggregates, circular sin/cos; **not** a hard-coded speed multiplier | P2 S1 / X S2 |
| Local ultrasound thickness | `last_uzt_thickness_mm`, but current quality audit rules out 67 suspicious measurements | avoid unless methodology clarified; do not use as valid target baseline | P2/X |
| Repairs: replacement of pipe sections | no verified event-date history in current as-of features | changes wall starting condition; future repairs prohibited | **P3 until verified** |
| Repair sleeves / clamps | no full as-of dated mapping | may change service conditions and measurement; do not impute | P3 |
| Late passport inspection severity, `pit_corrosion_intensity`, stated passport corrosion rate | source provenance/target dependence unverified | **QUARANTINE** until explicit non-leakage evidence | **X** |

**Major conclusion:** the entire physical factor inventory is considered; **physically possible** does not mean **measured** or **useful ML input**. Coverage, provenance, reproducibility and independent validation determine feature inclusion.

## 6. Concrete ML feature engineering for the current repository

### 6.1. Exact source-to-feature plan (feature aliases are proposed, not existing columns)

| Group | Actual existing canonical/output columns to inspect | Proposed derived features / transformation | S1 | S2 |
|---|---|---|---|---|
| Target join | `id`, `segment_id`, `prediction_date`, `case` | preserve exact target keys, no training label as input | yes | yes |
| Start & horizon | `nominal_thickness_mm`, `commissioning_date`, `age_years`, `last_vtd_thickness_mm`, `last_vtd_date`, `years_since_vtd`, `has_previous_vtd` | `h0_mm`, `horizon_years`, `baseline_missing`, `unobserved_years`; S2 h0 is nominal | VTD allowed | target VTD forbidden |
| Geometry | `diameter_mm`, `z_center`, `z_mean`, `z_min`, `z_max`, `slope_m_per_m` | `D_over_h`, slope windows, low-point, slope reversal, neighboring max/min, true segment length | yes | yes |
| Profile QA | `profile_extrapolated_flag`, `max_profile_gap_m`, `points_count` | quality masks for uncertain gradients and low points | yes | yes |
| Temperature | `t_mix_mean`, `t_mix_std`, `t_mix_valid_days`; `hydraulic_asof.t_mix_usable` | T in K, warm-day quantiles, `I_T`, Arrhenius integral and valid coverage | yes | yes |
| Watercut | `watercut_pct_mean`, `_std`, `_valid_days`; `watercut_pct_usable` | `W = watercut_pct/100`, high-water duration, `I_W`, T×W | yes | yes |
| Flow/velocity | `q_liquid_mean`, `v_liquid_true_mean`, `v_mix_mean`; their `_std`, `_valid_days` and as-of usable rows | flow stop/restart flags, velocity quantiles and `I_v`; compare liquid vs mix proxy | yes | yes |
| Hydro conditions | `p_start_mean`, `p_end_mean`, `density_liquid_work_mean`, `density_oil_mean`, `q_oil_mean`; `hydraulic_asof` | QA-approved pressure delta, 1-phase proxy Re/shear only if viscosity verified | yes | yes |
| Data completeness | `hydraulic_records_count`, `hydraulic_days_count`, `hydraulic_coverage_fraction`, `hydraulic_max_gap_days`, `hydraulic_source_date_min/max` | **recompute** coverage on each individual baseline→T horizon | yes | yes |
| Water chemistry | `ph_median`, `co2_water_median`, `total_mineralization_median`, `monitoring_records_count` | use missingness/age flags; current target chemistry coverage is zero | conditional | conditional |
| Previous VTD and angles | `processed/vtd_by_segment.*`; clean historical `vtd.csv`/`angle` | defect count, min, severity, circular statistics `sinθ`, `cosθ`, sector histogram, distance to past worst defect | yes with time QA | **never target-pipe** |
| Material/protection/soil | raw `passport.csv` **not in `STATIC_FEATURES`** | only after controlled typed as-of enrichment; group-invariance checks | optional | optional |
| Repairs/pigging/ECP | not verified as-of | no model columns until dated verified records exist | no | no |

**Important nuance:** many summary fields are identical across all 10 m segments of the same pipeline. Adding many such fields without local profile/defect structure cannot resolve intra-pipe ranking. Always compare within-pipe variation to between-pipe variation and use group-aware CV.

### 6.2. Training unit, target, and reconstruction

Use one training case keyed by `(id, segment_id, T_train, case)`, with **exactly one eligible measured min thickness label** on that historical inspection date. A valid label comes from `processed/training_targets.*` / `processed/vtd_by_segment.*` after target eligibility rules, not from a nominal-wall proxy. For every historical training case:

1. Select `T_train` as the historical inspection date and decide whether S1 (`previous VTD`) or S2 (`no target-pipe history`) is being simulated.
2. Request `build_features_asof` for **that target date and that case**, enforcing `timestamp < T_train` and the official exclusion rules; verify its returned schema. A target label is not an input feature.
3. Set `t0` from the last prior eligible VTD **per segment** in S1, or from `commissioning_date` in S2.
4. Compute/hypothesize integral exposure over `[t0, T_train)` using the actual QA-filtered rows and report unsupported gaps.
5. Attach label **only after** all features have been constructed and train/test groups established, so target-specific statistics cannot leak backwards.
6. Retain source quality flags and independent-group identifiers; never pseudo-replicate one survey as thousands of independent experiments.

Example documented by the repository (run with a **separate output root**, never overwrite production artifacts):

```bash
python -m src.data_prep --as-of 2022-07-06 --ids 1250022871 --case between_vtd --output-root scratch/validation_2022
python -m unittest discover -s tests -v
```

A first eligible VTD of a *different* permitted pipe can serve as an S2 training label with that pipe's diagnosis withheld from its features. For S1, train on truly sequential inspection pairs, not two depths from different defects just because they occupy the same segment.

### 6.3. Defect type, measurement and missingness

`Residual Wall Thickness` (canonical `residual_wall_thickness`) provides measurement-based minima; `Defect Depth in %` may have different campaign units and should not replace a trusted measured residual thickness. Count and type internal/external corrosion separately where valid; the target minimum may reflect both and the internal CO2 model alone does not explain external losses. Inspected non-defect entities should not be treated as corrosion. An empty VTD segment does **not** prove all local wall points equal the nominal thickness. For S1 missing prior thickness, retain an explicit flag and independently evaluate fallback uncertainty. Persist `source_row`, date and correction flags where possible.

## 7. Feature priority and weighting: experiments, not guessed truths

### 7.1. Ablation packages

| Package | Inputs | What it tests |
|---|---|---|
| `A0` | Scenario-specific h0, horizon, segment key/length, QA and missingness | no-physics baseline |
| `A1` | `A0` + temperature | added value of temperature / Arrhenius-like exposure |
| `A2` | `A1` + watercut + selected velocity and flow | dominant observed process regime |
| `A3` | `A2` + profile slope, local minimum, diameter | segment-level spatial differentiation |
| `A4` | `A3` + confirmed material/coating/inhibition/soil enrichment | structural and protection effects; beware confounding |
| `A5` | `A4` + extended QA-approved hydraulics, interactions and 1-phase proxies | whether hydrodynamic complexity actually helps |
| `A6-S1` | `A5` + past VTD counts/depth/angle | localized historical defect signal only in S1 |
| `A7` | chemistry/protective films/repairs only when verified new data exist | optional research, no synthetic values |

Test each package as an **incremental** candidate under the same folds; report missingness, performance, coefficient stability, individual group importance and whether improvements survive holding out a pipeline/campaign.

### 7.2. Three suggested feature-attention allocations

These are **planning allocations** for experiment time and ablation emphasis, **not** physical multipliers, ML input weights, or prescribed hyperparameters. Actual learned coefficients are estimated on training folds.

| Group | `Strategy CORE` (minimal physical) | `Strategy GEO` (segmentation emphasis) | `Strategy HISTORY-S1` |
|---|---:|---:|---:|
| Baseline thickness, horizon, geometry | 35% | 25% | 25% |
| Temperature, watercut, velocity/flow | 45% | 30% | 25% |
| Profile, elevation, low points | 15% | 35% | 15% |
| Historical VTD/defect distribution | 0% | 0% | 30% |
| Material/protection/extended QA | 5% | 10% | 5% |
| **Total** | **100%** | **100%** | **100%** |

**Important:** `Strategy HISTORY-S1` is illegal for target-pipe cold-start inference, and the 30% historical allocation must be **reassigned** to available groups for S2 (not filled with missing-target history). These values only help prioritize work, **not** mean a formula coefficient of 0.45 or an experimentally verified causal contribution.

### 7.3. Sample weighting (training loss) and class imbalance

Report separately:

- `W0`: uniform rows, with robust estimator if needed.
- `W1`: group-balanced weights, e.g. each **independent** survey/pipe receives the same total weight and then distribute within group: `w_i ∝ 1/n_group`. Renormalize weights to mean 1. If independent-group ID is ambiguous, document the grouping.
- `W2`: risk-sensitive weights for actually thin measured segments; define a threshold from **training labels only** (e.g. train bottom decile), upweight that group using a small candidate grid such as 1.0, 1.5, 2.0, 3.0 and compare on independent held-out metrics. Such weights are a modeling choice, not physically motivated constants.

Avoid joint tuning of feature groups and sample weights on the same holdout; use nested/repeated CV where feasible. Strongly correlated 10 m neighbors must not each count as independent evidence. Consider uncertainty-weighted losses only if per-label measurement uncertainty is known, not invented.

### 7.4. Physical-coefficient parameterizations and priors

- `r0`: positive via `r0=exp(a0)` or constrained optimizer; report fitted mm/year.
- `vartheta_T`: nonnegative **only within the explicit monotone-temperature assumption**; test unconstrained sensitivity and measurement coverage; do not fake positive physics if data show contradictory evidence.
- `beta_W`, `beta_v`: fit with L2 shrinkage, no mandatory sign absent evidence; watch confounding between flow, watercut, T, and season.
- `gamma`: strong regularization and very few predictors until cross-pipe generalization improves.
- `eta`: zero unless verified auxiliary features; avoid CO2/pH/HCO3 double-counting.
- `lambda` (blending): a validation-selected ensemble weight; not a physical parameter.

## 8. Candidate model families and validation design

| Model | Structure | Purpose |
|---|---|---|
| `C0` | `h_pred=h0` (no loss) | Absolute baseline and leakage sanity check |
| `M0` | `h_pred=h0-r0*(T-t0)` | Constant-rate baseline and unit-check |
| `M1` | Equations (7)–(8), no `F_Z/F_opt` | Parsimonious effective physics |
| `M2` | `M1` + 2–5 spatial/static factors | Profile and wall-data advantage |
| `M3` | `M1/M2` + out-of-fold Ridge/Huber residual | Low-variance hybrid correction |
| `M4` | Direct Ridge / boosting predictor of `h_min` | Independent accuracy benchmark, not physics |
| `M5` | Gradient-boosted hybrid residual if group CV supports it | Nonlinear refinement; avoid premature complexity |
| `M6` | Spatial angular/distribution or verified chemistry extension | Research only when demonstrated useful |

**Validation is the primary bottleneck:** only a small number of independent repeat-inspection histories exist. Splitting 10 m segments randomly inflates apparent generalization. For S1 simulate available sequential inspections, strictly reconstructing prior-only features at each historical target date. For S2 simulate each permissible first VTD with that pipe's VTD/UZT hidden from predictors. Use leave-one-pipe/survey-out where possible, forward-in-time holdout, spatial block holdout within pipe only as secondary diagnostic. Report the **number of independent held-out groups**, not just number of segments.

All preprocessing that learns from data (normalization, imputation, reference temperatures/velocities, feature selection, threshold selection, regression fit, boosting tuning and calibration) must operate inside train folds. Never fit an outlier threshold using held-out labels. Do not interpret high CV on one inspected pipe as guaranteed hidden-pipe accuracy.

### 8.1. Competition objective versus engineering metrics

The case documentation specifies **the square of Pearson correlation** separately for the two scenarios:

$$r^2_{Pearson}=\operatorname{corr}(y,\widehat y)^2.$$

This is **not** the same as `sklearn.metrics.r2_score` and can be high despite biased mm predictions. If either vector has zero variance, correlation is undefined; report it as such. Additional required/recommended metrics: `MAE_mm`, `RMSE_mm`, signed bias, median AE, `R²` as a separate metric, MAPE only for meaningful nonzero thicknesses, thin-segment decile error and top-risk recall. Check per-pipe/campaign metrics and physical bounds. For predictive intervals, use group-aware bootstraps or out-of-fold residual intervals with a warning about poor statistical coverage with few groups.

## 9. Implementation architecture — incremental additions to existing repository

**Do not replace** `src/data_prep.py`, `src/data_cleaner.py`, `src/temporal.py`, existing grids, configs or archived inputs by default.

Suggested new ML-layer interfaces (final names should follow actual checked-out architecture):

```text
src/ml/
  training_windows.py     # scenario-aware historical pseudo-targets using build_features_asof
  exposure_features.py    # QA-filtered interval integration and missing-history report
  physics_model.py        # positive rate, fit, predict, optional constraints
  residual_model.py       # out-of-fold correction model
  experiment.py           # group/time cross-validation and ablations
  predict.py              # production inference for current target grid
  submission.py           # assert result.csv contract and non-leakage
configs/ml_experiments.json
experiments/<run_id>/model_report.md
experiments/<run_id>/metrics.json
experiments/<run_id>/feature_manifest.json
experiments/<run_id>/folds.csv
tests/test_ml_pipeline.py
result.csv                # ONLY when complete and validated
```

These paths are **proposals**, not statements that these files already exist.

### 9.1. Minimum tests

- Exact current target IDs/dates/cases come from `configs/preprocessing.json` and 2,000 keys from `processed/target_segments_grid.*` (if unchanged).
- `timestamp >= prediction_date` excluded from **all** features, all fold-fitted transformations and monitoring data.
- S2 has zero target-pipe VTD/UZT-derived inputs, even when raw files contain them.
- Training features for a 2022 pseudo-case are **not** assembled from the production feature table computed for 2023.
- Correct alias mapping: `q`→`q_liquid`, `watercut`→`watercut_pct`, `Distance`→`distance`, `Angle`→`angle`; no silent in-place translation.
- `r` mm/year times `Delta t` year = mm; Arrhenius inputs in K; only dimensionless log/exp arguments.
- Sparse chemistry never imputed as zero without missingness flag/justification.
- No `pit_corrosion_intensity` or late diagnostic passport rate in allowed feature list.
- Unobserved historical periods reported; duplicate same-day calculations not counted as multiple days.
- `result.csv`: exact `id,segment_id,thickness_pred`; unique keys, finite numeric outputs, mm, expected count, no additional columns.
- Physical bounds checked and uncertainty/repairs flagged; raw un-clipped predictions retained in supplementary diagnostics.

### 9.2. Suggested development sequence

1. Confirm current commit, branch, Python environment/dependencies, run existing tests (do not claim success without running). Inspect artifact schemas, manifest and quality logs.
2. Construct **historical training windows** using existing as-of APIs and correct labels; document S1/S2 window coverage and realistic independent sample counts.
3. Implement `C0/M0/M1` and unobserved-period sensitivity; report leakage-safe metrics.
4. Add `M2` with the smallest validated local feature set; perform profile and operational ablations.
5. Fit `M3/M4/M5` only if stable holdout improvements; compare loss-weighting alternatives `W0/W1/W2`.
6. Produce final S1 and S2 forecasts on actual `target_features`/target grid. Export `result.csv`; provide separate feature attribution, uncertainty, thin-segment risk ranking and model report.

## 10. Official submission contract and QA

Required file:

```csv
id,segment_id,thickness_pred
```

No user data appears in the header example intentionally; add exactly one physical prediction in **millimeters** per target-grid key. Expected present configuration: 525 + 1,475 = **2,000 lines of data** (plus header). The repository grid defines `segment_id`; if the physical definition is needed, it is `floor(distance_from_pipe_start_m/10)` under the verified global coordinate. The last segment may be shorter. Use UTF-8, comma delimiters, decimal point, finite values, no duplicates, no NaN, no predicted thickness of the wrong pipe, and no diagnostic/interval columns in the submission. A supplementary risk report may rank the thinnest segments for ultrasonic follow-up, but must be separate.

## 11. Credible scientific references and interpretation

1. **S. Nešić (2007)**, *Key issues related to modelling of internal corrosion of oil and gas pipelines — A review*, Corrosion Science 49, 4308–4338. DOI: https://doi.org/10.1016/j.corsci.2007.06.006 . Review of CO2 corrosion, electrochemistry, wetting, flow, scales and local damage.
2. **ASTM G102-23**, *Standard Practice for Calculation of Corrosion Rates and Related Information from Electrochemical Measurements*. https://store.astm.org/g0102-23.html . Supports Faraday/equivalent penetration conversions.
3. **NORSOK M-506**, *CO2 Corrosion Rate Calculation Model*. https://standards.globalspec.com/std/10219751/M-506 . A domain-specific engineering model; requires appropriate chemistry, pH, gas and transport inputs, so do not fabricate its missing inputs or claim the MVP uses the standard's coefficients.
4. **H. Zhang, H. Lan, N. Lin (2019)**, *A numerical simulation of water distribution associated with internal corrosion induced by water wetting in upward inclined oil pipes*. DOI: https://doi.org/10.1016/j.petrol.2018.10.030 . Motivates tilt, phase distribution and wetting as spatial influences.
5. **M. Askari et al. (2021)**, *Downhole corrosion inhibitors for oil and gas production — a review*. DOI: https://doi.org/10.1016/j.apsadv.2021.100128 . Inhibitor mechanisms and limitations.

None of these sources validates the fitted `F_W`, `F_V`, or `F_Z` as universal laws or sets their coefficients for the dataset. **Evidence categories:** (i) exact physical identity; (ii) classical theory for specific conditions; (iii) semi-empirical model hypothesis; (iv) observed repository data/report; (v) unknown/unmeasured.

---

## Appendix A. Reusable starting prompt

Use the standalone English file **`PipeTwin_GitHub_ML_Start_Prompt_EN.md`** as the model's first message. Attach **this specification** and give the model the latest repository checkout/ZIP and GitHub URL. The prompt requires **Step 1: inspect and audit current `processed/` and current code**, not reimplement the parser, and then incrementally build time- and group-safe models for S1/S2.

**Suggested invocation:**

1. Open a new coding-capable model session with repository access.
2. Attach both English Markdown documents (or add them to the repo's `docs/` folder as a separate, reviewed change).
3. Send the content of `PipeTwin_GitHub_ML_Start_Prompt_EN.md` as the initial prompt.
4. Authorize changes or pushes explicitly only after code and experimental results have been checked.

**End of technical handoff.**

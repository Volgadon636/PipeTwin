# PipeTwin ML v2 — physics-informed cold start and gated spatial term

Status: implemented on branch `model_v2`. Companion run outputs are in
`experiments/ml_v2/`; the reproducible entry point is
`python -m src.ml.experiment_v2 --output-root experiments/ml_v2`.

This document explains **why** v2 changes the model, the equations used, the
validation that actually exists, and what remains unidentifiable. It does not
claim hidden-test accuracy.

## 1. What v1 did and why it was replaced

v1 (`src/ml/experiment.py`) fitted one constant nonnegative thinning rate per
scenario and applied it to every segment:

```
h(T) = clip(h0 - r * age_years, 0, h0)
```

For the cold-start pipe this produced **1.540 mm on every segment**:

* the rate 0.102 mm/year was fitted on warm (~11–13 °C), wet (~87 % watercut),
  flowing (~0.6 m/s) MAIN-2 pipes;
* it was extrapolated over 29 years to `1250009583`, which is cold (~6 °C),
  dry (~0.0005 water fraction) and slow (~0.065 m/s);
* a constant prediction has zero variance, so the organizer's Pearson² is
  undefined and the spatial risk ranking is meaningless.

v2 keeps the *form* required by the technical specification but makes the
environment explicit and only fits what can be identified.

## 2. Cold start (S2): environment-aware effective rate

Per valid exposure interval `n` (from `attach_exposure`, one-day forward hold):

```
F(t) = exp[ theta_T (1/T_ref - 1/T)
          + beta_W (W - W_ref)
          + beta_v ln((v + v0) / (v_ref + v0)) ]

exposure_k = age_k^(p-1) * [ sum_n F(t_n) dt_n + gap_term_k ]

loss_k    = r_ref * exposure_k
h_k(T)    = clip(h_nominal,k - loss_k, 0, h_nominal,k)
```

| Symbol | Meaning | Unit |
|---|---|---|
| `T`, `T_ref` | mixed temperature, training reference | K |
| `W`, `W_ref` | watercut / 100 | dimensionless |
| `v`, `v_ref` | mixture velocity, reference | m/s |
| `v0` | numerical velocity offset | m/s |
| `theta_T` | effective Arrhenius sensitivity (nonnegative prior) | K |
| `beta_W`, `beta_v` | log/watercut and velocity power sensitivities | dimensionless |
| `p` | age exponent (linear = 1) | dimensionless |
| `r_ref` | fitted reference rate | mm/year |
| `dt_n` | interval duration | years |

`gap_term` is explicit about unobserved history:

* `own_mean` (central): unobserved years run at the pipe's **own** observed mean
  `F`; this is the production assumption;
* `reference`: unobserved years run at `F = 1` (warm/wet reference);
* `observed_only`: unobserved years contribute nothing (lower bound, not a
  measurement claim).

### Identifiability

The repository contains exactly **two independent first-inspection
environments**: MAIN-2 2018 (warm/wet) and MAIN-3 2023 (cold/dry). Temperature,
watercut and velocity are almost perfectly confounded across them. Fitting
`theta_T`, `beta_W`, `beta_v` jointly would produce arbitrary, unstable
coefficients. Therefore:

* **only `r_ref` is fitted** (nonnegative weighted least squares, parent/campaign
  balanced `W1` weights);
* `theta_T = 2500 K`, `beta_W = 0.30`, `beta_v = 0.10`, `v0 = 0.10 m/s`,
  `p = 1` are **declared engineering priors** (`EnvironmentPriors`);
* a pre-declared 243-point grid (`theta_T` 0/2500/5000, `beta_W` 0/0.3/0.6,
  `beta_v` 0/0.1/0.2, `p` 0.75/1/1.25, three gap policies) is reported as a
  **model-scenario band**, never as calibrated confidence.

### Production result for `1250009583` @ 2022-07-01

| Quantity | Value |
|---|---:|
| Strict pre-cutoff training rows | 372 (3 MAIN-2 pipes, 2018-09-23) |
| Reference temperature | 285.69 K (12.54 °C) |
| Reference water fraction | 0.868 |
| Reference velocity | 0.571 m/s |
| Fitted `r_ref` | 0.1056 mm/year |
| Target mean environment factor | 0.549 |
| Mean exposure | 16.02 reference-years |
| Mean predicted thickness | **2.810 mm** |
| Scenario band p10 / median / p90 | 1.437 / 2.661 / 3.469 mm |
| Archive-training sensitivity (adds MAIN-3) | 2.735 mm |

The v1 value 1.540 mm is inside the lower half of this band; it corresponds to
transferring the warm/wet rate unchanged and is not supported for a cold/dry
pipe.

## 3. Between-VTD (S1): comparable-inspection rate calibration

Inspection pairs are built from `processed/vtd_by_segment` for consecutive
campaigns of each pipe. Detection density is the **sum of registered VTD
observations**, not the number of inspected segments. A pair is *comparable*
when the density ratio is within `[0.75, 1.33]`.

| Pair | Segments | Density ratio | Comparable |
|---|---:|---:|:--:|
| 1250009490 2018 → 2022 | 4 | 13.94 | no |
| 1250011604 2018 → 2022 | 32 | 11.66 | no |
| 1250022871 2018 → 2022 | 321 | 11.45 | no |
| 1250011604 2022 → 2023 | 47 | 1.21 | yes |
| 1250009490 2022 → 2023 | 5 | 1.65 | no |

Every pair *before* the 2023-12-01 cutoff is non-comparable: the 2018 campaign
registered about 11–14× fewer observations, so apparent thickening is a
detection artifact, not metal gain. The production rate is therefore fitted on
all pre-cutoff pairs and is **0.0 mm/year** (the unconstrained optimum is
negative and is clipped). The prediction transfers the last measured segment
minimum with no further thinning; 4 of 525 segments have no prior measurement
and keep the documented nearest-prior fallback.

The single dense, comparable pair (`1250011604` 2022 → 2023) shows a mean drop
of **−0.320 mm in 1.405 years** (0.228 mm/year) and a prior-thickness
persistence of Pearson² **0.0070**. It lies exactly on the production cutoff,
so under the repository contract it is **disclosed but not used**. This is the
single most important unresolved S1 question.

## 4. Within-pipe spatial term (M2)

Static profile features per 10 m segment: `z`, `slope`, `abs_slope`,
`position`, `local_low` (z − 21-segment rolling median), `pool_depth`
(z − 21-segment rolling minimum). Features are standardized **within each
pipe**; labels are centred within each campaign; a closed-form ridge
(`lambda = 10`) predicts a deviation from the pipe mean.

The term enters only as

```
h = clip(physics_level + amplitude * deviation, 0, h_nominal)
```

with `amplitude = 0.5` (shrinkage). It is **gated** by pooled
leave-one-campaign-out correlation; with negative or undefined evidence it is
set to zero.

Evidence: pooled leave-one-campaign-out Pearson **r = 0.126**, Pearson² =
**0.016** across 9 evaluable campaigns (one campaign has a single segment).
The most consistent single feature is `abs_slope` (positive in 8/8 campaigns),
i.e. steeper segments show a slightly thicker measured minimum; the effect is
small and is not promoted to a physical law. No spatial term is added for S1:
the last measured thickness already carries the segment identity, and a second
spatial layer would double-count it.

Impact on the production cold-start forecast: thickness standard deviation
rises from 0 to ~0.07 mm. That is intentionally small; it is the honest size
of the out-of-sample signal, not a decorative ranking.

## 5. Validation that actually exists

| Design | What it answers | Limitation |
|---|---|---|
| Cold-start leave-one-pipe-out (10 campaigns, 466 rows) | error and correlation of the full pipeline on unseen pipes | mixes environments inside each training fold |
| Cold-start leave-one-parent-out (2 folds) | the actual warm→cold transfer direction | only two environments, two folds |
| Spatial leave-one-campaign-out (9 evaluable) | whether profile information survives a new inspection | small n per campaign |
| S1 leave-one-pipe-out inspection pairs (3 pipes, 357 rows) | transfer of the last measured minimum | only the 2018→2022 transition is pre-cutoff |
| Forward-time held-parent folds | independent historical forecast | **0 usable folds** — unchanged from v1 |

Pooled cold-start error (out-of-pipe, within-pipe effect removed):

| Model | Rows | MAE | RMSE | Bias | Pooled Pearson² |
|---|---:|---:|---:|---:|---:|
| C0 (nominal) | 466 | 1.676 | 1.799 | +1.676 | undefined |
| M0 (v1 constant rate) | 466 | 0.433 | 0.538 | +0.047 | undefined |
| M1 (physics) | 466 | 0.429 | 0.561 | −0.098 | undefined |
| M1S (physics + spatial) | 466 | 0.429 | 0.558 | −0.094 | 0.0166 |

Cross-environment transfer: training on MAIN-2 and predicting MAIN-3
(the production direction) gives bias **+0.26…+0.66 mm** (predicted thickness
too high, i.e. loss too low) and MAE 0.26–0.66 mm. Training on MAIN-3 and
predicting MAIN-2 gives bias **−0.55…−0.95 mm** (loss too high). The central
production forecast sits between these directions.

S1 pair check (last measured minimum transferred across 2018→2022): pooled
within-pipe Pearson² **0.113**. The dense 2022→2023 pair shows persistence of
only **0.007**, so this number is optimistic for the actual 2023 target.

## 6. What remains unidentifiable

* Environment exponents: two confounded environments; no third independent
  first-inspection campaign with a different temperature/watercut/velocity.
* Repairs, pigging, inhibition, chemistry, cathodic protection, external
  corrosion: no dated, segment-mapped history. The fitted rate absorbs them.
* Early operating history: the cold-start pipe was commissioned in 1993 but
  hydraulic coverage begins in 2014 (~49 % of the 29-year horizon).
* UZT: all 67 readings exceed the nominal wall of the corresponding pipe by a
  large margin; they remain excluded, not rescaled.
* Measurement process: 2022 defect depths are integer percentages, 2023 has
  decimals, 2018 has fractions; the residual thickness is internally consistent
  within each campaign, but cross-campaign comparability is not established.

## 7. Open questions for the mentors

1. Is the 2023-12-01 campaign produced by the same tool/threshold as 2022? The
   dense 2022→2023 drop (−0.32 mm) is the only usable degradation signal for S1.
2. May a comparable pair from a *different* pipe on the target date calibrate
   the S1 rate, or does the cutoff contract forbid it?
3. Can any additional first-inspection campaigns be released to identify the
   temperature/watercut/velocity sensitivities?
4. Is there a dated repair/pigging/inhibition log for either target pipe?
5. Are the UZT readings known to refer to a different object or unit?

## 8. Reproduction

```bash
python -m src.ml.experiment_v2 --output-root experiments/ml_v2
python -m unittest discover -s tests -v
```

No new dependencies: NumPy and pandas only. No random seed is used.

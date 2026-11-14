# Stage 01 — Temperature Data Audit

## (a) Branch / HEAD SHA

| Field | Value |
|---|---|
| Branch | `model_v02` |
| HEAD SHA | `2efd775765bbfafa96330b32aa202b70d23466ec` |
| `git status` | clean |

---

## (b) Inspected Files

| File | Purpose |
|---|---|
| `src/temporal.py` | Cutoff-first hydraulic QA; generates `*_usable` fields and time-averages `*_mean` |
| `processed/hydraulic_asof.csv` | 4208 rows, raw + usable temperature fields per pipe per row |
| `processed/hydraulic_quality.csv` | QA action log per field/rule/row |
| `processed/target_features.csv` | 2000 segments, includes `t_mix_mean`, `t_start_mean`, `t_end_mean` |
| `experiments/ml_baseline/training_windows.csv` | 1080 historical training rows, all temperature means present |
| `src/ml/features.py` | Computes `temperature_c_mean` and `arrhenius_thermal_factor` |
| `data/cleaned/passport.csv` | Commissioning dates, `main_pipeline_id` |

---

## (c) Work Performed

### 1. Field Provenance and QA Semantics

| Field | Type | Source | Notes |
|---|---|---|---|
| `t_mix` | **OBSERVED** | raw hydraulic CSV | Measurement; exact sensor location unresolved (see `temporal.py` line 67 comment) |
| `t_start` | **OBSERVED** | raw hydraulic CSV | Inlet temperature |
| `t_end` | **OBSERVED** | raw hydraulic CSV | Outlet temperature |
| `t_mix_usable` | **OBSERVED (QA-filtered)** | `t_mix` after `hydraulic_asof()` | Nulled when zero-during-flow AND not 3-day persistent |
| `t_start_usable` | **OBSERVED (QA-filtered)** | same | No zero-during-flow events found → identical to `t_start` |
| `t_end_usable` | **OBSERVED (QA-filtered)** | same | Identical to `t_end` |
| `t_mix_mean` | **CALCULATED** | daily-median → equal-weight mean over days < cutoff | Single scalar per pipe per cutoff |
| `t_start_mean` | **CALCULATED** | same | Single scalar per pipe per cutoff |
| `t_end_mean` | **CALCULATED** | same | Single scalar per pipe per cutoff |
| `temperature_c_mean` | **CALCULATED (spatial)** | Shukhov exponential on `t_start_mean`/`t_end_mean`; fallback chain | Segment-resolved; see path table below |
| `arrhenius_thermal_factor` | **CALCULATED** | `exp(2500*(1/293.15 - 1/T_K))` from `temperature_c_mean` | `theta=2500 K` is a **legacy placeholder**, not a nominated physical constant |

> [!IMPORTANT]
> `ground_temp_c=4.0` and `fillna(20.0)` in `_compute_temperature_c` are **code fallbacks**, never measurements. They must not be reported as physical observations.

### 2. Temperature Count by Target

#### `1250009583` — `cold_start` — cutoff `2022-07-01` — MAIN-4

| Metric | Value |
|---|---|
| Total hydraulic rows | 1415 |
| `t_mix_usable` valid | **1415 / 1415** (100%) |
| `t_start_usable` valid | 1230 / 1415 (87%) |
| `t_end_usable` valid | 1230 / 1415 (87%) |
| All 3 T fields valid | 1230 / 1415 |
| T + W + v all valid | 1414 / 1415 |
| Rows with T but missing W or v | **1** |
| `t_mix_mean` (pipe-level) | **6.20 °C** |
| `t_start_mean` | **20.91 °C** |
| `t_end_mean` | **6.94 °C** |
| `temperature_c_mean` range (spatial) | **6.94 – 20.91 °C** (std = 3.94 °C) |
| `arrhenius_thermal_factor` range | **0.672 – 1.027** |
| Kelvin range | 280.09 – 294.06 K |

#### `1250022871` — `between_vtd` — cutoff `2023-12-01` — MAIN-2

| Metric | Value |
|---|---|
| Total hydraulic rows | 2793 |
| `t_mix_usable` valid | **2780 / 2793** (99.5%) |
| `t_start_usable` valid | 2756 / 2793 (98.7%) |
| `t_end_usable` valid | 2756 / 2793 (98.7%) |
| T + W + v all valid | 2780 / 2793 |
| Rows with T but missing W or v | **0** |
| `t_mix_mean` | **23.49 °C** |
| `t_start_mean` | **23.77 °C** |
| `t_end_mean` | **23.59 °C** |
| `temperature_c_mean` range (spatial) | **23.59 – 23.77 °C** (std = 0.052 °C) |
| `arrhenius_thermal_factor` range | **1.109 – 1.114** |
| Kelvin range | 296.74 – 296.92 K |

### 3. Celsius → Kelvin Conversion Check

```
t_mix_usable stored in Celsius: [0.0, 44.52] °C
Conversion: T_K = T_C + 273.15
Resulting K range: [273.15, 317.67] K
Clip in _compute_arrhenius_factor: lower=253.15 K → 0 rows affected
All values > 0 K: TRUE
```

✅ Conversion is correct. No anomalous readings requiring attention.

> [!NOTE]
> `t_mix_usable = 0.0 °C` (207 rows) are **not** QA-removed because they occur during zero-flow periods (`q_liquid = 0`). These are likely sensor readings at ambient/ground temperature, not erroneous. The QA rule `zero_temperature_during_flow` only fires when `q_liquid > 0`.

### 4. Rows Lost Because W or v Missing but T Valid

```
T valid but watercut_pct_usable missing: 0 rows
T valid but v_liquid_true_usable missing: 1 row
Total: 1 row
```

**A T-only Arrhenius experiment loses essentially zero data compared to the existing T+W+v joint exposure.** The `exposure_intervals()` gate (`watercut_pct_usable.between(0,100)` and `v_liquid_true_usable.ge(0)`) is the only blocker for 1 row.

### 5. `temperature_c_mean` Profile Path Determination

| Pipe | `t_start_mean` | `t_end_mean` | Path taken | Spatial variation |
|---|---|---|---|---|
| `1250009583` cold_start | 20.91 °C | 6.94 °C | **Shukhov exponential** (ratio = 0.174) | **Large** (std 3.94 °C) |
| `1250022871` between_vtd | 23.77 °C | 23.59 °C | **Shukhov exponential** (ratio = 0.991) | **Negligible** (std 0.052 °C) |

Both targets have `t_start_mean` and `t_end_mean` available → the Shukhov exponential path is activated. The fallback `4.0 °C` ground temperature is used as the soil reference but is **not a measurement**.

For `1250022871`, the inlet–outlet temperature difference is only **0.18 °C**, making `temperature_c_mean` effectively uniform along the pipe. The `arrhenius_thermal_factor` varies by only **0.006** across all 525 segments — it cannot differentiate spatial corrosion risk.

For `1250009583`, the gradient is **13.97 °C** (inlet 20.91 °C → outlet 6.94 °C). The Arrhenius factor varies meaningfully from **0.672 to 1.027**, which is a **35% kinetic acceleration spread** along the pipe length.

### 6. Exposure Horizon Coverage and Gaps

| Pipe | Commissioning | First obs | Pre-obs gap | Max intra-obs gap | Observed fraction |
|---|---|---|---|---|---|
| `1250009583` cold_start | 1993-06-20 | 2014-12-24 | **21.5 years (74%)** | 398 days | 26% of total life |
| `1250022871` between_vtd | 2000-04-22 | 2012-12-12 | 12.6 years | 195 days | ~54% of total life |

> [!CAUTION]
> For `1250009583`, **74% of the total pipe life (21.5 out of 29 years) has no hydraulic observations**. Under the `bounded_forward_hold_1_days` policy, this pre-2015 period is classified as **UNKNOWN** — not zero corrosion. Any Arrhenius experiment must report both observed-only and reference-gap results, and must not fill the missing decades with a constant daily measurement.

The 398-day intra-observation gap (within 2015–2022) is also significant: under `max_hold_days=1`, this gap will also contribute to `unobserved_years`.

---

## (d) Tests and Actual Outputs

All 17 tests from Stage 00 continue to pass (confirmed: `Ran 17 tests in 9.6s — OK`).

No new tests were run in this audit stage; test cases are proposed for Stage 02 (see below).

---

## (e) Deliverable Paths

| File | Description |
|---|---|
| [`experiments/temperature_v02_01/temperature_source_audit.md`](file:///C:/codes/nedra/pipetwins/experiments/temperature_v02_01/temperature_source_audit.md) | This report |
| [`experiments/temperature_v02_01/temperature_audit_data.json`](file:///C:/codes/nedra/pipetwins/experiments/temperature_v02_01/temperature_audit_data.json) | Machine-readable audit by `main_pipeline_id/id/prediction_date` |

---

## (f) Unresolved Blockers

### BLOCKER (Design decision): `arrhenius_thermal_factor` for `between_vtd` is nearly constant

The Shukhov exponential for `1250022871` has only a **0.18 °C inlet-outlet gradient** → Arrhenius factor varies by **0.006** across 525 segments. A temperature-based Arrhenius feature **cannot produce meaningful spatial differentiation** for this pipe.

**Implication:** Any Stage 02 T-only Arrhenius experiment that relies on `temperature_c_mean` for segment-level risk ranking will show near-zero spatial signal for `1250022871`. The M0 baseline (constant rate) already captures this correctly. The temperature feature may still be valid as a **global corrosion-rate modifier** (Arrhenius ≈ 1.11 vs reference), but not as a spatial differentiator.

This is not a code bug — it reflects the physical reality of the pipeline.

### Note: Pre-obs gap policy for cold_start

The 21.5-year pre-observation gap for `1250009583` is UNKNOWN by definition. Any experiment **must** report:
1. `observed_only` result (ignoring the gap)
2. `reference` gap-filling result (F=1 on unknown time)

Both must be presented with separate metrics — do not select one silently.

---

## (g) Go / No-Go for Stage 02

### ✅ GO for Stage 02

**Evidence is sufficient** to characterize which fields can feed a T-only Arrhenius experiment:

| Field | Can feed Arrhenius T-only experiment? | Condition |
|---|---|---|
| `t_mix_usable` | **YES** | 100% coverage for cold_start, 99.5% for between_vtd |
| `t_start_usable` / `t_end_usable` | **YES** | 87–99% coverage; both needed for Shukhov spatial profile |
| `temperature_c_mean` | **YES** | Already computed; spatially varying for cold_start |
| `arrhenius_thermal_factor` | **YES** (with theta candidates) | Currently uses placeholder theta=2500 K; Stage 02 must sweep Ea ∈ {0,25,40,55,70} kJ/mol |

**Required quality flags for Stage 02:**
- Use `t_mix_usable`/`t_start_usable`/`t_end_usable` (never raw `t_mix`/`t_start`/`t_end`)
- Do NOT require `watercut_pct_usable` or `v_liquid_true_usable` for T-only exposure
- Report pre-obs gap as UNKNOWN; do not fill with any temperature value
- Do NOT claim `ground_temp_c=4.0` or `fillna(20.0)` as measurements

**Proposed test cases for Stage 02** are in `temperature_audit_data.json` (8 test cases).

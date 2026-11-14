# PipeTwin provisional ML baseline and audit

## Scope and status

Implemented against `main` at `5f06c32e682fe43433213d6a51087733ccc27570` on 2026-11-14.
The three PDFs in `pdfs/`, both English instructions in `agent/`, mentor summary,
preprocessing source, tests, configuration, documentation, validation and manifest
were inspected. No original data, cleaned data, processed data, preprocessing
configuration or preprocessing implementation was changed. No commit or push was made.

The PDFs establish segment-minimum thickness in mm, two separate prediction cases,
the exact three-field submission, and **squared Pearson correlation** as the
organizer's metric. Coefficient of determination R² is reported separately.
The mentor summary/config resolve the target IDs and dates. The actual CSVs,
not the PDF's simplified source-file descriptions, determine feature availability.

The implementation provides a **working, provisional baseline**, not a completed
or independently validated hybrid model. Generated results are in
`experiments/ml_baseline/`. Reproduction does not require downloading raw data.

## What already existed, and what was added

Existing: source parsing, quality flags, 10 m grids, spatial mapping, strict
as-of hydraulic filtering, VTD/UZT histories, CSV/Parquet outputs and integrity checks.

Added under `src/ml/`:

| Module | Responsibility |
|---|---|
| `data.py` | Verify published CSV checksums and restore manifest types; no new raw parser |
| `audit.py` | Exact schemas, keys, examples, missingness, dates and source aliases |
| `windows.py` | Historical features through `build_features_asof`, labels joined afterward, initial conditions |
| `exposure.py` | Explicit bounded-hold intervals, gap accounting, nonlinear exposure integral |
| `models.py` | C0, fitted nonnegative M0, W0/W1/W2 sensitivity, gated regularized M1 |
| `evaluation.py` | Strict earlier-date/held-parent folds, distinct Pearson² and R², fallback diagnostic |
| `submission.py` | Exact key/column/finite/bounds validation and tie-aware risk ranking |
| `experiment.py` | End-to-end forecasting, reports, models, metrics, hashes and exports |

## Observed source-to-feature audit

Full exact columns, all missing counts and date ranges are in
`experiments/ml_baseline/source_audit.json`. The following is the decision summary;
counts are directly observed, not copied from a previous report.

| Published path (CSV and Parquet) | Rows | Join keys | Permitted use and restrictions |
|---|---:|---|---|
| `processed/segments_grid` | 4,668 | `id,segment_id` | Geometry, both cases |
| `processed/target_segments_grid` | 2,000 | `id,segment_id,prediction_date,case` | Authoritative submission keys; 525 S1 + 1,475 S2 |
| `processed/target_features` | 2,000 | `id,segment_id,prediction_date,case` | Production snapshot only; never joined to older labels |
| `processed/training_targets` | 1,080 | `id,date,segment_id` | Measured eligible labels; still filter strictly before each production/fold cutoff |
| `processed/vtd_by_segment` | 1,543 | `id,date,segment_id` | 66 missing thickness minima; S1 history only; apply eligibility |
| `processed/profile_by_segment` | 4,668 | `id,segment_id` | Complete geometry; static profile assumption |
| `processed/regime_mapped` | 2,000 | `id,segment_id,prediction_date` | Production hydraulic summaries; historical snapshots must be rebuilt |
| `processed/hydraulic_asof` | 4,208 | `id,date,prediction_date` | Use `*_usable`; `pit_corrosion_intensity` remains forbidden |
| `processed/hydraulic_quality` | 41 | `id,source_row,prediction_date,field,rule` | Audit decisions, not physical measurements |
| `processed/monitoring_mapped` | 2,000 | `id,segment_id,prediction_date` | Zero usable chemistry records, all three chemistry medians missing |
| `processed/quality_issues` | 70,668 | `file,source_row,field,rule` | Provenance only, never automatic model predictors |

Raw aliases already defined in `src/config.py` and the preprocessing manifest are
preserved: `Residual Wall Thickness` → `residual_wall_thickness` →
`min_residual_thickness_mm`; `Distance` → `distance`; `q` → `q_liquid`;
`watercut` → `watercut_pct`; `Angle` → `angle`.

S1 hydraulic history spans 2012-12-12 to 2023-11-30; S2 spans 2014-12-24 to
2022-06-30. These ranges are not complete exposure coverage. The S2 commissioning
date is 1993-06-20, leaving a long unobserved initial history. Under the explicit
one-day hold assumption, usable T/W/v covers approximately **13.30%** of its
29.03-year horizon. S1 coverage is approximately **83.19–91.42%**, depending on
each segment's last valid inspection date.

### Contradictions and uncertainty

- **Observed:** the raw passport contains 16 simple sections, not 16 independent
  pipelines. `main_pipeline_id` identifies four parent pipelines. Eligible labels
  cover only MAIN-2 and MAIN-3. MAIN-2 contains three labeled simple sections;
  MAIN-3 contains seven. The parent IDs are used as conservative validation groups.
- **Observed:** before 2022-07-01, only the 2018-09-23 campaign on MAIN-2 is usable
  for cold-start training: 372 labels. The 2022-07-06 campaign is five days too late.
- **Observed:** before 2023-12-01, S1 rate training has 357 matched prior/current
  segment minima, all from the MAIN-2 2022-07-06 campaign. Another 204 labeled
  segments lack a prior measured minimum and are excluded from rate fitting.
- **Observed:** 177 of those 357 matched minima increase between inspections.
  Negative loss is retained in fitting. Repairs, changing detected minima,
  measurement error or spatial alignment could explain this; the cause is unknown.
- **Observed:** four S1 target segments (17, 247, 401, 455) lack prior minima.
  Missing records do not establish nominal thickness.
- **Observed:** CSV checksums match all manifest outputs. `src/config.py` differs
  from the preprocessing manifest's saved code hash. The current config and
  reproduced target features match the published artifacts; provenance remains
  explicitly flagged rather than silently rewritten.
- **Stated in the published report:** 270 preprocessing checks passed previously.
  This is distinct from tests actually executed in this ML session.
- **Unknown:** repair chronology, instrument angle reference, chemical-unit/join
  resolution, independent campaign performance and hidden-test accuracy.

## Historical examples and leakage boundaries

**S1 example:** request `id=1250022871, prediction_date=2022-07-06,
case=between_vtd`. Features are rebuilt before joining the measured 2022 thickness
by `id,segment_id`. Eligible prior starts come from 2018-09-23. Hydraulic rows
must be strictly earlier than 2022-07-06 00:00. The later production snapshot is
not used to build this window.

**S2 example:** request `id=1250022871, prediction_date=2018-09-23,
case=cold_start`. The initial condition is 8 mm at 2000-04-22. All target-pipe
VTD/UZT histories are removed from the API inputs; the archive-wide diagnostic
conflict flag cannot influence age. The measured 2018 minimum is attached only
as the response. This campaign may train the 2022-07-01 cold-start forecast.

Concrete rows and their source dates are in `training_windows.csv`. Official
cold-start pipe 1250009583 is excluded from every training-label pool. For
cold-start fitting, the entire production parent is excluded. S1 can learn
from its own earlier inspection pairs, as intended by the case.

All fitted validation folds require both `train_date < validation_date` and
different parent pipelines. With this dataset there are **zero usable fitted-model
folds before either production cutoff**. No random segment split substitutes for
this missing evidence. C0 has no fitted parameters and can still be scored.

## Models and initial conditions

The selected provisional model is a predeclared baseline, not a validation winner:

`thickness_pred = clip(h0_mm - rate_mm_year * horizon_years, 0, h0_mm)`.

The rate minimizes weighted **unclipped** squared loss in mm, constrained to be
nonnegative. M0/W1 fitted rates are approximately 0.013034 mm/year for S1 and
0.101980 mm/year for S2. They are effective empirical rates across measured
segment minima, not identified CO2 corrosion constants. C0 uses zero rate.

W1 balances parents, then campaigns within parents; with one parent/campaign
it equals W0. W2 doubles the weights of rows below the training-label bottom
decile as a fixed sensitivity experiment. It is not scientifically prescribed
and is not selected based on held-out results.

S1 starts from each segment's last valid measured minimum. Missing segments copy
the nearest prior measured segment and its date, with an explicit assumption
flag and recorded distance. A secondary 100 m block-masking check on known
prior measurements gives MAE approximately **0.541 mm**. This does not validate
the four truly missing segments or constitute independent forecast accuracy.

S2 assumes nominal 4.5 mm at commissioning. M0 produces approximately **1.540 mm
for every segment**. This is a severe extrapolation from a different parent,
nominal thickness and shorter observed operating life. It cannot resolve spatial
risk or provide a useful within-pipe Pearson score. The risk export discloses
all ties; segment_id tie-breaks are administrative, not evidence of danger.
The top-10% file rounds up separately per pipe: 53 S1 rows and 148 S2 rows
(201 total), using `ceil(0.1 * pipe_segment_count)`.

M1 implements the specification's nonlinear T/W/v integral with train-only
reference conditions, temperature in Kelvin, watercut as a fraction and time
in years. It uses a deterministic regularized coordinate search with
nonnegative temperature sensitivity. Its minimum of three training parents is
an engineering guard, not a scientific sufficiency threshold. It is tested on
synthetic known-rate data and **not fitted to this insufficient dataset**.
M2/M3/M4 and learned residual corrections are deferred until a credible
independent evaluation becomes possible. No claimed M1 validation metrics exist.

The interval builder forward-holds valid readings for at most 24 hours, stopping
at the next record or prediction cutoff. This duration is an explicit assumption;
it is not measured exposure. It never backfills commissioning history. M1
supports observed-only sensitivity and unit-factor reference filling for gaps;
neither is claimed to recover the missing history. M0 instead assumes a constant
rate over the entire horizon and does not use those integrals.

## Factor status matrix

| Factor group | Actual availability | Current decision |
|---|---|---|
| Operating life, diameter, nominal thickness | Available passport static columns | Use nominal/commissioning baseline; no replacement history |
| Segment length and longitudinal position | Available full grid | Keys, fallback distance and diagnostics |
| Elevation, slope, low points | Available profile, some steep steps | Retain audited features; no unvalidated spatial effect fitted |
| Steel/material/construction | Archival passport; not static whitelist | No causal/as-of enrichment yet |
| Internal/external coating, weld insulation | Archival state without trustworthy change history | Dangerous for historical enrichment; exclude |
| Liquid/oil flow, watercut, temperature, speed | Available QA-filtered timestamps, incomplete history | Exposure bookkeeping and M1 implementation |
| Gas flow/gas factor/flow regime/fill coefficient | Archival; sparse or outside usable-field whitelist | Exclude pending unit/QA enrichment |
| Pressure/density/viscosity/Reynolds/shear | Some usable summaries; units/phase meaning incomplete | No CO2 partial-pressure inference or unverified shear calculation |
| Shutdowns/intermittency | Potentially derivable; gaps are not shutdowns | No invented zero-flow exposure |
| Water/oil/gas physicochemistry: pH, CO2, H2S, O2, ions, mineralization | Sparse source, zero eligible mapped target chemistry | UNAVAILABLE as target inputs; no synthetic zeros |
| Acid value, alkalinity, suspended solids, microbes | Sparse/unverified archival chemistry | No quantitative effect fitted |
| Inhibitors/chemical treatment/protection efficiency | Archival/sparse; no verified dosing history | Exclude |
| Soil/climate/groundwater/external environment | Partial archival descriptors; no verified histories | Exclude |
| Cathodic protection | No verified dated current/potential history | UNAVAILABLE |
| Cleaning/pigging/deposits/protective films | No usable event or film history | UNAVAILABLE; film omission is a case approximation |
| Repairs/replacements/sleeves | No reliable dated segment mapping | UNAVAILABLE; simple monotone clipping assumes no repair |
| Prior VTD minima and defect geometry/type | Available with measurement/chronology exclusions | Minimum only in S1; mixed mechanisms remain unresolved |
| Angles | Present but orientation/alignment unverified | No bottom-of-pipe rate multiplier; forbidden for S2 |
| UZT | All 67 readings exceed nominal; excluded by current QA | UNAVAILABLE as trusted baseline |
| Failures/events | No verified event table in the processed contract | UNAVAILABLE |
| `pit_corrosion_intensity`, diagnostic passport summaries, future inspections/repairs | Available archival fields but target-dependent/undated | QUARANTINE, never model inputs |

## Validation actually executed

- `python -m unittest discover -s tests -p test_ml.py -v`: **17 tests passed**.
  They check actual artifact joins/counts, cutoff and future-label injection,
  cold diagnostic injection, parent/time grouping, fallback provenance,
  checksum rejection, units/durations/gaps, weighting, metric definitions,
  synthetic physical fitting and submission failures.
- The end-to-end experiment completed and reread/validated its output: **2,000
  unique rows**, exact target keys, exactly three columns, finite bounded mm.
- A second run in `scratch/ml_repro` was compared byte-for-byte against all
  generated experiment files to check deterministic reproduction.
- Full `python -m unittest discover -s tests -v` was attempted but the existing
  preprocessing test module cannot import because **PyArrow is unavailable**.
  It is not claimed as passing. Network package installation was unavailable.
- Executed with the local runtime's Python 3.12, pandas 2.2.3 and NumPy 2.3.5.
  This differs from the preprocessing lock; actual versions are in the run manifest.
  The ML CSV reader allows this phase to run without changing preprocessing.

## Reproduce and continue

For ML-only work from the committed CSV artifacts, use Python 3.12 in a separate
environment and install `requirements-ml.txt`. Run:

```bash
python -m src.ml.experiment --output-root experiments/ml_baseline
python -m unittest discover -s tests -p test_ml.py -v
```

For reproduction from the original six CSVs, first install the original
`requirements.txt` environment and run `python -m src.data_prep`; then run the ML
command. Run the full test suite there, with PyArrow installed. The original
preprocessing environment was not reinstalled or independently retested here.

The next substantive work is to resolve the 177 apparent thickness increases
and request/identify additional independently dated parent-pipeline campaigns.
Spatial-block experiments may offer secondary diagnostics, but must not be sold
as independent-pipeline validation. An independently validated forecast should
replace this baseline before treating spatial ranking or uncertainty as reliable.

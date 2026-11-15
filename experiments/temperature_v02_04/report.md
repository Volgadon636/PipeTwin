# Stage 04: fixed Ea40 development model

Ea = **40 kJ/mol** is fixed by the user for development. Stage 03.1 did not scientifically select it. All other grid values are comparators; no new Ea selection occurs. No production promotion occurred.

`h_pred = clip(h0_mm - r_ref_mm_year * E_T, 0, h0_mm)`; `r_ref = max(0, sum(W1*E_T*(h0-h_obs))/sum(W1*E_T^2))`. The objective uses unclipped measured losses, including negative thinning. Rates are fit separately for each case and assumption. E_T uses only cutoff-specific t_mix_usable in Kelvin, R=8.314462618, Tref=293.15; no legacy Arrhenius/spatial term is multiplied in.

`result.csv` is EA40 with a one-day hold and F=1 assumed in unsupported history. It is an experimental candidate, not an official result. The observed-only export is an incomplete-history descriptor and must not be treated as a full-history physical forecast.

## Training rates (primary one-day hold)

| Case | Policy | Ea kJ/mol | Rate mm/year | Rows | Parents | Campaigns |
|---|---|---:|---:|---:|---:|---:|
| between_vtd | observed_only | 0 | 0.0162637525 | 357 | 1 | 1 |
| between_vtd | observed_only | 25 | 0.0114868528 | 357 | 1 | 1 |
| between_vtd | observed_only | 40 | 0.00908861443 | 357 | 1 | 1 |
| between_vtd | observed_only | 55 | 0.00707710411 | 357 | 1 | 1 |
| between_vtd | observed_only | 70 | 0.00543655588 | 357 | 1 | 1 |
| between_vtd | reference_gap | 0 | 0.0130335508 | 357 | 1 | 1 |
| between_vtd | reference_gap | 25 | 0.00977576103 | 357 | 1 | 1 |
| between_vtd | reference_gap | 40 | 0.00798302595 | 357 | 1 | 1 |
| between_vtd | reference_gap | 55 | 0.00638819712 | 357 | 1 | 1 |
| between_vtd | reference_gap | 70 | 0.00502068951 | 357 | 1 | 1 |
| cold_start | observed_only | 0 | 0.581855542 | 372 | 1 | 1 |
| cold_start | observed_only | 25 | 0.752174809 | 372 | 1 | 1 |
| cold_start | observed_only | 40 | 0.85632946 | 372 | 1 | 1 |
| cold_start | observed_only | 55 | 0.951736539 | 372 | 1 | 1 |
| cold_start | observed_only | 70 | 1.02701409 | 372 | 1 | 1 |
| cold_start | reference_gap | 0 | 0.10197981 | 372 | 1 | 1 |
| cold_start | reference_gap | 25 | 0.106197857 | 372 | 1 | 1 |
| cold_start | reference_gap | 40 | 0.108058205 | 372 | 1 | 1 |
| cold_start | reference_gap | 55 | 0.109448603 | 372 | 1 | 1 |
| cold_start | reference_gap | 70 | 0.110385957 | 372 | 1 | 1 |

## Ea40 target candidate summaries

Prediction differences are comparisons with B0, not errors against target labels.

| Case | Policy | Mean prediction mm | Mean delta vs B0 mm | Maximum absolute delta mm | Coverage | Unknown share |
|---|---|---:|---:|---:|---:|---:|
| between_vtd | observed_only | 6.196022 | -0.009394 | 0.009397 | 91.1961% | 8.8039% |
| between_vtd | reference_gap | 6.198429 | -0.006987 | 0.006990 | 91.1961% | 8.8039% |
| cold_start | observed_only | 2.996577 | 1.456992 | 1.456992 | 13.3086% | 86.6914% |
| cold_start | reference_gap | 1.590891 | 0.051307 | 0.051307 | 13.3086% | 86.6914% |

## Why E_a is presently unidentifiable

The current audit found 0 independent forward-time held-parent folds. Both eligible cohorts contain one parent and one campaign. Stage 03.1 showed practical ties and sensitivity to reference-gap filling. There is no independent accuracy claim, fitted material constant, or population confidence interval. Full local sensitivity rank does not establish practical identification.

Within-parent same-campaign section diagnostics remain documented in Stage 03.1; they are not relabelled as validation here. Temperature, time and operating histories are confounded, and chemistry is unmeasured. Independent parents/campaigns, denser temperature records, paired W/v and reliable repeated thickness measurements are needed.

Cold-start section-level exposure cannot create spatial risk rankings when h0 and the time window are shared. Constant predictions yield undefined Pearson squared, not zero. Production metrics are deliberately unavailable because production labels are not used. Four between-VTD target baselines use the existing nearest-prior assumption and remain flagged.

## Provenance and reproduction

The attached Stage 04 prompt defines the experiment scope; the user fixes Ea40. The English specification was absent; the supplied Russian v1.1 specification was read in full. Existing source wins over historical descriptions. The original M0 saved result is copied byte-for-byte into B0_original_M0.csv and checked against a reconstruction on the identical eligible rows. EA00/reference reproduces that reconstruction; EA00/observed-only need not.

Historical windows are built once, then each target snapshot is built once through make_snapshot. Existing RepositoryData verifies published input hashes. Intervals are frozen once per section/start/cutoff/hold and reused for all candidates. W/v missingness does not gate temperature; invalid next records end prior holds. Baseline provenance, interval hashes, support and all observed-loss quality flags are exported.

Run from repository root:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_04.run
.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_04.verify
```

Tests, exact output hashes and preservation checks are finalized by verify.py. Candidate CSVs have exactly id,segment_id,thickness_pred and 2000 keys (525 + 1475). prediction_diagnostics.csv retains the richer contract and evidence status. All outputs are isolated in this directory.

## Next-stage gate

GO for Stage 05 exploratory ablation/development with Ea40 fixed and limitations carried forward. NO-GO for claiming improved independent performance, a validated Ea, or production promotion. Known repository regression failures remain documented separately; this stage does not modify preprocessing or old tests to mask them.

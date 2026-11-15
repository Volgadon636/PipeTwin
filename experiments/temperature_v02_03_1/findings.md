# Executed numerical findings

These are exploratory historical fits and dependent section holdouts. No production target was used for selection.

## between_vtd: candidate_set_retained

One-day row-W1 observed support is 80.1364%; this is historical support, not the prior production statistic.

All fitted sensitivity matrices have effective normalized rank 2–2. Thermal exposure shapes are numerically different (proportionality residual versus EA00 ranges 0.00199267–0.0109472 across holds/policies). Full local rank does not establish practical identification.

- observed_only: SSE span 0.0100152% across the grid; fitted rates 0.00543656–0.0162638 mm/year; equal-section held-out MAE 0.412013–0.412068 mm. Fit tie set: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high. Transfer tie set: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high.
- reference: SSE span 0.00920769% across the grid; fitted rates 0.00502069–0.0130336 mm/year; equal-section held-out MAE 0.412013–0.412064 mm. Fit tie set: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high. Transfer tie set: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high.

Reference gaps contribute 7.7102%–19.8636% of exposure across candidates. This changes the relative candidate contrast; assumptions are never pooled.

Practical held-out ties prevent a common unique primary winner across both gap assumptions. EA00 is retained; extra thermal sensitivity is unsupported by these selection diagnostics.

## cold_start: candidate_set_retained

One-day row-W1 observed support is 17.5268%; this is historical support, not the prior production statistic.

All fitted sensitivity matrices have effective normalized rank 2–2. Thermal exposure shapes are numerically different (proportionality residual versus EA00 ranges 0.000609044–0.0462923 across holds/policies). Full local rank does not establish practical identification.

- observed_only: SSE span 3.66579% across the grid; fitted rates 0.581856–1.02701 mm/year; equal-section held-out MAE 0.357918–0.578409 mm. Fit tie set: EA00_null, EA25_low. Transfer tie set: EA00_null.
- reference: SSE span 0.165168% across the grid; fitted rates 0.10198–0.110386 mm/year; equal-section held-out MAE 0.358204–0.363313 mm. Fit tie set: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high. Transfer tie set: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high.

Reference gaps contribute 82.4732%–89.2790% of exposure across candidates. This changes the relative candidate contrast; assumptions are never pooled.

Practical held-out ties prevent a common unique primary winner across both gap assumptions. EA00 is retained; extra thermal sensitivity is unsupported by these selection diagnostics.

## Interpretation

Between-VTD rate decreases as Ea increases while SSE is almost unchanged: the rate absorbs most thermal scaling. Cold-start observed temperatures are colder on average than the reference in these histories, so higher Ea reduces exposure and requires a larger fitted rate. Reference gaps weaken that contrast substantially.

The between-VTD loss includes 177 negative-thinning rows, retained exactly as required. Baseline/measurement contradictions limit interpretation; no observations were clipped out of the fitting objective.

Neither case supports a unique thermal choice. Cold-start observed-only evidence opposes stronger thermal sensitivities, but that preference is not practically distinguishable with reference gaps. No candidate is rejected as a physical hypothesis. All five remain in the conservative retained set.

section_perturbations.csv recomputes every candidate ranking after omitting each held-out section, without refitting or presenting confidence intervals. grouped_transfer_metrics.csv exposes opposing folds; hold_shape_diagnostics.csv compares shorter holds directly to the primary exposure shape. Selection gates requiring a unique proposal are null/not assessed when no such proposal exists, not falsely reported as rank failure.

Additional evidence needed: independent parents and later campaigns with strict earlier calibration labels; denser timestamped temperature histories spanning distinct trajectories; paired valid W/v to diagnose confounding; reliable repeat thickness measurements and measured baselines. Stage 04 may develop against the retained grid, but must not claim an experimentally selected Ea or promote production results.

## Descriptive paired associations

Each section-timestamp contributes once per case. Missing W/v does not reduce temperature exposure. Correlation is descriptive; shared operating dates and parent histories are not independent replicates.

| Case | Pair | Paired valid / temperature timestamps | Pearson r |
|---|---|---:|---:|
| between_vtd | t_mix_usable / elapsed_years | 3331 / 3331 | 0.69966 |
| between_vtd | t_mix_usable / watercut_pct_usable | 3331 / 3331 | -0.016521 |
| between_vtd | t_mix_usable / v_liquid_true_usable | 3331 / 3331 | -0.057667 |
| between_vtd | elapsed_years / watercut_pct_usable | 3331 / 3331 | 0.017688 |
| between_vtd | elapsed_years / v_liquid_true_usable | 3331 / 3331 | -0.22074 |
| between_vtd | watercut_pct_usable / v_liquid_true_usable | 3331 / 3331 | -0.034912 |
| cold_start | t_mix_usable / elapsed_years | 3598 / 3598 | 0.084032 |
| cold_start | t_mix_usable / watercut_pct_usable | 3598 / 3598 | 0.055249 |
| cold_start | t_mix_usable / v_liquid_true_usable | 3598 / 3598 | 0.012027 |
| cold_start | elapsed_years / watercut_pct_usable | 3598 / 3598 | 0.09288 |
| cold_start | elapsed_years / v_liquid_true_usable | 3598 / 3598 | 0.25528 |
| cold_start | watercut_pct_usable / v_liquid_true_usable | 3598 / 3598 | -0.34922 |

# Experiment Report: baseline_control

Primary ML Model: **M0**
Git commit: `8770be9693de299f0b9b4c27db1ad7ae9ce517af` on branch `model_v02`.

## Forecast Summary

### Target 1250022871 (between_vtd)
- Segment count: **525**
- Thickness min / max / mean: **3.502 / 7.182 / 6.205 mm**
- Thickness std: **0.5560 mm** (non-degenerate spatial distribution)

### Target 1250009583 (cold_start)
- Segment count: **1475**
- Thickness min / max / mean: **1.540 / 1.540 / 1.540 mm**
- Thickness std: **0.0000 mm** (non-degenerate spatial distribution)

## Generated Files
- `result.csv`: Official submission file (id, segment_id, thickness_pred)
- `risk_ranking.csv`: Priority ranking for UZT inspection (top 10% risky segments)
- `risk_top_10pct.csv`: Filtered top 10% critical segments
- `prediction_diagnostics.csv`: Complete segment feature breakdown with unclipped and rate predictions
- `candidate_predictions.csv`: Side-by-side comparison across C0, M0 (W0/W1/W2), Ridge, LightGBM, CatBoost, Ensemble
- `metrics.json`: Forward cross-validation performance metrics (Pearson r2, R2, MAE, RMSE, MAPE)
- `run_manifest.json`: Cryptographic audit trail of inputs, code hashes, and execution environment

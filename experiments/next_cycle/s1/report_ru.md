# S1 next cycle: controls, last-baseline learners and honest limits

Exploratory-цикл. LOPO на трёх простых участках одного parent pipeline — spatial-transfer диагностика, не свежий независимый holdout. Скрытые ответы и post-cutoff метки не использовались.

## Данные и протокол

- fit: 357 строк, три простых участка (4 / 32 / 321), одна кампания `2022-07-06`.
- Признаки строятся только по ВТД строго раньше даты ответа.
- Новая core-модель использует один last baseline; S1B — отдельное правило.
- Old S1C — reference, не выбранный winner.

## Сравнение на одинаковых 357 fit-eligible ключах

| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | top-10% recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s1a | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 | 0.1133 | 0.2895 |
| s1b | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 | 0.1133 | 0.2895 |
| old_s1c | 0.4698 | 0.6180 | -0.0016 | 0.0913 | 0.0083 | -0.1473 | 0.0097 | 0.1053 |
| train_median | 0.4282 | 0.5979 | 0.1507 | -0.0483 | 0.0023 | -0.0740 | n/a | 0.0789 |
| affine | 0.4171 | 0.5711 | 0.0762 | 0.2050 | 0.0420 | 0.0202 | 0.0612 | 0.2895 |
| ridge_history | 0.4454 | 0.5966 | 0.0622 | 0.1382 | 0.0191 | -0.0692 | 0.0199 | 0.1316 |
| ridge_geometry | 0.4511 | 0.6034 | 0.0564 | 0.1042 | 0.0109 | -0.0938 | 0.0119 | 0.0789 |
| rate_signed | 0.4511 | 0.6034 | 0.0564 | 0.1042 | 0.0109 | -0.0938 | 0.0119 | 0.0789 |
| rate_nonnegative | 0.4580 | 0.5993 | -0.1160 | 0.3191 | 0.1018 | -0.0791 | 0.1016 | 0.2895 |
| catboost | 0.4449 | 0.6101 | 0.1274 | 0.1839 | 0.0338 | -0.1183 | 0.0355 | 0.1842 |
| mix_025 | 0.4658 | 0.6131 | 0.0511 | 0.3179 | 0.1010 | -0.1291 | 0.1020 | 0.2895 |
| mix_050 | 0.4426 | 0.5909 | 0.0529 | 0.2859 | 0.0817 | -0.0487 | 0.0830 | 0.2632 |
| mix_100 | 0.4511 | 0.6034 | 0.0564 | 0.1042 | 0.0109 | -0.0938 | 0.0119 | 0.0789 |
| mix_000 | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 | 0.1133 | 0.2895 |

`rate_signed` OOF численно совпал с `ridge_geometry`, потому что в train один и тот же dt=3.7837. На production фактический dt различается (1.4045/5.1882), поэтому rate-прогноз отличается. Это не доказывает корректное масштабирование скорости по времени.

## Macro по участкам

| Модель | macro MAE | macro Pearson r² | macro R² |
|---|---:|---:|---:|
| s1a | 0.4095 | 0.1413 | -0.5859 |
| old_s1c | 0.4347 | 0.0530 | -0.5933 |
| ridge_geometry | 0.4008 | 0.0582 | -0.2951 |
| rate_signed | 0.4008 | 0.0582 | -0.2951 |
| catboost | 0.3935 | 0.0140 | -0.2533 |

## All-pipeline 561 строк (fallback = S1B)

| Модель | n | MAE | bias | within-section r² |
|---|---:|---:|---:|---:|
| s1a | 561 | 0.5305 | -0.0684 | 0.0611 |
| s1b | 561 | 0.5305 | -0.0684 | 0.0611 |
| old_s1c | 561 | 0.5124 | -0.1008 | 0.0093 |
| ridge_geometry | 561 | 0.5005 | -0.0639 | 0.0053 |
| rate_signed | 561 | 0.5005 | -0.0639 | 0.0053 |
| catboost | 561 | 0.4965 | -0.0187 | 0.0072 |

## Weight ablation (history-only)

| Схема | MAE | within-section r² | Kish ESS | group totals |
|---|---:|---:|---:|---|
| row_uniform | 0.4454 | 0.0199 | 357.00 | {"1250009490": 4.0, "1250011604": 32.0, "1250022871": 321.0} |
| inverse_sqrt_section_size | 0.4490 | 0.0233 | 218.00 | {"1250009490": 27.91971481625562, "1250011604": 78.96887870147548, "1250022871": 250.11140648226893} |
| inverse_section_size | 0.4662 | 0.0245 | 31.65 | {"1250009490": 119.0, "1250011604": 119.0, "1250022871": 119.0} |

## Spatial block diagnostic

Статус: `same_pipe_spatial_interpolation`; 100 м blocks, buffer 20 м, same-pipe interpolation, не прогноз будущей кампании.

| Модель | MAE | within-section r² |
|---|---:|---:|
| s1a | 0.5068 | 0.1223 |
| s1b | 0.5068 | 0.1223 |
| ridge_geometry | 0.4185 | 0.1633 |
| rate_signed | 0.4185 | 0.1633 |
| catboost | 0.4179 | 0.1309 |

Block-CV (same-pipe interpolation, 100 м + 20 м buffer): within-section r² S1A=0.1223, ridge_geometry/rate_signed=0.1633, catboost=0.1309; top-10% recall S1A=0.303, ridge/rate=0.455. Это не LOPO между трубами; LOPO выше сигнал не подтверждает.

## Forward folds

`strict_forward_folds_available=False`. Every eligible S1 label has one prior campaign; 2023-12-01 labels are on the cutoff and excluded.

## Selection

default = **s1a**. no candidate improved both within-section spatial signal and MAE; S1A remains default.
best_spatial = **mix_025**; best_mae = **mix_050**.
Никакой кандидат не объявляется доказанным winner; production result не перезаписан.

## Production candidates

| Candidate | mean | std | min | max | above last | fallback | top33 overlap S1A | segment 135 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1a | 6.2250 | 0.5548 | 3.5200 | 7.2000 | 0 | 4 | 33 | 3.5200 |
| s1b | 6.0894 | 0.5812 | 3.5200 | 7.0400 | 0 | 4 | 27 | 3.5200 |
| old_s1c | 6.0261 | 0.2815 | 4.4981 | 7.0664 | 149 | 4 | 12 | 5.1281 |
| ridge_history | 6.0177 | 0.2384 | 4.8649 | 6.8000 | 121 | 4 | 27 | 5.0989 |
| ridge_geometry | 6.0604 | 0.2964 | 4.7981 | 7.0496 | 147 | 4 | 27 | 4.9903 |
| rate_signed | 6.1618 | 0.4434 | 4.0658 | 6.9095 | 147 | 4 | 32 | 4.0658 |
| rate_nonnegative | 6.1320 | 0.4994 | 3.5200 | 6.9095 | 0 | 4 | 33 | 3.5200 |
| catboost | 6.0685 | 0.3910 | 3.6657 | 6.8000 | 172 | 4 | 26 | 3.8493 |
| mix_025 | 6.1839 | 0.4807 | 3.8876 | 7.0044 | 147 | 4 | 33 | 3.8876 |
| mix_050 | 6.1427 | 0.4109 | 4.2391 | 6.8848 | 147 | 4 | 31 | 4.2552 |
| mix_100 | 6.0604 | 0.2964 | 4.7981 | 7.0496 | 147 | 4 | 27 | 4.9903 |

## Files

- `metrics_pooled.json`, `metrics_per_fold.csv`, `oof_fit_predictions.csv`, `oof_all_pipeline.csv`
- `weight_ablation.csv`, `block_cv/`, `forward_fold_audit.json`, `selected_candidates.json`
- `candidates/<name>/predictions_s1.csv`, `prediction_diagnostics.csv`, `result.csv`, `summary.json`
- `figures/`

```bash
python -m src.ml.next_s1 --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s1
```

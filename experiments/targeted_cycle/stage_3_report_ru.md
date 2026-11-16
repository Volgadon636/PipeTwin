# Stage 3 — S1 fixed 100 m block / 20 m buffer

Generated: `2026-11-15T21:42:54+00:00`. Physical blocks: `center_m=(start+end)/2`, `block=floor(center_m/100)`, buffer 20 m; training excludes the entire target block plus buffer. No future answer is used in features.

## 1. Protocol and keys

- Target pipe `1250022871`, 321 measured fit-eligible segments.
- Test keys: measured target segments whose center lies in the held-out 100 m block.
- Global models train on all eligible simple sections outside block+buffer; target-only models train only on target-pipe rows outside block+buffer.
- All imputers/scalers/features/coefficients are fit inside each train fold.
- LOPO is not used here; Stage 4 robustness and temporal Stage 5 are not started.

## 2. Metrics on the same 321 OOF keys

| Model | MAE | RMSE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail global MAE | top10 global | top10 per-block | within-block r | within-block r² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s1a | 0.5068 | 0.6632 | 0.0632 | 0.3497 | 0.1223 | -0.2375 | 0.5604 | 0.5962 | 0.8923 | 0.3030 | 0.3077 | 0.2467 | 0.0609 |
| train_median | 0.4381 | 0.6026 | 0.0881 | n/a | n/a | -0.0219 | 0.0000 | 0.5962 | 1.3690 | 0.0606 | 0.3269 | n/a | n/a |
| affine_global | 0.4245 | 0.5696 | 0.0137 | 0.2976 | 0.0885 | 0.0873 | 0.1936 | 0.5962 | 1.1493 | 0.3030 | 0.3077 | 0.2222 | 0.0494 |
| affine_target | 0.4265 | 0.5702 | 0.0027 | 0.2961 | 0.0877 | 0.0853 | 0.2054 | 0.5962 | 1.1302 | 0.3030 | 0.3077 | 0.2218 | 0.0492 |
| global_ridge_geometry | 0.4187 | 0.5464 | 0.0082 | 0.4031 | 0.1625 | 0.1599 | 0.2691 | 0.5962 | 1.0090 | 0.4545 | 0.3462 | 0.2491 | 0.0621 |
| target_ridge_geometry | 0.4214 | 0.5482 | 0.0027 | 0.4018 | 0.1614 | 0.1545 | 0.2892 | 0.5962 | 0.9857 | 0.4545 | 0.3462 | 0.2521 | 0.0636 |
| global_ridge_local | 0.4166 | 0.5461 | 0.0079 | 0.4049 | 0.1640 | 0.1610 | 0.2730 | 0.5962 | 0.9983 | 0.4242 | 0.3269 | 0.2500 | 0.0625 |
| catboost_reference | 0.4185 | 0.5637 | 0.0040 | 0.3604 | 0.1299 | 0.1059 | 0.3071 | 0.5962 | 1.0112 | 0.2727 | 0.3654 | 0.1978 | 0.0391 |

`top10 global`: k=ceil(0.1*321)=33 on the pooled target sample; `top10 per-block`: k=ceil(0.1*n_block) per block aggregated as sum(overlap)/sum(k). They have different denominators and are not interchangeable.

## 3. Paired block MAE difference vs S1A

| Model | mean ΔMAE | median ΔMAE | q05 | q95 | blocks better | blocks |
|---|---:|---:|---:|---:|---:|---:|
| affine_global | -0.0945 | -0.0746 | -0.3353 | 0.1577 | 41 | 52 |
| affine_target | -0.0920 | -0.0686 | -0.3162 | 0.1577 | 41 | 52 |
| catboost_reference | -0.1000 | -0.0927 | -0.3333 | 0.1099 | 41 | 52 |
| global_ridge_geometry | -0.0881 | -0.0649 | -0.3873 | 0.1224 | 36 | 52 |
| global_ridge_local | -0.0877 | -0.0672 | -0.3562 | 0.0934 | 40 | 52 |
| target_ridge_geometry | -0.0839 | -0.0582 | -0.3760 | 0.1430 | 36 | 52 |
| train_median | -0.0912 | -0.0765 | -0.5558 | 0.1630 | 33 | 52 |

## 4. Errors by baseline thickness bins

| Model | Bin | n | MAE | bias |
|---|---|---:|---:|---:|
| s1a | [0.0, 4.0) | 1 | 2.4000 | -2.4000 |
| s1a | [4.0, 5.0) | 5 | 1.2020 | -0.5300 |
| s1a | [5.0, 6.0) | 119 | 0.5401 | -0.2071 |
| s1a | [6.0, 7.0) | 172 | 0.4157 | 0.1830 |
| s1a | [7.0, 8.0) | 24 | 0.7713 | 0.7713 |
| train_median | [0.0, 4.0) | 1 | 0.0000 | 0.0000 |
| train_median | [4.0, 5.0) | 5 | 1.2640 | 0.8800 |
| train_median | [5.0, 6.0) | 119 | 0.5378 | 0.3240 |
| train_median | [6.0, 7.0) | 172 | 0.3696 | -0.0611 |
| train_median | [7.0, 8.0) | 24 | 0.2800 | -0.1733 |
| affine_global | [0.0, 4.0) | 1 | 0.9673 | -0.9673 |
| affine_global | [4.0, 5.0) | 5 | 1.2141 | 0.3636 |
| affine_global | [5.0, 6.0) | 119 | 0.4885 | 0.0751 |
| affine_global | [6.0, 7.0) | 172 | 0.3741 | -0.0437 |
| affine_global | [7.0, 8.0) | 24 | 0.2804 | 0.0884 |
| affine_target | [0.0, 4.0) | 1 | 1.0359 | -1.0359 |
| affine_target | [4.0, 5.0) | 5 | 1.2110 | 0.3262 |
| affine_target | [5.0, 6.0) | 119 | 0.4900 | 0.0536 |
| affine_target | [6.0, 7.0) | 172 | 0.3763 | -0.0492 |
| affine_target | [7.0, 8.0) | 24 | 0.2829 | 0.0976 |
| global_ridge_geometry | [0.0, 4.0) | 1 | 0.9506 | -0.9506 |
| global_ridge_geometry | [4.0, 5.0) | 5 | 1.0460 | 0.3320 |
| global_ridge_geometry | [5.0, 6.0) | 119 | 0.4761 | 0.0462 |
| global_ridge_geometry | [6.0, 7.0) | 172 | 0.3694 | -0.0313 |
| global_ridge_geometry | [7.0, 8.0) | 24 | 0.3348 | 0.0751 |
| target_ridge_geometry | [0.0, 4.0) | 1 | 1.0034 | -1.0034 |
| target_ridge_geometry | [4.0, 5.0) | 5 | 1.0391 | 0.2750 |
| target_ridge_geometry | [5.0, 6.0) | 119 | 0.4805 | 0.0283 |
| target_ridge_geometry | [6.0, 7.0) | 172 | 0.3710 | -0.0292 |
| target_ridge_geometry | [7.0, 8.0) | 24 | 0.3363 | 0.0899 |
| global_ridge_local | [0.0, 4.0) | 1 | 0.9633 | -0.9633 |
| global_ridge_local | [4.0, 5.0) | 5 | 1.0610 | 0.3000 |
| global_ridge_local | [5.0, 6.0) | 119 | 0.4717 | 0.0443 |
| global_ridge_local | [6.0, 7.0) | 172 | 0.3659 | -0.0284 |
| global_ridge_local | [7.0, 8.0) | 24 | 0.3500 | 0.0672 |
| catboost_reference | [0.0, 4.0) | 1 | 2.1159 | -2.1159 |
| catboost_reference | [4.0, 5.0) | 5 | 1.1434 | 0.1568 |
| catboost_reference | [5.0, 6.0) | 119 | 0.4960 | -0.0195 |
| catboost_reference | [6.0, 7.0) | 172 | 0.3493 | 0.0071 |
| catboost_reference | [7.0, 8.0) | 24 | 0.3088 | 0.1548 |

## 5. Within-block centered correlation

- `s1a`: pooled r=0.2467, r²=0.0609, defined blocks=51, undefined/small=1.
- `train_median`: pooled r=n/a, r²=n/a, defined blocks=0, undefined/small=52.
- `affine_global`: pooled r=0.2222, r²=0.0494, defined blocks=51, undefined/small=1.
- `affine_target`: pooled r=0.2218, r²=0.0492, defined blocks=51, undefined/small=1.
- `global_ridge_geometry`: pooled r=0.2491, r²=0.0621, defined blocks=51, undefined/small=1.
- `target_ridge_geometry`: pooled r=0.2521, r²=0.0636, defined blocks=51, undefined/small=1.
- `global_ridge_local`: pooled r=0.2500, r²=0.0625, defined blocks=51, undefined/small=1.
- `catboost_reference`: pooled r=0.1978, r²=0.0391, defined blocks=51, undefined/small=1.

## 6. Прямые ответы

1. **Ridge vs affine:** global Ridge MAE 0.4187, affine_global MAE 0.4245; Pearson r² 0.1625 против 0.0885; within-block r² 0.0621 против 0.0494; top10 global 0.4545 против 0.3030. Да, Ridge превосходит affine по MAE/r²/top10 в этом 100m diagnostic; CatBoost имеет чуть меньший MAE (0.4185), но худшие r²/top10 и lower-tail.
2. **Local vs global:** target Ridge MAE 0.4214, global Ridge 0.4187; Pearson r² 0.1614 против 0.1625; within-block r² 0.0636 против 0.0621. Local fit не даёт убедительного преимущества: global чуть лучше по MAE/r², local — на тысячные по within-block r²; это exploratory, не выбор.
3. **Level vs form:** Ridge снижает MAE с 0.5068 до 0.4187, Pearson r² с 0.1223 до 0.1625, top10 global с 0.3030 до 0.4545. Но lower-tail global MAE ухудшается (0.8923 → 1.0090), а within-block r² растёт лишь с 0.0609 до 0.0621. Следовательно, подтверждён в основном уровень/общий top-decile, а не устойчивая локальная пространственная форма.

## 7. Archive comparison

`archive_comparison.json`: {"rows_both": 321, "rows_only_current": 0, "rows_only_archive": 0, "s1a_max_abs_diff": 0.0, "global_ridge_max_abs_diff": 0.07982826600551984, "catboost_max_abs_diff": 0.10080499517674735, "block_mapping_equal": true, "old_block_definition": "segment_id // 10 (archive)", "current_block_definition": "center_m // 100 (physical)", "note": "Archive used only for protocol comparison; no selection."}
Старый archive block-CV использовал segment_id // 10; текущий — physical center. Совпадение проверено только на общих ключах; старые прогнозы не использовались для выбора.

## 8. Artifacts and commands

```text
experiments/targeted_cycle/stage_3_s1_100m_20m/
  fold_assignments.json, fold_training_keys.csv, fold_fit_metadata.json
  oof_predictions.csv, metrics_overall.csv, metrics_per_block.csv
  paired_block_mae.csv, paired_block_mae_summary.json
  baseline_thickness_bins.csv, within_block_correlation.json, tail_metrics.json
  models/{global_ridge_geometry,target_ridge_geometry,catboost_reference,affine_global,affine_target,train_median}
  target_predictions/<model>/predictions_s1.csv
```

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s1 train --root . --data-root C:\Users\Администратор\Documents\PipeTwin_next\experiments\next_cycle\data_generation --output-root experiments/targeted_cycle/stage_3_s1_100m_20m
```

## 9. Tests

- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_3_s1 -v`: 6 tests, OK.
- OOF keys: 321 unique target keys; all model columns finite; fold train/test disjoint; physical block bounds and buffer checked.

## 10. Limitations

- 100 m/20 m is the primary scheme; stability on 200/50 and 500/100 is Stage 4 and not yet claimed.
- Same-pipe block CV is spatial interpolation, not a forward-time forecast.
- Historical campaign is one campaign; target production has a different deadline/horizon.
- Affine preserves order before clipping; clipping can change ranking; clipping counts are saved.
- Target labels are unavailable; no target accuracy is claimed.

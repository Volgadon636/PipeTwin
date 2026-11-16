# Stage 2 — full-grid S2 thermal contract and full-grid centering

Generated: `2026-11-15T20:00:30+00:00`. No retrospective or target labels were used for fit/selection.

## 1. Что исправлено

- Тепловой сценарий теперь строится по полной физической сетке каждого `id/cutoff/case` до присоединения labels; measured rows получают значения по строгим ключам.
- Добавлены `t_in_segment_c`, `t_out_segment_c`, `local_delta_t_c`, `cumulative_cooling_from_inlet_c`; `scenario_cooling_c` сохранён как прежний тип локального cooling-признака.
- S2 spatial correction: `h_raw = h_physics + lambda*(g(X) - mean_full_grid(g(X)))`, `lambda=1`; центрирование по длине сегмента на полной сетке, без labels. Арифметическое среднее также сохранено.
- Параметры тепла не подбирались; CatBoost не переобучался; retrospective не использовалась для параметров или выбора.

## 2. Thermal before/after

- Измеренных строк сравнено: **466**.
- max |Δ t_out| на measured keys: **0.539957 °C**.
- max |Δ local cooling|: **0.000450331 °C**.
- mean |Δ local cooling|: **1.95919e-05 °C**.
- Старое поведение строило сценарий по разреженным measured rows; новое — по полной сетке. Это contract fix, не доказательство роста accuracy.

Thermal by campaign:

| id | date | rows | max abs Δt_out | max abs Δcooling | valid |
|---|---|---:|---:|---:|---:|
| 1250009490 | 2018-09-23 | 4 | 0.00089073 | 0.00000007 | 4 |
| 1250009584 | 2023-08-12 | 19 | 0.53995721 | 0.00045033 | 19 |
| 1250009587 | 2023-08-12 | 3 | 0.00412289 | 0.00000313 | 3 |
| 1250011604 | 2018-09-23 | 34 | 0.01537464 | 0.00000228 | 34 |
| 1250022871 | 2018-09-23 | 334 | 0.21958276 | 0.00003483 | 334 |
| 1250028931 | 2023-08-12 | 4 | n/a | n/a | 0 |
| 1250028932 | 2023-08-12 | 9 | n/a | n/a | 0 |
| 1250028933 | 2023-08-12 | 1 | n/a | n/a | 0 |
| 1250028934 | 2023-08-12 | 31 | n/a | n/a | 0 |
| 1250028935 | 2023-08-12 | 27 | n/a | n/a | 0 |

Для 1250022871 2018 max Δt_out ≈ 0.2196 °C и max Δlocal cooling ≈ 3.48e-5 °C, как в Stage 1/аудите. Пять 2023-кампаний имеют `missing_inlet_temperature` и сохраняются как invalid без подстановки; target full grid valid.

## 3. Thermal contract tests

| Тест | Результат |
|---|---|
| row_order_invariance | True |
| label_subset_invariance | True |
| full_grid_to_subset | True |
| gap_detected | True |
| overlap_detected | True |
| analytic_exponential | True |
| local_sum_equals_total | True |
| zero_flow_invalid | True |

## 4. S2 strict OOF на одинаковых 372 ключах

| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | lower-tail MAE | top10 recall | clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| physics_priors | 0.4544 | 0.5689 | -0.1129 | -0.0326 | 0.0011 | -0.0640 | n/a | 0.8493 | 0.1538 | nan |
| old_physics_spatial | 0.4482 | 0.5649 | -0.1100 | 0.0617 | 0.0038 | -0.0490 | 0.0124 | 0.8358 | 0.1282 | nan |
| spatial_no_temperature_uncentered | 0.4482 | 0.5649 | -0.1100 | 0.0617 | 0.0038 | -0.0490 | 0.0124 | 0.8358 | 0.1282 | 0.0 |
| spatial_no_temperature_centered | 0.4495 | 0.5665 | -0.1178 | 0.0618 | 0.0038 | -0.0550 | 0.0124 | 0.8278 | 0.1282 | 0.0 |
| spatial_centered_corrected_temperature | 0.4498 | 0.5668 | -0.1180 | 0.0601 | 0.0036 | -0.0561 | 0.0118 | 0.8272 | 0.1538 | 0.0 |

Centered no-temperature меняет MAE незначительно (0.4495 против 0.4482), signed r почти не меняется (0.06179 против 0.06165); length-weighted mean `g_centered` ≈ 1e-17, arithmetic mean ≈ -3.9e-5. Это contract fix уровня, не доказательство улучшения spatial ranking. Temperature-centered MAE 0.4498, within-section r² 0.0118; также не объявляется улучшением.

## 5. Target predictions: raw / clipped / centering

| Candidate | mean | std | min | max | clipping | mean shift after clip |
|---|---:|---:|---:|---:|---:|---:|
| physics_priors | 2.8095 | 0.0000 | 2.8095 | 2.8095 | 0 | 0.0000 |
| spatial_no_temperature_uncentered | 2.8143 | 0.1461 | 2.0086 | 3.7936 | 0 | 0.0000 |
| spatial_no_temperature_centered | 2.8095 | 0.1461 | 2.0038 | 3.7888 | 0 | 0.0000 |
| spatial_centered_corrected_temperature | 2.8096 | 0.1195 | 1.9522 | 3.7771 | 0 | 0.0000 |

Preserved references (no refit):

| Reference | mean | std | rows |
|---|---:|---:|---:|
| old_physics_spatial | 2.8143 | 0.1461 | 1475 |
| old_physics_spatial_temperature | 2.5907 | 0.1236 | 1475 |
| physics_priors | 2.8095 | 0.0000 | 1475 |
| v2_reference | 2.8095 | 0.0715 | 1475 |

Target labels отсутствуют; MAE/корреляция на target не вычислялись. Старый uncentered temperature-spatial имел mean ≈2.5907, новый centered — ≈2.80956: это устранение level shift, не улучшение spatial accuracy.

## 6. Frozen 2023 retrospective (не selection)

| Модель | MAE | bias | Pearson r | Pearson r² | R² |
|---|---:|---:|---:|---:|---:|
| physics_priors | 0.2905 | 0.2905 | -0.0068 | 0.0000 | -0.7160 |
| spatial_no_temperature_uncentered | 0.3188 | 0.3152 | 0.2020 | 0.0408 | -0.8125 |
| spatial_no_temperature_centered | 0.3146 | 0.3103 | 0.2021 | 0.0408 | -0.7870 |
| spatial_centered_corrected_temperature | 0.3141 | 0.3100 | 0.2007 | 0.0403 | -0.7856 |

Retrospective уже просмотрена командой; она не является untouched holdout и не использовалась для выбора.

## 7. Train/target conditions

| Variable | strict mean | target mean |
|---|---:|---:|
| nominal_thickness_mm | 8.0000 | 4.5000 |
| age_years | 18.4203 | 29.0294 |
| t_mix_mean | 12.3847 | 6.1986 |
| watercut_pct_mean | 86.7868 | 0.0554 |
| v_mix_mean | 0.5760 | 0.0647 |
| coverage_fraction | 0.1753 | 0.1330 |
| scenario_cooling_c | 0.0012 | 0.0088 |
| cumulative_cooling_from_inlet_c | 0.2544 | 8.2422 |

## 8. Artifacts and commands

```text
experiments/targeted_cycle/stage_2_s2_fixes/
  thermal_contract/{grid_checks,before_after_features,before_after_summary,tests}.json/csv
  oof_predictions.csv, metrics_strict.csv, metrics_strict_per_section.csv
  models/<candidate>/{model_artifact.joblib,model_metadata.json,predictions_s2.csv,result.csv}
  post_cutoff_retrospective/{predictions,metrics}.json
  target_candidates_summary.json, domain_shift.json, run_manifest.json
```
Источник данных задаётся явно через `--data-root`; тесты требуют `PIPETWIN_NEXT_DATA_ROOT` и не пропускаются молча. Для запуска использован LF-worktree той же проверенной генерации, потому что основной checkout имеет CRLF.

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s2 train --root . --data-root C:/Users/Администратор/Documents/PipeTwin_next/experiments/next_cycle/data_generation --output-root experiments/targeted_cycle/stage_2_s2_fixes
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s2 predict --root . --data-root C:/Users/Администратор/Documents/PipeTwin_next/experiments/next_cycle/data_generation --output-root experiments/targeted_cycle/stage_2_predict --candidate spatial_no_temperature_centered
```

## 9. Тесты

- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_2_s2_fixes -v`: 9 тестов, OK (label invariance, row order, full→subset, gap/overlap, analytic, local sum, zero flow, zero mean, full/partial inference, save/load predict, metadata/bounds).
- OOF/target/result checks: 372 strict keys, 1475 target keys, 2000 result keys, S1B фиксирован с max abs diff 0.0.

## 10. Ограничения и решения

- Полная сетка всё ещё использует steady-state сценарий от средних входных условий; это не измеренная 29-летняя история.
- Центрирование устраняет сдвиг уровня, но не доказывает улучшение Pearson/ранжирования.
- Strict train: один parent/одна кампания; независимый forward fold недоступен.
- 2023 retrospective — использованная диагностика, не holdout.
- Требуют разбора: принимать ли centered spatial как challenger, и какой thermal feature (local vs cumulative) использовать в будущем; сейчас основной before/after использует прежний local cooling.

# S2 next cycle: strict cold-start baselines and challengers

Strict train: только cold-start метки строго раньше `2022-07-01`. Кампании 2023-08-12 используются только в замороженной retrospective-диагностике, не в fit / feature selection / gate / выборе модели.

## Eligibility и domain shift

- strict rows: **372**; три простых участка одного parent `MAIN-2`, одна кампания `2018-09-23`.
- strict: nominal 8 мм, age ≈18.42 года; target: nominal 4.5 мм, age ≈29.03 года.
- target T/W/v: cold/dry/slow; strict: warm/wet/faster. Это сильный domain shift.
- bounded exposure строгого train покрывает ≈17.5% истории; target ≈13.3%.
- Все spatial/temperature transforms обучаются внутри fold только на train.

## Strict OOF (leave-one-simple-section-out)

| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | pred std | top-10% recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_effective_rate | 0.4398 | 0.5540 | -0.0443 | -0.0302 | 0.0009 | -0.0090 | n/a | 0.0160 | 0.1538 |
| physics_priors | 0.4544 | 0.5689 | -0.1129 | -0.0326 | 0.0011 | -0.0640 | n/a | 0.0659 | 0.1538 |
| physics_velocity_consistent | 0.4544 | 0.5689 | -0.1129 | -0.0326 | 0.0011 | -0.0640 | n/a | 0.0659 | 0.1538 |
| physics_spatial | 0.4482 | 0.5649 | -0.1100 | 0.0617 | 0.0038 | -0.0490 | 0.0124 | 0.0972 | 0.1282 |
| physics_spatial_temperature | 0.4521 | 0.5690 | -0.1186 | 0.0513 | 0.0026 | -0.0641 | 0.0117 | 0.1074 | 0.1282 |
| catboost_residual | 0.4427 | 0.5586 | -0.0516 | 0.0342 | 0.0012 | -0.0258 | 0.0040 | 0.0932 | 0.1538 |

Macro по участкам:

| Модель | macro MAE | macro Pearson r² | macro R² |
|---|---:|---:|---:|
| constant_effective_rate | 0.3582 | n/a | -0.1593 |
| physics_priors | 0.3870 | n/a | -0.4758 |
| physics_spatial_temperature | 0.4047 | 0.2840 | -0.8029 |
| catboost_residual | 0.3942 | 0.2509 | -0.6778 |

## Temperature / velocity comparison

- `physics_priors` использует существующую v2-семантику: reference/`v_mix_mean`, интервалы exposure — `v_liquid_true`. Это несогласованно.
- `physics_velocity_consistent` использует `v_liquid_true` и в reference, и в exposure.
- `physics_spatial_temperature` добавляет scenario cooling `-ΔT` из `heat_transfer.py` на тех же folds; это scenario proxy от средних условий, не 29-летний измеренный профиль.

| Сравнение | MAE | within-section r² |
|---|---:|---:|
| physics_priors | 0.4544 | n/a |
| physics_velocity_consistent | 0.4544 | n/a |
| physics_spatial | 0.4482 | 0.0124 |
| physics_spatial_temperature | 0.4521 | 0.0117 |

`physics_velocity_consistent` после refit r_ref численно совпал с physics: нормировка компенсируется, поэтому это не независимое доказательство пользы.
`physics_spatial_temperature` сдвигает target mean до 2.5907 мм; на strict OOF и 2023 retrospective MAE не улучшается (0.4521 / 0.3178 против 0.4544 / 0.2905 у physics).

## Frozen post-cutoff retrospective

Модели обучены только на strict 2018 и предсказывают 2023-08-12 кампании других труб. Это проверка cross-environment переноса, не выбор модели.

| Модель | n | MAE | bias | within-section r² |
|---|---:|---:|---:|---:|
| constant_effective_rate | 94 | 0.4288 | -0.3294 | n/a |
| physics_priors | 94 | 0.2905 | 0.2905 | n/a |
| physics_velocity_consistent | 94 | 0.2905 | 0.2905 | n/a |
| physics_spatial | 94 | 0.3188 | 0.3152 | 0.0295 |
| physics_spatial_temperature | 94 | 0.3178 | 0.3137 | 0.0295 |
| catboost_residual | 94 | 0.2913 | 0.2861 | 0.0318 |

## Production S2 candidates (no target labels)

| Candidate | mean | std | min | max | finite | within [0,4.5] |
|---|---:|---:|---:|---:|---|---|
| constant_effective_rate | 1.5396 | 0.0000 | 1.5396 | 1.5396 | True | True |
| physics_priors | 2.8095 | 0.0000 | 2.8095 | 2.8095 | True | True |
| physics_velocity_consistent | 2.8095 | 0.0000 | 2.8095 | 2.8095 | True | True |
| physics_spatial | 2.8143 | 0.1461 | 2.0086 | 3.7936 | True | True |
| physics_spatial_temperature | 2.5907 | 0.1236 | 1.7247 | 3.5565 | True | True |
| catboost_residual | 2.8268 | 0.0733 | 2.6491 | 3.0386 | True | True |

v2 reference: mean ≈2.8095 мм, std ≈0.0715 мм; его старый OOF использовал 2023 кампании и не является strict baseline.

## Selection

default = **physics_priors**; level_winner = **constant_effective_rate**; spatial_winner = **physics_spatial**.
physics_priors is the predeclared strict level model; constant/velocity/spatial results do not establish a transferable improvement on the target domain.
Никакой S2 candidate не объявляется доказанным улучшением на target: target labels отсутствуют.

## Files

- `training_table.csv`, `eligibility.json`, `fold_assignments.json`
- `oof_predictions.csv`, `metrics_per_fold.csv`, `metrics_pooled.json`, `domain_shift.json`
- `post_cutoff_retrospective/` — frozen 2023 diagnostic
- `candidates/<name>/predictions_s2.csv`, `prediction_diagnostics.csv`, `result.csv`, `model_artifact.joblib`
- `figures/`

```bash
python -m src.ml.next_s2 --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s2
```

# PipeTwin next cycle: S1 diagnostics and S2 strict candidates

Generated: `2026-11-15T17:35:06+00:00`. Branch/commit — в `experiment_registry.json` и manifests.

Это exploratory-цикл. Ни один кандидат не объявляется доказанным production winner; `experiments/ml_v2/result.csv` не перезаписан. Push/deploy не выполнялись.

## Данные и воспроизводимость

- Согласованная генерация: `C:\Users\Администратор\Documents\PipeTwin_next\experiments\next_cycle\data_generation`.
- S1: 525 ключей, `1250022871`, `2023-12-01`; S2: 1475, `1250009583`, `2022-07-01`.
- S1ABC reproduction: max abs diff result.csv = `0.0`; artifact hash `e998606bc75b97ff9b88360b0d6a11a10dfefa466090abfa7ca3afbe61ae83cf`.
- Старый `target_segments_grid.csv` был отвергнут по checksum (лишний `delta_t_c`); новая генерация manifest-verified и используется для всех новых сравнений.

## S1: что проверено и что изменилось

Основная spatial-цель: within-section Pearson / Pearson², не только MAE. LOPO по трём простым участкам одного parent — spatial-transfer диагностика. Строгих forward folds нет: у каждого допустимого S1-ответа только одна прошлая кампания.

| S1 model | MAE | RMSE | bias | Pearson r | Pearson r² | ordinary R² | within-section r² | top-10% recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s1a | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 | 0.1133 | 0.2895 |
| s1b | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 | 0.1133 | 0.2895 |
| old_s1c | 0.4698 | 0.6180 | -0.0016 | 0.0913 | 0.0083 | -0.1473 | 0.0097 | 0.1053 |
| ridge_history | 0.4454 | 0.5966 | 0.0622 | 0.1382 | 0.0191 | -0.0692 | 0.0199 | 0.1316 |
| ridge_geometry | 0.4511 | 0.6034 | 0.0564 | 0.1042 | 0.0109 | -0.0938 | 0.0119 | 0.0789 |
| rate_signed | 0.4511 | 0.6034 | 0.0564 | 0.1042 | 0.0109 | -0.0938 | 0.0119 | 0.0789 |
| rate_nonnegative | 0.4580 | 0.5993 | -0.1160 | 0.3191 | 0.1018 | -0.0791 | 0.1016 | 0.2895 |
| catboost | 0.4449 | 0.6101 | 0.1274 | 0.1839 | 0.0338 | -0.1183 | 0.0355 | 0.1842 |
| mix_025 | 0.4658 | 0.6131 | 0.0511 | 0.3179 | 0.1010 | -0.1291 | 0.1020 | 0.2895 |
| mix_050 | 0.4426 | 0.5909 | 0.0529 | 0.2859 | 0.0817 | -0.0487 | 0.0830 | 0.2632 |
| mix_100 | 0.4511 | 0.6034 | 0.0564 | 0.1042 | 0.0109 | -0.0938 | 0.0119 | 0.0789 |
| affine | 0.4171 | 0.5711 | 0.0762 | 0.2050 | 0.0420 | 0.0202 | 0.0612 | 0.2895 |
| train_median | 0.4282 | 0.5979 | 0.1507 | -0.0483 | 0.0023 | -0.0740 | n/a | 0.0789 |

S1 default: **s1a**. no candidate improved both within-section spatial signal and MAE; S1A remains default. best_spatial=`mix_025`, best_mae=`mix_050`.

Почему старый S1C терял сигнал (подтверждено этим циклом):

1. `s1b_prediction` и `last_vtd_thickness_mm` были идентичны во всём fit; Ridge делил эффект между дублями, а в production эти величины уже различались. Новая core-модель использует один last baseline.
2. Горизонт в train был константным (3.7837 года), поэтому старый learner не мог масштабировать поправку по времени. Rate-модель вынесена отдельно; при константном dt signed-rate OOF алгебраически совпал с delta-моделью, а production различается из-за фактического dt 1.4045/5.1882.
3. `last_minus_min` и campaign count были константны; их эффект не идентифицируется.
4. Равный вес простых участков давал Kish ESS ≈31.65; row-uniform/inverse-sqrt проверены как ablation. Ни одна схема не восстановила within-section сигнал S1A.
5. Block-CV на целевой трубе (100 м + 20 м buffer) показывает потенциальный same-pipe эффект: within-section r² у ridge_geometry/rate_signed выше, чем у S1A; но LOPO между трубами этого не подтверждает. Это spatial interpolation, не будущий прогноз.

All-pipeline 561 (fallback = S1B): S1A MAE `0.5305`, old S1C `0.5124`, ridge_geometry `0.5005`, catboost `0.4965`; within-section сигнал при этом падает, поэтому S1A оставлен default.

## S2: strict cold-start

Strict train: только `2018-09-23` (372 строки, 3 простых участка, один parent). 2023-08-12 кампании — только frozen retrospective, не fit/selection/gate.

| S2 model | MAE | RMSE | bias | Pearson r | Pearson r² | R² | within-section r² | pred std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_effective_rate | 0.4398 | 0.5540 | -0.0443 | -0.0302 | 0.0009 | -0.0090 | n/a | 0.0160 |
| physics_priors | 0.4544 | 0.5689 | -0.1129 | -0.0326 | 0.0011 | -0.0640 | n/a | 0.0659 |
| physics_velocity_consistent | 0.4544 | 0.5689 | -0.1129 | -0.0326 | 0.0011 | -0.0640 | n/a | 0.0659 |
| physics_spatial | 0.4482 | 0.5649 | -0.1100 | 0.0617 | 0.0038 | -0.0490 | 0.0124 | 0.0972 |
| physics_spatial_temperature | 0.4521 | 0.5690 | -0.1186 | 0.0513 | 0.0026 | -0.0641 | 0.0117 | 0.1074 |
| catboost_residual | 0.4427 | 0.5586 | -0.0516 | 0.0342 | 0.0012 | -0.0258 | 0.0040 | 0.0932 |

S2 default: **physics_priors**; level_winner=`constant_effective_rate`, spatial_winner=`physics_spatial`.
Уровень: physics воспроизводит v2 production mean 2.809539; constant даёт 1.539584. Пространственный рисунок: только spatial candidates дают ненулевой std, но strict OOF within-section r² мал (0.004–0.012), а frozen 2023 retrospective ухудшает MAE по сравнению с physics-only. Поэтому spatial correction не объявлена улучшением.

Temperature: `physics_spatial_temperature` добавляет scenario cooling из `heat_transfer.py` и сдвигает target mean до 2.5907; на strict OOF и 2023 retrospective это не улучшает MAE. Velocity-consistent variant после refit r_ref численно совпал с physics (компенсация), поэтому сам по себе не доказывает пользу.

Frozen post-cutoff 2023 retrospective (не selection): physics MAE `0.2905`, bias `0.2905`; spatial MAE `0.3188`; catboost MAE `0.2913`. Physics level переносится лучше constant, но систематически смещён.

## Explicit assembly

`s1a` + `physics_priors`: `experiments/next_cycle/assembled/s1a__physics_priors/result.csv`. S1 и S2 половинки проверены на точное сохранение; 2000 уникальных ключей. Это candidate pair, не «лучший» result.

## Artifacts

- `experiments/next_cycle/audit/data_audit.json`
- `experiments/next_cycle/s1/` — OOF, folds, weight ablation, block CV, candidates, figures, report
- `experiments/next_cycle/s2/` — strict OOF, eligibility, domain shift, retrospective, candidates, figures, report
- `experiments/next_cycle/selected_candidates.json` (aggregate)
- `experiments/next_cycle/s1/selected_candidates.json`, `s2/selected_candidates.json`
- `experiments/next_cycle/experiment_registry.json`

## Limitations

- Three S1 simple sections and one S2 parent: no independent forward-time validation.
- S1 production S1B uses two campaigns while training has one; S1C transfer remains unverified.
- S2 target has no labels; target accuracy is not computed. Domain shift (4.5 mm / 29 yr / cold-dry) is severe.
- Temperature scenario is a steady-state proxy from mean inputs, not a measured 29-year profile.
- Post-cutoff 2023 labels are used only for the frozen retrospective; no strict-model selection.

## Commands

```bash
python -m src.data_prep --output-root experiments/next_cycle/data_generation
python -m src.ml.next_s1 train --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s1
python -m src.ml.next_s2 train --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s2
python -m src.ml.next_compare --root . --output-root experiments/next_cycle --s1-candidate s1a --s2-candidate physics_priors
```

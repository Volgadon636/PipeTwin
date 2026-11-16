# S1A / S1B / S1C: as-of baselines и Ridge-поправка

Статус: экспериментальная реализация S1. S2 не переобучался и не менялся.
Изменения находятся в локальной ветке `s1abc-ridge` (от `9af2fbc`, tree =
`model_v2` / `15155a3`), отдельный worktree:
`C:\Users\Администратор\Documents\PipeTwin_s1abc`.

## 1. Контракт данных

Для каждого примера с датой ответа `T` признаки строятся только из
`processed/vtd_by_segment` с `date < T`, `target_eligible=True` и непустым
`min_residual_thickness_mm`. Историческое обучение S1 использует только метки
`T < 2023-12-01`; в доступных данных это одна кампания `2022-07-06`.
Измерения на/после cutoff и скрытые ответы не участвуют в fit, выборе признаков,
hyperparameters или post-processing.

Общий fallback для S1A, S1B и S1C:

1. Если у сегмента есть собственный прошлый ВТД — используется он.
2. Иначе берётся последний прошлый ВТД ближайшего измеренного сегмента **той же
   трубы**; при точном равенстве расстояния — меньший `segment_id`.
3. Если у трубы вообще нет прошлых ВТД — используется
   `nominal_at_commissioning` с отдельным флагом.

Подставленное значение не считается собственным измерением. В диагностике
сохраняются `fallback_segment_id`, `fallback_distance_m`,
`fallback_measurement_date`, `baseline_source`, `s1a_missing`, `s1b_missing`.

### S1A

`prediction_S1A = last eligible own prior VTD thickness`, иначе общий fallback.
Это правило, не модель: `training_required=false`.

### S1B

`prediction_S1B = min(eligible measured thickness(k,t) for t < T)` по всем
прошлым кампаниям. При точном равенстве минимума выбирается самая ранняя дата
источника. Если собственного прошлого замера нет — значение S1A fallback
**без дополнительного изменения правила**. Дополнительная скорость не
вычитается. `training_required=false`.

### S1C

```text
b(k,T) = S1B(k,T)
target_delta_mm = measured_thickness(k,T) - b(k,T)
prediction_S1C = clip(b(k,T) + ridge.predict(X(k,T)), 0, nominal_thickness_mm)
```

Поправка может иметь оба знака; она не ограничена сверху значением S1B и не
является физической скоростью коррозии. Bounds policy S1C — `[0, nominal]`;
прежняя policy `[0, h0]` сохранена для S1A/S1B и остальных моделей.

## 2. Признаки S1C

Объявленный порядок (`feature_contract.json`):

1. `s1b_prediction`
2. `last_vtd_thickness_mm`
3. `age_last_vtd_years`
4. `age_min_vtd_years`
5. `prior_vtd_campaign_count`
6. `last_minus_min_mm`
7. `neighbor_prior_median_mm`
8. `neighbor_prior_count`
9. `neighbor_missing`
10. `slope_m_per_m`
11. `abs_slope_m_per_m`
12. `relative_position`
13. `local_low_m`

Соседние агрегаты строятся только по последним прошлым замерам сегментов той
же трубы в радиусе 20 м между центрами; собственный сегмент исключён.
`local_low_m` — отклонение `z_center` от 21-сегментного rolling median.

Train-only preprocessing: median imputation, удаление всех пустых и константных
колонок, StandardScaler. Всё обучается только на train fold. На полном fit
константными оказались `age_last_vtd_years`, `age_min_vtd_years`,
`last_minus_min_mm`, `prior_vtd_campaign_count`; они исключены, а не заменены
фиктивными значениями. `S1B` и последний замер при одной прошлой кампании
совпадают, поэтому их коэффициенты не идентифицируются раздельно. Это
ограничение зафиксировано в `feature_contract.json`.

Фиксированные параметры: `alpha=10.0`, `fit_intercept=True`,
`solver='cholesky'`, равный суммарный вес каждого простого участка с
нормировкой среднего веса к 1. Hyperparameter search не выполнялся.

## 3. Проверка качества

Основная проверка — leave-one-simple-section-out на 357 исторических строках
(3 простых участка, одна кампания `2022-07-06`, parent `MAIN-2`). В каждом fold
обучение исключает весь проверяемый простой участок; preprocessing и модель
обучаются только на train fold. Это **spatial-transfer diagnostic**, не
forward-time прогноз. Строгих forward-time folds до cutoff нет: у каждого
допустимого S1-ответа только одна прошлая кампания.

S1A и S1B на этих 357 строках совпадают по построению: их различает только
вторая прошлая кампания, а её в обучении нет. S1C и контроль
`S1A_ridge_control` также совпали, потому что при одной кампании S1B = last.
Поэтому historical test не различает правило S1B и S1A; quality S1B в
двухкампанейском production-режиме помечено как
`not_verified_in_two_prior_campaign_regime`.

| Модель | Состояние | n | MAE, мм | RMSE, мм | bias, мм | Pearson r | Pearson r² | R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1A | rule, no training | 357 | 0.4982 | 0.6523 | +0.0493 | 0.3355 | 0.1125 | −0.2780 |
| S1B | rule, no training | 357 | 0.4982 | 0.6523 | +0.0493 | 0.3355 | 0.1125 | −0.2780 |
| S1C | fit complete, OOF | 357 | 0.4698 | 0.6180 | −0.0016 | 0.0913 | 0.0083 | −0.1473 |
| S1A_ridge_control | control OOF | 357 | 0.4698 | 0.6180 | −0.0016 | 0.0913 | 0.0083 | −0.1473 |

S1C снизил MAE, но Pearson r² упал, а R² остался отрицательным. Это **не
доказанное улучшение**: поправка в основном сжимает уровень, а не
восстанавливает ранжирование. Артефакт сохранён, S1C не объявляется
победителем. Внешние MAE 0.551 / r² 0.221 не использовались.

Macro MAE по простым участкам: S1A/S1B 0.4095 мм, S1C 0.4347 мм; pooled MAE:
0.4982 → 0.4698 мм. Macro- и per-section метрики: `model_comparison_per_section.csv`,
`metrics_pooled.json`, `metrics_per_fold.csv`.

## 4. Production S1

`1250022871`, cutoff `2023-12-01`, 525 ключей:

- S1A: 521 собственный прошлый ВТД, 4 fallback; mean 6.2250 мм.
- S1B: 157 сегментов изменились относительно S1A; mean Δ = −0.135619 мм;
  S1B ≤ S1A на всех сегментах с собственным замером.
- Fallback: segment 17 → neighbor 16 (2018-09-23); 247 → 246; 401 → 400;
  455 → 454 (2022-07-06), расстояние 10 м.
- S1C: Ridge применён к 521 сегменту, 4 fallback; clipping изменил 0 значений;
  mean 6.0261 мм; Δ(S1C−S1B) mean −0.0633 мм, 199 положительных и 322
  отрицательных поправки; 149 прогнозов выше h0, но 0 выше nominal.

S2 `1250009583`, 1475 строк, скопирован без изменений из
`experiments/ml_v2/result.csv`; max |diff| после записи = 0.0. Общий
`result.csv` — 2000 строк.

## 5. Артефакты и команды

```bash
# отдельное окружение Python 3.12
python -m venv .venv-s1
.venv-s1/Scripts/pip install -r requirements-s1.txt

python -m src.ml.s1a --root . --output-root experiments/s1abc/s1a
python -m src.ml.s1b --root . --output-root experiments/s1abc/s1b
python -m src.ml.s1c train --root . --output-root experiments/s1abc/s1c --alpha 10.0
python -m src.ml.s1c predict --root . --output-root experiments/s1abc/s1c
python -m src.ml.s1_metrics --root . --output-root experiments/s1abc

# всё последовательно
python -m src.ml.s1_all --root . --output-root experiments/s1abc
```

Каждый stage directory содержит `predictions_s1.csv`,
`prediction_diagnostics.csv`, `result.csv`, `summary.json`,
`run_manifest.json`, `report_ru.md` и `metrics_summary.csv` рядом с
`result.csv`.

S1C дополнительно содержит `training_table.csv`, `fit_table.csv`,
`fold_assignments.csv`, `oof_predictions.csv`, `metrics_per_fold.csv`,
`metrics_pooled.json`, `feature_contract.json`, `exclusion_counts.json`,
`model_metadata.json`, полный `model_artifact.joblib`, `model_equivalence.json`,
`summary_train.json`, `report_training_ru.md`.

Artifact можно загрузить без labels:

```python
from src.ml.s1_model import load_artifact, predict_s1_ridge
artifact = load_artifact('experiments/s1abc/s1c/model_artifact.joblib')
# predict_s1_ridge(artifact, production_frame_without_labels)
```

Predict не читает `processed/training_targets` и не вызывает `fit`; это
проверено тестом и отдельным process recheck.

## 6. Проверки и ограничения

Проверено: S1A использует последний прошлый ВТД; S1B — минимум только прошлых
ВТД; 2022-ответ не входит в исторические признаки; cutoff/будущие строки не
меняют production features и predictions; held-out labels не меняют fold model;
train-only imputer/scaler; константные/пустые признаки; synthetic Ridge
восстанавливает полезную поправку; save→load→predict эквивалентен; bounds
per-model; S2 join точен. Запущены реальные stage-команды, обучение и отдельный
inference из artifact.

Ограничения:

- Нет strict forward-time folds до cutoff; LOPO — spatial transfer на одной
  кампании.
- 3 простых участка одного parent pipeline `MAIN-2` не являются независимыми
  трубопроводами; статистическая значимость не заявляется.
- Production S1B использует две прошлые кампании (2018, 2022), обучение S1C —
  только одну; перенос S1C в этот режим не проверен.
- `alpha=10` — заранее зафиксированный инженерный старт, не оптимум.
- `processed/target_segments_grid.csv` не прошёл manifest checksum (even after
  LF normalisation). Он не использовался; 525/1475 ключей взяты из
  manifest-verified `processed/segments_grid.csv` и сверены с
  `processed/target_features.csv`. Checksum не переписывался.
- В main checkout `core.autocrlf=true` вызывал CSV checksum mismatch; работа
  выполнена в отдельном worktree с `core.autocrlf=false`.
- `AGENTS.md` в проекте отсутствует; `code-review-graph` в окружении не найден.
  Связи проверялись по исходникам и тестам.
- Скрытые ответы не использовались; внешняя оценка 0.551 / 0.221 не является
  критерием настройки.

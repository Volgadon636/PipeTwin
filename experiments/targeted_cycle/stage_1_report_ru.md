# Stage 1 — аудит исходной точки

Сгенерировано: `2026-11-15T18:57:22+00:00`. Обучение моделей не запускалось.

## 1. Что проверено и изменено

- Репозиторий: `C:\Program Files\PipeTwin`.
- Ветка: `momo`, HEAD `ab57c280061ff919cce4c4240051a4760a0f8019`, origin/momo `ab57c280061ff919cce4c4240051a4760a0f8019`, git tracked-dirty: False, untracked entries: 2.
- Код `src/ml/next_*`, `s1_next.py`, `s2_next.py` присутствует. Совпадение code hashes: 3 raw + 11 LF-normalized из 14.
- Data generation: 25 outputs, hash match: 0 raw + 25 LF.
- Архив `scratch/ml_repro/next_cycle.zip`: НЕ найден. Проверка внешнего архива невозможна; использованы checked-in `experiments/next_cycle` и их manifests.
- Новые результаты сохранены в `experiments/targeted_cycle`; добавлен только read-only `src/ml/targeted_audit.py`. Старые результаты не изменялись.
- Доступные модели: CatBoost 1.2.10, scikit-learn Ridge, joblib; новых алгоритмов/зависимостей не добавлялось.
- Источники прогнозов: `experiments/next_cycle/s1/*`, `block_cv/*`, `s2/*`, `post_cutoff_retrospective/*`; SHA в `source_audit.json`.

### Расхождения с исходным документом

- В документе указан detached HEAD `855da18`; фактически checkout `ab57c28`, ветка `momo`, origin/momo совпадает.
- В документе указан архив `scratch/ml_repro/next_cycle.zip`; в проекте его нет.
- Код-манифест относится к commit `6aaa8f6`; `git diff --name-only 6aaa8f6..HEAD -- src` пуст. В дереве есть коммиты артефактов и удаление `experiments/s1abc.zip`, но код next_cycle не менялся.
- Line endings смешаны: часть code/manifest совпадает только после CRLF→LF; это зафиксировано, не скрыто.

## 2. Данные, роли и cutoff

| Роль | Данные | Строки | Допустимое использование |
|---|---|---:|---|
| historical_development | S1 fit-eligible | 357 | train/OOF, cutoff < 2023-12-01 |
| historical_development | S1 all historical | 561 | OOF/all-pipeline |
| historical_development | S2 strict | 372 | strict train, cutoff < 2022-07-01 |
| existing_retrospective | S2 2023 | 94 keys × 6 models | diagnostic only |
| pending_external_evaluation | S1 target truth | 525 keys | truth unavailable |
| pending_external_evaluation | S2 target truth | 1475 keys | truth unavailable |
| production_features_only | S1 target features | 525 | inference only |
| production_features_only | S2 target features | 1475 | inference only; no target VTD/UZT |

## 3. Пересчёт метрик без fit

Keys/masks проверены: S1 LOPO — 357 строк, S1 block-CV — 321, S2 strict — 372, S2 retrospective — 94 уникальных ключа на модель. LOPO и block-CV не объединялись.

### S1 LOPO (357; три участка 4/32/321)

| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail MAE | tail recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s1a | 0.498223 | 0.652255 | 0.049315 | 0.335451 | 0.112527 | -0.278008 | 0.550719 | 0.576967 | 0.828566 | 0.289474 |
| s1b | 0.498223 | 0.652255 | 0.049315 | 0.335451 | 0.112527 | -0.278008 | 0.550719 | 0.576967 | 0.828566 | 0.289474 |
| old_s1c | 0.469801 | 0.618014 | -0.001609 | 0.091346 | 0.008344 | -0.147346 | 0.280355 | 0.576967 | 1.145926 | 0.105263 |
| affine | 0.417090 | 0.571107 | 0.076174 | 0.205042 | 0.042042 | 0.020209 | 0.080019 | 0.576967 | 1.221409 | 0.289474 |
| ridge_history | 0.445390 | 0.596601 | 0.062180 | 0.138211 | 0.019102 | -0.069217 | 0.239537 | 0.576967 | 1.186051 | 0.131579 |
| ridge_geometry | 0.451148 | 0.603420 | 0.056392 | 0.104238 | 0.010866 | -0.093799 | 0.238080 | 0.576967 | 1.205247 | 0.078947 |
| rate_signed | 0.451148 | 0.603420 | 0.056392 | 0.104238 | 0.010866 | -0.093799 | 0.238080 | 0.576967 | 1.205247 | 0.078947 |
| rate_nonnegative | 0.458012 | 0.599348 | -0.116048 | 0.319107 | 0.101829 | -0.079086 | 0.400351 | 0.576967 | 0.796826 | 0.289474 |
| catboost | 0.444919 | 0.610149 | 0.127380 | 0.183889 | 0.033815 | -0.118332 | 0.291631 | 0.576967 | 1.195953 | 0.184211 |
| mix_025 | 0.465837 | 0.613075 | 0.051084 | 0.317861 | 0.101036 | -0.129081 | 0.455414 | 0.576967 | 0.914431 | 0.289474 |
| mix_050 | 0.442607 | 0.590859 | 0.052854 | 0.285872 | 0.081723 | -0.048737 | 0.366521 | 0.576967 | 1.000591 | 0.263158 |
| mix_100 | 0.451148 | 0.603420 | 0.056392 | 0.104238 | 0.010866 | -0.093799 | 0.238080 | 0.576967 | 1.205247 | 0.078947 |

### S1 block-CV (321; 52 blocks; 100 m / 20 m buffer)

| Модель | MAE | Pearson r² | ordinary R² | std pred | std truth | lower-tail MAE | tail recall | baseline overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1a | 0.506808 | 0.122282 | -0.237474 | 0.560415 | 0.596159 | 0.892288 | 0.303030 | 1.000000 |
| s1b | 0.506808 | 0.122282 | -0.237474 | 0.560415 | 0.596159 | 0.892288 | 0.303030 | 1.000000 |
| ridge_geometry | 0.418534 | 0.163340 | 0.160921 | 0.268943 | 0.596159 | 1.009018 | 0.454545 | 0.515152 |
| rate_signed | 0.418534 | 0.163340 | 0.160921 | 0.268943 | 0.596159 | 1.009018 | 0.454545 | 0.515152 |
| catboost | 0.417871 | 0.130854 | 0.107413 | 0.306834 | 0.596159 | 1.007885 | 0.272727 | 0.606061 |

Block-level paired MAE (same 52 blocks):

| Сравнение | блоков, где MAE ниже S1A | всего блоков |
|---|---:|---:|
| ridge_geometry | 36 | 52 |
| rate_signed | 36 | 52 |
| catboost | 41 | 52 |

`baseline overlap` — пересечение предсказанного нижнего k с нижним k S1A, а не true recall. Это разные диагностики.

### S2 strict (372; три участка 334/34/4)

| Модель | MAE | RMSE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail MAE | tail recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_effective_rate | 0.439772 | 0.554044 | -0.044262 | -0.030151 | 0.000909 | -0.009034 | 0.016013 | 0.551558 | 0.912964 | 0.153846 |
| physics_priors | 0.454416 | 0.568922 | -0.112907 | -0.032619 | 0.001064 | -0.063952 | 0.065861 | 0.551558 | 0.849298 | 0.153846 |
| physics_velocity_consistent | 0.454416 | 0.568922 | -0.112907 | -0.032619 | 0.001064 | -0.063952 | 0.065861 | 0.551558 | 0.849298 | 0.153846 |
| physics_spatial | 0.448168 | 0.564923 | -0.109953 | 0.061652 | 0.003801 | -0.049047 | 0.097153 | 0.551558 | 0.835795 | 0.128205 |
| physics_spatial_temperature | 0.452141 | 0.568974 | -0.118555 | 0.051317 | 0.002633 | -0.064148 | 0.107429 | 0.551558 | 0.827475 | 0.128205 |
| catboost_residual | 0.442719 | 0.558616 | -0.051561 | 0.034188 | 0.001169 | -0.025757 | 0.093240 | 0.551558 | 0.905404 | 0.153846 |

### S2 existing 2023 retrospective (94 keys, 7 ids)

| Модель | MAE | bias | Pearson r | Pearson r² | R² | std pred | std truth | lower-tail MAE | tail recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_effective_rate | 0.428834 | -0.329386 | n/a | n/a | -0.920302 | 0.000000 | 0.343353 | 0.358116 | 0.230769 |
| physics_priors | 0.290496 | 0.290496 | -0.006770 | 0.000046 | -0.716005 | 0.002958 | 0.343353 | 0.916193 | 0.230769 |
| physics_velocity_consistent | 0.290496 | 0.290496 | -0.006770 | 0.000046 | -0.716005 | 0.002958 | 0.343353 | 0.916193 | 0.230769 |
| physics_spatial | 0.318829 | 0.315155 | 0.202042 | 0.040821 | -0.812539 | 0.105155 | 0.343353 | 0.886996 | 0.230769 |
| physics_spatial_temperature | 0.317787 | 0.313703 | 0.200997 | 0.040400 | -0.805330 | 0.104987 | 0.343353 | 0.886906 | 0.230769 |
| catboost_residual | 0.291293 | 0.286096 | 0.185243 | 0.034315 | -0.660187 | 0.068555 | 0.343353 | 0.882520 | 0.384615 |

Top-10%: k=ceil(0.1*n) по каждой секции; ties разрешаются более низким `segment_id`. Для LOPO 357 суммарное k по секциям = 38, global k=36; для block 321 k=33; для S2 strict 372 суммарное k=39, global k=38; для retrospective 94 global k=10. `lower-tail MAE` — MAE на истинно нижних k сегментах; `tail recall` — пересечение истинного и предсказанного нижнего k. Это не top-risk официального scorer.

### Сверка с числами исходного документа

| Проверка | План | Пересчёт | Совпало |
|---|---:|---:|---|
| S1 block S1A MAE | 0.506808 | 0.506808 | True |
| S1 block Ridge MAE | 0.418534 | 0.418534 | True |
| S1 block CatBoost MAE | 0.417871 | 0.417871 | True |
| S1 block S1A r² | 0.122282 | 0.122282 | True |
| S1 block Ridge r² | 0.163340 | 0.163340 | True |
| S1 block CatBoost r² | 0.130854 | 0.130854 | True |
| S1 LOPO S1A MAE | 0.498223 | 0.498223 | True |
| S1 LOPO old S1C MAE | 0.469801 | 0.469801 | True |
| S1 LOPO old S1C r² | 0.008344 | 0.008344 | True |

Числа S1 block и S1 LOPO совпали с документом. S2 strict/retrospective пересчитаны из сохранённых predictions; документ не приводит полную таблицу S2.

## 4. Фактические наблюдения

- S1 LOPO: next−last > 0 у 177/357, < 0 у 178/357, =0 у 2/357; mean next−last = −0.049315 мм. Это не доказательство физического восстановления металла.
- S1 train dt: 3.783710 года у всех 357. Production: 1.404517 у 508 сегментов, 5.188227 у 13, 4 fallback.
- S2 strict: age 18.420260, nominal 8 мм, 372 строки, один parent/одна кампания. Target: age 29.029432, nominal 4.5 мм; VTD history отсутствует, но есть 1 UZT-строка с 3 замерами все выше nominal (eligible=0). Она не является допустимой историей.
- S2 retrospective: 2023-08-12, 94 уникальных ключа, 7 труб; это уже просмотренная диагностика.
- Geometry/coordinates: S1 target 525 сегментов, 5248.4 м, непрерывная сетка; S2 target 1475, 14745.1 м. VTD distances mapped to this grid; raw coordinate origin/direction and tool metadata are not independently verified. У S1 target один profile_extrapolated_flag; max profile gap 4.43 м. У S2 max gap 7.84 м.
- S1 2018→2022 density ratio: target 11.45×; другие fit sections 13.94× и 11.66×. Кампании несопоставимы по плотности; 2022→2023 для 1250011604 — 1.21×, для 1250009490 — 1.65×. S1 target 2023-12-01 labels отсутствуют.
- S2 2018 и 2023 — разные трубы/условия; прямое сравнение плотности кампаний не является валидацией переноса. Defect count, repairs, tool/threshold metadata недоступны.

## 5. Scorer и вопросы

- В PDF: основной показатель — Pearson r² отдельно по каждому кейсу; дополнительно MAE, RMSE, R², графики, top-10%, интервалы. Общей формулы score в проекте нет.
- Локального scorer/кода итогового score не найдено. Финальный score не вычислялся.
- Вопросы менторам: `mentor_questions_ru.md`.

## 6. Тесты и проверки Stage 1

- Пересчёт метрик выполнен только из сохранённых CSV, без fit/train.
- Проверены уникальность ключей, ожидаемые 357/321/372/94, совпадение ключей с fit/training table, отсутствие target id в S2 strict/retrospective, block=segment_id//10, fold=id, finite predictions.
- `python -m unittest tests.test_ingestion.MetricTests -v`: 2 теста, OK (undefined correlation, signed r vs r², tie policy).
- Полный тестовый набор проекта в Stage 1 не перезапускался: этап read-only; новые модели не обучались. Это ограничение, а не заявление о непройденных тестах.

## 7. Команды повторения

```bash
python -m src.ml.targeted_audit --root . --output-root experiments/targeted_cycle
```

Артефакты:
- `experiments/targeted_cycle/source_audit.json`
- `experiments/targeted_cycle/data_roles.json`
- `experiments/targeted_cycle/metrics_recheck/`
- `experiments/targeted_cycle/experiment_plan.json`
- `experiments/targeted_cycle/mentor_questions_ru.md`
- `experiments/targeted_cycle/stage_1_report_ru.md`

## 8. Решения, требующие разбора

1. Подтвердить отсутствие архива и принять checked-in outputs как источник Stage 1.
2. Подтвердить primary metric (Pearson r² per case) и отсутствие общей формулы score.
3. Подтвердить, что 2023 retrospective уже использована и не является untouched holdout.
4. Подтвердить допустимость same-pipe block-CV как диагностики, не как forward validation.
5. Подтвердить, что S1B/старый S1C не выбираются заново по результатам Stage 1.

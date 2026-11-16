# PipeTwin: подготовка данных для прогноза толщины

Скрипты читают шесть исходных CSV, проверяют связи и единицы, строят сетку по 10 м, сохраняют измеренную остаточную толщину и готовят признаки на две даты из резюме встречи:

| ID | Дата прогноза | Сценарий | Сегментов |
|---|---|---|---:|
| 1250022871 | 01.12.2023 | Между ВТД | 525 |
| 1250009583 | 01.07.2022 | Без истории ВТД/УЗТ | 1475 |

Предобработка описана ниже. Добавлен отдельный воспроизводимый ML-baseline в `src/ml/`; запуск и ограничения приведены в следующем разделе. `processed/target_features.csv` содержит признаки для всех 2000 целевых сегментов, а не предсказания толщины.

## Основной прогноз H18 / H31

Основная сборка использует H18 для S1 между ВТД и H31 для S2 cold start.
Команда `train` сохраняет модели, а `predict` строит итоговый CSV из
сохранённого комплекта. Используйте окружение из `requirements-next.txt`.

```powershell
python -m src.ml.production train --data-root scratch/odo_alignment_verified --output-root production/models_run
python -m src.ml.production predict --data-root scratch/odo_alignment_verified --model-root production/models_run --output-root production/forecast_run
```

На текущих данных H18 использует исходное поле `Distance`. Проверенную карту
привязки ВТД 2022 года можно передать через `--verified-s1-anchors`.
H31 строит пространственную поправку для S2 на полной сетке из рельефа и
разрешённой истории других участков. Итог содержит 525 строк S1 и 1 475 строк
S2, в миллиметрах и с прежними тремя столбцами.

Проверенная сборка находится в [production/h18_h31_verified/result.csv](production/h18_h31_verified/result.csv).
Переключатели методов, восстановление входов, метрики, ограничения и откат
описаны в [документации основного прогноза](docs/production_forecast.md).

## ML baseline (English)

The first forecasting implementation reuses the published artifacts and existing
`build_features_asof` API. It reconstructs historical windows, fits separate
nonnegative constant-rate baselines for both cases, accounts for exposure gaps,
and validates the exact submission keys and physical/assumed bounds.

```bash
python -m src.ml.experiment --output-root experiments/ml_baseline
python -m unittest discover -s tests -p test_ml.py -v
```

For ML-only execution from the checked-in CSVs, use Python 3.12 and
`pip install -r requirements-ml.txt` in a separate environment. Full raw-data
preprocessing still uses the original `requirements.txt` and commands below.

Outputs include `experiments/ml_baseline/result.csv` (2,000 rows), candidate
forecasts, source audit, historical windows, metrics, risk ranks and a run manifest.
The [model card](docs/ml_model_card.md) and [run report](experiments/ml_baseline/report.md)
explain assumptions and test results.

**Status: provisional baseline.** There are no eligible independent forward-time
fitted-model validation folds before either production cutoff. The cold-start
baseline is constant along the pipe and cannot establish spatial risk ranking.
The nonlinear physical model is implemented and synthetically tested, but gated
off because independent training/validation evidence is insufficient. No hybrid
accuracy or calibrated confidence intervals are claimed.

## ML v2: environment-aware cold start (branch `model_v2`)

v2 addresses the main defect of the v1 baseline: it transferred a warm, wet
MAIN-2 thinning rate to the cold, dry, slow cold-start pipe and produced one
constant value (1.540 mm) for all 1,475 segments. v2 keeps the specification's
multiplicative environment structure, fits only the reference rate, declares the
remaining exponents as explicit priors, and reports a 243-point sensitivity grid.
It also adds a within-pipe profile term that is gated by leave-one-campaign-out
correlation, and separates comparable from non-comparable inspection pairs in
the between-VTD rate calibration.

```bash
python -m src.ml.experiment_v2 --output-root experiments/ml_v2
python -m unittest tests.test_ml_v2 -v
```

Headline results (see `experiments/ml_v2/report.md` and
[docs/model_v2.md](docs/model_v2.md)):

| Case | v1 | v2 | Evidence status |
|---|---|---|---|
| `1250009583` cold start @ 2022-07-01 | 1.540 mm constant | 2.810 mm mean, sd 0.07 mm | cross-environment LOO; hidden test not available |
| `1250022871` between VTD @ 2023-12-01 | rate 0.013 mm/yr | rate 0.0 (no comparable pre-cutoff pair) | 3-pipe LOO pair check, Pearson² 0.113 (optimistic) |

v2 is still a provisional, physics-informed forecast. It does **not** claim a
validated hidden-test score, calibrated intervals, or a proven spatial ranking.
The document lists the open mentor questions, including whether the dense
2022→2023 campaign may be used for S1 calibration.

## S1A / S1B / S1C: as-of baselines and Ridge correction

The S1 experiment is implemented in a separate local branch/worktree
(`s1abc-ridge`, based on the `model_v2` tree). It does not retrain or change S2.
S1A is last-prior persistence, S1B is the minimum of all prior VTD campaigns,
and S1C is a small Ridge correction to S1B. The full contract, artifacts and
honest limitations are in [docs/s1abc.md](docs/s1abc.md).

```bash
python -m src.ml.s1a --root . --output-root experiments/s1abc/s1a
python -m src.ml.s1b --root . --output-root experiments/s1abc/s1b
python -m src.ml.s1c train --root . --output-root experiments/s1abc/s1c --alpha 10.0
python -m src.ml.s1c predict --root . --output-root experiments/s1abc/s1c
python -m src.ml.s1_metrics --root . --output-root experiments/s1abc
# sequential
python -m src.ml.s1_all --root . --output-root experiments/s1abc
```

S1C is fitted on 357 historical rows (3 simple sections, one campaign
`2022-07-06`, one parent pipeline). The only validation is leave-one-simple-
pipe-out spatial transfer, not a forward-time forecast. It lowers MAE but also
lowers Pearson r²; this is saved as a negative/limited result, not promoted as
a proven improvement. S2 stays unchanged from the explicit v2 `result.csv`.

## Next cycle: S1 diagnostics and strict S2 candidates

The next cycle adds a consistent regenerated preprocessing tree, a limited S1
diagnostic set (last-baseline Ridge, rate, CatBoost, weight/block ablations) and
strict-as-of S2 cold-start candidates (constant, physics priors, spatial,
temperature-scenario and CatBoost). It deliberately keeps S1A as the S1 default
and physics priors as the S2 default because no candidate improves both level
and spatial signal. See [docs/next_cycle.md](docs/next_cycle.md).

```bash
python -m src.data_prep --output-root experiments/next_cycle/data_generation
python -m src.ml.next_s1 train --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s1
python -m src.ml.next_s2 train --root . --data-root experiments/next_cycle/data_generation --output-root experiments/next_cycle/s2
python -m src.ml.next_compare --root . --output-root experiments/next_cycle --s1-candidate s1a --s2-candidate physics_priors
```

Artifacts: `experiments/next_cycle/` (OOF, folds, metrics, artifacts, figures,
explicit assembly). The old production result is not overwritten and nothing is
pushed.

## Ingestion v2: evidence, adapters and audits

The parser/ingestion layer was extended without changing the scientific model or
the public `build_features_asof` interface.  The target remains the measured
minimum residual wall thickness in mm per 10 m segment; `result.csv` remains the
exact three-column export.

```bash
python -m src.data_prep --inspect-only
python -m src.data_prep --output-root scratch/verified_run
python -m src.mentor_ingest --output docs/mentor
python scripts/calculate_segment_temperature.py \
  --features scratch/verified_run/processed/target_features.csv \
  --output-root scratch/temperature_scenario --sensitivity
```

Key additions:

- Semantic source IDs (`passport`, `profile`, `vtd`, `uzt`, `hydraulic_regime`,
  `corrosion_monitoring`, `submission_template`) selected by schema fingerprints,
  not filenames.  The legacy key `regime.csv` still means **corrosion monitoring**;
  Techmod is a separate `hydraulic_regime` source.  Conflicting copies fail and
  require an explicit `source_overrides` entry.
- Loss-aware parsing: UTF-8/BOM and Windows-1251, scored delimiter detection,
  duplicate header detection, exact integer IDs, per-value date parsing,
  ambiguous slash-date quarantine, decimal comma / thousands / NBSP policy,
  censored values such as `<0.1` (never coerced to zero) and malformed-record
  quarantine with logical record references.
- `reports/source_inventory.json`, `reports/schema_mapping.json`,
  `reports/feature_inventory.csv` (every supplied field), `reports/row_reconciliation.json`,
  `reports/join_audit.json`, `reports/leakage_audit.json`, `reports/lineage.json`,
  `reports/template_validation.json` and `reports/run_manifest.json`.
- `processed/rejected_records.*`, `processed/lineage_keys.*`, and a separate
  spreadsheet-safe quality export.  Raw source bytes remain immutable.
- A separate `temperature_scenarios.*` artifact; the verified grid is never
  mutated with scenario `delta_t_c`.
- Mentor ingestion stores source hashes/revisions, verbatim locators, accepted
  decisions/requirements/questions/actions and generated follow-up drafts.  It
  does not send messages or assume unsupplied recordings.
- Evaluation now reports signed Pearson r, Pearson squared, separate R2, MAE,
  RMSE, bias, MAPE with the existing near-zero suppression policy, an explicitly
  labelled eligible-subset MAPE, and top-fraction overlap with tie disclosure.

If no authoritative `sample_result.csv` exists, validation explicitly uses the
**derived target grid**.  A file named `sample_result.csv` is not treated as a
template unless its contents are exactly `id,segment_id,thickness_pred`.

## Запуск

Проверено на Python 3.13. Точные версии зависимостей записаны в `requirements.txt`. Случайных операций в подготовке нет.

```powershell
cd C:\Users\Honor\Desktop\PipeTwin
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data_prep
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Также поддерживается `python src/data_prep.py`. Скрипты работают на Windows и Linux. Из корня проекта достаточно `python -m src.data_prep` в окружении с установленными зависимостями.

Другие пути:

```powershell
python -m src.data_prep --data-dir "C:\Users\Honor\Downloads" --output-root "scratch\original_names_check"
```

Исторический срез для собственной валидации:

```powershell
python -m src.data_prep --as-of 2022-07-06 --ids 1250022871 --case between_vtd --output-root scratch\validation_2022
```

Для проверки без диагностики используйте `--case cold_start`. Всегда задавайте отдельный `--output-root` для экспериментов, чтобы не перезаписывать основной результат. Исторические признаки можно получать через `build_features_asof` в Python. Разметку для обучения на срезе нужно ограничить датами раньше среза; случайное перемешивание строк истории не является временной валидацией.

### Выравнивание одометров ВТД

Передайте план пар прогонов с ID простого участка, датой исходного и опорного ВТД.

```powershell
python -m src.data_prep --odometer-alignment-plan configs/odometer_alignment.example.json --output-root scratch/odo_alignment
```

Модуль строит переменный сдвиг по началам труб между швами
(`Distance − Relative Distance`). При недостатке надёжных совпадений он
сопоставляет локальные профили плотности дефектов. Для прогонов с
заведомо несовместимой нумерацией можно указать `"method": "density"`.
Сдвиг применяется до разбиения на 10 м. Преобразованные точки за
границей трубы учитываются в аудите и исключаются из агрегации.

- `processed/vtd_odometer_coordinates.*` содержит исходные и выровненные координаты с ключом исходной строки.
- `processed/vtd_aligned_by_segment.*` содержит измеренные минимумы после выравнивания и дату опорной системы координат.
- `reports/odometer_alignment.json` и `processed/vtd_alignment_profiles.*` содержат ориентиры, статистику сдвига, плотность дефектов и Pearson r/r² на общих сегментах до и после выравнивания. Это диагностика двух известных ВТД.
- Поле `feature_cutoff` в плане создаёт отдельный `processed/target_features_odo_aligned.*` для совпадающего запроса S1. Опорное ВТД должно быть строго раньше этой даты. Срез включает только перечисленные исходные прогоны, переведённые в выбранную систему координат, и сам опорный прогон.

Основные `vtd_by_segment`, `training_targets` и `target_features` используют исходную предобработку. В примере история 2018 года переводится в систему координат 2022 года. Для целевого участка координаты ВТД 2023 года отсутствуют.

### Перцентили остаточной толщины

```powershell
python -m src.data_prep --vtd-quantiles --odometer-alignment-plan configs/odometer_alignment.example.json --output-root scratch/vtd_quantiles
```

`--vtd-quantiles` публикует `processed/vtd_segment_state.*` по полной
сетке каждого допустимого прогона. При наличии плана выравнивания
`processed/vtd_aligned_segment_state.*` содержит те же показатели после
переноса координат. В состоянии сегмента есть минимум, среднее, P01,
P05, P10, P50, число дефектов, число пригодных и восстановленных значений
толщины, максимальная глубина в мм и флаги покрытия.

Для исторических признаков используется измеренная остаточная толщина.
Если исходное значение отсутствует, толщину можно восстановить из
допустимой глубины и паспортной стенки. Такие записи считаются отдельно.
Сегмент без зарегистрированных дефектов получает номинал и флаг
`nominal_fallback` только внутри покрытия прогона. При дефектах без
пригодной толщины или без покрытия пустой сегмент остаётся пропуском.
В сегменте с единственным значением все перцентили равны этому значению.

`processed/vtd_quantile_ablation.*` сравнивает все пять квантилей с
измеренным минимумом следующего ВТД на общих ключах. Показываются
Pearson r, Pearson r² и MAE отдельно для всех доступных оценок и
для оценок только по измерениям. В выровненной диагностике следующий
прогон служит опорой координат. Отдельные
`processed/target_features_vtd_quantiles.*` для S1 используют самый
поздний допустимый ВТД строго до даты прогноза в исходной системе координат.
Метки `training_targets` остаются измеренными минимумами.

## Исходные файлы и имена

| В папке проекта | Исходное имя организаторов | Содержание |
|---|---|---|
| passport.csv | pot_south.csv | Паспорт |
| profile.csv | altitudes_south.csv | Высотный профиль |
| vtd.csv | vtd_south_train.csv | ВТД |
| uzt.csv | uzt_south.csv | УЗТ |
| regime.csv | repcorr_south.csv | Физико-химия и коррозионный мониторинг |
| tech_mode_for_embedings_with_length_with_temperatures_ids16.csv | То же | Гидравлический режим |

Можно использовать любой из двух наборов имён. Разные версии одного источника под двумя именами вызывают ошибку. Пропущенный обязательный файл также вызывает ошибку, а не тихо пропускается. `sample_result.csv` не является входом этого набора: реальный шаблон ответа не предоставлен.

Содержимое исходных CSV не меняется. Проверяются контрольные суммы до и после расчёта. Все преобразования выполняются в памяти и сохраняются в `data/cleaned`, `processed`, `reports`.

## Что делает каждый этап

1. **Загрузка:** явные схемы полей, поддержка десятичной точки/запятой по значениям, разбор дат по строкам, уникальные названия столбцов. Неизвестные архивные поля сохраняются текстом; они не становятся признаками автоматически.
2. **Проверки:** единицы диаметра и глубины дефекта, хронология, связи ID, границы, качество УЗТ и мониторинга. Исключения имеют источник, номер записи CSV и причину.
3. **Сетка:** все 16 ID, 4668 сегментов, последний сегмент укорочен. Точка ровно на конечной границе относится к последнему сегменту; точки вне трубы не переносятся в неё.
4. **Измерения:** минимум ВТД вычисляется по `Residual Wall Thickness`, а не по паспортному номиналу и проценту глубины. Перед минимумом отбрасываются наблюдения с остатком выше паспортного номинала простого участка (флаг `above_nominal_flag`) и с неразрешимым номиналом (`nominal_unresolved_flag`); исходные строки сохраняются в аудите. Отсутствие измерения остаётся пропуском. Таблицы истории сохраняют даты.
5. **Признаки на дату:** сначала отсекается будущее, затем применяется правило трёх календарных дней, выполняется агрегация режима и присоединяется допустимая диагностика. При холодном старте ВТД и УЗТ полностью исключаются из входов.
6. **Проверка экспорта:** границы, полнота, измеренные минимумы, отсутствие будущих событий, CSV/Parquet. Публикация результатов происходит после всех проверок.

## Правила очистки и допущения

Подробности находятся в `docs/cleaning_rules.md`. Основное:

- `Distance` — координата ВТД. `Distance_raw` не используется для привязки. 251 запись вне трубы помещается в `processed/vtd_rejected`, включая четыре отрицательных расстояния. Причину происхождения этих координат мы не объявляем установленной.
- Поле глубины ВТД 23.09.2018 нормализуется из доли в проценты. Правило ограничено указанной датой в конфигурации и проверяется по соотношению локальной толщины и измеренного остатка. Исходное значение сохраняется в `defect_depth_pct_raw`. Это не меняет измеренную остаточную толщину.
- Для цели используются только положительные остаточные толщины, не превышающие **паспортный номинал** простого участка более чем на 0,02 мм. Это правило второго совещания: оно применяется к каждому наблюдению до расчёта минимума сегмента. Превышение сохраняется с флагом `above_nominal_flag` и в журнале качества, но не участвует в минимуме. Локальная стенка строки ВТД сравнивается отдельно (`above_local_wall_flag`) и не заменяет паспортный номинал. Если паспортный номинал отсутствует, сравнение помечается `nominal_unresolved_flag`. Восстановление отсутствующей толщины через глубину по умолчанию выключено: неизвестная метка не выдумывается.
- ВТД до паспортной даты ввода сохраняются, но не включаются в `training_targets`. Даты не исправляются предположением.
- Все 67 УЗТ выше паспортного номинала: значения сохранены, помечены, исключены из признаков. Причина не подтверждена; заявления об усиленных точках нет.
- В мониторинге 412 строк имеют отсутствующий/неизвестный ID; у 83 остальных конфликтует направление. Автоматическое присоединение этих записей или перенос их медиан на другие трубы не выполняется.
- `d_mm` в гидравлике преобразуется в миллиметры после сверки с паспортом. Исходное значение сохраняется. Давление и остальные величины не пересчитываются по догадке.
- Отсечение консервативное: `timestamp < T 00:00`. Расчёты в день T не используются, поскольку время оцениваемого ВТД неизвестно.
- Гидравлические значения в полном архиве сохраняются. Для каждого среза создаются отдельные `*_usable` поля и журнал фильтрации. Мягкие аномалии, наблюдаемые три последовательных календарных дня до T, сохраняются с флагами. Несколько строк одного дня не заменяют три дня; пропуск дня разрывает последовательность. Жёстко невозможные отрицательные расходы/плотности, нулевая плотность при потоке и обводнённость вне 0–100% не становятся валидными от длительности.
- Внутрисуточные расчёты сворачиваются медианой, затем считаются средние по наблюдаемым дням. Пропущенные дни не заполняются будущими значениями. Отдельно показываются покрытие и максимальный разрыв наблюдений.
- Пороги в `configs/preprocessing.json` — настраиваемые эвристики аудита, не инженерные пределы. Устойчивый подозрительный режим не объявляется доказанно правильным.
- `t_mix` не принуждается к диапазону `t_start`–`t_end`: физический смысл пока не подтверждён. Рост давления к концу трубы автоматически не удаляется. Линейная интерполяция давления вдоль трубы не выдаётся за гидравлическую модель.
- Профиль интерполируется по фактическим расстояниям. Средняя высота считается с весом по длине, уклон — по разнице высот границ сегмента и его реальной длине. Крутые ступени помечаются, но не сглаживаются произвольно.
- Поздние диагностические поля паспорта, покрытия без истории изменения и расчётная `pit_corrosion_intensity` исключены из признаков. Номинал, диаметр, дата ввода и профиль приняты статическими при отсутствии истории замен; это ограничение остаётся.

## Основные результаты

| Путь | Назначение |
|---|---|
| data/cleaned/* | Типизированные архивные данные, исходные значения и флаги; не все строки допустимы для модели |
| processed/segments_grid.* | Полная сетка всех 16 ID |
| processed/target_segments_grid.* | 2000 сегментов двух целевых труб с датами |
| processed/vtd_by_segment.* | История измеренных минимумов ВТД, включая отсутствующие/сомнительные метки с флагами |
| processed/training_targets.* | Подмножество допустимых наблюдений; для фолда всё равно требуется отсечение по дате |
| processed/uzt_by_segment.* | Все измерения УЗТ и отдельно пригодный минимум; сомнительные значения не становятся метками |
| processed/profile_by_segment.* | Высоты, уклоны и покрытие профиля всех сегментов |
| processed/target_features.* | Основная таблица признаков двух сценариев на соответствующие T |
| processed/regime_mapped.* | Режим только для целевых срезов, без усреднения после T |
| processed/monitoring_mapped.* | Покрытие и допустимая физико-химия; нулевое покрытие видно явно |
| processed/hydraulic_asof.* | Использованные до T записи режима, оригинальные и `*_usable` величины |
| processed/hydraulic_quality.* | Действие по каждой аномалии на каждом срезе |
| processed/quality_issues.* | Строки источников, правила и действия при нормализации |
| processed/*_rejected.* | Карантин позиций/ключей без потери исходной записи |
| processed/pipes_summary.* | Фактическое покрытие по ID |
| processed/rejected_records.* | Единый аудит карантина/malformed/дублей с исходной ссылкой |
| processed/lineage_keys.* | Source-row ranges и lineage ID для каждой целевой строки |
| reports/source_inventory.json | Роль, путь, SHA-256, схема и ограничения источника |
| reports/schema_mapping.json | Сырые→канонические поля, версии адаптеров и drift |
| reports/feature_inventory.csv | Статус каждого исходного и производного поля |
| reports/leakage_audit.json | Cutoff, cold start, диагностические поля и мониторинг |
| reports/template_validation.json | Проверка шаблона или явная проверка по производной сетке |
| reports/lineage.json | Агрегаты, правила и source references |
| reports/data_quality.md | Читаемый отчёт, предупреждения и покрытие |
| reports/validation.json | Выполненные проверки |
| reports/run_manifest.json | Входные хэши, конфигурация, версии, схемы, runtime, хэши результатов |
| docs/mentor/* | Решения, вопросы, действия и follow-up drafts |
| docs/ingestion_v2_verification.md | Наблюдаемые результаты проверки v2 |

Каждая таблица сохраняется в CSV и Parquet. CSV использует точку, UTF-8 и пустое поле для пропуска. Все пары проверяются после сохранения. Служебные журналы не являются признаками модели.

## Структура кода

```text
configs/preprocessing.json   целевые ID/T и явные параметры качества
src/config.py                legacy-схемы, имена источников и белый список признаков
src/source_registry.py       семантические типы, fingerprint-классификация, конфликты копий
src/schema_registry.py       версии схем и полный инвентарь полей/признаков
src/parsing.py               loss-aware чтение, ID/даты/числа и карантин
src/format_detector.py       адаптеры, совместимое чтение и происхождение строк
src/data_cleaner.py          нормализация, флаги, карантин и row dispositions
src/units.py                 явные правила единиц и неразрешённые единицы
src/grid_builder.py          сетка и пограничные координаты
src/temporal.py              отсечение, три календарных дня, покрытие окон
src/segment_mapper.py        измеренные минимумы, prior-inspection min и признаки на дату
src/quality.py               schema drift, row reconciliation, join/coverage отчёты
src/leakage.py               аудит cutoff, cold start и диагностических полей
src/lineage.py               feature/source lineage без копирования строк
src/cache.py                 content-addressed cache keys
src/fluid.py                 Reynolds только при явной вязкости
src/heat_transfer.py         изолированный steady-state тепловой сценарий
src/mentor_ingest.py         решения/вопросы/действия и follow-up drafts
src/storage.py               CSV/Parquet, rollback-safe publish_tree
src/validator.py             проверки содержимого и покрытия
src/data_prep.py             общий запуск и публикация проверенных файлов
tests/test_preprocessing.py  регрессионные и интеграционные тесты
tests/test_ingestion.py      новые контракты ingestion/mentor/metrics/export
```

Для воспроизводимости повторный запуск с одинаковыми входами, конфигурацией и версиями должен давать те же CSV/Parquet. Изменение даты T требует нового расчёта признаков, а не фильтра уже усреднённой таблицы.

# Основной прогноз PipeTwin: H18 для S1 и H31 для S2

Выбранная пользователем сборка использует H18 `geometry_type_distribution`
для S1 между ВТД и H31 `spatial_gaussian_process` только для S2 cold start.
Рабочий результат: [production/h18_h31_verified/result.csv](../production/h18_h31_verified/result.csv).
Обученные модели: `production/h18_h31_verified_models/forecast_models.joblib`.
[Отчёт интеграции](../experiments/r2_pool/integration/h18_h31/report_ru.md)
содержит повторные измерения, ограничения и результаты проверок.

## Запуск

Команды выполняются из `C:/Program Files/PipeTwin`. Рабочее окружение:
Python 3.12.8, scikit-learn 1.7.2; версии зафиксированы в `requirements-next.txt`.
Локальная `.venv` с другой версией scikit-learn не подходит к сохранённым моделям.
Указанные ниже выходные папки должны быть новыми: существующие результаты
защищены от перезаписи. Публикация каждой папки выполняется после успешного
формирования и проверки её содержимого.

```powershell
$python = 'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe'
& $python -B -m src.ml.production train --data-root scratch/odo_alignment_verified --output-root production/models_run
& $python -B -m src.ml.production predict --data-root scratch/odo_alignment_verified --model-root production/models_run --output-root production/forecast_run
```

`predict` загружает сохранённый комплект и проверяет его SHA-256; обучение и
чтение `processed/training_targets` для прогноза не требуются. Признаки на дату
строятся из разрешённой истории, паспорта и рельефа. Манифесты содержат версии
зависимостей, хеши исходного кода и фактически прочитанных входов.

Если подготовленные входы нужно восстановить, в отдельную папку запускается:

```powershell
& $python -B -m src.data_prep --config configs/production_preprocessing.json --output-root scratch/production_data_rebuild
& $python -B -m src.ml.production train --data-root scratch/production_data_rebuild --output-root production/models_rebuilt
& $python -B -m src.ml.production predict --data-root scratch/production_data_rebuild --model-root production/models_rebuilt --output-root production/forecast_rebuilt
```

`production_preprocessing.json` сохраняет конфигурацию существующей сборки
`scratch/odo_alignment_verified`. Её дополнительные таблицы привязки одометра
не используются как скрытый новый метод: H18 берёт точки из исходного `Distance`.
Исходные CSV в `data/` и прежние экспериментальные результаты сохраняются.

## Формулы и контракты

**S1 / H18.** Модель обучается на 357 допустимых исторических парах с ответами
2022-07-06, строго до 2023-12-01. Собственные предыдущие замеры предшествуют
каждому ответу. Дополнительные признаки — медианы `log1p` длины и ширины
прошлых дефектов, разность медианы и минимума толщины, междецильный размах,
доли записей потери металла и внешних дефектов. Угол не используется.
Imputation, удаление константных признаков и масштабирование обучаются на train.

```text
b = joint_blend(minimum, p05), нормированный по полной сетке
delta = Ridge(base_features + H18_features) - Ridge(base_features)
H18 = clip(b + delta, 0, nominal_thickness_mm)
```

Обе Ridge используют alpha=10 и равный суммарный вес простых участков.
Поправка добавляется только там, где есть прямой замер последнего прошлого ВТД;
остальные сегменты сохраняют `joint_blend`. Для целевой трубы используется ВТД
2022-07-06: 508 прямых сегментов, 17 с резервным значением из прошлого snapshot.
Нормировка требует полной сетки, укороченный набор ключей отклоняется.

Достоверной карты привязки ВТД 2022 года, известной до 2023-12-01, нет.
Поэтому целевой прогноз использует опубликованный `Distance`. При появлении
подтверждённой карты её можно передать через `--verified-s1-anchors path.csv`.
Нужны `id,source_m,reference_m,source_date,available_date,evidence`;
дата доступности строго до cutoff, источник 2022-07-06, монотонные опорные точки,
покрывающие всю сетку и все переносимые наблюдения. Геометрия соседних труб
на дату проверяемого ответа используется только в исторической ретроспективе.

**S2 / H31.** Обучение использует 372 первых измерения других участков от
2018-09-23, строго до 2022-07-01. ВТД/УЗТ целевой трубы исключены. Сначала
оценивается физический средний уровень с прежними priors и `own_mean` для
неизвестной ранней истории. Остаток центрируется внутри обучающего профиля.
Gaussian process использует нормированную координату, признаки рельефа и
train-only preprocessing прежней пространственной базы:

```text
K = X @ X_train.T + spatial_variance * Matern_3/2(position) * RBF(features)
H31 = clip(physics + GP_shape - length_weighted_mean_full_grid(GP_shape), 0, nominal)
```

Фиксированные параметры H31: `length_fraction=0.25`,
`feature_length_per_sqrt_dimension=1`, `spatial_variance=4`,
`noise_to_linear_variance=10`. Их не подбирали повторно по внешним ответам.
Прогноз требует полной непрерывной сетки; наличие собственных диагностических
признаков отклоняется. H31 недоступна как метод S1 в конфигурации и CLI.

## Состав результата и переключатели

Конфигурация `configs/production_forecast.json` выбирает `s1_method` и
`s2_method`. Допустимые значения: S1 — `baseline`/`H18`, S2 — `baseline`/`H31`.
CLI `predict --s1 ... --s2 ...` позволяет переопределить их без переобучения.
`previous_result` явно указывает прежний CSV для сравнения и отката.

Каждый прогноз сохраняет:

- `result.csv`: ровно `id,segment_id,thickness_pred`, миллиметры;
- 525 строк S1 (`1250022871`, 2023-12-01) и 1475 строк S2
  (`1250009583`, 2022-07-01), точное совпадение с целевой сеткой;
- `ablation/baseline.csv`, `h18_only.csv`, `h31_only.csv`, `h18_h31.csv`;
- `previous_result.csv`, точную копию предыдущего результата;
- `predictions_s1.csv`, `predictions_s2.csv` и `manifest.json`.

Ветви независимы: H18 меняет только S1, H31 — только S2. Полный состав
равен соответствующему одиночному методу внутри каждого кейса. Дополнительный
ансамбль и подбор весов не нужны. Старый baseline в целевой сборке восстановлен
побайтно; CSV абляций позволяют проверить это без скрытых ответов.

## Проверка и предел выводов

```powershell
& $python -B -m src.ml.production_validation --output-root production/validation_run
& $python -B -m scripts.verify_production_forecast --model-root production/models_run --forecast-root production/forecast_run --output-root scratch/reload_check_run
$env:PIPETWIN_NEXT_DATA_ROOT = (Resolve-Path 'experiments/next_cycle/data_generation').Path
& $python -B -m unittest discover -s tests -v
```

Валидация заново обучает рабочие реализации на замороженных разбиениях,
сверяет полные ключи и ответы и проверяет совпадение с архивными H18/H31.
Она сохраняет все профили, signed Pearson r, Pearson r², MAE/RMSE/bias,
разброс и причины неопределённости. Группы менее трёх точек не получают r².
Объединённые значения помечены `pooled_replay_only` и нужны для воспроизведения
старого отчёта. Общий балл S1+S2 не вычисляется.

На основных ретроспективах r² вырос с 0,458022 до 0,511492 для S1 и с
0,040832 до 0,059538 для S2. На дополнительном переносе S2 между участками
2018 года r² снизился с 0,008761 до 0,004031. Есть отрицательные корреляции
отдельных профилей и очень малые группы. Эти уже использованные development
проверки не являются независимым финальным holdout. Качество скрытого теста
неизвестно; результат имеет статус exploratory.

## Откат

Готовый старый CSV сохранён в
`production/h18_h31_verified/previous_result.csv`; его исходник
`experiments/incremental_cycle/e0/result.csv` не изменён. Для нового экспорта
прежних ветвей:

```powershell
& $python -B -m src.ml.production predict --model-root production/h18_h31_verified_models --s1 baseline --s2 baseline --output-root production/rollback_run
```

Исходные артефакты `production/h18_h31` и `production/h18_h31_models` также
сохранены. Никакого push/merge или внешней публикации интеграция не выполняет.

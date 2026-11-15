# Отчёт проверки подготовки данных

Это подготовка данных, не обученная модель и не файл прогноза.

Проверки структуры и содержимого: 306 успешно.
Полная сетка: 4668; строки целевых признаков: 2000.

| Источник | Сохранено строк |
|---|---:|
| passport.csv | 16 |
| profile.csv | 20180 |
| uzt.csv | 67 |
| vtd.csv | 48040 |
| regime.csv | 495 |
| tech_mode_for_embedings_with_length_with_temperatures_ids16.csv | 39052 |

| Карантин | Строк |
|---|---:|
| profile_rejected | 0 |
| vtd_rejected | 251 |
| uzt_rejected | 0 |

## Источники: роль, хэш и схема

| Путь | Тип | Строк | SHA-256 (12) | Schema | Ограничение |
|---|---|---:|---|---|---|
| passport.csv | passport | 16 | c613a97a4e3d | passport-v1 |  |
| profile.csv | profile | 20218 | 2b3f5d05d34f | profile-v1 |  |
| vtd.csv | vtd | 48291 | e4cae76524ea | vtd-v1 |  |
| uzt.csv | uzt | 67 | 25187e44c134 | uzt-v1 |  |
| tech_mode_for_embedings_with_length_with_temperatures_ids16.csv | hydraulic_regime | 39052 | 560ba4c56429 | hydraulic-v1 |  |
| regime.csv | corrosion_monitoring | 495 | 1c17535b437b | monitoring-v1 |  |

## Сверка строк (непересекающиеся disposition)

| Источник | Вход | Оставлено | Дубли | Карантин | Malformed | OK |
|---|---:|---:|---:|---:|---:|---|
| passport.csv | 16 | 16 | 0 | 0 | 0 | True |
| profile.csv | 20218 | 20180 | 38 | 0 | 0 | True |
| uzt.csv | 67 | 67 | 0 | 0 | 0 | True |
| vtd.csv | 48291 | 48040 | 0 | 251 | 0 | True |
| regime.csv | 495 | 495 | 0 | 0 | 0 | True |
| tech_mode_for_embedings_with_length_with_temperatures_ids16.csv | 39052 | 39052 | 0 | 0 | 0 | True |

## Единицы

| Правило | Поле | Строк | Статус |
|---|---|---:|---|
| UNIT-0001 | length_km -> length_m | 16 | verified |
| UNIT-0008 | defect_depth_pct fraction -> percent | 2473 | conditional_campaign |
| UNIT-0010 | Techmod d_mm m -> mm | 39052 | verified |
| UNIT-0011 | watercut_pct -> water_fraction | 0 | derived_only |

Неразрешённые единицы: UNIT-0009=residual_wall_thickness (unresolved), UNIT-0012=p_start/p_end (unresolved), UNIT-0013=viscosity_liquid_work (unresolved), UNIT-0014=q/q_oil/q_gas (unresolved), UNIT-0015=t_start/t_end/t_mix (observed_scale_only), UNIT-0017=co2_water (unresolved), UNIT-0018=fluid_density (unresolved), UNIT-0019=length (unresolved_scope)

## Мониторинг/физико-химия

Строк: 495; неизвестный ID: 412; конфликт направления: 83; допустимо к присоединению: 0; наблюдений: 0.
Политика: No automatic fuzzy join; no cross-pipe median transfer; point observations are not propagated.

## Шаблон результата

Статус: no_authoritative_template; grid: derived_target_grid; exact=None.

## Утечка/сроки

Passed: True; failures: []

## Инвентарь полей

Rows: 515; by status: {'available_but_excluded': 378, 'source_calculated': 69, 'measured': 31, 'derived_verified': 26, 'unresolved_semantics': 10, 'assumed_scenario': 1}

## Остаточные ограничения

- УЗТ выше номинала сохранены, но исключены из признаков до подтверждения методики.
- ВТД до ввода сохранены как наблюдения; такие метки не допускаются в training_targets.
- Спорные связи физико-химии не исправлены догадкой; переноса медиан между трубами нет.
- Пороговые значения гидравлики — параметры проверки качества, не пределы безопасной эксплуатации.
- Устойчивые три календарных дня подозрительные значения сохранены с флагом. Это не подтверждение их физической достоверности.
- Отсечение строго перед 00:00 даты прогноза; запись самого дня T не используется.
- Статические диаметр, номинал, ввод и профиль принимаются из паспорта/профиля. История замен не предоставлена.
- Поздние диагностические сведения паспорта и расчётная pit_corrosion_intensity не включаются в признаки.
- Архивные таблицы содержат всю историю. Для каждого обучающего фолда требуется отдельный вызов build_features_asof.
- Температурный сценарий не находится в target_features и не является измерением.
- Метрики модели и result.csv строятся на этапе моделирования; нулевые независимые forward-фолды не считаются успешной валидацией.

## Покрытие признаков по целевым датам

| ID | Дата | Сегментов | Дней режима | Последняя дата режима | Сегментов с прошлым ВТД | Записей физико-химии |
|---|---|---:|---:|---|---:|---:|
| 1250009583 | 2022-07-01 | 1475 | 1414 | 2022-06-30 07:09:35 | 0 | 0 |
| 1250022871 | 2023-12-01 | 525 | 2771 | 2023-11-30 07:09:51 | 521 | 0 |

## Срабатывания правил (не число независимых ошибок)

| Правило | Строк |
|---|---:|
| above_nominal_flag | 24 |
| before_commissioning_flag | 13511 |
| density_suspect_flag | 3 |
| depth_fraction_to_percent | 2473 |
| depth_residual_mismatch_flag | 6 |
| diameter_m_to_mm | 39052 |
| direction_conflict_flag | 83 |
| exact_duplicate | 38 |
| exposure_invalid_flag | 3 |
| exposure_mismatch_flag | 92 |
| negative_protective_effect_flag | 2 |
| outside_pipe_or_missing_key | 251 |
| residual_invalid_flag | 24 |
| residual_missing_flag | 347 |
| steep_profile_step | 61 |
| unmapped_id | 412 |
| unresolved_thickness_above_nominal | 67 |
| wall_nominal_mismatch_flag | 14267 |

Полные примеры и действия: processed/quality_issues.csv, processed/rejected_records.* и processed/hydraulic_quality.csv.
Машиночитаемые отчёты: reports/source_inventory.json, reports/schema_mapping.json, reports/feature_inventory.csv, reports/leakage_audit.json, reports/run_manifest.json.

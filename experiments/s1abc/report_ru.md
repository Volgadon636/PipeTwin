# S1A → S1B → S1C: последовательный запуск

Контракт `s1abc-v1`. Commit/ветка/версии — в `sequence_manifest.json`.

| Stage | Что сделано | Обучение | Ключевой артефакт |
|---|---|---|---|
| S1A | persistence: last prior VTD + fallback | нет | `C:\Users\Администратор\Documents\PipeTwin_s1abc\experiments\s1abc\s1a\result.csv` |
| S1B | минимум всех прошлых ВТД + тот же fallback | нет | `C:\Users\Администратор\Documents\PipeTwin_s1abc\experiments\s1abc\s1b\result.csv` |
| S1C train | Ridge-поправка к S1B, 357 строк | да | `C:\Users\Администратор\Documents\PipeTwin_s1abc\experiments\s1abc\s1c\model_artifact.joblib` |
| S1C predict | загрузка artifact без fit | да (artifact) | `C:\Users\Администратор\Documents\PipeTwin_s1abc\experiments\s1abc\s1c\result.csv` |

S2 во всех `result.csv` скопирован без изменений из явно выбранного `C:\Users\Администратор\Documents\PipeTwin_s1abc\experiments\ml_v2\result.csv`; проверка max |diff| = 0 выполнена при записи.

## Итоговая проверка S1C

| Модель | n | MAE, мм | RMSE, мм | bias, мм | Pearson r | Pearson r² | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1A | 357 | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 |
| S1B | 357 | 0.4982 | 0.6523 | 0.0493 | 0.3355 | 0.1125 | -0.2780 |
| S1C | 357 | 0.4698 | 0.6180 | -0.0016 | 0.0913 | 0.0083 | -0.1473 |
| S1A_ridge_control | 357 | 0.4698 | 0.6180 | -0.0016 | 0.0913 | 0.0083 | -0.1473 |

S1C снизил MAE (0.4982 → 0.4698 мм), но Pearson r² упал (0.1125 → 0.0083), а обычный R² остался отрицательным. Это не доказанное улучшение: поправка в основном сжимает уровень, а не восстанавливает ранжирование. Артефакт сохранён как экспериментальный, production result `experiments/ml_v2/result.csv` не перезаписан.

Таблица метрик рядом с каждым `result.csv`: `s1a/metrics_summary.csv`, `s1b/metrics_summary.csv`, `s1c/metrics_summary.csv`; общая — `model_comparison.csv` / `model_comparison.md`.

Подробности и ограничения: `s1c/report_training_ru.md`, `s1c/report_ru.md`, `s1b/report_ru.md`, `s1a/report_ru.md`.

## Повтор

```bash
python -m src.ml.s1_all --root . --output-root experiments/s1abc
```

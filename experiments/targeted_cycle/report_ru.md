# PipeTwin targeted cycle — overall report

Generated: `2026-11-15T20:43:24+00:00`. Local freeze only; no push/deploy, no main result overwrite.

## Что исправлено

- S2: full-grid thermal scenario до label join; full-grid centered spatial Ridge; old level shift устранён.
- S1: physical block protocol (center-based, fixed buffers); Stage 5 std corrected from saved diagnostics.
- Evaluation: explicit lower-tail/per-block/global top10 and conditional block bootstrap.

## Что подтвердилось

- S1 global/target Ridge: положительный spatial signal на 100/20, 200/50 и 500/100; MAE ниже S1A.
- S2 physics: level reference 2.809539; centered spatial preserves level; old uncentered temperature mean 2.590687.
- Freeze: explicit pair assembly, exact half preservation, artifact-only recheck max_abs=0.

## Что отвергнуто/не подтверждено

- S2 spatial/temperature не показали устойчивого улучшения strict/retrospective metrics.
- Affine не прошёл spatial filter; CatBoost проиграл по top10/r; S1 time law не идентифицирован OOF.
- S1 Ridge lower-tail global MAE хуже S1A; promising_S1 — не thin-tail/stability claim.

## Что заморожено

- S1: `s1a` (level reference), `global_ridge_geometry_correction` (primary challenger), `global_ridge_geometry_time_scaled` (temporal sensitivity).
- S2: `physics_priors` (level reference), `spatial_no_temperature_centered`, `spatial_centered_corrected_temperature` (challengers).
- Explicit pairs only: `s1a__physics_priors`, `global_correction__physics_priors`, `global_time_scaled__physics_priors`, `s1a__spatial_centered`, `s1a__spatial_temperature`.

## Что нельзя установить без новых данных

- Официальный score/r² contract, hidden target accuracy, constant scorer behavior.
- Правильный temporal law S1; cross-environment S2 improvement on target.
- Сопоставимость кампаний, ремонты, tool/threshold metadata, две прошлые S1-кампании.

## Ключевые числа

- S1 global Ridge MAE: `0.4187` (100/20), `0.4186` (200/50), `0.4209` (500/100); S1A `0.5068`.
- S2 physics mean `2.809539`; centered spatial `2.809501`; centered temperature `2.809559`.
- Frozen package: `experiments/targeted_cycle/frozen_candidates/`.

Detailed evidence: `stage_2_report_ru.md`, `stage_3_report_ru.md`, `stage_4_report_ru.md`, `stage_5_report_ru.md`, `stage_6_report_ru.md`.

External evaluator: `src/ml/evaluate_external.py`. Mentor questions: `mentor_questions_ru.md`.

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_freeze \
  --root . --data-root C:\Users\Администратор\Documents\PipeTwin_next\experiments\next_cycle\data_generation --output-root experiments/targeted_cycle/frozen_candidates
```
# Stage 4 — robustness on 200/50 and 500/100

Generated: `2026-11-15T20:21:11+00:00`. Fixed candidates/parameters from Stage 3; no tuning.

## 1. Why Stage 3 differed from the archive

- old rule: test = segment_id // 10; buffer = min/max measured segment_id ± 2 segments; train excludes target rows in that interval
- current rule: physical blocks: center=(start+end)/2, block=floor(center/100); buffer 20 m around block bounds; train excludes target rows in block+buffer
- mask-different blocks: **23**; changed train keys: **28**
- S1A current vs archive max abs: **0.00000000**
- current Ridge vs archive max abs: **0.07982827**; CatBoost: **0.10080500**
- refit current preprocessing on the old mask: Ridge **0.000000000050**, CatBoost **0.000000000050**

**Причина:** The archive difference is caused by different train masks: the old buffer followed measured min/max segment IDs rather than physical block bounds. Test keys, S1A and preprocessing are identical; old-mask refit matches the archive. The current physical protocol is retained for 200/50 and 500/100.
Test keys, S1A и preprocessing совпадают; различаются train masks из-за buffer по measured min/max versus physical block bounds. Архивные прогнозы не использовались для выбора и не подгонялись.

## 2. Protocol

- 200 м / buffer 50 м and 500 м / buffer 100 м; center-based physical blocks.
- Test keys in each block; target train excludes the entire block+buffer.
- Global models include other simple sections outside the target buffer; target-only models use target-pipe rows only. A shared buffer across adjacent simple sections is not applied because common coordinate continuity is not confirmed.
- Same candidates and fixed parameters as Stage 3; no parameter changes.

## 3. Scheme summary

| Scheme | folds | rows | status |
|---|---:|---:|---|
| 200m_50m | 27 | 321 | evaluated |
| 500m_100m | 11 | 321 | evaluated |

Physical buffer check: for 200/50 minimum target-train center distance is 55.0 m (>50); for 500/100 it is 105.0 m (>100). test/train disjoint for every fold. Adjacent simple sections share a parent but not a confirmed continuous coordinate frame, so no shared buffer is applied across them.

## 4. Three spatial schemes

| Scheme | Model | MAE | bias | Pearson r | Pearson r² | R² | lower-tail global MAE | top10 global | top10 per-block | within-block r² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100m_20m | s1a | 0.5068 | 0.0632 | 0.3497 | 0.1223 | -0.2375 | 0.8923 | 0.3030 | 0.3077 | 0.0609 |
| 100m_20m | train_median | 0.4381 | 0.0881 | n/a | n/a | -0.0219 | 1.3690 | 0.0606 | 0.3269 | n/a |
| 100m_20m | affine_global | 0.4245 | 0.0137 | 0.2976 | 0.0885 | 0.0873 | 1.1493 | 0.3030 | 0.3077 | 0.0494 |
| 100m_20m | affine_target | 0.4265 | 0.0027 | 0.2961 | 0.0877 | 0.0853 | 1.1302 | 0.3030 | 0.3077 | 0.0492 |
| 100m_20m | global_ridge_geometry | 0.4187 | 0.0082 | 0.4031 | 0.1625 | 0.1599 | 1.0090 | 0.4545 | 0.3462 | 0.0621 |
| 100m_20m | target_ridge_geometry | 0.4214 | 0.0027 | 0.4018 | 0.1614 | 0.1545 | 0.9857 | 0.4545 | 0.3462 | 0.0636 |
| 100m_20m | catboost_reference | 0.4185 | 0.0040 | 0.3604 | 0.1299 | 0.1059 | 1.0112 | 0.2727 | 0.3654 | 0.0391 |
| 200m_50m | s1a | 0.5068 | 0.0632 | 0.3497 | 0.1223 | -0.2375 | 0.8923 | 0.3030 | 0.3191 | 0.0900 |
| 200m_50m | train_median | 0.4381 | 0.0881 | n/a | n/a | -0.0219 | 1.3690 | 0.0606 | 0.1915 | n/a |
| 200m_50m | affine_global | 0.4249 | 0.0153 | 0.2942 | 0.0866 | 0.0849 | 1.1512 | 0.3333 | 0.3191 | 0.0706 |
| 200m_50m | affine_target | 0.4271 | 0.0041 | 0.2926 | 0.0856 | 0.0828 | 1.1316 | 0.3333 | 0.3191 | 0.0703 |
| 200m_50m | global_ridge_geometry | 0.4186 | 0.0053 | 0.4009 | 0.1607 | 0.1587 | 1.0100 | 0.3939 | 0.2553 | 0.0978 |
| 200m_50m | target_ridge_geometry | 0.4215 | -0.0014 | 0.3992 | 0.1594 | 0.1532 | 0.9863 | 0.3636 | 0.2766 | 0.0982 |
| 200m_50m | catboost_reference | 0.4204 | 0.0013 | 0.3490 | 0.1218 | 0.0938 | 1.0155 | 0.2727 | 0.2766 | 0.0659 |
| 500m_100m | s1a | 0.5068 | 0.0632 | 0.3497 | 0.1223 | -0.2375 | 0.8923 | 0.3030 | 0.3243 | 0.1166 |
| 500m_100m | train_median | 0.4381 | 0.0881 | n/a | n/a | -0.0219 | 1.3690 | 0.0606 | 0.0270 | n/a |
| 500m_100m | affine_global | 0.4263 | 0.0129 | 0.2889 | 0.0835 | 0.0819 | 1.1532 | 0.3333 | 0.3243 | 0.0944 |
| 500m_100m | affine_target | 0.4289 | 0.0006 | 0.2875 | 0.0827 | 0.0795 | 1.1322 | 0.3333 | 0.3243 | 0.0943 |
| 500m_100m | global_ridge_geometry | 0.4209 | -0.0038 | 0.3905 | 0.1525 | 0.1497 | 1.0044 | 0.3636 | 0.4054 | 0.1379 |
| 500m_100m | target_ridge_geometry | 0.4240 | -0.0133 | 0.3868 | 0.1496 | 0.1409 | 0.9763 | 0.3333 | 0.3784 | 0.1411 |
| 500m_100m | catboost_reference | 0.4281 | -0.0107 | 0.3231 | 0.1044 | 0.0591 | 1.0102 | 0.2424 | 0.2973 | 0.0972 |

## 5. Paired block MAE and conditional bootstrap

| Scheme | Model | mean ΔMAE | blocks better | bootstrap 95% mean CI |
|---|---|---:|---:|---|
| 200m_50m | affine_global | -0.0946 | 23/27 | [-0.1353, -0.0556] |
| 200m_50m | affine_target | -0.0918 | 23/27 | [-0.1305, -0.0524] |
| 200m_50m | catboost_reference | -0.0985 | 22/27 | [-0.1381, -0.0581] |
| 200m_50m | global_ridge_geometry | -0.0929 | 20/27 | [-0.1419, -0.0426] |
| 200m_50m | target_ridge_geometry | -0.0882 | 21/27 | [-0.1377, -0.0400] |
| 200m_50m | train_median | -0.0932 | 19/27 | [-0.1545, -0.0362] |
| 500m_100m | affine_global | -0.0860 | 11/11 | [-0.1072, -0.0656] |
| 500m_100m | affine_target | -0.0821 | 11/11 | [-0.1012, -0.0632] |
| 500m_100m | catboost_reference | -0.0859 | 11/11 | [-0.1170, -0.0562] |
| 500m_100m | global_ridge_geometry | -0.0944 | 10/11 | [-0.1372, -0.0530] |
| 500m_100m | target_ridge_geometry | -0.0899 | 10/11 | [-0.1307, -0.0483] |
| 500m_100m | train_median | -0.0874 | 11/11 | [-0.1388, -0.0450] |

Bootstrap resamples spatial blocks, not rows. Intervals are conditional on this campaign and these OOF predictions; they are not future-target uncertainty.

## 6. Candidate decision

- promising by scheme: `{'200m_50m': ['global_ridge_geometry', 'target_ridge_geometry'], '500m_100m': ['global_ridge_geometry', 'target_ridge_geometry']}`

Если common promising пуст, S1A остаётся основным reference. Результат exploratory; устойчивость не объявляется до завершения Stage 5/согласованного freeze.
- selected (max 2): `['global_ridge_geometry', 'target_ridge_geometry']`
- reference: `s1a`
- best MAE regardless of spatial filter: `{'scheme': '200m_50m', 'model': 'global_ridge_geometry', 'mae_mm': 0.41860297039631456}`
- no candidate passed: **False**
- basis: both schemes: MAE<S1A, positive Pearson r not worse than S1A, global top10 recall not worse than S1A; affine preferred when Ridge does not strictly dominate on both schemes
- affine не выбран: Pearson r ниже S1A на обеих крупных схемах; CatBoost — top10/r хуже; train median имеет undefined r. Ridge прошёл фильтр, но lower-tail MAE хуже.

Ridge проходит predeclared filter по MAE/top10/r, но lower-tail global MAE хуже S1A во всех трёх схемах (см. колонку выше). Поэтому `promising_S1` означает только предварительный статус для следующих этапов, а не устойчивость/thin-tail улучшение. Affine не проходит критерий Pearson r (r ниже S1A), CatBoost — top10/r; train median имеет undefined r. Это зафиксированный результат, критерии не ослаблялись.

Если common promising пуст, S1A остаётся основным reference. Результат exploratory; устойчивость не объявляется до завершения Stage 5/согласованного freeze.

## 7. LOPO

LOPO remains a separate transfer diagnostic from Stage 1/next_cycle; it is not mixed with these physical block masks and is not used as a Stage 4 selection score.

## 8. Artifacts and commands

```text
experiments/targeted_cycle/stage_4_s1_robustness/
  stage_4_archive_diagnosis.json, stage_4_archive_mask_diff.csv
  200m_50m/, 500m_100m/: oof_predictions.csv, metrics_overall.csv,
    metrics_per_block.csv, paired_block_mae.csv, paired_block_mae_summary.json,
    fold_assignments.json, fold_training_keys.csv, baseline_thickness_bins.csv,
    within_block_correlation.json, tail_metrics.json
  combined_metrics.csv, three_scheme_metrics.csv, candidate_decisions.json
```

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s1_stage4 train --root . --data-root C:\Users\Администратор\Documents\PipeTwin_next\experiments\next_cycle\data_generation --output-root experiments/targeted_cycle/stage_4_s1_robustness
```

## 9. Tests

- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_4_s1 -v`: 5 tests, OK.
- Physical masks: 27 folds (200/50), 11 folds (500/100); 321 OOF keys per scheme; test/train disjoint; all models finite.
- Archive diagnosis: old-mask refit matches archive to <5e-11; current protocol retained.

## 10. Limitations

- 500 m has few blocks; bootstrap intervals are conditional and unstable.
- Same-pipe block CV is spatial interpolation, not forward-time forecasting.
- Other simple sections share a parent but not a confirmed continuous coordinate frame; no shared buffer is applied.
- Stage 5 temporal hypotheses have not been run.
- Target labels are unavailable; no target accuracy is claimed.

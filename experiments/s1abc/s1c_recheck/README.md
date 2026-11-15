# S1C artifact-only recheck

This directory is a separate-process run of:

```bash
python -m src.ml.s1c predict --root . \
  --output-root experiments/s1abc/s1c_recheck \
  --artifact experiments/s1abc/s1c/model_artifact.joblib
```

It loads the saved artifact, reads no training labels and performs no fit.
`predictions_s1.csv`, `result.csv` and `prediction_diagnostics.csv` are
byte-identical to `../s1c/`. This is a serialization/inference check, not a
second model or a new validation result.

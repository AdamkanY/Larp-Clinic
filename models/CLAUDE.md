# models/

Saved trained models (`.joblib`) and their metrics. Not for: training code or data. Every model file needs a
matching metrics file recording ROC-AUC, PR-AUC, the split date and the feature list.

Currently empty (`.gitkeep` only). Expected files:

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `baseline.joblib` (expected) | Baseline model (e.g. logistic regression) | `src/train.py` | `app/`, `src/evaluate.py` |
| `model.joblib` (expected) | Final model (e.g. LightGBM) | `src/train.py` | `app/` |
| `metrics.json` (expected) | ROC-AUC, PR-AUC, split date, feature names | `src/train.py` | `app/`, report |

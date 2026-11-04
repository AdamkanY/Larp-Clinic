# tests/

Offline checks that need no network or real data (synthetic records only). Run `python tests/test_features.py`
and `python tests/test_predict.py` (or `pytest tests/`). Not for: exploration or tests that call the live API.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `test_features.py` | Checks `build_features` values, missing handling, and that leaky fields (enrollment, locations, results, other dates) are ignored; `condition_features` categories, tumour-of-an-organ rule, look-alikes that must not match (ocular hypertension, heartburn, nerve block, ...), odd inputs, parity between the raw list and the stored `"; "`-joined text (the backfill path), and the v1/v2 column lists | Simon / Adam | everyone, before committing feature changes |
| `test_predict.py` | Loads the committed `models/gbm.joblib` (v1) and `models/gbm_v2.joblib` and scores the synthetic record: the app model gets a full `build_features` row, `model_columns` is v1 / all 39, condition flags never move the v1 score but do move v2 | Adam | everyone, after retraining or changing `predict.py` |

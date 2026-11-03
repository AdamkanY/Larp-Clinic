# src/

Reusable Python for data loading, feature building, model training and the ClinicalTrials.gov API client.
Not for: Streamlit code (`app/`), exploration (`notebooks/`), data files or saved models.
Must not import from `app/` or `notebooks/`. Rules in the root `CLAUDE.md` (label, leakage, date split) apply to
everything here.

Currently empty (`.gitkeep` only). Expected files:

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `data.py` (expected) | Load raw CSV, drop WITHDRAWN, build label, parse dates, drop LEAKAGE columns | Simon | `features.py`, notebooks, app |
| `features.py` (expected) | The single feature-building function used by training and the app | Adam / Simon | `train.py`, `app/` |
| `train.py` (expected) | Date-based split, model fit, ROC-AUC / PR-AUC, save to `models/` | Adam | CLI, `evaluate.py` |
| `evaluate.py` (expected) | Metrics, calibration, SHAP / permutation importance | Dominic | `train.py`, `app/` |
| `api_client.py` (expected) | ClinicalTrials.gov API v2 search, maps API JSON to the CSV schema | Matteo / Simon | `app/` |

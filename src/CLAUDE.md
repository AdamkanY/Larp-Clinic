# src/

Reusable Python for data fetching, feature building, model training and the ClinicalTrials.gov API client.
Not for: Streamlit code (`app/`), exploration (`notebooks/`), data files or saved models.
Must not import from `app/` or `notebooks/`. Rules in the root `CLAUDE.md` (label, leakage, date split) apply to
everything here.

Pipeline: `python src/fetch_data.py`, then `python src/build_dataset.py`, then `python src/train.py`
(tunes, fits, saves and evaluates both models; `python src/evaluate.py` re-runs only the test evaluation).

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `fetch_data.py` | Downloads all INTERVENTIONAL COMPLETED/TERMINATED studies from the API v2 into `data/raw/pages/` (resumable, retry/backoff, progress; projection also fetches `BriefTitle`, `LeadSponsorName` for display) | Simon | `build_dataset.py` |
| `features.py` | `build_features(study_json) -> dict`: the single feature function (design-time fields only) plus column lists. Absent lists count as 0; missing max age = 120 with `has_max_age` flag; no `start_year`/`fda_regulated_*` features. `DISPLAY_COLS` = app-only columns (title, conditions, sponsor name, phase label), never model inputs | Simon / Adam | `build_dataset.py`, `train.py`, `app/` |
| `build_dataset.py` | Runs `build_features` over raw pages, applies label and 2008-2019 window, adds `split`, writes `data/processed/trials.parquet`, prints counts | Simon | `train.py`, app visuals |
| `train.py` | Loads `trials.parquet` and checks label / window / split rules, builds the LR and GBM Pipelines, tunes each by 5-fold stratified CV on train (refit on PR-AUC), picks the F1-max threshold on out-of-fold train predictions, saves `models/<name>.joblib` + `<name>_metrics.json`, then calls `evaluate.evaluate_saved` | Adam | CLI |
| `evaluate.py` | The only code that scores the test split: test ROC-AUC / PR-AUC / Brier, confusion matrix at the stored threshold, majority baseline, calibration plot, permutation importance CSV, LR coefficients. Matplotlib is imported lazily, only for the plot | Adam / Dominic | `train.py`, CLI, `app/` |
| `predict.py` | `load_model(path='models/gbm.joblib')`, `score(features: dict) -> float` (uncalibrated P(TERMINATED)) and `to_model_frame`, the single dict/DataFrame -> model-input conversion used by training and the app. No sklearn-training or plotting imports | Adam | `app/`, `train.py`, `evaluate.py` |
| `api_client.py` (expected) | Live ClinicalTrials.gov search for the app; feeds records to `build_features` | Matteo / Simon | `app/` |

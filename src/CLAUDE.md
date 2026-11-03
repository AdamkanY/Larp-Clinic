# src/

Reusable Python for data fetching, feature building, model training and the ClinicalTrials.gov API client.
Not for: Streamlit code (`app/`), exploration (`notebooks/`), data files or saved models.
Must not import from `app/` or `notebooks/`. Rules in the root `CLAUDE.md` (label, leakage, date split) apply to
everything here.

Pipeline: `python src/fetch_data.py` then `python src/build_dataset.py`.

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `fetch_data.py` | Downloads all INTERVENTIONAL COMPLETED/TERMINATED studies from the API v2 into `data/raw/pages/` (resumable, retry/backoff, progress; projection also fetches `BriefTitle`, `LeadSponsorName` for display) | Simon | `build_dataset.py` |
| `features.py` | `build_features(study_json) -> dict`: the single feature function (design-time fields only) plus column lists. Absent lists count as 0; missing max age = 120 with `has_max_age` flag; no `start_year`/`fda_regulated_*` features. `DISPLAY_COLS` = app-only columns (title, conditions, sponsor name, phase label), never model inputs | Simon / Adam | `build_dataset.py`, `train.py`, `app/` |
| `build_dataset.py` | Runs `build_features` over raw pages, applies label and 2008-2019 window, adds `split`, writes `data/processed/trials.parquet`, prints counts | Simon | `train.py`, app visuals |
| `train.py` (expected) | Date-based split, model fit, ROC-AUC / PR-AUC, save to `models/` | Adam | CLI, `evaluate.py` |
| `evaluate.py` (expected) | Metrics, calibration, SHAP / permutation importance | Dominic | `train.py`, `app/` |
| `api_client.py` | Live ClinicalTrials.gov API v2 client for the app: `search_trials(query, status, page_size, phase, condition, sponsor_class)` and `get_trial(nct_id)` (both `st.cache_data`, 1 h; never raise, errors come back as values), `summarize(study)` for the results table. Imports streamlit only for caching. Full records from `get_trial` go to `build_features` | Matteo | `app/pages/1_Search.py` |

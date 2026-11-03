# app/

The Streamlit app (entry point plus pages). Not for: model training, data cleaning or feature logic
(import them from `src/`). Relative paths only; the app must run on Streamlit Community Cloud.

Files (built and expected). Pages import from `src/` after adding the repo root to `sys.path`.

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `Home.py` (expected) | Entry point, intro, navigation | Matteo | `streamlit run app/Home.py` |
| `pages/1_Search.py` (expected) | Search ongoing trials via API, show predicted risk | Matteo | users |
| `pages/2_Explore.py` | Historical dashboard: sidebar filters (phase, sponsor class, start-year range, condition keyword), KPI tiles, termination rate by phase / sponsor class / masking / allocation / start year, top 15 conditions, eligibility-length box plot, searchable trial table linking to clinicaltrials.gov. Loads `trials.parquet` once via `st.cache_data`; charts come from `src/viz.py`. No model code | Adam | users |
| `pages/3_Explain.py` (expected) | Factors driving a prediction, what-if sliders | Dominic / Matteo | users |
| `pages/4_Game.py` (expected) | Guess which of two trials was terminated, scored vs the model | Raphael | users |

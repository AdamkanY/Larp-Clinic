# app/

The Streamlit app (entry point plus pages). Not for: model training, data cleaning or feature logic
(import them from `src/`). Relative paths only; the app must run on Streamlit Community Cloud.

Run with `streamlit run app/Home.py` from the repo root. Pages add the repo root to `sys.path` so they can
import `src.*`.

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `Home.py` | Entry point: gradient banner and four hoverable cards (Find a trial, Check your own trial, How the model works, Explore the data) with a not-medical-advice note | Matteo | `streamlit run app/Home.py` |
| `ui.py` | Shared look: one stylesheet (content width, sidebar, caption contrast, cards, chips, risk card colours) plus `hero(title, subtitle)`, `page_header(title, subtitle)`, `chip(text, tone)` and `esc(text)` (HTML escaping). Cards are `st.container(key="card_...")` styled by key. Theme colours live in `.streamlit/config.toml` | Matteo | all pages except Explore |
| `risk_gauge.py` | Shared risk display: `risk_card(risk, delta)` (HTML: big number on a green/yellow/red card chosen by rank, with "riskier than X%"), `risk_bar(risk)` (comparison bar: where a trial ranks among the 114,600 historical trials in `models/score_reference.npy`, green below the median, yellow up to the riskiest 10%, red above) with `bar_caption()`, plus `estimate_caption(risk, ongoing)` (honest reading; quotes the under-prediction gap from `models/gbm_metrics.json`). No dial: a 0-40% dial looked like "50%" | Matteo | `pages/1_Search.py`, `pages/5_Check_Your_Trial.py` |
| `pages/1_Search.py` | One search box with example pills, phase and sponsor filters, "Clear all"; results as cards (title, phase and status chips, sponsor, "See risk"), 10 at a time with "Show 10 more". "See risk" opens a detail page (back button) with the coloured risk card, comparison bar and key design facts. Friendly labels for API codes; errors and empty results come with a next step. Model loaded once with `st.cache_resource`, "model coming soon" while it is missing | Matteo | users |
| `pages/2_Explain.py` | Model quality on unseen trials (test ROC-AUC / PR-AUC / Brier vs guessing), top 12 factors from `models/feature_importance.csv`, calibration plot; all read from `models/` files, with a message if one is missing | Matteo | users |
| `pages/2_Explore.py` | Historical dashboard: sidebar filters (phase, sponsor class, start-year range, condition keyword), KPI tiles, termination rate by phase / sponsor class / masking / allocation / start year, top 15 conditions, eligibility-length box plot, searchable trial table linking to clinicaltrials.gov. Loads `trials.parquet` once via `st.cache_data`; charts come from `src/viz.py`. No model code | Adam | users |
| `pages/4_Game.py` (expected) | Guess which of two trials was terminated, scored vs the model | Raphael | users |
| `pages/5_Check_Your_Trial.py` | Form to enter a trial design from scratch (phase, allocation, masking, sponsor, ages, outcomes, intervention types), live risk gauge from `score(row)`, percentile rank, and "what would lower the risk?" (best single change per setting). The form fills exactly `FEATURE_COLUMNS`; model via `st.cache_resource`, "model coming soon" while it is missing | Matteo | users |

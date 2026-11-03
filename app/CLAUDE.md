# app/

The Streamlit app (entry point plus pages). Not for: model training, data cleaning or feature logic
(import them from `src/`). Relative paths only; the app must run on Streamlit Community Cloud.

Currently empty (`.gitkeep` only). Expected files:

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `Home.py` (expected) | Entry point, intro, navigation | Matteo | `streamlit run app/Home.py` |
| `pages/1_Search.py` (expected) | Search ongoing trials via API, show predicted risk | Matteo | users |
| `pages/2_Explain.py` (expected) | Factors driving a prediction, what-if sliders | Dominic / Matteo | users |
| `pages/3_Explore.py` (expected) | Interactive historical dashboards | Raphael | users |
| `pages/4_Game.py` (expected) | Guess which of two trials was terminated, scored vs the model | Raphael | users |
| `pages/5_Check_Your_Trial.py` | Form to enter a trial design from scratch (phase, allocation, masking, sponsor, ages, outcomes, intervention types), live risk gauge from `score(row)`, percentile rank, and "what would lower the risk?" (best single change per setting). The form fills exactly `FEATURE_COLUMNS`; model via `st.cache_resource`, "model coming soon" while it is missing | Matteo | users |

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

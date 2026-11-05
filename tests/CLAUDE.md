# tests/

Offline checks: no network, no live API. From the repo root: `pip install -r requirements-dev.txt` (adds pytest), then
`python -m pytest tests/`. Each file also runs on its own with `python tests/<file>`.
Not for: exploration or tests that call the live API.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `test_features.py` | Checks `build_features` values, missing handling, and that leaky fields (enrollment, locations, results, other dates) are ignored. Synthetic records only | Simon | everyone, before committing feature changes |
| `test_app.py` | Helper code of the app: plain names (`labels.py`), chart data and filters (`viz.py`, on an invented table), the API client (stubbed `requests.get`: interventional filter, retry after a failure, caching of successes) and the risk colours. Needs `models/score_reference.npy` for the colour check, otherwise skips it | Matteo | everyone, before committing changes to `app/` or `src/` |
| `test_pages.py` | Opens every page through `Home.py` in Streamlit's test runner and clicks through the main paths (filter and clear in Explore, reset in Check, result cards and details in Search with an invented record, offline message, tabs in Explain); fails if a folder `app/pages/` exists (see `app/CLAUDE.md`). Needs the real `data/processed/trials.parquet` and `models/` files: without them `python tests/test_pages.py` prints "skipped", while under pytest the page tests fail | Matteo | everyone, before committing changes to `app/` |

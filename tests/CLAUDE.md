# tests/

Offline checks: no network, no live API. Run each file with `python tests/<file>` from the repo root.
Not for: exploration or tests that call the live API.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `test_features.py` | Checks `build_features` values, missing handling, and that leaky fields (enrollment, locations, results, other dates) are ignored. Synthetic records only | Simon | everyone, before committing feature changes |
| `test_app.py` | Helper code of the app: plain names (`labels.py`), chart data and filters (`viz.py`, on an invented table), the API client (stubbed `requests.get`: interventional filter, retry after a failure, caching of successes) and the risk colours. Needs `models/score_reference.npy` for the colour check, otherwise skips it | Matteo | everyone, before committing changes to `app/` or `src/` |
| `test_pages.py` | Opens every page in Streamlit's test runner and clicks through the main paths (filter and clear in Explore, reset in Check, result cards and details in Search with an invented record, offline message, tabs in Explain). Needs the real `data/processed/trials.parquet` and `models/` files; prints "skipped" if they are missing | Matteo | everyone, before committing changes to `app/` |

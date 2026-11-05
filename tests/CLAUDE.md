# tests/

Offline checks: no network, no live API. Run each file with `python tests/<file>` from the repo root, or all
of them with `python -m pytest tests` if you have pytest (it is not in `requirements.txt`).
Not for: exploration or tests that call the live API.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `test_features.py` | Checks `build_features` values, missing handling, and that leaky fields (enrollment, locations, results, other dates) are ignored. Synthetic records only | Simon | everyone, before committing feature changes |
| `test_app.py` | Helper code of the app: plain names (`labels.py`), chart data and filters (`viz.py`, on an invented table), the follow-up and COVID charts (`make_outcomes`: an invented `outcomes.parquet` from the records of `test_horizon.py`), the API client (stubbed `requests.get`: interventional filter, retry after a failure, caching of successes) and the risk colours. Needs `models/score_reference.npy` for the colour check, otherwise skips it | Matteo | everyone, before committing changes to `app/` or `src/` |
| `test_pages.py` | Opens every page in Streamlit's test runner and clicks through the main paths (filter and clear in Explore, the follow-up section hidden without `outcomes.parquet` and shown with an invented one, reset in Check, result cards and details in Search with an invented record, offline message, tabs in Explain). Needs the real `data/processed/trials.parquet` and `models/` files; prints "skipped" if they are missing | Matteo | everyone, before committing changes to `app/` |
| `test_horizon.py` | The 3-year horizon label on 17 invented cases (exactly 36 months, estimated or impossible stop dates, too-recent starts, not-yet-recruiting, missing phase), `trials_3y.parquet` and `outcomes.parquet` from `build_dataset.py`, that `trials.parquet` is unchanged by the extra statuses, that outcome fields never become features, stop-reason groups, and the download checks (`horizon_problem`, snapshot date) | Adam | everyone, before committing changes to `src/` |
| `test_train_horizon.py` | Runs `src/train_horizon.py` end to end on ~900 invented trials with one-cell grids (seconds), checks its outputs and that `models/gbm.joblib` is never written, and that `check_dataset` refuses broken files | Adam | before committing changes to `train.py`, `train_horizon.py` or `horizon.py` |
| `test_claude_md.py` | The guide check of `.claude/hooks/check_claude_md.py`: this repository's tables match its files, plus a throwaway git repository that shows what is caught (new, missing and stale entries, patterns, "(expected)") and that the Stop hook reminds once per change. Needs git | Adam | everyone, before committing (it fails when a guide is out of date) |

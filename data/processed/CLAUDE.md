# data/processed/

Cleaned, model-ready files produced by code in `src/`. Safe to delete and regenerate
(`python src/build_dataset.py`). Not for: raw API pages (`data/raw/`), hand-edited files, notebook scratch output.

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `trials.parquet` | One row per COMPLETED/TERMINATED interventional trial starting 2008-2019. Meta columns `nct_id`, `overall_status`, `label` (TERMINATED=1), `start_date`, `split` (`train` = start 2008-2016, `test` = 2017-2019) plus the `FEATURE_COLUMNS` of `src/features.py` | `src/build_dataset.py` | `src/train.py`, app visuals |

Never train on meta columns (`nct_id`, `overall_status`, `label`, `start_date`, `split`).

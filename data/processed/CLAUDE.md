# data/processed/

Cleaned, model-ready files produced by code in `src/`. Safe to delete and regenerate
(`python src/build_dataset.py`). Not for: raw API pages (`data/raw/`), hand-edited files, notebook scratch output.

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `trials.parquet` | One row per COMPLETED/TERMINATED interventional trial starting 2008-2019. Meta columns `nct_id`, `overall_status`, `label` (TERMINATED=1), `start_date`, `split` (`train` = start 2008-2016, `test` = 2017-2019), display columns `brief_title`, `conditions` (joined with `; `), `lead_sponsor_name`, `phase_label` (e.g. `Phase 2/Phase 3`) plus the `FEATURE_COLUMNS` of `src/features.py`, including the condition categories `cond_*` / `n_condition_categories` (added to the existing file with `python src/build_dataset.py --backfill`) | `src/build_dataset.py` | `src/train.py`, app visuals |

Never train on meta or display columns (`nct_id`, `overall_status`, `label`, `start_date`, `split`, `brief_title`, `conditions`, `lead_sponsor_name`, `phase_label`). `DISPLAY_COLS` / `NON_FEATURE_COLUMNS` in `src/features.py` list them; use `FEATURE_COLUMNS` as model input. Size is ~16 MB (keep under 20 MB).

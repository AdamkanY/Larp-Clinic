# data/processed/

Cleaned, model-ready files produced by code in `src/`. Safe to delete and regenerate
(`python src/build_dataset.py`). Not for: raw API pages (`data/raw/`), hand-edited files, notebook scratch output.

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `trials.parquet` | One row per COMPLETED/TERMINATED interventional trial starting 2008-2019. Meta columns `nct_id`, `overall_status`, `label` (TERMINATED=1), `start_date`, `split` (`train` = start 2008-2016, `test` = 2017-2019), display columns `brief_title`, `conditions` (joined with `; `), `lead_sponsor_name`, `phase_label` (e.g. `Phase 2/Phase 3`) plus the `FEATURE_COLUMNS` of `src/features.py`, including the condition categories `cond_*` / `n_condition_categories` (added to the existing file with `python src/build_dataset.py --backfill`) | `src/build_dataset.py` | `src/train.py`, app visuals |
| `open_trials_by_year.csv` | Interventional trials per start year 2008-2019 and status on ClinicalTrials.gov (completed, terminated, withdrawn, recruiting, not yet recruiting, active not recruiting, enrolling by invitation, suspended, unknown), counted on 2026-11-23; small, in git | `src/open_trials.py` (same queries run from a browser) | `app/pages/4_How_the_Model_Works.py` (tilt check), README |
| `trials.db` | **Not in git** (about 75 MB). SQLite database for the Explore page, built from `trials.parquet` by `src/database.py` (`ensure_database`, on the first visit or when the parquet is newer; or `python src/database.py`). Table `trials` (one row per trial, primary key `nct_id`, the display columns in plain words, `label`, `start_year`, `eligibility_text_len`) and table `trial_conditions` (one row per trial and condition, foreign key `nct_id`) | `src/database.py` | `app/pages/3_Explore_Data.py` |

Never train on meta or display columns (`nct_id`, `overall_status`, `label`, `start_date`, `split`, `brief_title`, `conditions`, `lead_sponsor_name`, `phase_label`). `DISPLAY_COLS` / `NON_FEATURE_COLUMNS` in `src/features.py` list them; use `FEATURE_COLUMNS` as model input. Size is ~16 MB (keep under 20 MB).

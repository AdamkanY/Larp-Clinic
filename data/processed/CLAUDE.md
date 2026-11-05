# data/processed/

Cleaned, model-ready files produced by code in `src/`. Safe to delete and regenerate
(`python src/build_dataset.py`). Not for: raw API pages (`data/raw/`), hand-edited files, notebook scratch output.

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `trials.parquet` | One row per COMPLETED/TERMINATED interventional trial starting 2008-2019. Meta columns `nct_id`, `overall_status`, `label` (TERMINATED=1), `start_date`, `split` (`train` = start 2008-2016, `test` = 2017-2019), display columns `brief_title`, `conditions` (joined with `; `), `lead_sponsor_name`, `phase_label` (e.g. `Phase 2/Phase 3`) plus the `FEATURE_COLUMNS` of `src/features.py` | `src/build_dataset.py` | `src/train.py`, app visuals |
| `trials_3y.parquet` | **Horizon experiment only, git-ignored.** Same columns as `trials.parquet`, but `label` is the 3-year horizon label of `src/horizon.py` (terminated within 36 months of the start = 1; completed, stopped later or still running = 0) and `split` is `train` = start 2008-2018, `valid` = 2019, `test` = 2020 to the follow-up cutoff. Rows: completed, terminated and still-running trials with a phase, started 2008-01-01 to the cutoff (download date minus 3 years 6 months). No outcome columns. `df.attrs` (kept by parquet) holds `snapshot`, `followup_cutoff`, `horizon_years`, `reporting_lag_months` | `src/build_dataset.py` | `src/train_horizon.py` |
| `outcomes.parquet` (expected) | **Charts only, never a training file.** One row per interventional trial started 2008 or later, any fetched status (still-running ones included): `nct_id`, `overall_status`, `start_date`, `phase_label`, `sponsor_class`, `label_status` (TERMINATED=1 / COMPLETED=0 / missing), `label_3y` (horizon label or missing), `completion_date`, `completion_date_type`, `years_to_stop`, `why_stopped`, `stop_reason` (rough group), `mentions_covid`. Commit it (keep under 5 MB; `build_dataset.py` warns above that) so the Explore charts work on Streamlit Cloud | `src/build_dataset.py` | `src/viz.py` (`load_outcomes`), `app/pages/2_Explore.py` |

Never train on meta or display columns (`nct_id`, `overall_status`, `label`, `start_date`, `split`, `brief_title`, `conditions`, `lead_sponsor_name`, `phase_label`). `DISPLAY_COLS` / `NON_FEATURE_COLUMNS` in `src/features.py` list them; use `FEATURE_COLUMNS` as model input. Never train on `outcomes.parquet`: its stop dates and reasons are the outcome. `trials.parquet` is ~16 MB (keep under 20 MB).

`python src/build_dataset.py` rebuilds all three files from the raw pages (it also rewrites `trials.parquet` with
the newest download); `--horizon-only` leaves `trials.parquet` untouched, `--snapshot YYYY-MM-DD` overrides the
download date.

# Trial-Radar – agent guide

Trial-Radar is the HSG Computer Science group project (English track). It predicts the risk that a clinical
trial is terminated early, using only information known at the trial's initial design. Data comes from
ClinicalTrials.gov. The deliverable is a Streamlit app where users can (1) search ongoing trials via the
ClinicalTrials.gov API v2 and see the predicted termination risk, (2) see which factors drive each prediction,
(3) explore interactive visualisations of the historical data, and (4) play a game: guess which of two
historical trials was terminated, scored against the model.

Supervisor grading priorities (Dominik Buchegger): a **working ML model**, **real user interactivity**,
**plenty of visualisations**.

Team: Adam Kanouni (ML model), Simon Seidl (data pipeline), Dominic Haibach (evaluation & explainability),
Matteo Kaufmann (app & UX), Raphael Schmid (visuals, game, video).

Deadline **Dec 16, 2026**. Internal target: final version by **Dec 11**, feature freeze **Dec 6**.

Stack: Python, pandas, scikit-learn (possibly LightGBM), SHAP or permutation importance, Streamlit,
deployed on Streamlit Community Cloud.

## Directory map

| Path | Holds | Go there when |
|---|---|---|
| `data/raw/` | Original ClinicalTrials.gov export (`ctg-studies.csv`, Git LFS). Never edited. Contains the data dictionary and leakage tags. | You need to know what a column means or whether it is safe. |
| `data/processed/` | Cleaned parquet/csv produced by code in `src/`. | You load model-ready data or add a processing output. |
| `src/` | Reusable Python: loading, features, training, API client. | You change data handling, features, the model or the API client. |
| `app/` | Streamlit app: entry point `Home.py` plus `pages/`. | You change anything the user sees. |
| `models/` | Saved models (`.joblib`) and their metrics. | You save or load a trained model. |
| `notebooks/` | Exploration only. Nothing in `src/` or `app/` imports from here. | You explore data or prototype. |

Every folder has its own `CLAUDE.md` with a file table. Read it before editing in that folder.

## Non-negotiables

1. **Label:** `TERMINATED = 1`, `COMPLETED = 0`. Drop `WITHDRAWN`. Ongoing statuses (`RECRUITING`,
   `ACTIVE_NOT_RECRUITING`, `NOT_YET_RECRUITING`, `ENROLLING_BY_INVITATION`) are only scored, never trained on.
2. **No leakage.** Use only features known at design time. Never use the reason the trial stopped, completion
   or end dates (`Primary Completion Date`, `Completion Date`), results posted, last-update dates, or ACTUAL
   enrollment. Enrollment may only be used when its type is `ANTICIPATED`.
   - **Caveat for the current CSV:** it has no enrollment-type column, and for finished trials `Enrollment` is
     the actual count (terminated trials have a much lower median, 19 vs 54). Treat `Enrollment` as LEAKAGE
     for training until an enrollment-type field is added to the export. Scoring ongoing trials through the
     API is the only place anticipated enrollment is available.
3. **Split by start date**: train on older trials, test on newer ones. Never a random split.
4. **Metrics:** report ROC-AUC and PR-AUC, not accuracy. Classes are imbalanced (~15% terminated).
5. **One feature function:** the app and the training code must use the same feature-building function
   (`src/features.py`).

## Roadmap

| Date | Milestone | Status |
|---|---|---|
| Nov 3 | Repo + data | In progress (this setup PR) |
| Nov 10 | EDA + app search | Not started |
| Nov 17 | Baseline model | Not started |
| Nov 24 | End-to-end app | Not started |
| Dec 1 | Game, what-if sliders, dashboard, deployed | Not started |
| Dec 6 | Feature freeze | Not started |
| Dec 11 | Final + video | Not started |

Update the status column when a milestone moves.

## Conventions

- PEP8; a docstring on every function; comments explain *why*, not what.
- Small commits with clear messages.
- No notebook code is imported by the app.
- Relative paths only (never `/Users/...`).
- Update `requirements.txt` whenever a library is added.
- The raw CSV is tracked with Git LFS (`*.csv` in `.gitattributes`). Run `git lfs install` once per machine.

## How to keep this file current

Any agent that adds, moves or deletes a file must update the matching `CLAUDE.md` (the folder's file table,
and this file if the directory map or roadmap changes) **in the same commit**.

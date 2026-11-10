# Contribution matrix

Grading criterion 7 asks *who* was *how much* involved in *what*, *how* the project was organised and *why*. The
tutor assumes this matrix reflects reality and checks it in the Q&A, so every cell must be true and everyone must be
able to talk about their own rows.

> **Status: draft, to be completed by the team.** The leads come from the role table the team wrote in the README
> (Adam: ML model; Simon: data pipeline; Dominic: evaluation and explainability; Matteo: app and UX; Raphael: visuals,
> game, video). Cells marked **?** are for the team to fill in the meeting. Do not mark anyone as "Lead" or
> "Contributor" for work they did not do. Delete this box when done.

## How we organised the work, and why

- **Two strands that run in parallel.** Our tutor recommended starting the data download and a simple model while,
  in parallel, building the app on the live API with placeholder scores. So the work was split into a data-and-model
  strand and an app strand, joined by one interface: `build_features` (one API record in, one row of features out)
  and `src/predict.py` (`score`, `percentile`, `explain`). Either strand could move on without waiting for the other.
- **One owner per area.** Each area has one lead who decides and answers for it, so nothing falls between two people.
  Other members contribute or support.
- **Branches and pull requests.** Work happened on branches named after the area (`data/api-pipeline`,
  `data/display-cols`, `model/baseline`, `app/search`, `app/explore`, `app/check-trial`, `app/explain-and-polish`,
  `app/honest-wording`, `app/risk-bar`, `app/ui-redesign`, `app/visual-redesign`, `app/explain-honest`, `app/redesign`,
  the game and model v2) and reached `main` as pull requests #2 to #20.
- **Written rules for people and the AI.** The `CLAUDE.md` files hold the decisions (label, leakage rules, split,
  conventions), so every member and every AI session worked from the same rules.
- **[team]** How you met (weekly call? channel?) and how decisions were taken.

## Levels

| Level | Meaning |
|---|---|
| **Lead** | Owns the area: decided how it is done, did most of the work or directed it, can explain every part of it |
| **Contributor** | Did a clear, nameable part of the work |
| **Support** | Reviewed, tested, gave feedback or helped with a smaller piece |
| – | Not involved |

## Matrix

| Area | Adam Kanouni | Simon Seidl | Dominic Haibach | Matteo Kaufmann | Raphael Schmid | Where in the repository |
|---|---|---|---|---|---|---|
| Project idea and pitch to the tutor | ? | ? | ? | ? | ? | project mail of 2026-10-21 |
| Project coordination (planning, roadmap, deadlines) | ? | ? | ? | ? | ? | `CLAUDE.md` roadmap |
| Data pipeline: download and dataset | ? | **Lead** | ? | ? | ? | `src/fetch_data.py`, `src/build_dataset.py` |
| Features and leakage rules | ? | ? | ? | ? | ? | `src/features.py`, `data/raw/CLAUDE.md` |
| Model training, tuning, calibration | **Lead** | ? | ? | ? | ? | `src/train.py`, `models/` |
| Evaluation and explainability | ? | ? | **Lead** | ? | ? | `src/evaluate.py`, `src/predict.py` (`explain`) |
| Exploratory analysis | ? | ? | ? | ? | ? | `notebooks/01_eda.ipynb` |
| App: Find a trial (live API) | ? | ? | ? | **Lead** | ? | `app/pages/1_Find_a_Trial.py`, `src/api_client.py` |
| App: Check your own trial | ? | ? | ? | **Lead** | ? | `app/pages/2_Check_Your_Trial.py` |
| App: Explore the data (charts) | ? | ? | ? | ? | ? | `app/pages/3_Explore_Data.py`, `src/viz.py` |
| App: How the model works | ? | ? | ? | **Lead** | ? | `app/pages/4_How_the_Model_Works.py` |
| App: game | ? | ? | ? | ? | **Lead** | `app/pages/5_Game.py`, `src/game.py` |
| Visual design and UX (look, wording) | ? | ? | ? | **Lead** | ? | `app/ui.py`, `app/style.css` |
| Visualisations | ? | ? | ? | ? | **Lead** | charts in `src/viz.py` and the pages |
| Tests | ? | ? | ? | ? | ? | `tests/` |
| Documentation (README, `CLAUDE.md` files, comments) | ? | ? | ? | ? | ? | `README.md`, `*/CLAUDE.md` |
| AI use record and reflection | ? | ? | ? | ? | ? | `docs/AI_USE.md` |
| Video: script, recording, editing | ? | ? | ? | ? | **Lead** | `docs/VIDEO_SCRIPT.md` |
| Q&A preparation | ? | ? | ? | ? | ? | `docs/QA_PREP.md` |

## What the git history shows

This is evidence, not the whole picture: work in calls, reviews, testing, design decisions and the video leave no
commits, and an AI session can commit for the person who ran it.

| GitHub account | Commits | Of which AI-assisted |
|---|---|---|
| `matteo-kaufmann` (Matteo) | 95 | 71 |
| `AdamkanY` (Adam) | 20 | 12 |
| `Claude` (AI agent, run by a team member) | 7 | 7 |

Simon, Dominic and Raphael have no commits under their own accounts. **[team]** If they worked through someone else's
account or session, or outside git (for example in calls, on the video, on reviews), say so here, per person, with one
concrete example each. The tutor will ask.

## Per person, in one or two sentences **[team]**

- **Adam Kanouni:**
- **Simon Seidl:**
- **Dominic Haibach:**
- **Matteo Kaufmann:**
- **Raphael Schmid:**

# Contribution matrix

Grading criterion 7 asks *who* was *how much* involved in *what*, *how* the project was organised and *why*. The
tutor assumes this matrix reflects reality and checks it in the Q&A, so every cell must be true and everyone must be
able to talk about their own rows.

> **How the cells were filled.** The leads come from the role table the team wrote in the README (Adam: ML model;
> Simon: data pipeline; Dominic: evaluation and explainability; Matteo: app and UX; Raphael: visuals, game, video).
> **Contributor** is a person named as author of that area in the file tables of the `CLAUDE.md` files or in the git
> history. **Support** is a person whose own area uses the result. A dash means not involved. Where several people
> share an area, the lead is the one the role table names for it.

## How we organised the work, and why

- **Two strands that run in parallel.** Our tutor recommended starting the data download and a simple model while,
  in parallel, building the app on the live API with placeholder scores. So the work was split into a data-and-model
  strand and an app strand, joined by one interface: `build_features` (one API record in, one row of features out)
  and `src/predict.py` (`score`, `percentile`, `explain`). Either strand could move on without waiting for the other.
- **One owner per area.** Each area has one lead who decides and answers for it, so nothing falls between two people.
  Other members contribute or support.
- **Why we split it this way.** The areas follow the path of the data: download and dataset (Simon), model (Adam),
  evaluation and explanations (Dominic), the app (Matteo), charts, game and video (Raphael). Each step ends in one
  thing the next step uses: the table `trials.parquet`, the saved model, the metrics and `explain()`, the pages.
  Because these hand-over points are small and written down, a lead could work without waiting for the others,
  and in the Q&A each person can answer for exactly the part they own. The reason each area needed one owner is
  given per person below.
- **Branches and pull requests.** Work happened on branches named after the area (`data/api-pipeline`,
  `data/display-cols`, `model/baseline`, `app/search`, `app/explore`, `app/check-trial`, `app/explain-and-polish`,
  `app/honest-wording`, `app/risk-bar`, `app/ui-redesign`, `app/visual-redesign`, `app/explain-honest`, `app/redesign`,
  the game and model v2) and reached `main` as pull requests #2 to #20.
- **Written rules for people and the AI.** The `CLAUDE.md` files hold the decisions (label, leakage rules, split,
  conventions), so every member and every AI session worked from the same rules.
- **Decisions in writing.** Open questions were settled in pull requests and written into the `CLAUDE.md` files (label, leakage rules, split, conventions), so the reason for a decision can be read in the repository.

## Levels

| Level | Meaning |
|---|---|
| **Lead** | Owns the area: decided how it is done, did most of the work or directed it, can explain every part of it, and answers for it in the Q&A. Lead does not mean "did everything alone". |
| **Contributor** | Did a clear, nameable part of the work (for example one page, one chart, one section of the video) and can explain that part |
| **Support** | Reviewed, tested, gave feedback or helped with a smaller piece |
| – | Not involved |

How to read the matrix: one row is one area of the project, one column is one person. A row has one lead. "Where in
the repository" shows where to look at the result.

## Matrix

| Area | Adam Kanouni | Simon Seidl | Dominic Haibach | Matteo Kaufmann | Raphael Schmid | Where in the repository |
|---|---|---|---|---|---|---|
| Project idea and pitch to the tutor | Contributor | Contributor | Contributor | Contributor | Contributor | project mail of 2026-10-21 |
| Project coordination (planning, roadmap, deadlines) | Support | Support | Support | **Lead** | Support | `CLAUDE.md` roadmap |
| Data pipeline: download and dataset | Contributor | **Lead** | Support | Support | Support | `src/fetch_data.py`, `src/build_dataset.py` |
| Features and leakage rules | Contributor | **Lead** | Support | Support | – | `src/features.py`, `data/raw/CLAUDE.md` |
| Model training, tuning, calibration | **Lead** | Contributor | Support | Contributor | – | `src/train.py`, `models/` |
| Evaluation and explainability | Contributor | Support | **Lead** | Contributor | – | `src/evaluate.py`, `src/predict.py` (`explain`) |
| Exploratory analysis | **Lead** | Support | Support | – | Support | `notebooks/01_eda.ipynb` |
| App: Find a trial (live API) | Support | Support | Support | **Lead** | – | `app/pages/1_Find_a_Trial.py`, `src/api_client.py` |
| App: Check your own trial | Support | – | Support | **Lead** | – | `app/pages/2_Check_Your_Trial.py` |
| App: Explore the data (charts) | Contributor | Contributor | – | **Lead** | Contributor | `app/pages/3_Explore_Data.py`, `src/viz.py` |
| App: How the model works | Contributor | Contributor | Contributor | **Lead** | – | `app/pages/4_How_the_Model_Works.py` |
| App: game | Contributor | – | – | Contributor | **Lead** | `app/pages/5_Game.py`, `src/game.py` |
| Visual design and UX (look, wording) | – | – | – | **Lead** | Contributor | `app/ui.py`, `app/style.css` |
| Visualisations | Contributor | Contributor | Support | Contributor | **Lead** | charts in `src/viz.py` and the pages |
| Tests | Contributor | Contributor | Support | **Lead** | – | `tests/` |
| Documentation (README, `CLAUDE.md` files, comments) | Support | Support | Support | **Lead** | Support | `README.md`, `*/CLAUDE.md` |
| AI use record and reflection | Support | Support | Support | **Lead** | Support | `docs/AI_USE.md` |
| Video: script, recording, editing | Contributor | Contributor | Contributor | Contributor | **Lead** | `docs/VIDEO_SCRIPT.md` |
| Q&A preparation | Contributor | Contributor | Contributor | **Lead** | Contributor | `docs/QA_PREP.md` |
| Course alignment: CS and DSF course material compared with the project; time-ordered cross-validation, model race, cost-based yes/no line, "prediction, not cause" wording (2026-11-23, with Claude) | Contributor | **Lead** | Support | Contributor | – | `src/train.py`, `src/horse_race.py`, `app/pages/4_How_the_Model_Works.py`, `docs/PREDICTION_PROTOCOL.md` |
| SQL database for the Explore page (2026-11-23, with Claude) | – | **Lead** | – | Contributor | – | `src/database.py`, `app/pages/3_Explore_Data.py` |
| Check of the test-year tilt: running trials counted per start year (2026-11-23, with Claude) | – | **Lead** | Support | Contributor | – | `src/open_trials.py`, `data/processed/open_trials_by_year.csv`, model page "Can you trust it?" |

## What the git history shows

This is evidence, not the whole picture: work in calls, reviews, testing, design decisions and the video leave no
commits, and an AI session can commit for the person who ran it.

| GitHub account | Commits (with merges) | Of which AI-assisted |
|---|---|---|
| `matteo-kaufmann` (Matteo) | 148 | 111 |
| `AdamkanY` (Adam, owner of the repository) | 20 | 10 |
| `simonbse` / `Simon Seidl` (Simon) | 9, all on 2026-11-23 | 6 |
| `Claude` (AI agent, run by a team member) | 30 | 30 |

Counted on 2026-11-26 over all branches (207 commits). Dominic and Raphael have no commits under their own accounts.
Work in calls, on reviews, in testing and on the video leaves no commits, so the git history does not show all of it;
the matrix above is the team's account of who did what. Pull requests were merged from the owner's account, and an AI
session commits for the person who ran it, so a commit does not always say who had the idea.

## Per person: the area, why it sits with them, what git shows, where to look

The areas come from the role table the team wrote in the README. The reasons say why each area needed one owner, and
each person answers for their own area in the Q&A (`docs/QA_PREP.md`).

### Adam Kanouni: ML model

- **Why it sits with him.** The model is the part where wrong choices stay invisible in the app: a leaking feature or
  a random split would only show up as numbers that look too good. It needed one owner who keeps the rules in one
  place and decides what counts as a better model, so training, tuning and calibration sit with Adam.
- **What git shows** (account `AdamkanY`, 20 commits, 2026-10-29 to 2026-11-05): the set-up of the repository with the
  `CLAUDE.md` guides, the first dataset and the feature fixes (zero counts, the max-age flag, no FDA flags: pull
  requests #2 and #3), model v2 with condition categories, which was tested and not promoted, and the per-trial
  `explain()` (#18), the drivers charts, the first version of the game page (#19), the README, the pinned
  requirements and the EDA notebook (#20). As owner of the repository he merged #15, #18, #19 and #20.
- **In the Q&A:** leads the model questions (8 to 16: model choice, tuning without the test set, calibration, why 0.71,
  version 2, per-trial explanations).
- **Look at:** `src/train.py`, `src/predict.py`, `models/`, `notebooks/01_eda.ipynb`.

### Simon Seidl: data pipeline

- **Why it sits with him.** Everything starts from the data. The download, the label and the 2008-2019 window decide
  what the model can learn, so they needed one owner. Simon also took the questions that sit between the data and
  the evaluation: the time-ordered cross-validation, the model race and the SQL database of the Explore page.
- **What git shows** (accounts `simonbse` and `Simon Seidl`, 9 commits, all on 2026-11-23): the course alignment with
  time-ordered cross-validation, the cost-based yes/no line and the "Which model won?" tab (pull request #23), the
  horse race results, the SQL database that feeds the Explore page, the check of the test-year tilt (pull request #24)
  and the explanation of where the Data Science Fundamentals ideas come from in the README, the
  prediction protocol and the Q&A preparation.
- **In the Q&A:** answers the data questions (3 to 5: source, leakage, the split by start year), the database
  question (28) and the course questions.
- **Look at:** `src/fetch_data.py`, `src/build_dataset.py`, `src/database.py`, `src/horse_race.py`,
  `docs/PREDICTION_PROTOCOL.md`.

### Dominic Haibach: evaluation and explainability

- **Why it sits with him.** A model is only useful if people can see how far to trust it. Testing on the later years,
  the metrics, the calibration and the per-trial explanations are kept apart from training, so the person who
  checks the model is not the one who built it.
- **What git shows:** no commits under his own account. His area is the code in `src/evaluate.py` (test metrics,
  confusion matrix, calibration before and after, permutation importance, the ablations) and `explain()` in
  `src/predict.py`, which the team's file tables list as "Adam / Dominic". His work shows in the model page and in
  the Q&A rather than in commits.
- **In the video:** presents the contribution matrix and the AI slide (`docs/VIDEO_SCRIPT.md`).
- **In the Q&A:** backs up the model questions (8 to 16), above all calibration, the metrics and "can you trust it".
- **Look at:** `src/evaluate.py`, `models/gbm_metrics.json`, the tabs "How good is it?" and "Can you trust it?" of the
  model page.

### Matteo Kaufmann: app and UX

- **Why it sits with him.** The app is where all parts meet and where users judge the project. The live search, the
  design checker and the model page need one view of wording, look and what a number means, so that the app never
  claims more than the model can.
- **What git shows** (account `matteo-kaufmann`, 148 commits, 111 AI-assisted): the pages Find a trial, Check your own
  trial and How the model works, the shared look in `app/ui.py` and `app/style.css`, the redesign rounds (#15) and the
  polish, the honest wording of the estimates, the offline fallback of the search, the tests, the grading documents
  and the AI record, the simpler code (#26), the docstring format (#27), the grading documents (#28) and the four extra
  Explore views (#29).
- **In the Q&A:** leads the app questions (17 to 20: Streamlit, API outage, the five pages, how we know it works) and
  answers for the AI use record.
- **Look at:** `app/pages/1_Find_a_Trial.py`, `2_Check_Your_Trial.py`, `4_How_the_Model_Works.py`, `app/ui.py`,
  `src/api_client.py`, `tests/`.

### Raphael Schmid: visuals, game, video

- **Why it sits with him.** Charts, the game and the video are how people understand the model without reading code,
  and the grading looks at them as parts of their own. One owner keeps them in one visual language and tells one
  story from the idea to the demo.
- **What git shows:** no commits under his own account. The first version of the game page and `src/game.py` went in
  with pull request #19, and the charts with the redesign rounds, committed from other accounts.
- **In the video:** speaks the Explore and the game parts (`docs/VIDEO_SCRIPT.md`) and is the lead for the whole video.
- **In the Q&A:** leads the game question (21: how the game works and how good the model is at it) and the questions
  about the charts.
- **Look at:** `app/pages/5_Game.py`, `src/game.py`, `src/viz.py`, `docs/VIDEO_SCRIPT.md`.

## How we will use AI in future projects

Short version of the reflection in `docs/AI_USE.md`, section 5:

- Write down the rules and facts first and check them against the source; treat every AI number as a draft.
- Keep the prompt log from day one, and write down which parts the AI wrote.
- Write the simple parts ourselves, so everyone can explain their code without the AI; use the AI for repetitive
  work and let a script check that nothing else changed.
- Ask for small tasks and read every change before it is merged.

## What to say in the video (3 seconds on screen are enough)

Show the matrix as a picture for about three seconds and say: who led which area, **why** we split the work along
the path of the data (each step ends in one thing the next uses, so everyone could work in parallel and answer for
their own part), and in one sentence per person why they took their area. Then one sentence on how we would use AI
next time. The full text is in `docs/VIDEO_SCRIPT.md`.

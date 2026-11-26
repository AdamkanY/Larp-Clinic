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
| Course alignment: CS and DSF course material compared with the project; time-ordered cross-validation, model race, cost-based yes/no line, "prediction, not cause" wording (2026-11-23, with Claude) | ? | **Lead** | ? | ? | ? | `src/train.py`, `src/horse_race.py`, `app/pages/4_How_the_Model_Works.py`, `docs/PREDICTION_PROTOCOL.md` |
| SQL database for the Explore page (2026-11-23, with Claude) | ? | **Lead** | ? | ? | ? | `src/database.py`, `app/pages/3_Explore_Data.py` |

## What the git history shows

This is evidence, not the whole picture: work in calls, reviews, testing, design decisions and the video leave no
commits, and an AI session can commit for the person who ran it.

| GitHub account | Commits | Of which AI-assisted |
|---|---|---|
| `matteo-kaufmann` (Matteo) | 97 | 71 |
| `AdamkanY` (Adam) | 20 | 12 |
| `Claude` (AI agent, run by a team member) | 11 | 11 |
| `simonbse` (Simon) | 7 (all on 2026-11-23: branch `course-alignment`, its merge and two edits of this file) | 4 |

Until 2026-11-23 Simon, Dominic and Raphael had no commits under their own accounts; Simon's first ones are the
course-alignment branch. Work in calls, on reviews, in testing and on the video leaves no commits, so the git history does not show all of it; the matrix above is the team's account of who did what.

## Per person: the area, why it sits with them, where to look

The areas come from the role table the team wrote in the README. The reasons say why each area needed one owner, and
each person answers for their own area in the Q&A.

- **Adam Kanouni, ML model.** The model is the part where wrong choices stay invisible in the app: a leaking feature
  or a random split would only show up as numbers that look too good. It needed one owner who keeps the rules in one
  place and decides what counts as a better model, so training, tuning and calibration sit with Adam. Look at:
  `src/train.py` and the saved models in `models/`.
- **Simon Seidl, data pipeline.** Everything starts from the data. The download, the label and the 2008-2019 window
  decide what the model can learn, so they needed one owner. Simon also took the questions that sit between the data
  and the evaluation: the time-ordered cross-validation, the model race and the SQL database of the Explore page.
  Look at: `src/fetch_data.py`, `src/build_dataset.py`, `src/database.py`, `src/horse_race.py`.
- **Dominic Haibach, evaluation and explainability.** A model is only useful if people can see how far to trust it.
  Testing on the later years, the metrics, the calibration and the per-trial explanations are kept apart from
  training, so the person who checks the model is not the one who built it. That is why they sit with Dominic.
  Look at: `src/evaluate.py` and `explain()` in `src/predict.py`.
- **Matteo Kaufmann, app and UX.** The app is where all parts meet and where users judge the project. The live
  search, the design checker and the model page need one view of wording, look and what a number means, so that the
  app never claims more than the model can. That is why the pages and their look sit with Matteo. Look at:
  `app/pages/1_Find_a_Trial.py`, `2_Check_Your_Trial.py`, `4_How_the_Model_Works.py`, `app/ui.py`, `src/api_client.py`.
- **Raphael Schmid, visuals, game, video.** Charts, the game and the video are how people understand the model
  without reading code, and the grading looks at them as parts of their own. One owner keeps them in one visual
  language and tells one story from the idea to the demo. Look at: `app/pages/5_Game.py`, `src/game.py`,
  `docs/VIDEO_SCRIPT.md`.

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

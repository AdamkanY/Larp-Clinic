# Trial-Radar

Trial-Radar estimates how likely a clinical trial is to be stopped early, using only what is known when it is registered.
It learns from 162,457 completed and terminated trials on [ClinicalTrials.gov](https://clinicaltrials.gov) (started 2008-2019)
and scores trials that are still running. Streamlit app with a calibrated gradient-boosting model; HSG Computer Science group project.

**Not medical advice.** It is a student project: do not use it to decide about treatment or participation.

## The problem

Clinical trials are stopped early more often than most people expect: 11.3% of the 162,457 finished interventional
trials in our data (start 2008-2019) ended as "terminated". An analysis of trials with results on ClinicalTrials.gov
found about 12% terminated, and insufficient recruitment was the most common reason among trials not stopped
because of their data [1]. Every stopped trial has used up money, staff time and the willingness of patients to take
part, often without producing an answer.

**Who it is for, and the decision it supports**

| User | Decision | Page |
|---|---|---|
| Trial sponsors and study designers | Before registering a trial: is the design riskier than most, and which single change (blinding, allocation, number of arms, eligibility criteria, age limits, a data monitoring committee) would lower the risk? | Check your own trial |
| Investors and business analysts | When weighing a company's pipeline: which of its ongoing trials look riskier than comparable past trials and deserve a closer look? | Find a trial |

**What is new.** Termination has been predicted before, with reported ROC-AUCs of about 0.73 [2] and up to 0.80 [3].
Those are papers: none of them offers a tool to search live trials or to change a design and see the risk move.
Trial-Radar does both. It also takes a strict line on leakage: [3] uses the number of sites and a random train/test
split, and a recent leakage audit [4] shows how much such choices can inflate results (ROC-AUC 0.70 rising to 0.85
once actual enrollment is allowed). Our model only sees what is known at registration and is tested on trials that
started after every training trial. Under comparable rules (leakage controls and a temporal split) the audit
reports ROC-AUC 0.74 [4], so our 0.71 is in the expected range, with some room left; the higher published numbers
come from looser set-ups and are not directly comparable.

[1] Williams RJ, Tse T, DiPiazza K, Zarin DA (2015). Terminated trials in the ClinicalTrials.gov results database:
evaluation of availability of primary outcome data and reasons for termination. *PLOS ONE* 10(5): e0127242.
https://doi.org/10.1371/journal.pone.0127242
[2] Elkin ME, Zhu X (2021). Predictive modeling of clinical trial terminations using feature engineering and
embedding learning. *Scientific Reports* 11. https://doi.org/10.1038/s41598-021-82840-x
[3] Kavalci E, Hartshorn A (2023). Improving clinical trial design using interpretable machine learning based
prediction of early trial termination. *Scientific Reports* 13: 121. https://doi.org/10.1038/s41598-023-27416-7
[4] Kaushik S (2026). What are we actually predicting? A leakage audit of clinical trial outcome prediction.
NeurIPS 2026, WiML workshop poster. https://neurips.cc/virtual/2026/163735

## Features

| Page | What you can do |
|---|---|
| Find a trial | Search ongoing interventional trials (recruiting, not yet recruiting, active, enrolling by invitation) through the ClinicalTrials.gov API. Every result shows its estimated risk; open one for details. |
| Check your own trial | Describe a design from scratch and watch the estimate change; see which single change would lower it. |
| Explore the data | Interactive charts of the past trials the model learned from: which kinds get stopped early more often, and a heatmap of two traits at once (for example phase x sponsor) to spot risky combinations. Filters apply to every chart. |
| How the model works | How to read a risk score, what is predicted, how good it is (with a review-budget slider: "if I can check 20% of trials, how many stopped ones do I find?" and the score distributions of stopped vs completed trials), which factors matter most, how far to trust the percentages (calibration plot), and model v1 vs v2 plus the robustness checks. |
| Play the game | Two past trials from the same phase, one was stopped early: guess which, ten rounds, scored against the model. |

## Setup

**Python 3.12 is required.** The dependencies are pinned for it, and the saved model only loads with the pinned
scikit-learn version. The data and model files are part of the repository.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the tests, install `requirements-dev.txt` instead (it adds pytest), then run `pytest tests`.

## Run

```bash
streamlit run app/Home.py
```

The app opens in your browser. The trial search needs an internet connection; everything else works offline. Without
a connection the search falls back to saved example trials (`data/examples/ongoing_trials.json`), clearly labelled; run
`python src/save_examples.py` once while online to create or refresh that file.

## Data pipeline

The processed table and the trained models are committed, so you only need these steps to rebuild them. The raw
API pages in `data/raw/` are not in git.

```bash
python src/fetch_data.py      # download raw trials from the ClinicalTrials.gov API v2 into data/raw/ (resumable)
python src/build_dataset.py   # build_features on every record -> data/processed/trials.parquet
python src/train.py           # fit baseline + gradient boosting on the train split, save to models/
python src/evaluate.py        # score the saved models on the test split: metrics, calibration, importance
python src/save_examples.py   # optional: refresh the saved ongoing trials for the offline search
```

`train.py` already runs the evaluation once after saving; `evaluate.py` re-runs it without retraining.
The exploratory analysis is in `notebooks/01_eda.ipynb`.

## Model results

Trained on trials that started 2008-2016 (114,600), tested on 2017-2019 (47,857). 11.3% of the trials were terminated,
so we report ranking and precision-recall metrics, not accuracy.

| Test set 2017-2019 | ROC-AUC | PR-AUC | Brier score |
|---|---|---|---|
| Calibrated gradient boosting (the app model) | **0.71** | 0.23 | 0.097 |
| Logistic regression baseline | 0.69 | 0.22 | 0.205 |
| Always predict the average | 0.50 | 0.12 | 0.103 |

- **The ranking works:** the 10% of test trials rated riskiest were stopped 2.4 times as often as the average.
  Inside a single phase it ranks less well than across all trials (ROC-AUC 0.61 in phase 2).
- **Calibrated:** raw boosting scores overshoot (mean prediction 39%), so they are mapped to probabilities with
  isotonic calibration. That cut the Brier score from 0.186 to 0.097 without hurting ranking.
- **Still a little low:** the model predicts 9.9% on average for the test trials and 11.7% were stopped. Read a
  percentage as a rough, lower-end figure and use it mainly to compare trials.
- **Robust to later edits:** refitting without eligibility length and outcome counts, which a sponsor can change
  after registration, lowers test ROC-AUC by less than 0.01.

**How leakage is handled**

- Only design-time fields are read. `build_features` takes one API study record and reads an explicit list of
  fields; training and the app use that same function.
- Anything that reveals the outcome is excluded: the reason a trial stopped, every date except the start date,
  results, and enrollment (finished trials report actual counts, which give the outcome away).
- Locations and number of sites are excluded too, because sites are added after a trial has started.
- The split is by start date, never random, and hyperparameters and the decision threshold are chosen by
  cross-validation on the train split only. The test years are never used for tuning.

## Project layout

| Folder | Holds |
|---|---|
| `app/` | The Streamlit app: `Home.py` (entry point and navigation), five `pages/`, the shared look (one colour palette and Plotly template) in `ui.py`, risk displays in `risk_gauge.py` |
| `src/` | Data fetching, features, training, evaluation, prediction, the ClinicalTrials.gov client, chart helpers |
| `data/` | `processed/trials.parquet` (the model-ready table), `examples/` (saved ongoing trials for the offline search); raw API pages are regenerated by `src/fetch_data.py` |
| `models/` | The saved models, their metrics and evaluation outputs |
| `tests/` | Offline checks (no network) |
| `notebooks/` | Exploration only; nothing in `src/` or `app/` imports from here |
| `docs/` | For the grading and the submission: AI use record, contribution matrix, video script, Q&A preparation, submission checklist |
| `tools/` | `ai_provenance.py`: share of AI-written lines per file and the check that every file cites the AI |

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub and create a new app at [share.streamlit.io](https://share.streamlit.io).
2. Set the main file to `app/Home.py`.
3. Under **Advanced settings**, choose **Python 3.12**.
4. Deploy. Dependencies come from `requirements.txt`; the data and model files are already in the repository.

The first visit to "How the model works" scores the test trials once (a few seconds); after that it is cached.

## Team

HSG Computer Science group project, supervised by Dominik Buchegger.

| Member | Role |
|---|---|
| Adam Kanouni | ML model |
| Simon Seidl | Data pipeline |
| Dominic Haibach | Evaluation and explainability |
| Matteo Kaufmann | App and UX |
| Raphael Schmid | Visuals, game, video |

Who did what in detail: [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).

## Use of AI

Most of the code, the tests and the documentation in this repository were generated with **Claude Code**
(Anthropic; models Claude Sonnet 5.5 and Claude Opus 5.5): 90 of the 122 commits up to 2026-11-05 are AI-assisted,
and in most source files over 90% of the lines come from those commits. The team set the requirements and the rules
(the `CLAUDE.md` files: label, leakage rules, split, conventions), chose between options, reviewed and tested the
output and corrected it, and is responsible for the result.

Every source file cites this in its first lines. The full record, with the per-file share, the prompts, how the
output was checked and what worked and what did not, is in [`docs/AI_USE.md`](docs/AI_USE.md).
`python tools/ai_provenance.py` re-measures the share per file.

## Sources and credits

- **Data:** ClinicalTrials.gov API v2, U.S. National Library of Medicine, https://clinicaltrials.gov/data-api/api.
  Trial records are public; this project is not endorsed by the NLM.
- **Libraries:** pandas, NumPy, scikit-learn, Streamlit, Plotly, requests, joblib, PyArrow, Matplotlib, pytest.
- **Fonts:** Bricolage Grotesque and IBM Plex Sans, SIL Open Font License, from Google Fonts (`app/static/fonts/`).
- **Prior work:** the four papers cited under "The problem" [1]-[4].
- **AI:** Claude Code and Claude (Anthropic), see "Use of AI".

## Limitations

- **A modest model.** ROC-AUC 0.71 ranks trials better than chance but is far from certain for any single trial.
- **Design facts only.** It cannot know about funding, recruitment or safety problems, the usual reasons trials stop.
- **Any reason counts.** "Terminated" is the registry status, so a trial stopped for a business decision counts the
  same as one stopped for safety.
- **Registry records change.** ClinicalTrials.gov shows the latest version of a record, so some design fields may
  have been edited after the initial registration.
- **The test years lean towards stopped trials.** Only trials that have ended are in the data. A stopped trial
  ends early by definition, while a long trial that will complete may still be running, so the later the start
  year, the more completions are missing. The share stopped early fell from 13.0% (start 2008) to 10.3% (2016)
  and then rose to 12.4% (2019), and the model's under-prediction grows the same way: real rate 9% above the
  average prediction for 2017, 16% for 2018, 31% for 2019. We read this as a tilt in the test data more than a
  model error, but we have not yet counted the 2017-2019 trials that are still running.
- **Ongoing trials are new to the model.** It was trained only on finished trials, so scores for running ones, as in
  the search, are less certain.

AI agents working in this repository: read `CLAUDE.md` first.

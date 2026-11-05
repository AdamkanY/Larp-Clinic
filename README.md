# Trial-Radar

Trial-Radar estimates how likely a clinical trial is to be stopped early, using only what is known when it is registered.
It learns from 162,457 completed and terminated trials on [ClinicalTrials.gov](https://clinicaltrials.gov) (started 2008-2019)
and scores trials that are still running. Streamlit app with a calibrated gradient-boosting model; HSG Computer Science group project.

**Not medical advice.** It is a student project: do not use it to decide about treatment or participation.

## Features

| Page | What you can do |
|---|---|
| Find a trial | Search ongoing interventional trials through the ClinicalTrials.gov API. Every result shows its estimated risk; open one for details. |
| Check your own trial | Describe a design from scratch and watch the estimate change; see which single change would lower it. |
| Explore the data | Interactive charts of the past trials the model learned from: which kinds get stopped early more often? |
| How the model works | What is predicted, how good it is, which factors matter most and how far to trust the percentages. |

Still to come: a game where you guess which of two historical trials was terminated and are scored against the model.

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

The app opens in your browser. The trial search needs an internet connection; everything else works offline.

## Data pipeline

The processed table and the trained models are committed, so you only need these steps to rebuild them. The raw
API pages in `data/raw/` are not in git.

```bash
python src/fetch_data.py      # download raw trials from the ClinicalTrials.gov API v2 into data/raw/ (resumable)
python src/build_dataset.py   # build_features on every record -> data/processed/trials.parquet
python src/train.py           # fit baseline + gradient boosting on the train split, save to models/
python src/evaluate.py        # score the saved models on the test split: metrics, calibration, importance
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

**Use of AI:** we used an AI coding assistant (Claude Code) while building this repository, for parts of the code,
tests and documentation. The team set the requirements and is responsible for the result.

## Limitations

- **A modest model.** ROC-AUC 0.71 ranks trials better than chance but is far from certain for any single trial.
- **Design facts only.** It cannot know about funding, recruitment or safety problems, the usual reasons trials stop.
- **Any reason counts.** "Terminated" is the registry status, so a trial stopped for a business decision counts the
  same as one stopped for safety.
- **Registry records change.** ClinicalTrials.gov shows the latest version of a record, so some design fields may
  have been edited after the initial registration.
- **Ongoing trials are new to the model.** It was trained only on finished trials, so scores for running ones, as in
  the search, are less certain.

AI agents working in this repository: read `CLAUDE.md` first.

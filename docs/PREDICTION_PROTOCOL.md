# Trial-Radar and the prediction protocol

The Data Science Fundamentals course (DSF, deck P1 "Prediction in Practice") lists ten questions that every
predictive model must answer before anyone relies on it. This file answers them for Trial-Radar, one by one, with
the place in the code where each answer is implemented. Use it for the video and the Q&A: almost every model
question is one of these ten.

Numbers are from `models/gbm_metrics.json` and `models/horse_race.json` (run of @@DATE@@). If the model is
retrained, update them.

| # | Question (DSF P1) | Trial-Radar's answer | Where in the code |
|---|---|---|---|
| 1 | **Question:** does the prediction serve a concrete decision? | Two users. A sponsor or study designer decides how to design a trial before registering it; an investor or analyst decides which of a company's ongoing trials deserve a closer look. Both need a ranking of trials by the risk of being stopped early. | `app/Home.py` (`AUDIENCES`), README "The problem" |
| 2 | **Cutoff:** who acts, when, on what information? | The cutoff is the moment a trial is registered. Only what is in the design at that moment may be used. | `CLAUDE.md` non-negotiable 2 |
| 3 | **Inputs:** is every input known at the cutoff? | Yes: 29 design fields (phase, blinding, allocation, arms, age limits, eligibility length, sponsor type, ...). Excluded on purpose: why the trial stopped, every date except the start date, results, enrollment (finished trials report the *actual* number), number of sites (added after the start). | `src/features.py` (`build_features` reads an explicit field list), `data/raw/CLAUDE.md` |
| 4 | **Target:** does it measure what the decision needs? Unit? Horizon? | One row = one trial. Target: status *Terminated* (1) against *Completed* (0). Withdrawn and running trials are dropped. The horizon is the whole life of the trial. Limit: the reason for stopping is not used, so a business decision counts the same as a safety stop. | `src/build_dataset.py`, `src/train.py` (`check_dataset`) |
| 5 | **Split:** random, by time or by group? | **By time**, because the app scores trials that start after the ones it learned from. Test = trials started 2017-2019, train = 2008-2016. Inside the training trials, tuning also uses **time-ordered folds** (`TimeSeriesSplit`, an expanding window): each check trains on earlier trials and validates on the next, later block (DSF decks 10 and 16). | `src/train.py` (`SPLITS`, `make_cv`) |
| 6 | **Baseline:** what does a simple rule achieve? | Always predicting the average rate: ROC-AUC 0.50, PR-AUC @@BASE_PR@@ (the share of stopped test trials). It runs as a horse in the race as well. | `src/evaluate.py` (`majority_baseline`), `src/horse_race.py` |
| 7 | **Metric:** does it reflect the cost of errors where they matter? | ROC-AUC (ranking) and PR-AUC (finding the rare stopped trials); not accuracy, because "always completed" is already right @@ACC@@ of the time. Brier score for the percentages. For a yes/no decision the line comes from the costs: flag a trial if its risk is above c_FP / (c_FP + c_FN) (DSF deck 11); the model page has a slider for the cost ratio and shows the confusion matrix it leads to. | `src/evaluate.py`, `app/pages/4_How_the_Model_Works.py` ("From a score to a yes/no") |
| 8 | **Leakage:** checked against the catalogue? | After the cutoff: excluded fields (question 3). The answer in disguise: none of the inputs is computed from the status. Preprocessing on all data: every imputer, scaler and encoder sits inside a scikit-learn `Pipeline`, so it is fitted on the training folds only. Same entity on both sides: duplicates removed by NCT number. Reusing the test set: the test split is scored once, by `evaluate.py`, after every choice is made. Edited fields: ablations without the fields a sponsor can change after the start cost under 0.01 ROC-AUC. | `src/train.py` (pipelines), `src/build_dataset.py` (duplicates), `src/evaluate.py` (`run_ablations`) |
| 9 | **Sanity checks:** data, predictions, errors by segment? | `check_dataset` stops training if labels, dates or splits are wrong; the model page shows the ranking quality per phase and the prediction against reality per risk group and per start year. Finding: the percentages run low on the test years, most likely because recent trials that will complete are still running and therefore missing. | `src/train.py` (`check_dataset`), `app/pages/4_How_the_Model_Works.py` |
| 10 | **Use:** can it be rerun, what if inputs are missing? | The whole pipeline reruns from the API with four commands (fetch, build, train, horse race), with fixed random seeds. In the app, missing fields are imputed like in training, unknown categories become missing instead of crashing, and if the API is down the search falls back to a saved snapshot. | README "Run it", `src/api_client.py` (offline fallback) |

## The horse race (DSF deck 16)

Six models ran on the same training trials, the same five time-ordered folds and the same metric (PR-AUC), each with
its standard settings and with a small tuned grid. The test split was not used.

@@RACE_TABLE@@

@@RACE_VERDICT@@

Rules we followed (DSF deck 16): every horse tuned the same way inside the cross-validation; a lead only counts if
it is larger than the spread across the folds; the chosen model is refitted on all training trials and scored
**once** on the test trials (@@TEST_LINE@@).

## Prediction, not cause (DSF decks 01, 17 and the project rules)

The model learns which designs *went with* more stopped trials in the past. It does not learn what a change would
*cause*. That is why the Check page asks "Which single change lowers the estimate?" and not "What would lower the
risk?", and why the factor chart says "important does not mean causal". Example to use in the Q&A: many umbrellas on
the street predict rain, but handing out umbrellas does not make it rain.

## Methods that are in neither course

They are allowed, but at least one person must be able to explain each on the spot (DSF project rules: "every piece
must be explainable").

| Method | In one sentence | Who explains it |
|---|---|---|
| `HistGradientBoostingClassifier` with native categories and early stopping | Boosting (DSF deck 17) on binned numbers; categories are split directly instead of one-hot coded; it stops adding trees when an inner 10% validation set stops improving. | **[team]** |
| `class_weight="balanced"` | Each stopped trial counts about 8 times as much in training, so the rare class is not ignored (CS course 6.4.1, uneven class distribution). | **[team]** |
| Isotonic calibration (`CalibratedClassifierCV`) | A step function that maps the raw scores to the observed share of stopped trials, learned on held-out folds; the ranking stays the same. | **[team]** |
| Permutation importance | Shuffle one input in the test trials and measure how much worse the ranking gets (DSF deck 17: importance is predictive, not causal). | **[team]** |
| Per-trial drivers (perturbation) | Set one input of a trial to the value of a typical training trial and measure how far the estimate moves. | **[team]** |

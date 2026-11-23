# Q&A preparation (Fri Dec 11)

After the video every group gets questions from the tutor. The point of the Q&A is to check that the team
understands what it built and that the contribution matrix is real. **Everyone must be able to answer the
questions about their own part in their own words, and at least the "must know" questions below.**

Rule for the session: if you do not know an answer, say so and say who in the team owns that part. Do not guess.

All numbers are from `models/gbm_metrics.json` and the test split (trials started 2017-2019, 47,857 trials,
11.7% stopped early). If the model is retrained, update this file.

## Must know (everyone)

**1. What problem does the app solve, and for whom?**
Clinical trials are stopped early more often than people expect (11.3% in our data). Two users: trial sponsors
and study designers, who can test a design before registering it and see which single change goes with a lower
estimate in past trials ("Check your own trial"); and investors or business analysts, who can screen a company's ongoing trials
("Find a trial").

**2. What does "stopped early" mean exactly?**
The ClinicalTrials.gov status *Terminated*, compared with *Completed*. Withdrawn trials (nobody enrolled) and
running trials are not in the training data. The reason for stopping is not used: a business decision counts the
same as a safety stop.

**3. Where does the data come from?**
The ClinicalTrials.gov API v2. `src/fetch_data.py` downloaded all interventional completed and terminated trials;
`src/build_dataset.py` keeps those that started 2008-2019 (162,457). The "Find a trial" page calls the same API
live for ongoing trials. The "Explore the data" page reads the historical trials from a SQLite database with SQL
(question 28).

**4. What is data leakage, and how did you avoid it?**
Leakage means the model sees information that would not be available at the time of the prediction, so it looks
better in testing than it can ever be in use. Examples we excluded: the field "why stopped" (only filled for
stopped trials), every date except the start date, results, and enrollment (finished trials report the *actual*
number of participants, and stopped trials have few). We also excluded the number of sites, because sites are
added after a trial starts. All features come from one function, `build_features`, used by training and the app.

**5. Why do you split by start year and not at random?**
Because that is how the app is used: it learns from the past and scores newer trials. A random split would mix
2019 trials into training and test the model on 2008 trials, which flatters it.

**6. Why ROC-AUC and PR-AUC and not accuracy?**
Only 11.7% of the test trials were stopped. A "model" that always says "completed" is right 88% of the time and
useless. ROC-AUC (0.71) says how well the model ranks: pick one stopped and one completed trial at random, and the
model scores the stopped one higher 71% of the time (guessing: 50%). PR-AUC (0.23 against 0.12 for guessing) looks
only at how well it finds the rare stopped trials.

**7. Is the model good?**
Clearly better than guessing, far from certain. The 10% of test trials it rated riskiest were stopped 28% of the
time, 2.4 times the average; the safest 10%: 3%. If you review the riskiest 20% of trials first you find 43% of all
stopped trials (random picking: 20%). But most "red" trials still finish.

## Model questions (Adam leads, Dominic backs up)

**8. Which model, and why that one?**
Gradient boosting (`HistGradientBoostingClassifier` from scikit-learn): many small decision trees, each one
correcting the errors of the ones before. It handles missing values and categories (phase, blinding) natively and
finds combinations of features without us writing them. We did not just pick it: six models ran a fair race on the
same time-ordered folds (question 24). Boosting scored best, the random forest was practically tied, logistic
regression (test ROC-AUC 0.69), KNN and a single tree were behind. On the test trials boosting reached 0.71 with
honest probabilities.

**9. How did you tune it without touching the test set?**
Time-ordered cross-validation on the training trials only (2008-2016), `TimeSeriesSplit` in scikit-learn: the
training trials are sorted by start date and cut into six blocks; the model trains on block 1 and is checked on
block 2, then trains on blocks 1-2 and is checked on block 3, and so on, five checks in all (an "expanding window",
DSF decks 10 and 16). So it is always checked on trials that started *later* than the ones it learned from, like in
the app. A shuffled 5-fold split would let it learn from 2016 trials to predict 2009 ones. A small grid of 12
settings (learning rate, tree size, regularisation) was compared on PR-AUC. The grid was flat (all 12 settings
between 0.208 and 0.211 PR-AUC, less than the spread across the checks), so tuning was not what made the model work.
The test years were scored once, at the end.

**10. What is calibration, and why did you need it?**
We trained with `class_weight="balanced"`, which makes the model pay extra attention to the rare stopped trials. That
helps the ranking but inflates the percentages: the raw model predicted 40% on average when 11.7% were stopped.
Isotonic calibration learns a monotone mapping from raw scores to real rates on held-out folds of the training data,
so "15%" means about 15 of 100 such trials stop. The Brier score (squared error of the percentages) fell from 0.192
to 0.097; the ranking stayed the same.

**11. The papers report 0.73-0.80. Why is yours only 0.71?**
The set-ups are not comparable. Kavalci & Hartshorn (2023) report about 0.80 with a random train/test split and the
number of sites as a feature; both make results look better, and we excluded both on purpose. A leakage audit
(Kaushik 2026) shows how much this matters: ROC-AUC rose from 0.70 to 0.85 once actual enrollment was allowed. Under
leakage controls and a temporal split that audit reports 0.74, so 0.71 is in the expected range, with some room left.
Kavalci & Hartshorn also turn the eligibility text into thousands of yes/no features; we only use its length.

**12. Why does the model predict too low on the test trials?**
On the 2017-2019 trials it predicts 9.8% on average and 11.7% were stopped. The gap grows with the start year (real
rate 9% above the prediction in 2017, 16% in 2018, 31% in 2019). Partly a tilt in the data: only finished trials
are included, a stopped trial ends early by definition, and many long trials that will complete are still running,
so they are missing. We checked it: we asked ClinicalTrials.gov how many trials of each start year are still running.
For the training years it is under 4%, for 2017 5.6%, 2018 7.3%, 2019 10.3%. Even if all of them complete, 2019 still
ends up at least 11.1% stopped against 9.4% predicted (+18% instead of +31%). So the tilt explains about half the gap;
the rest is a real change or hidden among the 23% of 2019 trials whose status is "unknown". It is on the model page
("Can you trust it?", "We counted them").

**13. You tested a version 2. Why is it not in the app?**
Version 2 adds condition areas (cancer, heart, ...). Before training we fixed a rule: switch only if test ROC-AUC rises
by at least 0.005 and cross-validation agrees. It rose by 0.004, so we kept version 1. Fixing the rule first stops us
from picking whatever looks best on the test set.

**14. Can the model be fooled by later edits to a record?**
ClinicalTrials.gov shows the latest version of each record, so a sponsor may have edited the eligibility text or the
outcomes after the start. We refitted the model without those fields: test ROC-AUC drops by less than 0.01, so the
model does not depend on them.

**15. How does "what pushes this trial's risk" work?**
For one trial we replace one factor at a time with the value of a typical training trial and measure how much the
model's estimate (in log-odds of the uncalibrated model) moves. The longer the bar, the more that factor matters for
this trial. These are patterns in past trials, not causes, and the bars do not add up to the total. We did not use
SHAP because it does not follow the model's native category splits.

**16. Which factors matter most?**
By permutation importance (shuffle one feature in the test data and see how much worse the ranking gets): whether
healthy volunteers are accepted, the length of the eligibility criteria, the age limits, the phase and whether a drug
is tested.

## App questions (Matteo leads)

**17. Why Streamlit, and how does it work?**
Streamlit turns a Python script into a web page. On every click or input the whole script runs again from top to
bottom. Slow steps are cached: the data and the model load once (`st.cache_data`, `st.cache_resource`), and API
answers are cached for an hour.

**18. What happens if the ClinicalTrials.gov API is down?**
The search shows a plain message and offers example trials stored in the repository, so the rest of the demo still
works. Every other page runs offline.

**19. How do the five pages fit the use case?**
Find a trial (search live trials, each with a risk), Check your own trial (design form with "which single change
lowers the estimate"), Explore the data (filters, bar charts, a heatmap of two traits), How the model works (quality,
a review-budget slider, a cost slider with the confusion matrix, the model race, calibration, versions), Play the game
(guess which of two trials was stopped, against the model).

**20. How do you know the app works?**
48 offline tests (`pytest tests`): the feature function, the predictions, the model protocol, the database, the tilt check, the chart helpers and every page clicked
through in Streamlit's test runner with the API mocked.

## Game questions (Raphael leads)

**21. How does the game work, and how good is the model at it?**
Ten rounds; each shows one stopped and one completed trial from the test years, of the same phase, in random order.
The model "picks" the one with the higher score. On a fixed sample of 2,000 such pairs it is right 67% of the time.
Same-phase pairs are harder than the overall ROC-AUC of 0.71 suggests, because phase is one of the things the model
uses; that is on purpose, so you cannot win by looking at the phase alone.

## Team and AI questions (everyone)

**22. Who did what?**
Answer from `docs/CONTRIBUTIONS.md`, for your own rows. Be ready to show one file you wrote or reviewed and explain it.

**23. How did you use AI?**
Answer from `docs/AI_USE.md`: which tool, for what, how we checked its output (tests, reviews, the rules in the
`CLAUDE.md` files), what worked, what did not, and what we would do differently.

## Course questions (everyone)

These connect the project to what we learned: the CS course (Ecoreps summary chapters) and Data Science Fundamentals
(DSF). DSF = *Data Science Fundamentals*, an additional HSG course outside Computer Science. Simon attended most of its lectures and brought its methods into the project where they solve a concrete problem of ours; the rest of the team learned them from him and from this file. If asked why we use material from another course: the CS course gives the basics (KNN,
confusion matrix, cross-validation); a real prediction model needs more, and each DSF method we use answers a problem
of our data (time-ordered split: the app scores newer trials; PR-AUC: stopped trials are rare; cost line: a missed
stop costs more than a false alarm; model race: a justified choice of model). The full list of DSF's ten "prediction protocol" questions, answered for our project, is in
`docs/PREDICTION_PROTOCOL.md`.

**24. How did you choose between models?**
With a "horse race" (DSF deck 16): baseline, logistic regression, decision tree, random forest, k-nearest neighbours
(the classifier of the CS course, with min-max scaling) and gradient boosting, all on the same training trials, the
same five time-ordered checks and the same score (PR-AUC), each with standard and with tuned settings. Boosting
0.211, forest 0.210, logistic regression 0.192, KNN 0.190, tree 0.187, baseline 0.109. Boosting's lead over the
forest (0.001) is smaller than the spread across the checks (0.005), so they are practically tied; we keep boosting
(best score, handles missing values and categories, calibration and explanations are built on it). The test set was
not used for this choice. Nice detail: a tree without limits learns the training trials by heart (0.116, barely
better than guessing), and KNN with k = 5 is too noisy (0.137, k = 201 gives 0.190): overfitting and the
bias-variance trade-off in action.

**25. How do you turn a probability into a yes/no?**
With a threshold, and the threshold comes from the costs of the two errors, not from the model (DSF deck 11): flag a
trial if its risk is above c_FP / (c_FP + c_FN). If a missed stopped trial costs 5 false alarms, the line is
1/6 = 17%. The model page has a slider for this ratio and shows the confusion matrix on the test trials (CS course
6.4.2). This only works because the percentages are calibrated. F1, which we also report, is a compromise when you
do not know the costs.

**26. Does "which single change lowers the estimate" mean the change lowers the risk?**
No. The model learned which designs went with more stopped trials in the past (prediction), not what a change would
cause. A setting can simply go with other things (money, experience, the kind of drug) that make the real
difference. That is why the heading speaks of the estimate and the caption says "a pattern, not proof of cause".
Umbrellas predict rain; handing out umbrellas does not make it rain (DSF deck 01).

**27. Which parts of the project are not from the courses?**
Gradient boosting with native categories, `class_weight="balanced"`, isotonic calibration, permutation importance
and the per-trial drivers. Each one has a one-sentence explanation in `docs/PREDICTION_PROTOCOL.md`; the team assigns
one person per method who can explain it on the spot.

**28. Where do you use a database, and why?**
On "Explore the data". `src/database.py` turns the trials table into a SQLite database (`data/processed/trials.db`)
with two tables: `trials` (one row per trial, primary key `nct_id`) and `trial_conditions` (one row per trial and
condition, `nct_id` as foreign key). The second table exists because a record lists several conditions in one
cell ("Asthma; COPD"); one value per cell is the first normal form (CS course 3.3). The page then asks SQL for its
numbers: `WHERE` for the filters, `COUNT` and `AVG` for the key numbers, `GROUP BY` for the bars, a `JOIN` with
`ORDER BY ... LIMIT 15` for the most common conditions, all through `pd.read_sql` (CS course 3.4.7). The filters go
in as `?` parameters, never pasted into the SQL text, so typing `'; DROP TABLE trials; --` into the keyword box is
just a search that finds nothing (no SQL injection). The folded box "Under the hood" at the bottom of the page shows
every query. The model still trains from the parquet file: the database is for asking questions, the parquet for
training.

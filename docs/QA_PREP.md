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
and study designers, who can test a design before registering it and see which single change would lower the
risk ("Check your own trial"); and investors or business analysts, who can screen a company's ongoing trials
("Find a trial").

**2. What does "stopped early" mean exactly?**
The ClinicalTrials.gov status *Terminated*, compared with *Completed*. Withdrawn trials (nobody enrolled) and
running trials are not in the training data. The reason for stopping is not used: a business decision counts the
same as a safety stop.

**3. Where does the data come from?**
The ClinicalTrials.gov API v2. `src/fetch_data.py` downloaded all interventional completed and terminated trials;
`src/build_dataset.py` keeps those that started 2008-2019 (162,457). The "Find a trial" page calls the same API
live for ongoing trials.

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
finds combinations of features without us writing them. We compared it with a logistic regression baseline
(ROC-AUC 0.69); boosting reached 0.71 with honest probabilities.

**9. How did you tune it without touching the test set?**
5-fold cross-validation on the training trials only (2008-2016): the training set is cut into five parts, the model
is trained on four and scored on the fifth, five times. A small grid of 12 settings (learning rate, tree size,
regularisation) was compared on PR-AUC. The grid was flat (all within one standard deviation), so tuning was not
what made the model work. The test years were scored once, at the end.

**10. What is calibration, and why did you need it?**
We trained with `class_weight="balanced"`, which makes the model pay extra attention to the rare stopped trials. That
helps the ranking but inflates the percentages: the raw model predicted 39% on average when 11.7% were stopped.
Isotonic calibration learns a monotone mapping from raw scores to real rates on held-out folds of the training data,
so "15%" means about 15 of 100 such trials stop. The Brier score (squared error of the percentages) fell from 0.186
to 0.097; the ranking stayed the same.

**11. The papers report 0.73-0.80. Why is yours only 0.71?**
The set-ups are not comparable. Kavalci & Hartshorn (2023) report about 0.80 with a random train/test split and the
number of sites as a feature; both make results look better, and we excluded both on purpose. A leakage audit
(Kaushik 2026) shows how much this matters: ROC-AUC rose from 0.70 to 0.85 once actual enrollment was allowed. Under
leakage controls and a temporal split that audit reports 0.74, so 0.71 is in the expected range, with some room left.
Kavalci & Hartshorn also turn the eligibility text into thousands of yes/no features; we only use its length.

**12. Why does the model predict too low on the test trials?**
On the 2017-2019 trials it predicts 9.9% on average and 11.7% were stopped. The gap grows with the start year (real
rate 9% above the prediction in 2017, 16% in 2018, 31% in 2019). Most likely a tilt in the data: only finished trials
are included, a stopped trial ends early by definition, and many long trials from 2019 that will complete are still
running, so they are missing. We say so on the model page and in the README; we have not yet counted the running
trials to prove it.

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
Find a trial (search live trials, each with a risk), Check your own trial (design form with "what would lower the
risk"), Explore the data (filters, bar charts, a heatmap of two traits), How the model works (quality, a review-budget
slider, calibration, versions), Play the game (guess which of two trials was stopped, against the model).

**20. How do you know the app works?**
40 offline tests (`pytest tests`): the feature function, the predictions, the chart helpers and every page clicked
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

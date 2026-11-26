# Video script (target 3:50, hard limit 4:00)

Rules from the grading scheme, so nothing costs points:

- **At most 4:00.** Longer gives at most 1 point for the video. Aim for 3:50 to leave room.
- **Own voices only.** An AI-generated voice-over gives 0 points for the video. No text-to-speech, no voice cloning.
- **Must contain:** the idea, the development process and teamwork, the contributions, the aids, the use of AI,
  the results and a short demo of the app.
- The contribution matrix and the AI slide may be shown for a few seconds; the tutors pause the video.

The video also carries the points for the visualisations, the interactions, the comments and the contribution
matrix, so it must **show** them (feedback of 2026-11-26):

- **Variety of charts, not one bar chart after another:** bars, the heatmap, the map of conditions, the bubbles, the
  flow diagram, the drivers chart of one trial, the calibration and gain curves, the game.
- **Variety of interactions:** the search, the form with the live risk, the filters, the sliders, the game.
- **The code with its comments:** a short shot of a file with the docstring and the comments in view.
- **AI use in more depth:** what we used it for, what we wrote ourselves, how we checked it, what we do next time.
- **The matrix:** who led what, and why we split the work like this.

Each member speaks their own part, which also prepares the Q&A. About 140 spoken words per minute; the word counts
below add up to about 520. Record the demo first (screen recording of the app at `localhost:8501`, browser at 100%
zoom, window 1440 px wide), then the voices over it. Rehearse once with a stopwatch.

| Time | Speaker | On screen | Spoken text (about) |
|---|---|---|---|
| 0:00-0:20 | Matteo | Home page, the 11.3% hero and tick scale | "More than one in ten clinical trials is stopped before its planned end. Each one has used up money, staff and the trust of patients, often without an answer. We built Trial-Radar: an app that estimates, from the design alone, how likely a trial is to be stopped early." (55 words) |
| 0:20-0:35 | Matteo | Home, "Who it is for" band | "It is for two kinds of users: sponsors who design a trial and want to check it before registering it, and investors who want to know which trials in a company's pipeline deserve a closer look." (37 words) |
| 0:35-1:00 | Simon | Diagram or slide: ClinicalTrials.gov API → 162,457 trials → features → model → app | "Our data comes from the ClinicalTrials.gov API: more than 160,000 finished trials that started between 2008 and 2019. The hard part was leakage. A finished trial reports how many people it actually enrolled, and stopped trials have few, so that field would give the answer away. We only use what is known at registration." (55 words) |
| 1:00-1:25 | Matteo | Demo: Find a trial, type "breast cancer", open one result, the drivers chart | "Here you search ongoing trials live. Every result gets a risk and a colour that compares it with past trials. In the details you see which factors push this trial's risk up or down." (35 words) |
| 1:25-1:45 | Matteo | Demo: Check your own trial, change blinding, look at "Which single change lowers the estimate?" | "A sponsor can enter a design and watch the estimate move. The app tries every design choice a sponsor controls and shows which single change lowers the estimate most: a pattern in past trials, not a promise." (37 words) |
| 1:45-2:10 | Raphael | Demo: Explore. Set a filter, then show bars, the heatmap "Phase × lead sponsor" and, under "More ways to look at it", the map of conditions, the bubbles and the flow diagram | "Explore shows the history behind the model. The filters change every chart at once. Bars show which kinds of trials stop more often, the heatmap shows which combinations do, and the map, the bubbles and the flow diagram show the same data in other ways." (46 words) **[team]** only if the new charts (branch `app/more-views`) are merged; otherwise cut the last clause. |
| 2:10-2:35 | Adam | Demo: How the model works, ROC-AUC tile, then move the review slider to 20% | "We trained gradient boosting, tuned it with cross-validation and calibrated it so the percentages are honest. We test on trials that started after every training trial. ROC-AUC is 0.71. In practice: if you review the riskiest 20% of trials, you find 42% of all stopped ones, twice as many as by chance. It is a ranking tool, not a crystal ball." (62 words) |
| 2:35-2:50 | Raphael | Demo: the game, one round | "And you can play against the model: two trials, one was stopped. On two thousand pairs the model is right two times out of three. Can you beat it?" (30 words) |
| 2:50-3:00 | Matteo | A source file in the editor, docstring and comments in view (for example `src/features.py`) | "Every file starts with a docstring that cites the AI, every function says what goes in and what comes out, and every block has a comment. A script checks this format." (30 words) |
| 3:00-3:20 | Dominic | Slide: contribution matrix (`docs/CONTRIBUTIONS.md`), picture for about 3 seconds | "We split the work along the path of the data: data, model, evaluation, app, visuals. Each step ends in one thing the next step uses, so everyone could work in parallel and answers for their own part. **[team]** One sentence per person on why they took their area." (45 words + the team's sentences) **[team]** add one honest sentence about what was hard in the teamwork. |
| 3:20-3:45 | Dominic | Slide: AI use and list of aids (`docs/AI_USE.md`, section 6), picture for about 3 seconds | "Most of the code was written with Claude Code. We set the rules, checked and corrected the output, and wrote some simple parts ourselves; they are listed in our AI record. Written rules and tests worked best. Unchecked wording and a wrong deadline did not. Next time we keep the prompt log from day one and let scripts check the AI's work." (60 words) **[team]** adjust to your real experience, name the parts written by hand. |
| 3:45-3:50 | all / Matteo | Home page, team names | "Thank you. Trial-Radar: know the risk before the trial starts." (10 words) |

## Slides to prepare

1. Data flow (Simon's part): API → 162,457 trials (2008-2019) → design-time features only → model → app.
2. Contribution matrix: the table from `docs/CONTRIBUTIONS.md`, readable at full screen. Add the "lead / contributor"
   meaning and, per person, one line on why they took their area.
3. AI use and list of aids: tools and models, share of AI-generated code, how it was checked, what worked and what not,
   what we wrote by hand, what we do next time (from `docs/AI_USE.md`, sections 1, 2, 3.7, 5 and 6).
4. Code shot: a file with the docstring and comments readable (not a slide, a screen recording of the editor).

## Before uploading

- [ ] Length at most 4:00 (check in the player, not the editor)
- [ ] Only human voices, recorded by us
- [ ] Every number spoken matches the app (11.3%, 162,457, 0.71, 42% at 20%, two out of three in the game)
- [ ] Contribution and AI slides each visible for at least 3 seconds
- [ ] Several kinds of charts and several kinds of interactions are shown, not one bar chart after another
- [ ] The code shot shows a docstring with Input and Output and the comments
- [ ] Exported as MP4, uploaded to Canvas as a file (a YouTube link alone is not accepted)

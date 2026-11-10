# Video script (target 3:45, hard limit 4:00)

Rules from the grading scheme, so nothing costs points:

- **At most 4:00.** Longer gives at most 1 point for the video. Aim for 3:45 to leave room.
- **Own voices only.** An AI-generated voice-over gives 0 points for the video. No text-to-speech, no voice cloning.
- **Must contain:** the idea, the development process and teamwork, the contributions, the aids, the use of AI,
  the results and a short demo of the app.
- The contribution matrix and the AI slide may be shown for a few seconds; the tutors pause the video.

Each member speaks their own part, which also prepares the Q&A. About 140 spoken words per minute; the word counts
below add up to about 500. Record the demo first (screen recording of the app at `localhost:8501`, browser at 100%
zoom, window 1440 px wide), then the voices over it. Rehearse once with a stopwatch.

| Time | Speaker | On screen | Spoken text (about) |
|---|---|---|---|
| 0:00-0:20 | Matteo | Home page, the 11.3% hero and tick scale | "More than one in ten clinical trials is stopped before its planned end. Each one has used up money, staff and the trust of patients, often without an answer. We built Trial-Radar: an app that estimates, from the design alone, how likely a trial is to be stopped early." (55 words) |
| 0:20-0:35 | Matteo | Home, "Who it is for" band | "It is for two kinds of users: sponsors who design a trial and want to check it before registering it, and investors who want to know which trials in a company's pipeline deserve a closer look." (37 words) |
| 0:35-1:00 | Simon | Diagram or slide: ClinicalTrials.gov API → 162,457 trials → features → model → app | "Our data comes from the ClinicalTrials.gov API: more than 160,000 finished trials that started between 2008 and 2019. The hard part was leakage. A finished trial reports how many people it actually enrolled, and stopped trials have few, so that field would give the answer away. We only use what is known at registration." (55 words) |
| 1:00-1:30 | Matteo | Demo: Find a trial, type "breast cancer", open one result | "Here you search ongoing trials live. Every result gets a risk and a colour that compares it with past trials. In the details you see which factors push this trial's risk up or down." (35 words) |
| 1:30-1:55 | Matteo | Demo: Check your own trial, change blinding, look at "What would lower the risk?" | "A sponsor can enter a design and watch the estimate move. The app also tries every design choice a sponsor controls and shows which single change would lower the risk most." (32 words) |
| 1:55-2:15 | Raphael | Demo: Explore, the heatmap with "Phase × lead sponsor", then a filter | "Explore shows the history behind the model: which kinds of trials stop more often, and with the heatmap, which combinations, for example phase 2 trials run by individual investigators." (30 words) |
| 2:15-2:45 | Adam | Demo: How the model works, ROC-AUC tile, then move the review slider to 20% | "We trained gradient boosting, tuned it with cross-validation and calibrated it so the percentages are honest. We test on trials that started after every training trial. ROC-AUC is 0.71. In practice: if you review the riskiest 20% of trials, you find 43% of all stopped ones, twice as many as by chance. It is a ranking tool, not a crystal ball." (62 words) |
| 2:45-3:00 | Raphael | Demo: the game, one round | "And you can play against the model: two trials, one was stopped. On two thousand pairs the model is right two times out of three. Can you beat it?" (30 words) |
| 3:00-3:20 | Dominic | Slide: contribution matrix (`docs/CONTRIBUTIONS.md`) | "We split the work into a data-and-model strand and an app strand, as our tutor suggested, joined by one feature function. Each area had one lead; the matrix shows who led and who contributed." (35 words) **[team]** add one honest sentence about what was hard in the teamwork. |
| 3:20-3:40 | Dominic | Slide: AI use and list of aids (`docs/AI_USE.md`, section 6) | "Most of the code was generated with Claude Code. Our job was to set the rules, check the output and correct it. That worked best with written rules and tests. It worked badly where we trusted it unchecked: wording that claimed too much, and a wrong deadline." (49 words) **[team]** adjust to your real experience. |
| 3:40-3:45 | all / Matteo | Home page, team names | "Thank you. Trial-Radar: know the risk before the trial starts." (10 words) |

## Slides to prepare

1. Data flow (Simon's part): API → 162,457 trials (2008-2019) → design-time features only → model → app.
2. Contribution matrix: the table from `docs/CONTRIBUTIONS.md`, readable at full screen.
3. AI use and list of aids: tools and models, share of AI-generated code, how it was checked, what worked and what not
   (from `docs/AI_USE.md`, sections 1, 2, 5 and 6).

## Before uploading

- [ ] Length at most 4:00 (check in the player, not the editor)
- [ ] Only human voices, recorded by us
- [ ] Every number spoken matches the app (11.3%, 162,457, 0.71, 43% at 20%, two out of three in the game)
- [ ] Contribution and AI slides each visible for at least 3 seconds
- [ ] Exported as MP4, uploaded to Canvas as a file (a YouTube link alone is not accepted)

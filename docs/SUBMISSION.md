# Submission checklist

From the course slides ("Deliverables and their presentation", "Checklist: AI Use, Citations, and Declaration of
Authorship") and the grading scheme. One person uploads; everyone checks this list once.

## Dates

| When | What |
|---|---|
| Sun Dec 6 | Feature freeze: no new features, only fixes |
| Tue Dec 8 | Final code merged, video recorded and checked |
| Wed Dec 9 | Upload to Canvas (one day of buffer) |
| **Thu Dec 10, 23:59** | **Hard deadline for the Canvas upload** |
| Fri Dec 11 | Video shown in our exercise group, then Q&A. **Everyone must attend.** |
| Wed Dec 16 | Only if we are in the top 3: presentation in the lecture (mandatory then) |

## What to upload to Canvas (the files themselves; links to GitHub or YouTube are not enough)

1. **The video** (MP4, at most 4:00, own voices). See `docs/VIDEO_SCRIPT.md`.
2. **The source code as a zip** that runs as it is, with data and models included:

   ```bash
   git checkout main && git pull
   git archive --format=zip --output=trial-radar.zip HEAD
   ```

   `git archive` packs exactly the committed files: code, `data/processed/trials.parquet`, `data/examples/`, `models/`,
   fonts, README and docs (about 20 MB), without `.venv` or caches.
3. **The declaration of authorship**, one for the whole group, filled in and signed: the HSG form linked in the course
   slides and in the grading sheet (II.A.07 "Declaration of Authorship", on the HSG student SharePoint). Download it
   from there; the version on the SharePoint is the valid one. Only the team can fill in and sign it. If the form
   asks for the aids used (Hilfsmittelverzeichnis), copy them from `docs/AI_USE.md`, section 6: AI tools and models,
   data, software, fonts, literature and course material. Say "none" for other AI tools only if that is true.

## Test the zip before uploading

On a laptop that did not build the project (ideally Windows):

```bash
unzip trial-radar.zip -d trial-radar-test && cd trial-radar-test
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest tests                        # all tests should pass
streamlit run app/Home.py           # click through all five pages
```

Then switch off Wi-Fi and open "Find a trial": the saved example trials must appear (see below).

## Content checks (each one is a grading point or a formal requirement)

- [ ] **AI use declared and cited.** Every Python file starts on line 1 with the "AI use" docstring
      (`python tools/ai_provenance.py --check`), `docs/AI_USE.md` is complete (no **[team]** left, including the list
      of the simple parts the team wrote by hand in section 3.7) and its table is re-measured
      (`python tools/ai_provenance.py`).
- [ ] **Docstring format.** `python tools/check_docstrings.py` passes: every function has a summary, an `Input:` and
      an `Output:` paragraph.
- [ ] **AI on the list of aids in the video** (slide from `docs/AI_USE.md`, section 6).
- [ ] **Other people's work cited.** Data source, libraries, fonts and the four papers are in the README
      ("Sources and credits" and "The problem").
- [ ] **Contribution matrix complete** (`docs/CONTRIBUTIONS.md`, no **?** left) and shown in the video.
- [ ] **Saved example trials** for the offline fallback: `data/examples/ongoing_trials.json` was saved and committed
      on 2026-11-26 (28 trials). Check that it is in the zip, and re-run `python src/save_examples.py` while online
      shortly before the Q&A so the date is recent.
- [ ] **Numbers in the README, the app and the video agree** (162,457 trials, 11.3%, ROC-AUC 0.71, 42% at 20%).
- [ ] **Deployed app (optional, not graded):** redeploy on Streamlit Community Cloud from `main` (README, "Deploy").
      A private repository works too.

## Q&A on Dec 11

Everyone reads `docs/QA_PREP.md` and can answer the "must know" questions and the questions about their own rows of
the contribution matrix.

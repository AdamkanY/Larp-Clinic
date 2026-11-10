# data/examples/

A small, committed snapshot of real **ongoing** trials for the offline fallback of the "Find a trial" page: when the
ClinicalTrials.gov API cannot be reached, the page shows these trials instead, under a banner that says they are
saved examples and when they were downloaded. Not for: training data (ongoing trials are never trained on), invented
records, or anything large.

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `ongoing_trials.json` | `{"saved_on": "YYYY-MM-DD", "studies": [...]}`: full API v2 records of about 30 ongoing interventional trials, the first six for each example search of the page (breast cancer, diabetes, asthma, depression, COVID-19), unchanged | `python src/save_examples.py` (run while online, then commit) | `src/api_client.py` (`load_examples`, `search_examples`, the fallback in `get_trial`) |

The file is not in the repository until someone runs the script once; without it the page shows the plain
"Could not reach ClinicalTrials.gov" message as before. Re-run the script before a demo so the date is recent.

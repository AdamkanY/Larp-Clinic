"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Save a few real ongoing trials, so the page "Find a trial" still has something to show without internet.

Run it once from the project folder while online, then commit the file:

    python src/save_examples.py

The script runs the five example searches that the page offers (breast cancer, diabetes, asthma, depression,
COVID-19) and keeps the first ongoing interventional trials of each one. Their full records go into
data/examples/ongoing_trials.json together with today's date. The records are saved exactly as the API returns
them, so build_features() can score them like live results. Run it again before the demo to refresh the date.
"""
# ---- 1. Imports ----
import json  # write the snapshot file
import sys  # change the module search path
from datetime import date  # today's date for the snapshot
from pathlib import Path  # file paths

import requests  # send the search request to the API

# The other files of src/ are imported by their plain name, so this works from the terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import BASE_URL, EXAMPLES_PATH, ONGOING

# ---- 2. Settings ----
SEARCHES = ["breast cancer", "diabetes", "asthma", "depression", "COVID-19"]  # the example buttons of the page
PER_SEARCH = 6  # 30 trials in all: enough for a demo and small enough to commit (well under 1 MB)
TIMEOUT = 30  # seconds we wait for an answer


def fetch(query, n):
    """Ask the API for the first n ongoing interventional trials for one keyword. Returns full records.

    We call requests directly and not search_trials() from api_client.py. That function hides errors from the
    user, but here we want to see an error at once.

    Input: query is the search keyword (string); n is how many trials to ask for (integer).

    Output: a list of at most n full study records, as dictionaries exactly as the API returns them.
    """
    # The same filters as the page: only ongoing, interventional trials.
    params = {
        "query.term": query,
        "filter.overallStatus": ",".join(ONGOING),
        "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
        "pageSize": n,
    }
    # Send the request; a bad answer raises an error at once.
    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json().get("studies", [])


def main():
    """Run every example search, keep each trial only once and write the snapshot file.

    Input: none.

    Output: nothing. It writes data/examples/ongoing_trials.json and prints the progress.
    """
    # Collect the trials of all searches.
    studies = []
    seen = set()  # the ids we already have, because a trial can match two keywords
    for query in SEARCHES:
        for study in fetch(query, PER_SEARCH):
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nct_id and nct_id not in seen:
                seen.add(nct_id)
                studies.append(study)
        print(f"{query}: {len(studies)} trials so far")

    # Save the trials together with today's date, so the page can say how old the examples are.
    snapshot = {"saved_on": date.today().isoformat(), "studies": studies}
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(json.dumps(snapshot), encoding="utf-8")
    print(f"Saved {len(studies)} trials to {EXAMPLES_PATH.relative_to(EXAMPLES_PATH.parents[2])}")


# ---- 3. Run the script ----
if __name__ == "__main__":
    main()

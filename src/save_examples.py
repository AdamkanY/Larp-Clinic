# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Save a small snapshot of real ongoing trials for the offline fallback of the "Find a trial" page.

Run once from the repo root while online, then commit the file:

    python src/save_examples.py

It asks the ClinicalTrials.gov API v2 for the five example searches the page offers (breast cancer, diabetes,
asthma, depression, COVID-19), keeps the first few ongoing interventional trials of each and writes their full
records to `data/examples/ongoing_trials.json` with the download date. The records are stored exactly as the API
returns them, so `build_features` scores them like live results. Re-run it before the demo to refresh the date.
"""
import json
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import BASE_URL, EXAMPLES_PATH, ONGOING  # noqa: E402

SEARCHES = ["breast cancer", "diabetes", "asthma", "depression", "COVID-19"]  # the page's example pills
PER_SEARCH = 6  # 30 trials in all: enough to demo, small enough to commit (well under 1 MB)
TIMEOUT = 30


def fetch(query: str, n: int) -> list:
    """The first n ongoing interventional trials for one keyword, as full API records.

    Uses requests directly (not the app's cached client): this runs outside Streamlit and must fail loudly.
    """
    params = {"query.term": query, "filter.overallStatus": ",".join(ONGOING),
              "filter.advanced": "AREA[StudyType]INTERVENTIONAL", "pageSize": n}
    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json().get("studies", [])


def main() -> None:
    """Fetch every example search, drop duplicates across searches and write the snapshot file."""
    studies, seen = [], set()
    for query in SEARCHES:
        for study in fetch(query, PER_SEARCH):
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nct_id and nct_id not in seen:  # a trial can match two keywords; keep it once
                seen.add(nct_id)
                studies.append(study)
        print(f"{query}: {len(studies)} trials so far")
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(json.dumps({"saved_on": date.today().isoformat(), "studies": studies}),
                             encoding="utf-8")
    print(f"Saved {len(studies)} trials to {EXAMPLES_PATH.relative_to(EXAMPLES_PATH.parents[2])}")


if __name__ == "__main__":
    main()

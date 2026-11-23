# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Asks the ClinicalTrials.gov API (version 2) for ongoing trials. The page "Find a trial" uses this file.

search_trials() runs a search and get_trial() loads one trial. Both never raise an error: if the API cannot be
reached, they return a value that the page can turn into a message.

Offline fallback: data/examples/ongoing_trials.json is a small snapshot of real ongoing trials, written by
"python src/save_examples.py". Without internet (the exercise room, an outage during the demo) the page searches
this file instead and labels the results clearly as saved examples.
"""
import json
from pathlib import Path

import requests
import streamlit as st

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "examples" / "ongoing_trials.json"
# What the search page tells users when the API cannot be reached, whatever the technical reason is.
OFFLINE_MESSAGE = "Could not reach ClinicalTrials.gov right now."
# The statuses of trials that are still running. These are the trials the app scores.
ONGOING = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
SPONSOR_CLASSES = ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "INDIV", "NETWORK", "OTHER"]
TIMEOUT = 20  # seconds we wait for an answer


@st.cache_data(ttl=3600, show_spinner=False)
def _get_json(url, params=None):
    """Send one request to the API and return the answer (JSON) as a dictionary.

    Streamlit remembers the answer for one hour, so the same search is not sent twice. A failed request raises
    an error here. An error is never remembered, so the next try asks the API again.
    """
    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def search_trials(query="", status=None, page_size=20, phase=None, condition="", sponsor_class=None):
    """Search ongoing trials. Returns {"studies": [...], "total": number, "error": None or a message}.

    Never raises an error, so the page can always show something. If the API cannot be reached, "error" is one
    plain sentence for the user and "detail" holds the technical reason for us. The page does not show the detail.
    """
    # Without a list of statuses we search all ongoing ones.
    if status:
        status = list(status)
    else:
        status = ONGOING

    params = {
        "filter.overallStatus": ",".join(status),
        "pageSize": page_size,
        "countTotal": "true",
    }
    if query:
        params["query.term"] = query
    if condition:
        params["query.cond"] = condition

    # Only interventional studies: the model learned from those, so its score means nothing for observational
    # ones. The API has no simple filter for study type, phase and sponsor class, so they go into one expression.
    advanced = ["AREA[StudyType]INTERVENTIONAL"]
    if phase:
        advanced.append(f"AREA[Phase]{phase}")
    if sponsor_class:
        advanced.append(f"AREA[LeadSponsorClass]{sponsor_class}")
    params["filter.advanced"] = " AND ".join(advanced)

    try:
        data = _get_json(BASE_URL, params)
    except (requests.RequestException, ValueError) as error:
        # The error text (a long URL, proxy and SSL messages) means nothing to a user, so it goes into "detail".
        detail = f"{type(error).__name__}: {error}"
        return {"studies": [], "total": 0, "error": OFFLINE_MESSAGE, "detail": detail}
    return {"studies": data.get("studies", []), "total": data.get("totalCount", 0), "error": None}


def get_trial(nct_id):
    """Load the full record of one trial. Returns None if the trial is not found.

    If the API cannot be reached, we look for the trial in the saved examples. This way a result from the offline
    list can still be opened.
    """
    try:
        return _get_json(f"{BASE_URL}/{nct_id}")
    except (requests.RequestException, ValueError):
        snapshot = load_examples()
        if snapshot is None:
            return None
        for study in snapshot["studies"]:
            if summarize(study)["NCT ID"] == nct_id:
                return study
        return None


def load_examples(path=None):
    """The saved example trials as {"saved_on": "YYYY-MM-DD", "studies": [...]}. None if there is no snapshot.

    The file is small, so it is read again on every call.
    """
    # EXAMPLES_PATH is looked up here and not in the "def" line, so a test can point it to its own file.
    if not path:
        path = EXAMPLES_PATH

    # A missing or broken file is the same for us as no snapshot.
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return None

    # The same goes for a file without any studies in it.
    if not isinstance(data, dict):
        return None
    studies = data.get("studies")
    if not studies:
        return None
    return {"saved_on": data.get("saved_on", "unknown"), "studies": studies}


def search_examples(query="", phase=None, sponsor_class=None, page_size=20, path=None):
    """Search the saved example trials. The result looks like the one of search_trials(), plus "saved_on".

    The keyword must appear in the title or in the conditions (upper and lower case do not matter). This is
    simpler than the full-text search of the API, which we do not have offline. Phase and sponsor class must
    match exactly. Returns None if there is no snapshot.
    """
    snapshot = load_examples(path)
    if snapshot is None:
        return None

    keyword = query.strip().lower()
    matches = []
    for study in snapshot["studies"]:
        row = summarize(study)

        # Filter 1: the keyword.
        if keyword:
            text = f"{row['Title']} {row['Conditions']}".lower()
            if keyword not in text:
                continue

        # Filter 2: the phase. A trial without a phase counts as "NA", like in the API.
        design = study.get("protocolSection", {}).get("designModule", {})
        phases = design.get("phases") or ["NA"]
        if phase and phase not in phases:
            continue

        # Filter 3: the sponsor class.
        if sponsor_class and row["Sponsor type"] != sponsor_class:
            continue

        matches.append(study)

    # "total" counts all matches, but only the first page_size trials are returned (like one page of the API).
    return {"studies": matches[:page_size], "total": len(matches), "error": None, "saved_on": snapshot["saved_on"]}


def summarize(study):
    """Pick the few fields of one study that the result list shows and return them as a flat dictionary."""
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})

    row = {}
    row["NCT ID"] = identification.get("nctId", "")
    row["Title"] = identification.get("briefTitle", "")

    # The phases stay API codes, joined by commas (for example "PHASE1, PHASE2"). No phase is "NA", like in
    # the API. "or []" is needed because a record can hold null instead of a list.
    phases = protocol.get("designModule", {}).get("phases") or []
    row["Phase"] = ", ".join(phases) or "NA"

    row["Status"] = protocol.get("statusModule", {}).get("overallStatus", "")

    sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    row["Sponsor"] = sponsor.get("name", "")
    row["Sponsor type"] = sponsor.get("class", "")

    conditions = protocol.get("conditionsModule", {}).get("conditions") or []
    row["Conditions"] = ", ".join(conditions)
    return row

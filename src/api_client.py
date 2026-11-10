# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Thin client for the ClinicalTrials.gov API v2 (search and single-trial lookup), with an offline fallback.

The fallback is a small snapshot of real ongoing trials in `data/examples/ongoing_trials.json`, written by
`python src/save_examples.py`. When the API cannot be reached (no internet in the exercise room, an outage during
the demo), the search page can still show and score those trials, clearly labelled as saved examples.
"""
import json
from pathlib import Path

import requests
import streamlit as st

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "examples" / "ongoing_trials.json"
# What the search page tells users when the API cannot be reached, whatever the technical reason.
OFFLINE_MESSAGE = "Could not reach ClinicalTrials.gov right now."
ONGOING = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
SPONSOR_CLASSES = ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "INDIV", "NETWORK", "OTHER"]
TIMEOUT = 20


@st.cache_data(ttl=3600, show_spinner=False)
def _get_json(url, params=None):
    """GET and decode JSON. Raises on any failure, so st.cache_data never stores an error for an hour."""
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def search_trials(query="", status=None, page_size=20, phase=None, condition="", sponsor_class=None):
    """Search ongoing interventional trials; returns {"studies": [...], "total": int, "error": str | None}.

    Never raises: network problems and empty results come back as a value so the page can show a message.
    On a failure "error" is one plain sentence for users and "detail" holds the technical reason (exception and
    URL) for debugging; the page keeps the detail out of sight. Successful responses are cached for an hour,
    failures are not.
    """
    status = list(status) if status else ONGOING
    params = {
        "filter.overallStatus": ",".join(status),
        "pageSize": page_size,
        "countTotal": "true",
    }
    if query:
        params["query.term"] = query
    if condition:
        params["query.cond"] = condition
    # Only interventional studies: the model learned from those, and its score means nothing for observational ones.
    # Phase and sponsor class have no plain filter parameter either, so they join the advanced expression.
    advanced = ["AREA[StudyType]INTERVENTIONAL"]
    if phase:
        advanced.append(f"AREA[Phase]{phase}")
    if sponsor_class:
        advanced.append(f"AREA[LeadSponsorClass]{sponsor_class}")
    params["filter.advanced"] = " AND ".join(advanced)
    try:
        data = _get_json(BASE_URL, params)
    except (requests.RequestException, ValueError) as exc:
        # The exception text (a long URL, proxy and SSL messages) means nothing to a user, so it goes to "detail".
        return {"studies": [], "total": 0, "error": OFFLINE_MESSAGE, "detail": f"{type(exc).__name__}: {exc}"}
    return {"studies": data.get("studies", []), "total": data.get("totalCount", 0), "error": None}


def get_trial(nct_id):
    """Fetch the full record of one trial; returns the study dict or None if not found or unreachable.

    If the API cannot be reached, a trial from the saved examples is returned instead, so a result opened from
    the offline list still has its detail view.
    """
    try:
        return _get_json(f"{BASE_URL}/{nct_id}")
    except (requests.RequestException, ValueError):
        snapshot = load_examples() or {"studies": []}
        return next((s for s in snapshot["studies"] if summarize(s)["NCT ID"] == nct_id), None)


def load_examples(path=None):
    """The saved example trials as {"saved_on": "YYYY-MM-DD", "studies": [...]}, or None if there is no snapshot.

    Read on every call (it is small) so a test can point `path` at its own file.
    """
    try:
        data = json.loads(Path(path or EXAMPLES_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    studies = data.get("studies") if isinstance(data, dict) else None
    return {"saved_on": data.get("saved_on", "unknown"), "studies": studies} if studies else None


def search_examples(query="", phase=None, sponsor_class=None, page_size=20, path=None):
    """Search the saved example trials the way `search_trials` searches the API, as far as the snapshot allows.

    Same return shape as `search_trials`, plus "saved_on". The keyword is a plain case-insensitive match on title
    and conditions (the API's full-text search is not available offline); phase and sponsor class match exactly.
    Returns None if there is no snapshot.
    """
    snapshot = load_examples(path)
    if snapshot is None:
        return None
    words = query.strip().lower()
    matches = []
    for study in snapshot["studies"]:
        row = summarize(study)
        if words and words not in f"{row['Title']} {row['Conditions']}".lower():
            continue
        phases = study.get("protocolSection", {}).get("designModule", {}).get("phases") or ["NA"]
        if phase and phase not in phases:
            continue
        if sponsor_class and row["Sponsor type"] != sponsor_class:
            continue
        matches.append(study)
    return {"studies": matches[:page_size], "total": len(matches), "error": None, "saved_on": snapshot["saved_on"]}


def summarize(study):
    """Flatten a study into the few fields shown in the results table."""
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    return {
        "NCT ID": ident.get("nctId", ""),
        "Title": ident.get("briefTitle", ""),
        # Raw API codes joined by commas (for example "PHASE1, PHASE2"); a missing phase is "NA", like the API's own.
        # `or []`: a record may carry an explicit null instead of leaving the key out.
        "Phase": ", ".join(p.get("designModule", {}).get("phases") or []) or "NA",
        "Status": p.get("statusModule", {}).get("overallStatus", ""),
        "Sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        "Sponsor type": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("class", ""),
        "Conditions": ", ".join(p.get("conditionsModule", {}).get("conditions") or []),
    }

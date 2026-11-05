"""Thin client for the ClinicalTrials.gov API v2 (search and single-trial lookup)."""
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CATEGORY_CODES  # noqa: E402

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
ONGOING = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
# Filter choices come from the codes the whole app offers. The API filters on single phases, so the joined
# values of the phase feature ("PHASE1/PHASE2") are left out.
PHASES = [code for code in CATEGORY_CODES["phase"] if "/" not in code]
SPONSOR_CLASSES = list(CATEGORY_CODES["sponsor_class"])
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
    Successful responses are cached for an hour, failures are not.
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
        return {"studies": [], "total": 0, "error": f"Could not reach ClinicalTrials.gov: {exc}"}
    return {"studies": data.get("studies", []), "total": data.get("totalCount", 0), "error": None}


def get_trial(nct_id):
    """Fetch the full record of one trial; returns the study dict or None if not found or unreachable."""
    try:
        return _get_json(f"{BASE_URL}/{nct_id}")
    except (requests.RequestException, ValueError):
        return None


def summarize(study):
    """Flatten a study into the few fields shown in the results table."""
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    return {
        "NCT ID": ident.get("nctId", ""),
        "Title": ident.get("briefTitle", ""),
        # Raw API codes joined by commas (for example "PHASE1, PHASE2"); a missing phase is "NA", like the API's own.
        "Phase": ", ".join(p.get("designModule", {}).get("phases", [])) or "NA",
        "Status": p.get("statusModule", {}).get("overallStatus", ""),
        "Sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        "Sponsor type": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("class", ""),
        "Conditions": ", ".join(p.get("conditionsModule", {}).get("conditions", [])),
    }

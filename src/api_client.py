"""Thin client for the ClinicalTrials.gov API v2 (search and single-trial lookup)."""
import requests
import streamlit as st

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
ONGOING = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]
PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
SPONSOR_CLASSES = ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "INDIV", "NETWORK", "OTHER"]
TIMEOUT = 20


@st.cache_data(ttl=3600, show_spinner=False)
def search_trials(query="", status=None, page_size=20, phase=None, condition="", sponsor_class=None):
    """Search ongoing trials; returns {"studies": [...], "error": str | None}.

    Never raises: network problems and empty results come back as a value so the page can show a message.
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
    # Phase and sponsor class have no plain filter parameter, so they go into the advanced expression.
    advanced = []
    if phase:
        advanced.append(f"AREA[Phase]{phase}")
    if sponsor_class:
        advanced.append(f"AREA[LeadSponsorClass]{sponsor_class}")
    if advanced:
        params["filter.advanced"] = " AND ".join(advanced)
    try:
        resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return {"studies": [], "total": 0, "error": f"Could not reach ClinicalTrials.gov: {exc}"}
    return {"studies": data.get("studies", []), "total": data.get("totalCount", 0), "error": None}


@st.cache_data(ttl=3600, show_spinner=False)
def get_trial(nct_id):
    """Fetch the full record of one trial; returns the study dict or None if not found or unreachable."""
    try:
        resp = requests.get(f"{BASE_URL}/{nct_id}", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def summarize(study):
    """Flatten a study into the few fields shown in the results table."""
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    return {
        "NCT ID": ident.get("nctId", ""),
        "Title": ident.get("briefTitle", ""),
        # The API uses "NA" for "not applicable"; show one spelling for both that and a missing phase.
        "Phase": ", ".join(p.get("designModule", {}).get("phases", [])).replace("NA", "N/A") or "N/A",
        "Status": p.get("statusModule", {}).get("overallStatus", ""),
        "Sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        "Sponsor type": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("class", ""),
        "Conditions": ", ".join(p.get("conditionsModule", {}).get("conditions", [])),
    }

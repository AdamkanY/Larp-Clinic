"""Download clinical trial data from ClinicalTrials.gov and save it as CSV.

Only finished interventional trials are kept:
  - COMPLETED  -> terminated_early = 0
  - TERMINATED -> terminated_early = 1

Usage:
    pip install requests pandas
    python scripts/download_data.py            # all trials (takes a while)
    python scripts/download_data.py --limit 5000   # quick test
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "trials.csv"


def get(d, *keys, default=None):
    """Safely walk nested dicts: get(study, "a", "b") == study["a"]["b"]."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def flatten(study):
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    sponsor = p.get("sponsorCollaboratorsModule", {})
    arms = p.get("armsInterventionsModule", {})
    elig = p.get("eligibilityModule", {})
    locs = get(p, "contactsLocationsModule", "locations", default=[]) or []
    oversight = p.get("oversightModule", {})

    interventions = arms.get("interventions", []) or []
    countries = sorted({l.get("country") for l in locs if l.get("country")})

    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "overall_status": status.get("overallStatus"),
        "terminated_early": int(status.get("overallStatus") == "TERMINATED"),
        "why_stopped": status.get("whyStopped"),
        "start_date": get(status, "startDateStruct", "date"),
        "completion_date": get(status, "completionDateStruct", "date"),
        "phases": "|".join(design.get("phases", []) or []),
        "allocation": get(design, "designInfo", "allocation"),
        "intervention_model": get(design, "designInfo", "interventionModel"),
        "primary_purpose": get(design, "designInfo", "primaryPurpose"),
        "masking": get(design, "designInfo", "maskingInfo", "masking"),
        "enrollment": get(design, "enrollmentInfo", "count"),
        "enrollment_type": get(design, "enrollmentInfo", "type"),
        "lead_sponsor": get(sponsor, "leadSponsor", "name"),
        "sponsor_class": get(sponsor, "leadSponsor", "class"),
        "n_collaborators": len(sponsor.get("collaborators", []) or []),
        "n_arms": len(arms.get("armGroups", []) or []),
        "intervention_types": "|".join(sorted({i.get("type", "") for i in interventions})),
        "conditions": "|".join(get(p, "conditionsModule", "conditions", default=[]) or []),
        "sex": elig.get("sex"),
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "n_locations": len(locs),
        "n_countries": len(countries),
        "countries": "|".join(countries),
        "has_dmc": oversight.get("oversightHasDmc"),
        "fda_regulated_drug": oversight.get("isFdaRegulatedDrug"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max number of trials")
    args = parser.parse_args()

    params = {
        "filter.overallStatus": "COMPLETED,TERMINATED",
        "query.term": "AREA[StudyType]INTERVENTIONAL",
        "pageSize": 1000,
        "countTotal": "true",
    }
    rows, page = [], 0
    with requests.Session() as s:
        while True:
            r = s.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if page == 0:
                print(f"Total matching trials: {data.get('totalCount')}")
            rows += [flatten(st) for st in data.get("studies", [])]
            page += 1
            print(f"  page {page}: {len(rows)} trials so far")
            token = data.get("nextPageToken")
            if not token or (args.limit and len(rows) >= args.limit):
                break
            params["pageToken"] = token

    df = pd.DataFrame(rows[: args.limit] if args.limit else rows)
    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} trials to {OUT_FILE}")
    print(df["overall_status"].value_counts())


if __name__ == "__main__":
    main()

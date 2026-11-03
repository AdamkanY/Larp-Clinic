"""The single feature-building function shared by training and the app.

``build_features`` takes ONE ClinicalTrials.gov API v2 study record (the dict
under ``studies[i]``) and returns a flat dict of design-time features plus a few
non-feature columns (id, status, label, start date).

Leakage rule: every value is read through an explicit path below. Enrollment,
locations, whyStopped, results, completion/last-update dates and every date
other than ``startDateStruct`` are never read, so they cannot leak in even
when the record contains them (live trials from the app carry the full record).
"""

import re
from datetime import date

# Intervention types in the API enum; one 0/1 column each (multi-hot).
INTERVENTION_TYPES = [
    "DRUG", "DEVICE", "BIOLOGICAL", "PROCEDURE", "RADIATION", "BEHAVIORAL",
    "GENETIC", "DIETARY_SUPPLEMENT", "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST",
    "OTHER",
]

# Columns that identify or describe a row but are NEVER fed to the model.
META_COLUMNS = ["nct_id", "overall_status", "label", "start_date"]

CATEGORICAL_FEATURES = [
    "phase", "allocation", "intervention_model", "primary_purpose", "masking",
    "sponsor_class", "sex",
]
BINARY_FEATURES = [
    "has_collaborators", "has_dmc", "fda_regulated_drug",
    "fda_regulated_device", "healthy_volunteers",
] + [f"intervention_{t.lower()}" for t in INTERVENTION_TYPES]
NUMERIC_FEATURES = [
    "n_arms", "min_age_years", "max_age_years", "eligibility_text_len",
    "n_primary_outcomes", "n_secondary_outcomes", "n_conditions", "start_year",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

LABELS = {"TERMINATED": 1, "COMPLETED": 0}

# Hours/minutes only occur for very young infants; kept for completeness.
_AGE_UNIT_YEARS = {
    "year": 1.0, "month": 1 / 12, "week": 7 / 365.25, "day": 1 / 365.25,
    "hour": 1 / 8766, "minute": 1 / 525960,
}
_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\b", re.I)
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def _get(obj, *path):
    """Walk nested dicts and return None as soon as a key is missing."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _count(items):
    """Length of a list, or None when the field is absent (not the same as 0)."""
    return len(items) if isinstance(items, list) else None


def _flag(value):
    """Map an API boolean (or its string form) to 1/0, None if absent."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return int(value.strip().lower() == "true")
    return None


def parse_age_years(text):
    """Convert strings like '18 Years' or '6 Months' to years; 'N/A'/missing -> None."""
    if not isinstance(text, str):
        return None
    match = _AGE_RE.match(text)
    if not match:
        return None
    return float(match.group(1)) * _AGE_UNIT_YEARS[match.group(2).lower()]


def parse_start_date(text):
    """Parse 'YYYY-MM' or 'YYYY-MM-DD' to a date (month precision -> day 1).

    Returns None for anything else (year-only, garbage, impossible dates);
    those rows are treated as invalid and dropped from the dataset.
    """
    if not isinstance(text, str):
        return None
    match = _DATE_RE.match(text.strip())
    if not match:
        return None
    year, month, day = int(match.group(1)), int(match.group(2)), match.group(3)
    try:
        return date(year, month, int(day) if day else 1)
    except ValueError:
        return None


def build_features(study_json: dict) -> dict:
    """Build design-time features for one API v2 study record.

    Args:
        study_json: one element of the API's ``studies`` list.

    Returns:
        Flat dict with META_COLUMNS and FEATURE_COLUMNS keys. Missing source
        fields become None. ``label`` is 1/0 for TERMINATED/COMPLETED and None
        for any other status (e.g. ongoing trials that are only scored).
    """
    proto = _get(study_json, "protocolSection") or {}

    status = _get(proto, "statusModule", "overallStatus")
    start = parse_start_date(_get(proto, "statusModule", "startDateStruct", "date"))

    design = _get(proto, "designModule", "designInfo") or {}
    phases = _get(proto, "designModule", "phases")
    interventions = _get(proto, "armsInterventionsModule", "interventions")
    collaborators = _get(proto, "sponsorCollaboratorsModule", "collaborators")
    sponsor_module = _get(proto, "sponsorCollaboratorsModule")
    criteria = _get(proto, "eligibilityModule", "eligibilityCriteria")

    row = {
        "nct_id": _get(proto, "identificationModule", "nctId"),
        "overall_status": status,
        "label": LABELS.get(status),
        "start_date": start.isoformat() if start else None,

        "phase": "/".join(sorted(phases)) if phases else None,
        "allocation": design.get("allocation"),
        "intervention_model": design.get("interventionModel"),
        "primary_purpose": design.get("primaryPurpose"),
        "masking": _get(design, "maskingInfo", "masking"),
        "n_arms": _count(_get(proto, "armsInterventionsModule", "armGroups")),
        "sponsor_class": _get(sponsor_module, "leadSponsor", "class"),
        # No collaborators is a real value (0), but only if the module exists.
        "has_collaborators": (int(bool(collaborators)) if sponsor_module else None),
        "has_dmc": _flag(_get(proto, "oversightModule", "oversightHasDmc")),
        "fda_regulated_drug": _flag(_get(proto, "oversightModule", "isFdaRegulatedDrug")),
        "fda_regulated_device": _flag(_get(proto, "oversightModule", "isFdaRegulatedDevice")),
        "healthy_volunteers": _flag(_get(proto, "eligibilityModule", "healthyVolunteers")),
        "sex": _get(proto, "eligibilityModule", "sex"),
        "min_age_years": parse_age_years(_get(proto, "eligibilityModule", "minimumAge")),
        "max_age_years": parse_age_years(_get(proto, "eligibilityModule", "maximumAge")),
        "eligibility_text_len": len(criteria) if isinstance(criteria, str) else None,
        "n_primary_outcomes": _count(_get(proto, "outcomesModule", "primaryOutcomes")),
        "n_secondary_outcomes": _count(_get(proto, "outcomesModule", "secondaryOutcomes")),
        "n_conditions": _count(_get(proto, "conditionsModule", "conditions")),
        "start_year": start.year if start else None,
    }

    present = {i.get("type") for i in interventions or [] if isinstance(i, dict)}
    for kind in INTERVENTION_TYPES:
        # Unknown intervention list -> None, otherwise a clean 0/1.
        row[f"intervention_{kind.lower()}"] = (
            int(kind in present) if interventions is not None else None
        )
    return row

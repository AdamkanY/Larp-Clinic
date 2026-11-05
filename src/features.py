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

# Condition categories: keyword patterns matched (case-insensitive, at word starts)
# against each registered condition string. A trial can fall into several
# categories (multi-hot) or none. Only these flags reach the model, never the
# free text itself, so a model cannot memorise individual trials by name.
CONDITION_KEYWORDS = {
    "oncology": [
        r"cancers?\b", r"carcinom", r"tumou?r", r"neoplas", r"lymphom", r"leuka?emi", r"myelom",
        r"sarcom", r"melanom", r"gliom", r"glioblastom", r"\w*blastoma", r"metasta", r"malignan",
        r"oncolog", r"mesotheliom", r"myelodysplas", r"carcinoid",
    ],
    "cardio": [
        r"heart", r"cardi", r"coronary", r"myocard", r"hypertens", r"atrial fibrillation",
        r"arrhythm", r"stroke", r"cerebrovascular", r"vascular", r"atheroscl", r"thrombo",
        r"embolism", r"angina", r"aortic", r"aneurysm", r"venous", r"peripheral arter",
    ],
    "neuro": [
        r"alzheimer", r"parkinson", r"dementia", r"epilep", r"seizure", r"multiple sclerosis",
        r"migraine", r"headache", r"neuropath", r"neurolog", r"neurodegenerat", r"brain injur",
        r"spinal cord", r"amyotrophic", r"als\b", r"huntington", r"cognitive", r"cerebral palsy",
        r"neuralgia", r"concussion",
    ],
    "infectious": [
        r"infecti", r"hiv\b", r"aids\b", r"hepatitis", r"tubercul", r"malaria", r"influenza",
        r"flu\b", r"h1n1", r"covid", r"sars", r"viral\b", r"virus", r"bacteri", r"sepsis",
        r"septic", r"pneumonia",
        r"vaccin", r"hpv\b", r"papilloma", r"herpes", r"fungal", r"candid", r"parasit",
        r"cytomegalovirus", r"ebola", r"dengue", r"cholera", r"measles", r"pertussis",
        r"meningococc", r"pneumococc", r"rsv\b", r"gonorrh", r"chlamydia", r"syphilis",
        r"streptococc", r"staphylococc", r"helicobacter", r"clostridi",
    ],
    "metabolic": [
        r"diabet", r"obes", r"overweight", r"metabolic", r"insulin", r"glucose", r"glycemi",
        r"glycaemi", r"lipid", r"cholesterol", r"hyperlipid", r"dyslipid", r"thyroid", r"nafld\b",
        r"nash\b", r"fatty liver", r"gout", r"weight loss",
    ],
    "psych": [
        r"depress", r"anxiety", r"schizophren", r"bipolar", r"psychos", r"psychotic",
        r"psychiatr", r"mental", r"autis", r"adhd\b", r"attention deficit", r"ptsd\b",
        r"stress disorder", r"substance", r"alcohol", r"opioid use", r"opioid dependence",
        r"addict", r"eating disorder", r"anorexia nervosa", r"bulimi", r"obsessive", r"suicid",
        r"cocaine", r"cannabis", r"nicotine", r"smoking", r"tobacco", r"panic",
    ],
    "respiratory": [
        r"asthma", r"copd\b", r"chronic obstructive", r"pulmonary", r"lung", r"respirat",
        r"bronch", r"cystic fibrosis", r"pneumonia", r"emphysema", r"airway", r"sleep apn",
    ],
    "healthy": [r"healthy"],
}
CONDITION_CATEGORIES = list(CONDITION_KEYWORDS)
_CONDITION_RES = {
    category: re.compile(r"\b(?:" + "|".join(patterns) + ")", re.I)
    for category, patterns in CONDITION_KEYWORDS.items()
}
CONDITION_FLAG_FEATURES = [f"cond_{c}" for c in CONDITION_CATEGORIES]
# Every column condition_features adds (the v2 features; see FEATURE_SETS in train.py).
CONDITION_FEATURES = CONDITION_FLAG_FEATURES + ["n_condition_categories"]

# Columns that identify or describe a row but are NEVER fed to the model.
META_COLUMNS = ["nct_id", "overall_status", "label", "start_date"]

# Human-readable columns for the app (cards, tables, game). NEVER model inputs:
# free text / identifiers would let a model memorise trials. Overlaps META
# (nct_id, start_date, overall_status) on purpose; the app reads them from here.
DISPLAY_COLS = [
    "nct_id", "brief_title", "conditions", "lead_sponsor_name", "phase_label",
    "start_date", "overall_status",
]
# Display columns that are not already META columns (extra parquet columns).
EXTRA_DISPLAY_COLS = [c for c in DISPLAY_COLS if c not in META_COLUMNS]
# Everything in the parquet that must be excluded when training.
NON_FEATURE_COLUMNS = META_COLUMNS + EXTRA_DISPLAY_COLS + ["split"]

CATEGORICAL_FEATURES = [
    "phase", "allocation", "intervention_model", "primary_purpose", "masking",
    "sponsor_class", "sex",
]
BINARY_FEATURES = [
    "has_collaborators", "has_dmc", "healthy_volunteers", "has_max_age",
] + [f"intervention_{t.lower()}" for t in INTERVENTION_TYPES] + CONDITION_FLAG_FEATURES
NUMERIC_FEATURES = [
    "n_arms", "min_age_years", "max_age_years", "eligibility_text_len",
    "n_primary_outcomes", "n_secondary_outcomes", "n_conditions", "n_condition_categories",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

# Stand-in for a missing maximum age (no upper limit).
MAX_AGE_FILL = 120.0

LABELS = {"TERMINATED": 1, "COMPLETED": 0}

_PHASE_NAMES = {
    "EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase 1", "PHASE2": "Phase 2",
    "PHASE3": "Phase 3", "PHASE4": "Phase 4", "NA": "Not Applicable",
}

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
    """Length of a list; an absent list counts as 0 (nothing was registered)."""
    return len(items) if isinstance(items, list) else 0


def _flag(value):
    """Map an API boolean (or its string form) to 1/0, None if absent."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return int(value.strip().lower() == "true")
    return None


def phase_label(phases):
    """Readable phase, e.g. ['PHASE2','PHASE3'] -> 'Phase 2/Phase 3'; absent -> None."""
    if not phases:
        return None
    return "/".join(_PHASE_NAMES.get(p, p) for p in sorted(phases))


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


def condition_features(conditions: list[str]) -> dict:
    """Keyword multi-hot flags for the condition categories of one trial.

    Args:
        conditions: the registered condition strings (``conditionsModule.conditions``);
            None or an empty list means none were registered. Non-strings are ignored.

    Returns:
        ``{"cond_oncology": 0/1, ..., "cond_healthy": 0/1, "n_condition_categories": k}``
        where k is the number of categories with at least one matching condition.
        An absent list gives all zeros, like the other list counts.
    """
    texts = [c for c in conditions or [] if isinstance(c, str)]
    row = {
        f"cond_{category}": int(any(pattern.search(t) for t in texts))
        for category, pattern in _CONDITION_RES.items()
    }
    row["n_condition_categories"] = sum(row.values())
    return row


def build_features(study_json: dict) -> dict:
    """Build design-time features for one API v2 study record.

    Args:
        study_json: one element of the API's ``studies`` list.

    Returns:
        Flat dict with META_COLUMNS, DISPLAY_COLS and FEATURE_COLUMNS keys. Missing source
        fields become None, except counts (0) and max age (120 + flag). ``label`` is 1/0 for TERMINATED/COMPLETED and None
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

    max_age = parse_age_years(_get(proto, "eligibilityModule", "maximumAge"))
    conditions = _get(proto, "conditionsModule", "conditions")

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
        "healthy_volunteers": _flag(_get(proto, "eligibilityModule", "healthyVolunteers")),
        "sex": _get(proto, "eligibilityModule", "sex"),
        "min_age_years": parse_age_years(_get(proto, "eligibilityModule", "minimumAge")),
        # Missing max age means "no upper limit": flag it and use 120 as a
        # stand-in so the column has no gaps.
        "has_max_age": int(max_age is not None),
        "max_age_years": max_age if max_age is not None else MAX_AGE_FILL,
        "eligibility_text_len": len(criteria) if isinstance(criteria, str) else None,
        "n_primary_outcomes": _count(_get(proto, "outcomesModule", "primaryOutcomes")),
        "n_secondary_outcomes": _count(_get(proto, "outcomesModule", "secondaryOutcomes")),
        "n_conditions": _count(conditions),

        # Display-only (see DISPLAY_COLS); not in FEATURE_COLUMNS.
        "brief_title": _get(proto, "identificationModule", "briefTitle"),
        "conditions": "; ".join(
            c for c in conditions or [] if isinstance(c, str)) or None,
        "lead_sponsor_name": _get(sponsor_module, "leadSponsor", "name"),
        "phase_label": phase_label(phases),
    }

    present = {i.get("type") for i in interventions or [] if isinstance(i, dict)}
    for kind in INTERVENTION_TYPES:
        # Unknown intervention list -> None, otherwise a clean 0/1.
        row[f"intervention_{kind.lower()}"] = (
            int(kind in present) if interventions is not None else None
        )
    row.update(condition_features(conditions if isinstance(conditions, list) else None))
    return row


def study_from_design(design: dict) -> dict:
    """Turn a trial design typed into a form into a minimal API v2 study record.

    The record only holds the design-time fields ``build_features`` reads, so
    ``build_features(study_from_design(d))`` applies exactly the training rules
    (age parsing, the 120 stand-in for "no maximum age", multi-hot intervention
    and condition flags, list counts) instead of a hand-written copy of them.

    Args:
        design: any subset of these keys (absent or None = not stated):
            ``phase`` (API code, e.g. "PHASE2" or "PHASE1/PHASE2"), ``allocation``,
            ``intervention_model``, ``primary_purpose``, ``masking``,
            ``sponsor_class``, ``sex`` (API codes); ``n_arms``,
            ``n_primary_outcomes``, ``n_secondary_outcomes`` (ints);
            ``has_collaborators``, ``has_dmc``, ``healthy_volunteers`` (bools);
            ``min_age_years``, ``max_age_years`` (years; None max = no upper
            limit); ``eligibility_text_len`` (characters); ``intervention_types``
            (list of ``INTERVENTION_TYPES``); ``conditions`` (list of strings).

    Returns:
        A dict shaped like one element of the API's ``studies`` list, with
        overall status NOT_YET_RECRUITING (so it never gets a label).
    """
    def years(value):
        """Age in the API's text form, e.g. 18 -> '18 Years'; None stays absent."""
        return None if value is None else f"{float(value):g} Years"

    def placeholders(n):
        """A list of n empty entries: build_features only counts these lists."""
        return [{}] * int(n or 0)

    info = {
        "allocation": design.get("allocation"),
        "interventionModel": design.get("intervention_model"),
        "primaryPurpose": design.get("primary_purpose"),
        "maskingInfo": {"masking": design.get("masking")},
    }
    length = design.get("eligibility_text_len")
    sections = {
        "statusModule": {"overallStatus": "NOT_YET_RECRUITING"},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": design["phase"].split("/") if design.get("phase") else None,
            "designInfo": info,
        },
        "armsInterventionsModule": {
            "armGroups": placeholders(design.get("n_arms")),
            "interventions": [{"type": t} for t in design.get("intervention_types") or []],
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"class": design.get("sponsor_class")},
            "collaborators": [{}] if design.get("has_collaborators") else [],
        },
        "oversightModule": {"oversightHasDmc": design.get("has_dmc")},
        "conditionsModule": {"conditions": list(design.get("conditions") or [])},
        "outcomesModule": {
            "primaryOutcomes": placeholders(design.get("n_primary_outcomes")),
            "secondaryOutcomes": placeholders(design.get("n_secondary_outcomes")),
        },
        "eligibilityModule": {
            "healthyVolunteers": design.get("healthy_volunteers"),
            "sex": design.get("sex"),
            "minimumAge": years(design.get("min_age_years")),
            "maximumAge": years(design.get("max_age_years")),
            # Only the length is a feature; the text itself is a stand-in.
            "eligibilityCriteria": None if length is None else "x" * int(length),
        },
    }
    return {"protocolSection": sections}


def features_from_design(design: dict) -> dict:
    """Feature dict for a design typed into a form: ``build_features`` on ``study_from_design``.

    This is the helper the "Check your own trial" page uses, so a typed-in design
    and a real API record go through the same feature code.
    """
    return build_features(study_from_design(design))

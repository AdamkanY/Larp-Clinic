"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Turns one trial from ClinicalTrials.gov into the numbers and categories the model uses.

build_features() gets ONE study record from the API (one element of the "studies" list) and returns a flat
dictionary. Training and the app both call this function, so a trial is always described in the same way.

Leakage rule: we only read fields that are known when a trial is registered. Enrollment, locations, whyStopped,
results and every date except the start date are never read here, so they cannot reach the model.
"""

import re  # regular expressions: finding keywords in condition texts and reading ages and dates
from datetime import date  # the date type for the start date

# ---------------------------------------------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------------------------------------------

# Intervention types of the API. Each one becomes its own 0/1 column, because a trial can have several.
INTERVENTION_TYPES = [
    "DRUG", "DEVICE", "BIOLOGICAL", "PROCEDURE", "RADIATION", "BEHAVIORAL",
    "GENETIC", "DIETARY_SUPPLEMENT", "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST",
    "OTHER",
]
# The column name of every intervention type, for example "DRUG" -> "intervention_drug".
INTERVENTION_FEATURES = []
for kind in INTERVENTION_TYPES:
    INTERVENTION_FEATURES.append("intervention_" + kind.lower())

# Condition categories. Each registered condition is searched for these keywords (at the start of a word, upper
# and lower case do not matter). A trial can belong to several categories or to none. Only the 0/1 flags reach
# the model, never the text itself, so the model cannot learn single trials by name.
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

# One search pattern per category: "\b" means "start of a word", "|" means "or".
CONDITION_PATTERNS = {}
CONDITION_FLAG_FEATURES = []
for category in CONDITION_CATEGORIES:
    keywords = "|".join(CONDITION_KEYWORDS[category])
    CONDITION_PATTERNS[category] = re.compile(r"\b(" + keywords + ")", re.IGNORECASE)
    CONDITION_FLAG_FEATURES.append("cond_" + category)

# All columns that condition_features() adds (the "v2" features, see FEATURE_SETS in train.py).
CONDITION_FEATURES = CONDITION_FLAG_FEATURES + ["n_condition_categories"]

# Columns that describe a row but are NEVER given to the model.
META_COLUMNS = ["nct_id", "overall_status", "label", "start_date"]

# Columns the app shows to people (cards, tables, the game). Also NEVER given to the model. nct_id, start_date
# and overall_status are listed here again on purpose, because the app reads them from this list.
DISPLAY_COLS = [
    "nct_id", "brief_title", "conditions", "lead_sponsor_name", "phase_label",
    "start_date", "overall_status",
]
# The display columns that are not meta columns already.
EXTRA_DISPLAY_COLS = ["brief_title", "conditions", "lead_sponsor_name", "phase_label"]
# Everything in the data file that must be left out when training.
NON_FEATURE_COLUMNS = META_COLUMNS + EXTRA_DISPLAY_COLS + ["split"]

# The model inputs, in three groups: categories, 0/1 flags and numbers.
CATEGORICAL_FEATURES = [
    "phase", "allocation", "intervention_model", "primary_purpose", "masking",
    "sponsor_class", "sex",
]
BINARY_FEATURES = (
    ["has_collaborators", "has_dmc", "healthy_volunteers", "has_max_age"]
    + INTERVENTION_FEATURES
    + CONDITION_FLAG_FEATURES
)
NUMERIC_FEATURES = [
    "n_arms", "min_age_years", "max_age_years", "eligibility_text_len",
    "n_primary_outcomes", "n_secondary_outcomes", "n_conditions", "n_condition_categories",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

# A trial without a maximum age gets 120 years, so the column has no gaps.
MAX_AGE_FILL = 120.0

# What we predict: 1 = stopped early, 0 = finished.
LABELS = {"TERMINATED": 1, "COMPLETED": 0}

# Readable names for the phase codes of the API.
PHASE_NAMES = {
    "EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase 1", "PHASE2": "Phase 2",
    "PHASE3": "Phase 3", "PHASE4": "Phase 4", "NA": "Not Applicable",
}

# How many years one unit of age is. Hours and minutes only occur for newborns.
YEARS_PER_UNIT = {
    "year": 1.0, "month": 1 / 12, "week": 7 / 365.25, "day": 1 / 365.25,
    "hour": 1 / 8766, "minute": 1 / 525960,
}
# "18 Years" -> number 18 and unit "year".
AGE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\b", re.IGNORECASE)
# "2015-03" or "2015-03-17" -> year, month and (maybe) day.
DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------

def get_field(record, *keys):
    """Go down a nested dictionary key by key.

    Input: record is a dictionary (or anything else); keys are the key names to follow, one after the other.

    Output: the value at the end of the path. None as soon as a key is missing or a level is not a dictionary.
    """
    # Follow the path one key at a time and stop with None if the path breaks.
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def count_items(items):
    """Count the entries of a list. A missing list counts as 0, because then nothing was registered.

    Input: items is a list or None (anything that is not a list counts as missing).

    Output: the length of the list as an int, or 0 if it is not a list.
    """
    if isinstance(items, list):
        return len(items)
    return 0


def yes_no_flag(value):
    """Turn True/False (or the text "true"/"false") into 1/0.

    Input: value is a bool, a text such as "true" or "False", or anything else (for example None).

    Output: 1 for true, 0 for false, None for anything else.
    """
    # A real bool first, then the text form.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return 1
        if text == "false":
            return 0
    return None


def phase_label(phases):
    """Make a readable phase text, for example ['PHASE2', 'PHASE3'] -> 'Phase 2/Phase 3'.

    Input: phases is a list of API phase codes, or None.

    Output: the readable names joined by "/" (a text), or None if there are no phases.
    """
    if not phases:
        return None
    # Sorted, so the same phases always give the same text.
    names = []
    for phase in sorted(phases):
        names.append(PHASE_NAMES.get(phase, phase))
    return "/".join(names)


def parse_age_years(text):
    """Turn an age text of the API into years: '18 Years' -> 18.0 and '6 Months' -> 0.5.

    Input: text is the age text of the API (for example "18 Years"), or None.

    Output: the age in years as a float. None for 'N/A', for other texts or for a missing value.
    """
    if not isinstance(text, str):
        return None
    match = AGE_PATTERN.match(text)
    if not match:
        return None
    # Number times the length of its unit in years.
    number = float(match.group(1))
    unit = match.group(2).lower()
    return number * YEARS_PER_UNIT[unit]


def parse_start_date(text):
    """Turn 'YYYY-MM' or 'YYYY-MM-DD' into a date (without a day we take day 1).

    Everything else (only a year, nonsense, impossible dates) gives None. Such trials are dropped from the data.

    Input: text is the start date text of the API, or None.

    Output: a datetime.date, or None if the text is missing or not a valid date.
    """
    if not isinstance(text, str):
        return None
    match = DATE_PATTERN.match(text.strip())
    if not match:
        return None
    # Day 1 if the API gave only year and month.
    year = int(match.group(1))
    month = int(match.group(2))
    day = 1
    if match.group(3):
        day = int(match.group(3))
    # date() raises an error for impossible dates such as month 13.
    try:
        return date(year, month, day)
    except ValueError:
        return None


def condition_features(conditions):
    """Make a 0/1 flag per condition category for one trial, plus how many categories matched.

    Example result: {"cond_oncology": 1, "cond_cardio": 0, ..., "n_condition_categories": 1}

    Input: conditions is the list of registered condition texts. None or an empty list gives all zeros.

    Output: a dictionary with the columns "cond_<category>" (0 or 1) and "n_condition_categories" (the sum).
    """
    # Keep only real texts.
    texts = []
    for condition in conditions or []:
        if isinstance(condition, str):
            texts.append(condition)

    # A category is 1 as soon as one condition text contains one of its keywords.
    row = {}
    for category in CONDITION_CATEGORIES:
        found = 0
        for text in texts:
            if CONDITION_PATTERNS[category].search(text):
                found = 1
                break
        row["cond_" + category] = found
    row["n_condition_categories"] = sum(row.values())
    return row


# ---------------------------------------------------------------------------------------------------------------
# The feature function
# ---------------------------------------------------------------------------------------------------------------

def build_features(study_json):
    """Build the row for one study record of the API.

    The result has the meta columns, the display columns and all FEATURE_COLUMNS. Missing fields become None,
    except counts (0) and the maximum age (120 plus the flag has_max_age = 0). "label" is 1 for TERMINATED,
    0 for COMPLETED and None for every other status (running trials are only scored, not used for training).

    Input: study_json is one study record of the ClinicalTrials.gov API v2 (a nested dictionary).

    Output: a flat dictionary with one entry per column (meta, design features and display columns).
    """
    proto = get_field(study_json, "protocolSection") or {}

    # Read every part of the record once. Each read names its exact path (the leakage rule above).
    status = get_field(proto, "statusModule", "overallStatus")
    start = parse_start_date(get_field(proto, "statusModule", "startDateStruct", "date"))
    design = get_field(proto, "designModule", "designInfo") or {}
    phases = get_field(proto, "designModule", "phases")
    interventions = get_field(proto, "armsInterventionsModule", "interventions")
    sponsor_module = get_field(proto, "sponsorCollaboratorsModule")
    collaborators = get_field(proto, "sponsorCollaboratorsModule", "collaborators")
    criteria = get_field(proto, "eligibilityModule", "eligibilityCriteria")
    max_age = parse_age_years(get_field(proto, "eligibilityModule", "maximumAge"))
    conditions = get_field(proto, "conditionsModule", "conditions")

    # No collaborators is a real answer (0), but only if the sponsor part exists at all.
    has_collaborators = None
    if sponsor_module:
        has_collaborators = int(bool(collaborators))

    # No maximum age means "no upper limit": we store the flag 0 and 120 years as a stand-in.
    has_max_age = 1
    if max_age is None:
        has_max_age = 0
        max_age = MAX_AGE_FILL

    # Only the length of the eligibility text is used, never the text itself.
    eligibility_text_len = None
    if isinstance(criteria, str):
        eligibility_text_len = len(criteria)

    # The conditions as one text for the app, for example "Asthma; COPD".
    condition_texts = []
    for condition in conditions or []:
        if isinstance(condition, str):
            condition_texts.append(condition)

    # Put all values into one flat row.
    row = {
        # Meta columns (never model inputs).
        "nct_id": get_field(proto, "identificationModule", "nctId"),
        "overall_status": status,
        "label": LABELS.get(status),
        "start_date": start.isoformat() if start else None,

        # Design features.
        "phase": "/".join(sorted(phases)) if phases else None,
        "allocation": design.get("allocation"),
        "intervention_model": design.get("interventionModel"),
        "primary_purpose": design.get("primaryPurpose"),
        "masking": get_field(design, "maskingInfo", "masking"),
        "n_arms": count_items(get_field(proto, "armsInterventionsModule", "armGroups")),
        "sponsor_class": get_field(sponsor_module, "leadSponsor", "class"),
        "has_collaborators": has_collaborators,
        "has_dmc": yes_no_flag(get_field(proto, "oversightModule", "oversightHasDmc")),
        "healthy_volunteers": yes_no_flag(get_field(proto, "eligibilityModule", "healthyVolunteers")),
        "sex": get_field(proto, "eligibilityModule", "sex"),
        "min_age_years": parse_age_years(get_field(proto, "eligibilityModule", "minimumAge")),
        "has_max_age": has_max_age,
        "max_age_years": max_age,
        "eligibility_text_len": eligibility_text_len,
        "n_primary_outcomes": count_items(get_field(proto, "outcomesModule", "primaryOutcomes")),
        "n_secondary_outcomes": count_items(get_field(proto, "outcomesModule", "secondaryOutcomes")),
        "n_conditions": count_items(conditions),

        # Display columns (never model inputs).
        "brief_title": get_field(proto, "identificationModule", "briefTitle"),
        "conditions": "; ".join(condition_texts) or None,
        "lead_sponsor_name": get_field(sponsor_module, "leadSponsor", "name"),
        "phase_label": phase_label(phases),
    }

    # One 0/1 column per intervention type. If the list is missing we do not know, so the flags are None.
    types_in_trial = []
    for intervention in interventions or []:
        if isinstance(intervention, dict):
            types_in_trial.append(intervention.get("type"))
    for kind in INTERVENTION_TYPES:
        column = "intervention_" + kind.lower()
        if interventions is None:
            row[column] = None
        elif kind in types_in_trial:
            row[column] = 1
        else:
            row[column] = 0

    # The condition flags come last.
    if not isinstance(conditions, list):
        conditions = None
    row.update(condition_features(conditions))
    return row


# ---------------------------------------------------------------------------------------------------------------
# A design typed into the form of the app
# ---------------------------------------------------------------------------------------------------------------

def empty_entries(n):
    """Make a list with n empty entries. build_features only counts such lists, the content does not matter.

    Input: n is the wanted number of entries (a whole number, or None for 0).

    Output: a list of n empty dictionaries.
    """
    return [{}] * int(n or 0)


def age_text(years):
    """Write an age the way the API writes it, for example 18 -> '18 Years'. None stays None.

    Input: years is an age in years (a number), or None.

    Output: the age text (for example "18 Years"), or None if years is None.
    """
    if years is None:
        return None
    return f"{float(years):g} Years"


def study_from_design(design):
    """Turn the values of the form on "Check your own trial" into a small record shaped like the API's.

    This way a typed-in design goes through build_features() like a real trial, and we do not need a second copy
    of the rules (age text, the 120 stand-in, the 0/1 columns, the counts).

    The status is NOT_YET_RECRUITING, so the record never gets a label.

    Input: design is a dictionary. Every key is optional (missing or None = not stated):
    phase (API code such as "PHASE2" or "PHASE1/PHASE2"), allocation, intervention_model, primary_purpose,
    masking, sponsor_class, sex (API codes); n_arms, n_primary_outcomes, n_secondary_outcomes (whole numbers);
    has_collaborators, has_dmc, healthy_volunteers (True/False); min_age_years, max_age_years (years, no maximum
    = None); eligibility_text_len (number of characters); intervention_types (list); conditions (list of texts).

    Output: a dictionary shaped like one study record of the API ({"protocolSection": {...}}).
    """
    # "PHASE1/PHASE2" -> ["PHASE1", "PHASE2"], the form the API uses.
    phases = None
    if design.get("phase"):
        phases = design["phase"].split("/")

    # One entry per chosen intervention type.
    interventions = []
    for kind in design.get("intervention_types") or []:
        interventions.append({"type": kind})

    # Only "is the list empty or not" matters for has_collaborators.
    collaborators = []
    if design.get("has_collaborators"):
        collaborators = [{}]

    # Only the length of the eligibility text is a feature, so a row of "x" with the right length is enough.
    criteria = None
    if design.get("eligibility_text_len") is not None:
        criteria = "x" * int(design["eligibility_text_len"])

    # Build the nested record with the same paths that build_features() reads.
    protocol = {
        "statusModule": {"overallStatus": "NOT_YET_RECRUITING"},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": phases,
            "designInfo": {
                "allocation": design.get("allocation"),
                "interventionModel": design.get("intervention_model"),
                "primaryPurpose": design.get("primary_purpose"),
                "maskingInfo": {"masking": design.get("masking")},
            },
        },
        "armsInterventionsModule": {
            "armGroups": empty_entries(design.get("n_arms")),
            "interventions": interventions,
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"class": design.get("sponsor_class")},
            "collaborators": collaborators,
        },
        "oversightModule": {"oversightHasDmc": design.get("has_dmc")},
        "conditionsModule": {"conditions": list(design.get("conditions") or [])},
        "outcomesModule": {
            "primaryOutcomes": empty_entries(design.get("n_primary_outcomes")),
            "secondaryOutcomes": empty_entries(design.get("n_secondary_outcomes")),
        },
        "eligibilityModule": {
            "healthyVolunteers": design.get("healthy_volunteers"),
            "sex": design.get("sex"),
            "minimumAge": age_text(design.get("min_age_years")),
            "maximumAge": age_text(design.get("max_age_years")),
            "eligibilityCriteria": criteria,
        },
    }
    return {"protocolSection": protocol}


def features_from_design(design):
    """Build the feature row for a design typed into the form (used by the page "Check your own trial").

    Input: design is the dictionary of form values described in study_from_design().

    Output: the flat feature dictionary from build_features().
    """
    return build_features(study_from_design(design))

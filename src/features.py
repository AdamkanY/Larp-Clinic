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

# Condition categories found by keyword rules in the free-text condition list
# (see condition_features); one 0/1 column each (multi-hot).
CONDITION_CATEGORIES = [
    "oncology", "cardio", "neuro", "infectious", "metabolic", "psych",
    "respiratory", "healthy",
]
CONDITION_FLAGS = [f"cond_{name}" for name in CONDITION_CATEGORIES]

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
INTERVENTION_FLAGS = [f"intervention_{t.lower()}" for t in INTERVENTION_TYPES]
# New columns go at the END of each list: dropping V2_FEATURES then gives the
# v1 model's inputs in exactly their old order (train.py relies on this).
BINARY_FEATURES = [
    "has_collaborators", "has_dmc", "healthy_volunteers", "has_max_age",
] + INTERVENTION_FLAGS + CONDITION_FLAGS
NUMERIC_FEATURES = [
    "n_arms", "min_age_years", "max_age_years", "eligibility_text_len",
    "n_primary_outcomes", "n_secondary_outcomes", "n_conditions",
    "n_condition_categories", "n_intervention_types",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES
# Features added in model v2 (condition text and intervention breadth); every
# other feature column is a v1 feature.
V2_FEATURES = CONDITION_FLAGS + ["n_condition_categories", "n_intervention_types"]
V1_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in V2_FEATURES]

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

# Keyword rules for the condition categories, matched against one lower-cased
# condition (clause) at a time. Built from the most frequent of the ~49,000
# distinct condition strings of the 2008-2019 trials, then checked for false
# hits; the lookarounds keep out look-alikes such as "ocular hypertension",
# "respiratory depression", "non-alcoholic fatty liver", "bariatric surgery
# candidate", "neuromuscular blockade", "heartburn" and "mycosis fungoides".
_TUMOUR_TERMS = [
    r"cancer", r"carcinoma", r"neoplas", r"tumou?r", r"lymphoma", r"leuka?emi",
    r"myeloma", r"melanoma", r"sarcoma", r"glioma", r"blastoma", r"astrocytoma",
    r"ependymoma", r"meningioma", r"mesothelioma", r"carcinoid", r"adenoma", r"metasta",
    r"oncolog", r"(?<!neuroleptic )malignan(?!t hyperthermia|t hypertension)",
    r"myelodysplas", r"myeloproliferative", r"myelofibrosis", r"polycyth(?:a)?emia vera",
    r"essential thrombocyth", r"macroglobulin", r"mycosis fungoides", r"s[eé]zary",
    r"hodgkin", r"seminoma", r"plasmacytoma", r"(?<!lympho)histiocytosis", r"gammopathy",
    r"lymphoproliferative", r"chemotherap", r"radiotherap", r"radiation therapy",
    r"radiation[- ]induced", r"\bbrca",
    r"\b(?:n?sclc|aml|cml|cll|mds|hcc|rcc|crc|gist|nhl|dlbcl|tnbc|mpn|hnscc|scchn|scc)\b",
]
# Per category: (disease terms, organ terms). Disease terms always count;
# organ terms are ignored in a clause that names a tumour, so "Non-small cell
# lung cancer" and "Brain metastases" are oncology only, while "Lung
# transplant" is respiratory and "Depression in cancer patients" is oncology
# and psych. Stroke and dementia count as neuro.
_CONDITION_TERMS = {
    "oncology": (_TUMOUR_TERMS, []),
    "cardio": ([
        r"heart(?!burn)", r"cardi(?!a\b)", r"coronary", r"myocard", r"angina", r"atrial",
        r"\bventricular", r"arrh?ythmi", r"fibrillation",
        r"(?<!ocular )(?<!intraocular )(?<!intracranial )(?<!portal )(?<!intra-abdominal )"
        r"hypertensi(?!on,? (?:ocular|portal|intracranial))",
        r"blood pressure", r"eclampsia", r"atheroscl", r"arterioscl", r"aort", r"mitral",
        r"tricuspid", r"thrombos[ie]s", r"thromboembol", r"embolism", r"embolus",
        r"\b(?:dvt|vte|n?stemi|cad|chf|cabg)\b", r"peripheral arter", r"peripheral vascular",
        r"\bvascular diseases?", r"limb isch(?:a)?emia", r"(?<!neurogenic )claudication",
        r"varicose", r"venous insufficiency", r"endothelial (?:dys)?function",
    ], []),
    "neuro": ([
        r"(?<!heat )stroke(?! volume)", r"epilep", r"seizure", r"(?<!electro)convulsi",
        r"alzheimer", r"dementia", r"parkinson", r"multiple sclerosis", r"amyotrophic",
        r"lateral sclerosis", r"\bals\b", r"huntington", r"migraine", r"headache",
        r"neuropath", r"neuralg", r"neuritis", r"neurodegenerat", r"neurolog",
        r"neuromyelitis", r"neurocognitive", r"neurodevelopment", r"myasthenia",
        r"muscular dystroph", r"duchenne", r"spinal muscular", r"ataxia", r"dystonia",
        r"tremor", r"concussion", r"encephalopath", r"encephalitis", r"encephalomyelitis",
        r"meningitis", r"guillain", r"restless legs", r"narcolep", r"palsy", r"hydrocephalus",
        r"cognitive (?:impairment|decline|dysfunction|disorder|deficit)", r"impaired cognition",
        r"cognition disorder", r"delirium", r"\bcoma\b", r"radiculopath", r"sciatica",
        r"carpal tunnel", r"spasticity", r"paralysis", r"(?<!gastro)paresis", r"hemipleg",
        r"parapleg", r"tetrapleg", r"quadripleg", r"aphasia", r"\bcidp\b",
        r"neuromuscular (?:disease|disorder)", r"motor neuron",
        r"nerve (?:injur|damage|lesion|compression)", r"subarachnoid", r"intracerebral",
    ], [
        r"brain", r"cerebr", r"spinal cord", r"\bcns\b", r"central nervous", r"nervous system",
        r"intracranial", r"meningeal",
    ]),
    "infectious": ([
        r"(?<!non-)(?<!non)(?<!non )infect", r"\bseps[ie]s", r"\bseptic", r"bacter", r"viral",
        r"virus", r"\bhiv", r"human immunodeficiency", r"(?<!hearing )\baids\b",
        r"hepatitis [a-e]\b", r"\b(?:hbv|hcv|cmv|ebv|rsv|hpv|mrsa|uti|stis?)\b", r"tubercul",
        r"malaria", r"influenza", r"\bflu\b", r"pneumonia", r"coccal", r"streptococ",
        r"staphylococ", r"coccidioid", r"clostridi", r"difficile", r"fungal", r"fungemia",
        r"\bmycoses\b", r"onychomycosis", r"(?<!\w)mycosis(?! fungoides)",
        r"\bcandid(?:a|iasis|iases|emia|osis)\b", r"aspergill", r"cryptococ", r"tinea",
        r"parasit", r"helminth", r"covid", r"sars-cov", r"coronavirus", r"vaccin",
        r"(?<!iso)immuni[sz]ation", r"herpe", r"zoster", r"varicella", r"cytomegalo",
        r"epstein-barr", r"syncytial", r"papilloma ?virus", r"ebola", r"zika", r"dengue",
        r"chikungunya", r"yellow fever", r"cholera", r"typhoid", r"shigell", r"salmonell",
        r"escherichia", r"e\. ?coli", r"rotavirus", r"norovirus", r"measles", r"mumps",
        r"rubella", r"pertussis", r"whooping", r"diphtheria", r"tetanus", r"polio", r"rabies",
        r"anthrax", r"smallpox", r"\bm(?:onkey)?pox", r"chlamydia", r"gonorrh", r"syphilis",
        r"trichomon", r"leishmania", r"trypanosom", r"chagas", r"schistosom", r"hookworm",
        r"filaria", r"onchocerc", r"toxoplasm", r"cryptosporid", r"giardia", r"amoeb",
        r"leprosy", r"\blyme\b", r"otitis media", r"cellulitis", r"abscess", r"osteomyelitis",
        r"endocarditis", r"pharyngitis", r"tonsillitis", r"gastroenteritis", r"vaginosis",
        r"\bpylori\b", r"nosocomial", r"antibiotic", r"antimicrobial", r"antiretroviral",
        r"antimalarial", r"antifungal", r"sexually transmitted",
        r"(?:japanese|tick[- ]borne|equine) encephalitis", r"encephalitis, (?:japanese|tick)",
    ], []),
    "metabolic": ([
        r"diabet(?!es insipidus)", r"insulin(?!oma)", r"glyc(?:a)?emi", r"obes", r"overweight",
        r"bariatric", r"adiposity", r"body weight", r"^weight$",
        r"(?<!birth )\bweight (?:loss|gain|reduction|management|maintenance|control|change)",
        r"(?<!phospho)lipid(?:a)?emi", r"lipidosis", r"lipid metabolism", r"lipid disorder",
        r"^lipids?$", r"lipoprotein", r"cholesterol", r"triglycerid",
        r"metabolic (?:syndrome|disease|disorder|disturbance|risk)", r"cardiometabolic",
        r"metabolism and nutrition", r"nutritional and metabolic", r"inborn error",
        r"metabolism, inborn", r"glucose (?:intolerance|tolerance|metabolism)",
        r"impaired fasting glucose", r"blood (?:glucose|sugar)", r"fatty liver",
        r"\b(?:nafld|nash|masld|mash|pku)\b", r"steatohepatitis", r"steatosis", r"\bgout",
        r"hyperuric", r"phenylketon", r"lysosomal", r"fabry", r"gaucher", r"pompe",
        r"mucopolysacchar", r"glycogen storage", r"lipodystrophy",
    ], []),
    "psych": ([
        r"(?<!respiratory )depress", r"anxi", r"schizo", r"bipolar", r"psychos[ie]s",
        r"psychotic", r"\bmental", r"psychiatr", r"psycholog", r"post[- ]?traumatic stress",
        r"\bptsd\b", r"stress disorder", r"obsessive", r"\bocd\b", r"\bpanic", r"\bphobi",
        r"agoraphobi", r"\badhd\b", r"attention deficit", r"autis", r"asperger",
        r"eating disorder", r"anorexia nervosa", r"bulimi", r"binge",
        r"(?<!non)(?<!non-)(?<!non )alcohol(?:ism| use| abuse| dependen| drinking| consumption"
        r"| withdrawal| intoxication| addiction|-related disorder| related disorder| misuse)",
        r"^alcohol$", r"smok", r"tobacco", r"nicotine", r"cigarette",
        r"opioid[- ](?:use|dependen|abuse|addiction|withdrawal|misuse|overdose|related disorder)",
        r"heroin", r"cocaine", r"amphetamine", r"cannabis use", r"marijuana use", r"addict",
        r"substance[- ](?:use|abuse|related|dependen|misuse)",
        r"drug (?:abuse|addiction|dependen|use disorder)", r"gambling", r"suicid", r"self-harm",
        r"insomnia", r"sleep initiation", r"personality disorder", r"tourette",
        r"\btic disorder", r"conduct disorder", r"\bmood",
    ], []),
    "respiratory": ([
        r"asthma", r"\bcopd\b", r"chronic obstructive", r"emphysema", r"bronchitis",
        r"bronchiectasis", r"bronchiolitis", r"bronchospasm", r"bronchopulmonary",
        r"pneumonia", r"pneumonitis", r"pneumothorax", r"cystic fibrosis",
        r"\b(?:ipf|ild|ards)\b", r"sarcoidosis", r"respiratory distress", r"apn(?:o)?ea",
        r"dyspn(?:o)?ea", r"cough", r"wheez", r"rhinitis", r"sinusitis", r"nasal polyp",
        r"hay fever", r"airway (?:obstruction|disease|inflammation|hyperresponsiveness|remodeling)",
        r"reactive airway", r"mechanical(?:ly)? ventilat", r"ventilator",
    ], [
        r"lung", r"\bpulmonary", r"pleura", r"bronch", r"respiratory",
    ]),
    "healthy": ([
        r"\bhealthy\b", r"\bheathy\b", r"\bvolunteers?\b",
        r"\bnormal (?:healthy )?(?:volunteers?|subjects?|controls?|physiology)\b",
    ], []),
}
_TUMOUR_RE = re.compile("|".join(_TUMOUR_TERMS))
_CONDITION_RES = {
    name: (re.compile("|".join(disease)), re.compile("|".join(organ)) if organ else None)
    for name, (disease, organ) in _CONDITION_TERMS.items()
}
assert list(_CONDITION_RES) == CONDITION_CATEGORIES


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
    """Multi-hot condition-category flags from a trial's free-text condition list.

    Each condition is matched against keyword rules (``_CONDITION_TERMS``) for
    oncology, cardio(vascular), neuro(logical), infectious, metabolic
    (diabetes, obesity, lipids, fatty liver, inborn errors), psych(iatric,
    including substance use and insomnia), respiratory and healthy
    (volunteers). A trial can be in several categories, or in none (e.g. pain,
    osteoarthritis, kidney disease).

    The list is joined with "; " and split on every semicolon before
    matching. The result therefore depends only on the joined text, so the
    ``conditions`` column of ``trials.parquet`` (the same "; "-joined text)
    split on "; " gives exactly what ``build_features`` computed from the raw
    list, even when a condition name contains a semicolon.

    Args:
        conditions: ``conditionsModule.conditions`` of one API record. None
            means no conditions; non-string items are skipped; a single string
            counts as one condition.

    Returns:
        ``{"cond_<category>": 0/1 for each of CONDITION_CATEGORIES,
        "n_condition_categories": number of flags set}``.
    """
    if isinstance(conditions, str):
        conditions = [conditions]
    text = "; ".join(c for c in conditions or [] if isinstance(c, str)).lower()
    hits = set()
    for clause in (part.strip() for part in text.split(";")):
        if not clause:
            continue
        tumour = bool(_TUMOUR_RE.search(clause))
        for name, (disease, organ) in _CONDITION_RES.items():
            if disease.search(clause) or (organ is not None and not tumour
                                          and organ.search(clause)):
                hits.add(name)
    flags = {f"cond_{name}": int(name in hits) for name in CONDITION_CATEGORIES}
    flags["n_condition_categories"] = len(hits)
    return flags


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
        "n_conditions": _count(_get(proto, "conditionsModule", "conditions")),

        # Display-only (see DISPLAY_COLS); not in FEATURE_COLUMNS.
        "brief_title": _get(proto, "identificationModule", "briefTitle"),
        "conditions": "; ".join(
            c for c in _get(proto, "conditionsModule", "conditions") or []
            if isinstance(c, str)) or None,
        "lead_sponsor_name": _get(sponsor_module, "leadSponsor", "name"),
        "phase_label": phase_label(phases),
    }

    present = {i.get("type") for i in interventions or [] if isinstance(i, dict)}
    for kind in INTERVENTION_TYPES:
        # Unknown intervention list -> None, otherwise a clean 0/1.
        row[f"intervention_{kind.lower()}"] = (
            int(kind in present) if interventions is not None else None
        )
    # Like the other counts, an absent list counts as 0. Only the enum types
    # are counted, so this always equals the sum of the intervention_* flags.
    row["n_intervention_types"] = sum(kind in present for kind in INTERVENTION_TYPES)
    row.update(condition_features(_get(proto, "conditionsModule", "conditions")))
    return row

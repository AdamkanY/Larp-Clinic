"""Plain-English names for the codes the ClinicalTrials.gov API and the processed data use, and for the model's
input columns.

One place for every page, so the same thing always carries the same words (for example "No phase" instead of
"NA", "Not applicable" or "N/A"). Pure Python: no Streamlit, no model code.
"""

NOT_STATED = "Not stated"

# Keys cover both the API spelling (PHASE1/PHASE2) and the spelling in trials.parquet (Phase 1/Phase 2).
PHASE = {
    "EARLY_PHASE1": "Early phase 1", "PHASE1": "Phase 1", "PHASE1/PHASE2": "Phase 1/2", "PHASE2": "Phase 2",
    "PHASE2/PHASE3": "Phase 2/3", "PHASE3": "Phase 3", "PHASE4": "Phase 4", "NA": "No phase", "N/A": "No phase",
    "Early Phase 1": "Early phase 1", "Phase 1": "Phase 1", "Phase 1/Phase 2": "Phase 1/2", "Phase 2": "Phase 2",
    "Phase 2/Phase 3": "Phase 2/3", "Phase 3": "Phase 3", "Phase 4": "Phase 4", "Not Applicable": "No phase",
}
PHASE_ORDER = ["Early phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "No phase"]

STATUS = {
    "RECRUITING": "Recruiting", "NOT_YET_RECRUITING": "Not yet recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting", "ENROLLING_BY_INVITATION": "By invitation",
    "COMPLETED": "Completed", "TERMINATED": "Stopped early",
}
SPONSOR = {
    "INDUSTRY": "Industry", "NIH": "NIH", "FED": "Other US federal", "OTHER_GOV": "Other government",
    "INDIV": "Individual", "NETWORK": "Research network", "OTHER": "Academic / other",
    "UNKNOWN": "Unknown", "AMBIG": "Unclear",
}
ALLOCATION = {"RANDOMIZED": "Randomized", "NON_RANDOMIZED": "Not randomized", "NA": "Single group"}
MASKING = {
    "NONE": "None (open label)", "SINGLE": "Single blind", "DOUBLE": "Double blind",
    "TRIPLE": "Triple blind", "QUADRUPLE": "Quadruple blind",
}
MODEL = {
    "SINGLE_GROUP": "Single group", "PARALLEL": "Parallel groups", "CROSSOVER": "Crossover",
    "FACTORIAL": "Factorial", "SEQUENTIAL": "Sequential",
}
PURPOSE = {
    "TREATMENT": "Treatment", "PREVENTION": "Prevention", "DIAGNOSTIC": "Diagnosis",
    "SUPPORTIVE_CARE": "Supportive care", "SCREENING": "Screening",
    "HEALTH_SERVICES_RESEARCH": "Health services research", "BASIC_SCIENCE": "Basic science",
    "DEVICE_FEASIBILITY": "Device feasibility", "OTHER": "Other",
}
SEX = {"ALL": "All", "FEMALE": "Female only", "MALE": "Male only"}

NAMES = {
    "phase": PHASE, "status": STATUS, "sponsor_class": SPONSOR, "allocation": ALLOCATION, "masking": MASKING,
    "intervention_model": MODEL, "primary_purpose": PURPOSE, "sex": SEX,
}


def label(kind, code):
    """Friendly name of an API code, for example label("masking", "DOUBLE") -> "Double blind".

    Unknown codes are cleaned up ("SOME_CODE" -> "Some code"); a missing value becomes "Not stated".
    """
    if code is None or code != code or code == "":  # None, NaN and the empty string
        return NOT_STATED
    return NAMES.get(kind, {}).get(code) or str(code).replace("_", " ").capitalize()

# Plain-English names for the model's input columns (FEATURE_COLUMNS in src/features.py). The Explain page,
# the per-trial driver charts and the "what would lower the risk" table all read them from here.
FEATURE_NAMES = {
    "phase": "Phase",
    "allocation": "Allocation (randomized or not)",
    "intervention_model": "Intervention model",
    "primary_purpose": "Primary purpose",
    "masking": "Blinding",
    "sponsor_class": "Sponsor type",
    "sex": "Sexes eligible",
    "has_collaborators": "Has collaborators",
    "has_dmc": "Has a data monitoring committee",
    "healthy_volunteers": "Accepts healthy volunteers",
    "has_max_age": "Has a maximum age",
    "intervention_drug": "Tests a drug",
    "intervention_device": "Tests a device",
    "intervention_biological": "Tests a biological",
    "intervention_procedure": "Tests a procedure",
    "intervention_radiation": "Tests radiation",
    "intervention_behavioral": "Behavioral intervention",
    "intervention_genetic": "Tests a genetic intervention",
    "intervention_dietary_supplement": "Tests a dietary supplement",
    "intervention_combination_product": "Tests a combination product",
    "intervention_diagnostic_test": "Tests a diagnostic test",
    "intervention_other": "Other intervention type",
    "cond_oncology": "Condition: cancer",
    "cond_cardio": "Condition: heart and blood vessels",
    "cond_neuro": "Condition: brain and nerves",
    "cond_infectious": "Condition: infection",
    "cond_metabolic": "Condition: diabetes and metabolism",
    "cond_psych": "Condition: mental health and addiction",
    "cond_respiratory": "Condition: lungs and breathing",
    "cond_healthy": "Studies healthy people",
    "n_arms": "Number of arms",
    "min_age_years": "Minimum age",
    "max_age_years": "Maximum age",
    "eligibility_text_len": "Length of eligibility criteria",
    "n_primary_outcomes": "Number of primary outcomes",
    "n_secondary_outcomes": "Number of secondary outcomes",
    "n_conditions": "Number of conditions",
    "n_condition_categories": "Number of condition areas",
}
# Same value as MAX_AGE_FILL in src/features.py (the stand-in for "no maximum age"); tests check they match.
NO_MAX_AGE = 120.0


def feature_name(column):
    """Plain name of a model input column, for example "has_dmc" -> "Has a data monitoring committee"."""
    return FEATURE_NAMES.get(column) or column.replace("_", " ").capitalize()


def feature_value(column, value):
    """A feature value in plain words, for example ("masking", "DOUBLE") -> "Double blind", ("has_dmc", 1) -> "Yes".

    Categories go through `label`, 0/1 flags become Yes/No, ages and lengths get their unit; gaps are "Not stated".
    """
    if column in NAMES:
        return label(column, value)
    if value is None or value != value:  # None and NaN
        return NOT_STATED
    if column.startswith(("has_", "intervention_", "cond_")) or column == "healthy_volunteers":
        return "Yes" if value else "No"
    if column == "max_age_years" and float(value) >= NO_MAX_AGE:
        return "No upper limit"
    if column.endswith("_age_years"):
        return f"{float(value):g} years"
    if column == "eligibility_text_len":
        return f"{int(value):,} characters"
    return f"{float(value):g}"

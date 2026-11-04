"""Plain-English names for the codes the ClinicalTrials.gov API and the processed data use.

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

"""Check your own trial: enter a design from scratch and see its predicted termination risk."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
from risk_gauge import bar_caption, estimate_caption, risk_bar, risk_card  # noqa: E402
from src.features import (CONDITION_CATEGORIES, CONDITION_FLAGS,  # noqa: E402
                          FEATURE_COLUMNS, INTERVENTION_TYPES, MAX_AGE_FILL,
                          condition_features)
from ui import chip, page_header  # noqa: E402

UNSET = "Not specified"
DEFAULT_CONDITIONS = "Type 2 Diabetes"
# Plain names for the condition areas that src.features.condition_features recognises.
AREA_NAMES = {
    "oncology": "Cancer", "cardio": "Heart and blood vessels", "neuro": "Brain and nerves",
    "infectious": "Infection", "metabolic": "Diabetes and metabolism", "psych": "Mental health",
    "respiratory": "Lungs and breathing", "healthy": "Healthy volunteers",
}

# Allowed values per categorical feature, as the ClinicalTrials.gov API spells them.
CHOICES = {
    "phase": ["EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2", "PHASE2", "PHASE2/PHASE3", "PHASE3", "PHASE4", "NA"],
    "allocation": ["RANDOMIZED", "NON_RANDOMIZED", "NA"],
    "intervention_model": ["SINGLE_GROUP", "PARALLEL", "CROSSOVER", "FACTORIAL", "SEQUENTIAL"],
    "primary_purpose": ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", "SCREENING",
                        "HEALTH_SERVICES_RESEARCH", "BASIC_SCIENCE", "DEVICE_FEASIBILITY", "OTHER"],
    "masking": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
    "sponsor_class": ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "INDIV", "NETWORK", "OTHER"],
    "sex": ["ALL", "FEMALE", "MALE"],
}
NAMES = {
    "phase": "Phase", "allocation": "Allocation", "intervention_model": "Intervention model",
    "primary_purpose": "Primary purpose", "masking": "Masking", "sponsor_class": "Sponsor type",
    "sex": "Sex eligible",
}
# A typical randomized phase 2 industry drug trial, so the page opens with a realistic example instead of blanks.
DEFAULTS = {
    "phase": "PHASE2", "allocation": "RANDOMIZED", "intervention_model": "PARALLEL",
    "primary_purpose": "TREATMENT", "masking": "DOUBLE", "sponsor_class": "INDUSTRY", "sex": "ALL",
}
# 0/1 design flags the what-if table may toggle.
FLAGS = {"has_collaborators": "Has collaborators", "has_dmc": "Has data monitoring committee"}


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server; None while src/predict.py or models/gbm.joblib is missing."""
    try:
        from src.predict import load_model
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


def pick(label, key, help_text=None):
    """Selectbox for one categorical feature; 'Not specified' becomes None (a missing value)."""
    options = [UNSET] + CHOICES[key]
    choice = st.selectbox(label, options, index=options.index(DEFAULTS[key]), key=key, help=help_text)
    return None if choice == UNSET else choice


def parse_conditions(text):
    """Split the conditions box on semicolons; commas stay, since names like 'Carcinoma, Hepatocellular' use them."""
    return [part.strip() for part in (text or "").split(";") if part.strip()]


def conditions_input(row):
    """Conditions box: fills n_conditions and the condition-area features (via condition_features) and shows the areas."""
    text = st.text_input(
        "Conditions studied", DEFAULT_CONDITIONS, key="conditions",
        help="Separate several conditions with semicolons, for example 'Breast Cancer; Depression'. "
             "They are grouped into broad areas such as cancer or mental health, as for registered trials.")
    conditions = parse_conditions(text)
    row["n_conditions"] = len(conditions)
    # The same keyword rules build_features applies to a registered trial's condition list.
    row.update(condition_features(conditions))
    areas = [AREA_NAMES[name] for name in CONDITION_CATEGORIES if row[f"cond_{name}"]]
    if not conditions:
        chips = chip("No condition entered", "gray")
    else:
        chips = "".join(chip(area) for area in areas) or chip("No specific area recognised", "gray")
    st.markdown(f"Recognised areas: {chips}", unsafe_allow_html=True)
    model = get_model()
    if model is not None:
        from src.predict import model_columns
        # The app model is v1, which only counts the conditions; the note goes away for a model that uses the areas.
        if not set(CONDITION_FLAGS) & set(model_columns(model)):
            st.caption("For information only: the current model uses how many conditions you enter, not their areas.")


def feature_row():
    """Render the form and return the feature dict in the exact shape src.features.build_features produces."""
    row = {}
    conditions_input(row)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Design")
        row["phase"] = pick("Phase", "phase")
        row["allocation"] = pick("Allocation", "allocation")
        row["intervention_model"] = pick("Intervention model", "intervention_model")
        row["primary_purpose"] = pick("Primary purpose", "primary_purpose")
        row["masking"] = pick("Masking (blinding)", "masking")
        row["n_arms"] = st.number_input("Number of arms / groups", 0, 20, 2)
    with c2:
        st.subheader("Sponsor and oversight")
        row["sponsor_class"] = pick("Lead sponsor type", "sponsor_class")
        row["has_collaborators"] = int(st.checkbox("Has collaborators", value=False))
        row["has_dmc"] = int(st.checkbox("Has a data monitoring committee", value=False))
        st.subheader("Outcomes")
        row["n_primary_outcomes"] = st.number_input("Primary outcomes", 0, 20, 1)
        row["n_secondary_outcomes"] = st.number_input("Secondary outcomes", 0, 50, 3)
    with c3:
        st.subheader("Participants")
        row["sex"] = pick("Sex eligible", "sex")
        row["healthy_volunteers"] = int(st.checkbox("Accepts healthy volunteers", value=False))
        row["min_age_years"] = st.number_input("Minimum age (years)", 0.0, 120.0, 18.0)
        row["has_max_age"] = int(st.checkbox("Has a maximum age", value=False))
        # No upper limit is stored as 120 plus the flag, exactly as build_features does.
        row["max_age_years"] = (
            st.number_input("Maximum age (years)", 0.0, 120.0, 65.0) if row["has_max_age"] else MAX_AGE_FILL
        )
        row["eligibility_text_len"] = st.number_input(
            "Length of eligibility criteria (characters)", 0, 30000, 1500, step=100,
            help="A short paragraph is about 500 characters; long inclusion/exclusion lists reach several thousand.",
        )
        types = st.multiselect("Intervention types", INTERVENTION_TYPES, default=["DRUG"])
    for kind in INTERVENTION_TYPES:
        row[f"intervention_{kind.lower()}"] = int(kind in types)
    row["n_intervention_types"] = len(types)
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    assert not missing, f"form does not fill: {missing}"
    return row


def lowering_changes(row, score, top=5):
    """Best single change per setting that lowers the risk, strongest first, as (description, new risk, change)."""
    base = score(row)
    best = []
    for key, values in CHOICES.items():
        tries = [(value, score({**row, key: value})) for value in values if value != row[key]]
        if tries:
            value, risk = min(tries, key=lambda t: t[1])
            best.append((f"{NAMES[key]}: {row[key] or UNSET} -> {value}", risk))
    for key, label in FLAGS.items():
        flipped = 1 - row[key]
        best.append((f"{label}: {'yes' if row[key] else 'no'} -> {'yes' if flipped else 'no'}",
                     score({**row, key: flipped})))
    best.sort(key=lambda t: t[1])
    return [(text, risk, risk - base) for text, risk in best[:top] if risk < base]


st.set_page_config(page_title="Check your trial - Trial-Radar", page_icon="🧪", layout="wide")
page_header("🧪 Check your own trial",
            "Describe a trial design and see how likely it is to be stopped early. Change any value and the estimate updates straight away.")
st.subheader("What does the trial study?")

row = feature_row()
st.divider()

if get_model() is None:
    st.info("Model coming soon. The estimate appears here once the model is trained.")
    st.stop()

from src.predict import score  # noqa: E402

try:
    risk = score(row)
except ValueError as exc:
    st.error(f"The model could not score this design: {exc}")
    st.stop()

previous = st.session_state.get("previous_risk")
st.session_state["previous_risk"] = risk

st.subheader("Estimated termination risk")
changed = previous is not None and abs(risk - previous) >= 0.0005  # hide a meaningless "+0.0%"
st.markdown(risk_card(risk, delta=risk - previous if changed else None), unsafe_allow_html=True)
bar = risk_bar(risk)
if bar is not None:
    st.plotly_chart(bar, width="stretch")
st.caption("Rough estimate for comparing trials, not an exact chance. Not medical advice.")
with st.expander("How to read this"):
    st.write(bar_caption())
    st.write(estimate_caption(risk))

st.subheader("What would lower the risk?")
changes = lowering_changes(row, score)
if changes:
    st.table(pd.DataFrame(
        [(text, f"{new:.1%}", f"{delta:+.1%}") for text, new, delta in changes],
        columns=["Single change", "New risk", "Change"],
    ).set_index("Single change"))
    st.caption(
        "Patterns in past trials, not proof of cause: changing a design does not guarantee this effect. "
        "Rare options (for example sponsor type \"Individual\") rest on few trials and are less reliable."
    )
else:
    st.write("No single change in these settings lowers the estimate.")

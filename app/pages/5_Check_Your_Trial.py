"""Check your own trial: enter a design from scratch and see its predicted termination risk."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
from risk_gauge import gauge_caption, risk_gauge  # noqa: E402
from src.features import FEATURE_COLUMNS, INTERVENTION_TYPES, MAX_AGE_FILL  # noqa: E402

UNSET = "Not specified"

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


def feature_row():
    """Render the form and return the feature dict in the exact shape src.features.build_features produces."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Design")
        row = {"phase": pick("Phase", "phase")}
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
        row["n_conditions"] = st.number_input("Conditions studied", 0, 20, 1)
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
st.title("🧪 Check your own trial")
st.write(
    "Describe a clinical trial design below and see how likely it is to be terminated early. "
    "Change any value and the estimate updates straight away."
)

row = feature_row()
st.divider()

if get_model() is None:
    st.info("Model coming soon. The estimate appears here once the model is trained.")
    st.stop()

from src.predict import percentile, score  # noqa: E402

try:
    risk = score(row)
except ValueError as exc:
    st.error(f"The model could not score this design: {exc}")
    st.stop()

previous = st.session_state.get("previous_risk")
st.session_state["previous_risk"] = risk

left, right = st.columns([1, 1])
with left:
    st.subheader("Estimated termination risk")
    st.plotly_chart(risk_gauge(risk), width="stretch")
    st.caption(gauge_caption())
with right:
    st.metric("Risk", f"{risk:.1%}", delta=None if previous is None else f"{risk - previous:+.1%}",
              delta_color="inverse", help="Change compared with your previous input.")
    try:
        st.write(f"Riskier than **{percentile(risk):.0f}%** of historical trials (started 2008-2016).")
    except (ImportError, FileNotFoundError):
        pass  # models/score_reference.npy not there yet: show the probability alone.
    st.caption(
        f"Calibrated estimate: of past trials scored around {risk:.0%}, about {risk:.0%} were terminated early. "
        "Based only on design information known at registration; a student project, not medical advice."
    )

st.subheader("What would lower the risk?")
changes = lowering_changes(row, score)
if changes:
    st.table(pd.DataFrame(
        [(text, f"{new:.1%}", f"{delta:+.1%}") for text, new, delta in changes],
        columns=["Single change", "New risk", "Change"],
    ).set_index("Single change"))
    st.caption("Patterns in past trials, not proof of cause: changing a design does not guarantee this effect.")
else:
    st.write("No single change in these settings lowers the estimate.")

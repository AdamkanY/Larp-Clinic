"""Check your own trial: enter a design from scratch and see its predicted risk of stopping early."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
from risk_gauge import bar_caption, estimate_caption, risk_bar, risk_card  # noqa: E402
from src.features import FEATURE_COLUMNS, INTERVENTION_TYPES, MAX_AGE_FILL  # noqa: E402
from src.labels import label  # noqa: E402
from ui import page_header  # noqa: E402

UNSET = "Not specified"

# Allowed values per categorical feature, as the ClinicalTrials.gov API spells them; the page shows plain names.
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
    "primary_purpose": "Purpose", "masking": "Blinding", "sponsor_class": "Sponsor",
    "sex": "Sex eligible",
}
HELP = {
    "phase": "The development stage. 'No phase' is used for studies that are not drug trials, such as behavioural ones.",
    "allocation": "Randomized: chance decides who gets which treatment.",
    "intervention_model": "How participants are grouped: parallel groups, everyone gets everything in turn (crossover), and so on.",
    "primary_purpose": "What the trial mainly tries to do.",
    "masking": "How many parties (patients, doctors, assessors) are kept unaware of who gets which treatment.",
    "sponsor_class": "Who runs and pays for the trial.",
    "sex": "Which sexes may take part.",
}
# A typical randomized phase 2 industry drug trial, so the page opens with a realistic example instead of blanks.
DEFAULTS = {
    "phase": "PHASE2", "allocation": "RANDOMIZED", "intervention_model": "PARALLEL",
    "primary_purpose": "TREATMENT", "masking": "DOUBLE", "sponsor_class": "INDUSTRY", "sex": "ALL",
    "n_arms": 2, "has_collaborators": False, "has_dmc": False, "n_primary_outcomes": 1, "n_secondary_outcomes": 3,
    "n_conditions": 1, "healthy_volunteers": False, "min_age_years": 18.0, "has_max_age": False,
    "max_age_years": 65.0, "eligibility_text_len": 1500, "intervention_types": ["DRUG"],
}
# 0/1 design flags the what-if table may toggle.
FLAGS = {"has_collaborators": "Has collaborators", "has_dmc": "Has data monitoring committee"}
# Values that make no sense as advice: a drug trial cannot simply become "no phase", nor a randomized one "single group".
NOT_ADVISABLE = {"phase": {"NA"}, "allocation": {"NA"}}


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server; None while src/predict.py or models/gbm.joblib is missing."""
    try:
        from src.predict import load_model
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


def reset_form():
    """Put every input back to the example trial."""
    st.session_state.update({f"chk_{name}": value for name, value in DEFAULTS.items()})


def pick(name, text):
    """Dropdown for one categorical feature in plain words; 'Not specified' becomes None (a missing value)."""
    options = [UNSET] + CHOICES[name]
    choice = st.selectbox(text, options, key=f"chk_{name}", help=HELP[name],
                          format_func=lambda v: v if v == UNSET else label(name, v))
    return None if choice == UNSET else choice


def feature_row():
    """Render the form and return the feature dict in the exact shape src.features.build_features produces."""
    for name, value in DEFAULTS.items():
        st.session_state.setdefault(f"chk_{name}", value)
    with st.container(key="panel_design"):
        st.markdown("##### Design")
        a, b, c = st.columns(3)
        with a:
            row = {"phase": pick("phase", "Phase")}
            row["allocation"] = pick("allocation", "Allocation")
        with b:
            row["masking"] = pick("masking", "Blinding")
            row["intervention_model"] = pick("intervention_model", "Intervention model")
        with c:
            row["primary_purpose"] = pick("primary_purpose", "Purpose")
            row["n_arms"] = st.number_input("Number of groups (arms)", 0, 20, key="chk_n_arms")
        types = st.multiselect("Intervention types", INTERVENTION_TYPES, key="chk_intervention_types",
                               format_func=lambda t: label("", t), help="What is tested: a drug, a device, a behaviour change...")
    with st.container(key="panel_sponsor"):
        st.markdown("##### Sponsor and oversight")
        a, b, c = st.columns(3)
        with a:
            row["sponsor_class"] = pick("sponsor_class", "Lead sponsor")
        row["has_collaborators"] = int(b.checkbox("Has collaborators", key="chk_has_collaborators"))
        row["has_dmc"] = int(c.checkbox("Has a data monitoring committee", key="chk_has_dmc",
                                        help="An independent group that watches the safety data during the trial."))
    with st.container(key="panel_outcomes"):
        st.markdown("##### Outcomes")
        a, b, c = st.columns(3)
        row["n_primary_outcomes"] = a.number_input("Primary outcomes", 0, 20, key="chk_n_primary_outcomes",
                                                   help="The main questions the trial answers.")
        row["n_secondary_outcomes"] = b.number_input("Secondary outcomes", 0, 50, key="chk_n_secondary_outcomes")
        row["n_conditions"] = c.number_input("Conditions studied", 0, 20, key="chk_n_conditions")
    with st.container(key="panel_people"):
        st.markdown("##### Participants")
        a, b, c = st.columns(3)
        with a:
            row["sex"] = pick("sex", "Sex eligible")
            row["healthy_volunteers"] = int(st.checkbox("Accepts healthy volunteers", key="chk_healthy_volunteers"))
        with b:
            row["min_age_years"] = st.number_input("Minimum age (years)", 0.0, 120.0, key="chk_min_age_years")
            row["has_max_age"] = int(st.checkbox("Has a maximum age", key="chk_has_max_age"))
        with c:
            # No upper limit is stored as 120 plus the flag, exactly as build_features does.
            row["max_age_years"] = (st.number_input("Maximum age (years)", 0.0, 120.0, key="chk_max_age_years")
                                    if row["has_max_age"] else MAX_AGE_FILL)
            row["eligibility_text_len"] = st.number_input(
                "Length of eligibility criteria (characters)", 0, 30000, step=100, key="chk_eligibility_text_len",
                help="A short paragraph is about 500 characters; long inclusion/exclusion lists reach several thousand.",
            )
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
        tries = [(value, score({**row, key: value})) for value in values
                 if value != row[key] and value not in NOT_ADVISABLE.get(key, ())]
        if tries:
            value, risk = min(tries, key=lambda t: t[1])
            best.append((f"{NAMES[key]}: {label(key, row[key]) if row[key] else UNSET} → {label(key, value)}", risk))
    for key, name in FLAGS.items():
        flipped = 1 - row[key]
        best.append((f"{name}: {'yes' if row[key] else 'no'} → {'yes' if flipped else 'no'}",
                     score({**row, key: flipped})))
    best.sort(key=lambda t: t[1])
    return [(text, risk, risk - base) for text, risk in best[:top] if risk < base]


page_header("🧪 Check your own trial",
            "Describe a trial design and see how likely it is to be stopped early. Change any value and the estimate updates straight away.")

result_strip = st.container(key="sticky_result")  # filled once the risk is known, but shown above the form
row = feature_row()
st.button("Reset to the example trial", on_click=reset_form)

if get_model() is None:
    result_strip.info("The estimate is not available here yet: the model file is missing.")
    st.stop()

from src.predict import score  # noqa: E402

try:
    risk = score(row)
except ValueError as exc:
    result_strip.error(f"The model could not score this design: {exc}")
    st.stop()

previous = st.session_state.get("previous_risk")
st.session_state["previous_risk"] = risk
changed = previous is not None and abs(risk - previous) >= 0.0005  # hide a meaningless "+0.0%"
result_strip.markdown(risk_card(risk, delta=risk - previous if changed else None, compact=True), unsafe_allow_html=True)

st.subheader("Where this design stands")
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
        "Rare options (for example sponsor \"Individual\") rest on few trials and are less reliable."
    )
else:
    st.write("No single change in these settings lowers the estimate.")

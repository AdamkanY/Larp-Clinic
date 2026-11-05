"""Check your own trial: enter a design from scratch and see its predicted risk of stopping early."""
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
from risk_gauge import (bar_caption, drivers_caption, drivers_chart, estimate_caption,  # noqa: E402
                        risk_bar, risk_card)
from src.features import (CONDITION_FLAG_FEATURES, INTERVENTION_TYPES, condition_features,  # noqa: E402
                          features_from_design)
from src.labels import feature_name, feature_value, label  # noqa: E402
from ui import page_header, require_router  # noqa: E402

require_router("pages/5_Check_Your_Trial.py")

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
    "conditions": "Type 2 diabetes", "healthy_volunteers": False, "min_age_years": 18.0, "has_max_age": False,
    "max_age_years": 65.0, "eligibility_text_len": 1500, "intervention_types": ["DRUG"],
}
# "What would lower the risk" only tries levers a trial designer actually controls. Phase, sponsor type, purpose and
# the condition follow from the research question, so changing them is not advice.
LEVERS = ["masking", "allocation", "n_arms", "eligibility_text_len", "min_age_years", "max_age_years", "has_dmc"]
MIN_AGES = [0.0, 12.0, 18.0, 21.0, 40.0, 65.0]
MAX_AGES = [None, 50.0, 65.0, 75.0, 85.0]  # None = no upper limit
LENGTH_FACTORS = [0.5, 0.75, 1.25, 1.5]


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


def split_conditions(text):
    """'Asthma; COPD, Smoking' -> ['Asthma', 'COPD', 'Smoking']: commas, semicolons and new lines separate entries."""
    return [part.strip() for part in re.split(r"[;,\n]", text or "") if part.strip()]


def design_form():
    """Render the form and return the design in plain inputs (the keys src.features.study_from_design reads)."""
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
        row["has_collaborators"] = b.checkbox("Has collaborators", key="chk_has_collaborators")
        row["has_dmc"] = c.checkbox("Has a data monitoring committee", key="chk_has_dmc",
                                    help="An independent group that watches the safety data during the trial.")
    with st.container(key="panel_outcomes"):
        st.markdown("##### Outcomes")
        a, b, c = st.columns(3)
        row["n_primary_outcomes"] = a.number_input("Primary outcomes", 0, 20, key="chk_n_primary_outcomes",
                                                   help="The main questions the trial answers.")
        row["n_secondary_outcomes"] = b.number_input("Secondary outcomes", 0, 50, key="chk_n_secondary_outcomes")
        with c:
            text = st.text_input("Conditions studied", key="chk_conditions",
                                 help="Separate several conditions with commas, for example: asthma, COPD.")
        row["conditions"] = split_conditions(text)
        areas = [feature_name(col).removeprefix("Condition: ")
                 for col, on in condition_features(row["conditions"]).items() if col in CONDITION_FLAG_FEATURES and on]
        st.caption(f"Recognised condition areas: {', '.join(areas)}." if areas else
                   "No condition area recognised; the model then treats it like any other condition.")
    with st.container(key="panel_people"):
        st.markdown("##### Participants")
        a, b, c = st.columns(3)
        with a:
            row["sex"] = pick("sex", "Sex eligible")
            row["healthy_volunteers"] = st.checkbox("Accepts healthy volunteers", key="chk_healthy_volunteers")
        with b:
            row["min_age_years"] = st.number_input("Minimum age (years)", 0.0, 120.0, key="chk_min_age_years")
            has_max_age = st.checkbox("Has a maximum age", key="chk_has_max_age")
        with c:
            # None = no upper limit; features_from_design turns it into 120 plus the flag, as for real records.
            row["max_age_years"] = (st.number_input("Maximum age (years)", 0.0, 120.0, key="chk_max_age_years")
                                    if has_max_age else None)
            row["eligibility_text_len"] = st.number_input(
                "Length of eligibility criteria (characters)", 0, 30000, step=100, key="chk_eligibility_text_len",
                help="A short paragraph is about 500 characters; long inclusion/exclusion lists reach several thousand.",
            )
    row["intervention_types"] = types
    return row


def lever_options(design, lever):
    """The other values worth trying for one design lever; nothing that would describe an impossible design."""
    current = design.get(lever)
    if lever == "masking":
        options = CHOICES["masking"]
    elif lever == "allocation":
        # "Single group" is not advice, and one arm cannot be randomized.
        options = ["RANDOMIZED", "NON_RANDOMIZED"] if (design.get("n_arms") or 0) >= 2 else ["NON_RANDOMIZED"]
    elif lever == "n_arms":
        fewest = 2 if design.get("allocation") == "RANDOMIZED" else 1  # randomizing needs two groups
        options = [n for n in range((current or 0) - 2, (current or 0) + 3) if fewest <= n <= 20]
    elif lever == "eligibility_text_len":
        options = [int(round(current * f / 100) * 100) for f in LENGTH_FACTORS] if current else []
        options = [n for n in options if n > 0]
    elif lever == "min_age_years":
        top = design.get("max_age_years")
        options = [a for a in MIN_AGES if top is None or a < top]
    elif lever == "max_age_years":
        options = [a for a in MAX_AGES if a is None or a > (design.get("min_age_years") or 0)]
    else:  # has_dmc
        options = [not current]
    return [value for value in options if value != current]


def describe(lever, old, new):
    """One change in plain words, for example 'Blinding: None (open label) → Double blind'."""
    words = {"max_age_years": lambda v: feature_value("max_age_years", 120.0 if v is None else v)}
    show = words.get(lever, lambda v: feature_value(lever, v))
    return f"{feature_name(lever)}: {show(old)} → {show(new)}"


def lowering_changes(design, risk_of, top=5):
    """Best single change per design lever that lowers the risk, strongest first, as (description, new risk, change).

    risk_of scores a design dict. Only LEVERS are tried: never phase, sponsor type, purpose or the condition.
    """
    base = risk_of(design)
    best = []
    for lever in LEVERS:
        tries = [(value, risk_of({**design, lever: value})) for value in lever_options(design, lever)]
        if tries:
            value, risk = min(tries, key=lambda t: t[1])
            best.append((describe(lever, design.get(lever), value), risk))
    best.sort(key=lambda t: t[1])
    return [(text, risk, risk - base) for text, risk in best[:top] if risk < base]


page_header("🧪 Check your own trial",
            "Describe a trial design and see how likely it is to be stopped early. Change any value and the estimate updates straight away.")

result_strip = st.container(key="sticky_result")  # filled once the risk is known, but shown above the form
design = design_form()
st.button("Reset to the example trial", on_click=reset_form)

if get_model() is None:
    result_strip.info("The estimate is not available here yet: the model file is missing.")
    st.stop()

from src.predict import explain, model_features, score  # noqa: E402

MISMATCH = ("This design cannot be scored right now: the saved model and the app expect different inputs{detail}. "
            "Nothing is wrong with your entries; the model needs retraining (python src/train.py). "
            "The other pages still work.")
row = features_from_design(design)
missing = [c for c in model_features(get_model()) if c not in row]
if missing:
    result_strip.error(MISMATCH.format(detail=f" (the model also needs: {', '.join(map(feature_name, missing))})"))
    st.stop()
try:
    risk = score(row, get_model())
except ValueError:
    result_strip.error(MISMATCH.format(detail=""))
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

st.subheader("What drives this estimate")
try:
    drivers = explain(row)
except (FileNotFoundError, KeyError, ValueError):
    drivers = None
chart = drivers_chart(drivers or [])
if chart is not None:
    st.plotly_chart(chart, width="stretch", key="drivers_chart")
    st.caption(drivers_caption())
elif drivers == []:
    st.write("This design looks like a typical past trial in every factor, so nothing pushes it up or down.")
else:
    st.info("The breakdown is not available here yet: models/explain_reference.json is missing.")

st.subheader("What would lower the risk?")
changes = lowering_changes(design, lambda d: score(features_from_design(d), get_model()))
if changes:
    st.table(pd.DataFrame(
        [(text, f"{new:.1%}", f"{delta:+.1%}") for text, new, delta in changes],
        columns=["Single change", "New risk", "Change"],
    ).set_index("Single change"))
    st.caption(
        "Only settings a trial designer can choose are tried (blinding, allocation, arms, length of the eligibility "
        "criteria, age limits, data monitoring committee); phase, sponsor and condition follow from the research "
        "question. Patterns in past trials, not proof of cause: changing a design does not guarantee this effect."
    )
else:
    st.write("No single change in the design settings lowers the estimate.")

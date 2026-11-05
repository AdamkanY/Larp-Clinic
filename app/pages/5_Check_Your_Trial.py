"""Check your own trial: enter a design from scratch and see its predicted risk of stopping early."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
from risk_gauge import bar_caption, estimate_caption, risk_bar, risk_card  # noqa: E402
from src.features import CATEGORY_CODES, INTERVENTION_TYPES, features_from_design  # noqa: E402
from src.labels import label  # noqa: E402
from src.predict import get_model, score  # noqa: E402
from ui import page_header, require_router  # noqa: E402

require_router("pages/5_Check_Your_Trial.py")

UNSET = "Not specified"

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
# Yes/no design settings the what-if table may toggle.
FLAGS = {"has_collaborators": "Has collaborators", "has_dmc": "Has data monitoring committee"}
# Values that make no sense as advice: a drug trial cannot simply become "no phase", nor a randomized one "single group".
NOT_ADVISABLE = {"phase": {"NA"}, "allocation": {"NA"}}


def reset_form():
    """Put every input back to the example trial."""
    st.session_state.update({f"chk_{name}": value for name, value in DEFAULTS.items()})


def pick(name, text):
    """Dropdown of one categorical feature's API codes (CATEGORY_CODES) in plain words; 'Not specified' becomes None."""
    options = [UNSET] + CATEGORY_CODES[name]
    choice = st.selectbox(text, options, key=f"chk_{name}", help=HELP[name],
                          format_func=lambda v: v if v == UNSET else label(name, v))
    return None if choice == UNSET else choice


def design_form():
    """Render the form and return the design as the keyword arguments of src.features.features_from_design."""
    for name, value in DEFAULTS.items():
        st.session_state.setdefault(f"chk_{name}", value)
    with st.container(key="panel_design"):
        st.markdown("##### Design")
        a, b, c = st.columns(3)
        with a:
            design = {"phase": pick("phase", "Phase")}
            design["allocation"] = pick("allocation", "Allocation")
        with b:
            design["masking"] = pick("masking", "Blinding")
            design["intervention_model"] = pick("intervention_model", "Intervention model")
        with c:
            design["primary_purpose"] = pick("primary_purpose", "Purpose")
            design["n_arms"] = st.number_input("Number of groups (arms)", 0, 20, key="chk_n_arms")
        design["intervention_types"] = st.multiselect(
            "Intervention types", INTERVENTION_TYPES, key="chk_intervention_types",
            format_func=lambda t: label("", t), help="What is tested: a drug, a device, a behaviour change...")
    with st.container(key="panel_sponsor"):
        st.markdown("##### Sponsor and oversight")
        a, b, c = st.columns(3)
        with a:
            design["sponsor_class"] = pick("sponsor_class", "Lead sponsor")
        design["has_collaborators"] = b.checkbox("Has collaborators", key="chk_has_collaborators")
        design["has_dmc"] = c.checkbox("Has a data monitoring committee", key="chk_has_dmc",
                                       help="An independent group that watches the safety data during the trial.")
    with st.container(key="panel_outcomes"):
        st.markdown("##### Outcomes")
        a, b, c = st.columns(3)
        design["n_primary_outcomes"] = a.number_input("Primary outcomes", 0, 20, key="chk_n_primary_outcomes",
                                                      help="The main questions the trial answers.")
        design["n_secondary_outcomes"] = b.number_input("Secondary outcomes", 0, 50, key="chk_n_secondary_outcomes")
        design["n_conditions"] = c.number_input("Conditions studied", 0, 20, key="chk_n_conditions")
    with st.container(key="panel_people"):
        st.markdown("##### Participants")
        a, b, c = st.columns(3)
        with a:
            design["sex"] = pick("sex", "Sex eligible")
            design["healthy_volunteers"] = st.checkbox("Accepts healthy volunteers", key="chk_healthy_volunteers")
        with b:
            design["min_age_years"] = st.number_input("Minimum age (years)", 0.0, 120.0, key="chk_min_age_years")
            has_max_age = st.checkbox("Has a maximum age", key="chk_has_max_age")
        with c:
            # Unticked means no upper age limit: no maximum age in the record, as on ClinicalTrials.gov.
            design["max_age_years"] = (st.number_input("Maximum age (years)", 0.0, 120.0, key="chk_max_age_years")
                                       if has_max_age else None)
            design["eligibility_text_len"] = st.number_input(
                "Length of eligibility criteria (characters)", 0, 30000, step=100, key="chk_eligibility_text_len",
                help="A short paragraph is about 500 characters; long inclusion/exclusion lists reach several thousand.",
            )
    return design


def lowering_changes(design, score, top=5):
    """Best single change per setting that lowers the risk, strongest first, as (description, new risk, change).

    Each changed design goes through features_from_design too, so it is scored exactly as if it had been typed in.
    """
    def risk_with(**change):
        return score(features_from_design(**{**design, **change}))

    base = risk_with()
    best = []
    for key, values in CATEGORY_CODES.items():
        tries = [(value, risk_with(**{key: value})) for value in values
                 if value != design[key] and value not in NOT_ADVISABLE.get(key, ())]
        if tries:
            value, risk = min(tries, key=lambda t: t[1])
            best.append((f"{NAMES[key]}: {label(key, design[key]) if design[key] else UNSET} → {label(key, value)}",
                         risk))
    for key, name in FLAGS.items():
        flipped = not design[key]
        best.append((f"{name}: {'yes' if design[key] else 'no'} → {'yes' if flipped else 'no'}",
                     risk_with(**{key: flipped})))
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

try:
    risk = score(features_from_design(**design))
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
changes = lowering_changes(design, score)
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

"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Check your own trial: enter a design from scratch and see its predicted risk of stopping early.

Outline of the page code at the bottom of this file:
    1. Header and result strip
    2. Form: design, sponsor, outcomes, participants
    3. Scoring the design
    4. Where this design stands
    5. What drives this estimate
    6. Which single change lowers the estimate?
"""
import re  # splits the conditions text
import sys  # changes the import path
from pathlib import Path  # builds file paths that work on every computer

import pandas as pd  # tables (DataFrame)
import streamlit as st  # the web app framework

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
# Risk card, comparison bar, drivers chart and their captions (app/risk_gauge.py).
from risk_gauge import bar_caption, drivers_caption, drivers_chart, estimate_caption, risk_bar, risk_card
# Feature code shared with training: condition areas, intervention types and the form-to-features function.
from src.features import CONDITION_FLAG_FEATURES, INTERVENTION_TYPES, condition_features, features_from_design
from src.labels import feature_name, feature_value, label  # plain-English names for features, values and codes
from src.predict import explain, load_model, model_features, score  # model loading, score, drivers, model inputs
from ui import chart, page_header, require_router  # shared look: charts, header, router

# Send the page through Home.py if it was opened directly.
require_router("pages/2_Check_Your_Trial.py")

# Label of the empty choice in every dropdown; it becomes a missing value, as in a real record without that field.
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
# Help texts shown next to each dropdown (the question mark icon).
HELP = {
    "phase": ("The development stage. 'No phase' is used for studies that are not drug trials, such as "
              "behavioural ones."),
    "allocation": "Randomized: chance decides who gets which treatment.",
    "intervention_model": ("How participants are grouped: parallel groups, everyone gets everything in turn "
                           "(crossover), and so on."),
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
# "Which single change lowers the estimate?" only tries levers a trial designer actually controls. Phase, sponsor
# type, purpose and the condition follow from the research question, so changing them is not advice.
LEVERS = ["masking", "allocation", "n_arms", "eligibility_text_len", "min_age_years", "max_age_years", "has_dmc"]
# The values that search tries for the age limits and for the length of the eligibility criteria.
MIN_AGES = [0.0, 12.0, 18.0, 21.0, 40.0, 65.0]
MAX_AGES = [None, 50.0, 65.0, 75.0, 85.0]  # None = no upper limit
LENGTH_FACTORS = [0.5, 0.75, 1.25, 1.5]
# How many single changes the table shows at most.
MAX_CHANGES = 5

# The two halves of the message for a model that does not fit the form. A detail can go in between.
MISMATCH_START = "This design cannot be scored right now: the saved model and the app expect different inputs"
MISMATCH_END = (". Nothing is wrong with your entries; the model needs retraining (python src/train.py). "
                "The other pages still work.")


# ---------------------------------------------------------------------------------------------------------------
# Model and form helpers
# ---------------------------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server.

    Input: none.

    Output: the model object, or None while models/gbm.joblib cannot be loaded.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # A missing library or file means no model; the page then shows a message instead of a result.
    try:
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


def reset_form():
    """Put every input back to the example trial.

    Input: none.

    Output: nothing. It writes the example values into the session state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    for name, value in DEFAULTS.items():
        st.session_state[f"chk_{name}"] = value


def pick(name, text):
    """Draw a dropdown for one categorical feature in plain words. 'Not specified' becomes None (a missing value).

    Input: name is the feature name, a key of CHOICES (text); text is the label above the dropdown (text).

    Output: the chosen API code (text), or None if "Not specified" is chosen.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md

    def shown_name(code):
        """Give the text the dropdown shows for one option: the plain name of the API code.

        Input: code is an API code (text) or the "Not specified" option.

        Output: the plain name as text.
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        if code == UNSET:
            return UNSET
        return label(name, code)

    # The empty choice comes first, then the API codes.
    options = [UNSET] + CHOICES[name]
    choice = st.selectbox(text, options, key=f"chk_{name}", help=HELP[name], format_func=shown_name)
    if choice == UNSET:
        return None
    return choice


def intervention_name(code):
    """Give the plain name of an intervention type, for example 'DIETARY_SUPPLEMENT' becomes 'Dietary supplement'.

    Input: code is the API code of the intervention type (text).

    Output: the plain name as text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return label("", code)


def split_conditions(text):
    """Split the conditions text into a list, for example 'Asthma; COPD, Smoking' into three entries.

    Commas, semicolons and new lines separate entries.

    Input: text is what the user typed (text, may be None or empty).

    Output: a list of condition names (text), without empty entries.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Split at every separator and drop empty pieces.
    conditions = []
    for part in re.split(r"[;,\n]", text or ""):
        part = part.strip()
        if part:
            conditions.append(part)
    return conditions


# ---------------------------------------------------------------------------------------------------------------
# "Which single change lowers the estimate?"
# ---------------------------------------------------------------------------------------------------------------

def lever_options(design, lever):
    """List the other values worth trying for one design lever. Nothing that would describe an impossible design.

    Input: design is the design dict of the form; lever is the name of the setting to vary (text, one of LEVERS).

    Output: a list of values for that lever, without the current value.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Each lever has its own rule for which values make sense.
    current = design.get(lever)
    options = []
    if lever == "masking":
        options = CHOICES["masking"]
    elif lever == "allocation":
        # One arm cannot be randomized. "Single group" is never suggested, because it is not advice.
        if (design.get("n_arms") or 0) >= 2:
            options = ["RANDOMIZED", "NON_RANDOMIZED"]
        else:
            options = ["NON_RANDOMIZED"]
    elif lever == "n_arms":
        # Up to two arms fewer or more, at most 20. Randomizing needs at least two groups.
        fewest = 1
        if design.get("allocation") == "RANDOMIZED":
            fewest = 2
        arms = current or 0
        for n in range(arms - 2, arms + 3):
            if fewest <= n <= 20:
                options.append(n)
    elif lever == "eligibility_text_len":
        # Shorter and longer texts, rounded to full hundreds. An empty text has nothing to scale.
        if current:
            for factor in LENGTH_FACTORS:
                length = int(round(current * factor / 100) * 100)
                if length > 0:
                    options.append(length)
    elif lever == "min_age_years":
        # The minimum has to stay below the maximum age.
        top = design.get("max_age_years")
        for age in MIN_AGES:
            if top is None or age < top:
                options.append(age)
    elif lever == "max_age_years":
        # The maximum has to stay above the minimum age. None (no upper limit) is always possible.
        lowest = design.get("min_age_years") or 0
        for age in MAX_AGES:
            if age is None or age > lowest:
                options.append(age)
    else:  # has_dmc
        options = [not current]

    # The current value is not a change, so it is left out.
    changes = []
    for value in options:
        if value != current:
            changes.append(value)
    return changes


def describe(lever, old, new):
    """Describe one change in plain words, for example 'Blinding: None (open label) → Double blind'.

    Input: lever is the setting name (text); old is its current value; new is the value after the change.

    Output: the description as text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # "No maximum age" is None in the form, but the names in labels.py expect it as 120 years, like in the
    # training data.
    if lever == "max_age_years":
        if old is None:
            old = 120.0
        if new is None:
            new = 120.0
    return f"{feature_name(lever)}: {feature_value(lever, old)} → {feature_value(lever, new)}"


def risk_of(design):
    """Predict the risk of a design from the form. It goes through the same feature function as the training data.

    Input: design is the design dict of the form.

    Output: the predicted risk of stopping early (number between 0 and 1).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return score(features_from_design(design), get_model())


def new_risk(change):
    """Pick the new risk out of a (description, new risk) pair. Used to sort the changes.

    Input: change is a tuple (description, new risk).

    Output: the new risk (number).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return change[1]


def lowering_changes(design):
    """Find the best single change per design lever that lowers the risk, strongest first.

    Only LEVERS are tried: never phase, sponsor type, purpose or the condition.

    Input: design is the design dict of the form.

    Output: a list of tuples (description, new risk, change in risk), at most MAX_CHANGES long, only with changes that
    lower the risk.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Score the current design once, then every other value of every lever, one change at a time.
    base = risk_of(design)
    best = []
    for lever in LEVERS:
        # Keep only the value with the lowest risk for this lever.
        best_value = None
        best_risk = None
        for value in lever_options(design, lever):
            changed = dict(design)
            changed[lever] = value
            risk = risk_of(changed)
            if best_risk is None or risk < best_risk:
                best_value = value
                best_risk = risk
        if best_risk is not None:
            best.append((describe(lever, design.get(lever), best_value), best_risk))

    # Lowest new risk first. Of the first few, only changes that really lower the risk are shown.
    best.sort(key=new_risk)
    changes = []
    for text, risk in best[:MAX_CHANGES]:
        if risk < base:
            changes.append((text, risk, risk - base))
    return changes


# ---------------------------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------------------------

# ---- 1. Header and result strip ----
page_header("Check your own trial", "Describe a trial design and see how likely it is to be stopped early. "
                                    "Change any value and the estimate updates straight away.")

# The result strip is created before the form so it sits on top. It is filled further down, once the form has
# been read and the risk is known.
result_strip = st.container(key="sticky_result")

# ---- 2. Form ----
# First run: fill every input with the example trial. What the user already typed is kept.
for name, value in DEFAULTS.items():
    if f"chk_{name}" not in st.session_state:
        st.session_state[f"chk_{name}"] = value

# The form fills this dictionary. Its keys are the ones src.features.study_from_design reads.
design = {}

# ---- 2a. Panel 1 ----
# Panel 1: the design itself (phase, allocation, blinding, model, purpose, arms, intervention types).
with st.container(key="panel_design"):
    st.markdown("##### Design")
    a, b, c = st.columns(3)
    with a:
        design["phase"] = pick("phase", "Phase")
        design["allocation"] = pick("allocation", "Allocation")
    with b:
        design["masking"] = pick("masking", "Blinding")
        design["intervention_model"] = pick("intervention_model", "Intervention model")
    with c:
        design["primary_purpose"] = pick("primary_purpose", "Purpose")
        design["n_arms"] = st.number_input("Number of groups (arms)", 0, 20, key="chk_n_arms")
    design["intervention_types"] = st.multiselect(
        "Intervention types", INTERVENTION_TYPES, key="chk_intervention_types", format_func=intervention_name,
        help="What is tested: a drug, a device, a behaviour change...")

# ---- 2b. Panel 2 ----
# Panel 2: who runs the trial and who watches it.
with st.container(key="panel_sponsor"):
    st.markdown("##### Sponsor and oversight")
    a, b, c = st.columns(3)
    with a:
        design["sponsor_class"] = pick("sponsor_class", "Lead sponsor")
    design["has_collaborators"] = b.checkbox("Has collaborators", key="chk_has_collaborators")
    design["has_dmc"] = c.checkbox("Has a data monitoring committee", key="chk_has_dmc",
                                   help="An independent group that watches the safety data during the trial.")

# ---- 2c. Panel 3 ----
# Panel 3: what is measured and which conditions are studied.
with st.container(key="panel_outcomes"):
    st.markdown("##### Outcomes")
    a, b, c = st.columns(3)
    design["n_primary_outcomes"] = a.number_input("Primary outcomes", 0, 20, key="chk_n_primary_outcomes",
                                                  help="The main questions the trial answers.")
    design["n_secondary_outcomes"] = b.number_input("Secondary outcomes", 0, 50, key="chk_n_secondary_outcomes")
    with c:
        conditions_text = st.text_input("Conditions studied", key="chk_conditions",
                                        help="Separate several conditions with commas, for example: asthma, COPD.")
    design["conditions"] = split_conditions(conditions_text)
    # Show which condition areas the keywords recognise, so the user sees how the text is read.
    areas = []
    for column, found in condition_features(design["conditions"]).items():
        if column in CONDITION_FLAG_FEATURES and found:
            areas.append(feature_name(column).removeprefix("Condition: "))
    if areas:
        st.caption(f"Recognised condition areas: {', '.join(areas)}.")
    else:
        st.caption("No condition area recognised; the model then treats it like any other condition.")

# ---- 2d. Panel 4 ----
# Panel 4: who may take part.
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
        # The input for the maximum age only shows when the box is ticked. None = no upper limit;
        # features_from_design turns it into 120 plus the flag, as for real records.
        if has_max_age:
            design["max_age_years"] = st.number_input("Maximum age (years)", 0.0, 120.0, key="chk_max_age_years")
        else:
            design["max_age_years"] = None
        design["eligibility_text_len"] = st.number_input(
            "Length of eligibility criteria (characters)", 0, 30000, step=100, key="chk_eligibility_text_len",
            help="A short paragraph is about 500 characters; long inclusion/exclusion lists reach several thousand.",
        )

# ---- 2e. Reset button ----
st.button("Reset to the example trial", on_click=reset_form)

# ---- 3. Scoring the design ----
# Without a model file the form still works, but nothing can be scored.
if get_model() is None:
    result_strip.info("The estimate is not available here yet: the model file is missing.")
    st.stop()

# The same feature function as training (via study_from_design), then a check that the model gets every input
# it was trained on.
row = features_from_design(design)
missing = []
for column in model_features(get_model()):
    if column not in row:
        missing.append(feature_name(column))
if missing:
    result_strip.error(MISMATCH_START + f" (the model also needs: {', '.join(missing)})" + MISMATCH_END)
    st.stop()
try:
    risk = score(row, get_model())
except ValueError:
    result_strip.error(MISMATCH_START + MISMATCH_END)
    st.stop()

# Remember the last risk, so the strip can show how much the latest edit changed it. A change that would round
# to "+0.0%" is not shown.
previous = st.session_state.get("previous_risk")
st.session_state["previous_risk"] = risk
delta = None
if previous is not None and abs(risk - previous) >= 0.0005:
    delta = risk - previous
result_strip.markdown(risk_card(risk, delta=delta, compact=True), unsafe_allow_html=True)

# ---- 4. Where this design stands ----
# Where the design ranks among past trials.
st.subheader("Where this design stands")
bar = risk_bar(risk)
if bar is not None:
    chart(bar)
st.caption("Rough estimate for comparing trials, not an exact chance. Not medical advice.")
with st.expander("How to read this"):
    st.write(bar_caption())
    st.write(estimate_caption(risk))

# ---- 5. What drives this estimate ----
# Which factors push this design's risk up or down (src.predict.explain). None means it could not be worked out,
# an empty list means no factor differs from a typical past trial.
st.subheader("What drives this estimate")
try:
    drivers = explain(row)
except (FileNotFoundError, KeyError, ValueError):
    drivers = None
if drivers:
    chart(drivers_chart(drivers), key="drivers_chart")
    st.caption(drivers_caption())
elif drivers == []:
    st.write("This design looks like a typical past trial in every factor, so nothing pushes it up or down.")
else:
    st.info("The breakdown is not available here yet: models/explain_reference.json is missing.")

# ---- 6. Which single change lowers the estimate? ----
# The best single change per design lever, as a small table. The heading speaks of the model's estimate, not the
# real risk: the model learned patterns from past trials (prediction), not what a change would cause (DSF deck 01,
# "prediction or causal question?").
st.subheader("Which single change lowers the estimate?")
changes = lowering_changes(design)
if changes:
    # One table row per change, shown as percentages.
    table_rows = []
    for text, risk_after, change in changes:
        table_rows.append({"Single change": text, "New estimate": f"{risk_after:.1%}", "Change": f"{change:+.1%}"})
    st.table(pd.DataFrame(table_rows).set_index("Single change"))
    st.caption(
        "Only settings a trial designer can choose are tried (blinding, allocation, arms, length of the eligibility "
        "criteria, age limits, data monitoring committee); phase, sponsor and condition follow from the research "
        "question. The estimate drops because past trials with this setting were stopped less often. That is a "
        "pattern, not proof of cause: the setting may simply go with other things (money, experience, the kind of "
        "drug) that really make the difference, so changing it alone does not guarantee a lower real risk."
    )
else:
    st.write("No single change in the design settings lowers the estimate.")

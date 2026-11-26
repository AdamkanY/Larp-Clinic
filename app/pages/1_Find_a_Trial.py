"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Find an ongoing trial, see its predicted risk of stopping early in the list and open its details.

Outline of the page code at the bottom of this file:
    1. Page state and detail view
    2. Header, search box and filters
    3. Search on ClinicalTrials.gov (with offline fallback)
    4. Result count and note
    5. Result cards
    6. Show more
"""
import sys  # changes the import path
from pathlib import Path  # builds file paths that work on every computer

import streamlit as st  # the web app framework

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge and ui
# Risk card, comparison bar, drivers chart and their captions (app/risk_gauge.py).
from risk_gauge import bar_caption, drivers_caption, drivers_chart, estimate_caption, risk_bar, risk_card, risk_chip
# ClinicalTrials.gov search, single record download, offline snapshot search and a flat summary of one record.
from src.api_client import PHASES, SPONSOR_CLASSES, get_trial, search_examples, search_trials, summarize
from src.features import build_features  # turns one API record into model features (same code as in training)
from src.labels import label  # plain-English names for raw codes
from src.predict import explain, load_model, score  # model loading, risk score and per-trial drivers
from ui import chart, chip, esc, page_header, require_router  # shared look: charts, chips, safe html, header, router

# Send the page through Home.py if it was opened directly.
require_router("pages/1_Find_a_Trial.py")

# Results per "page" of the list. "Show more" asks the API for the next ten.
PAGE = 10
# Search ideas shown as pills under the search box.
EXAMPLES = ["breast cancer", "diabetes", "asthma", "depression", "COVID-19"]


# ---------------------------------------------------------------------------------------------------------------
# Model and scores
# ---------------------------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server.

    Input: none.

    Output: the model object, or None while models/gbm.joblib cannot be loaded.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # A missing library or file means no model; the page then shows a message instead of risks.
    try:
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


@st.cache_data(show_spinner=False, max_entries=500)
def risk_for(nct_id, last_update, _study):
    """Predict the risk of one trial from its record.

    Streamlit remembers the result per trial and record version (nct_id and last_update). The record itself is
    not part of the cache key, that is what the underscore in _study means.

    Input: nct_id is the trial ID (text); last_update is the date of the record's last update (text); _study is the
    full API record (dict).

    Output: the predicted risk of stopping early (number between 0 and 1), or None if the record cannot be scored.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Any failure (missing model, odd record) becomes None, so one bad record never breaks the result list.
    try:
        # The model only learned from interventional trials, so other studies get no risk.
        if _study["protocolSection"]["designModule"].get("studyType") != "INTERVENTIONAL":
            return None
        return score(build_features(_study))
    except (ImportError, FileNotFoundError, KeyError, ValueError, TypeError):
        return None


@st.cache_data(show_spinner=False, max_entries=200)
def drivers_for(nct_id, last_update, _study):
    """Work out what pushes this trial's risk up or down (src.predict.explain).

    Remembered like risk_for: per trial and record version, without looking at the record itself.

    Input: nct_id is the trial ID (text); last_update is the date of the record's last update (text); _study is the
    full API record (dict).

    Output: the list of drivers from src.predict.explain, or None if it cannot be worked out.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Any failure becomes None, so the page just skips the chart.
    try:
        return explain(build_features(_study))
    except (ImportError, FileNotFoundError, KeyError, ValueError, TypeError):
        return None


def last_update_of(study):
    """Read the date of the record's last update. It is part of the cache key, so a changed record is scored again.

    Input: study is the full API record (dict).

    Output: the date as text, or an empty text if the record has none.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Walk down the record; missing parts give an empty date.
    status = study.get("protocolSection", {}).get("statusModule", {})
    return status.get("lastUpdatePostDateStruct", {}).get("date", "")


# ---------------------------------------------------------------------------------------------------------------
# Names and labels
# ---------------------------------------------------------------------------------------------------------------

def pretty_phase(raw):
    """Turn the raw phase code into words, for example 'PHASE1, PHASE2' into 'Phase 1 / Phase 2'.

    Input: raw is the phase text from the API, codes separated by commas (text).

    Output: the phase in words as text; 'NA' or an empty text gives 'No phase'.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Name each code on its own.
    names = []
    for code in raw.split(","):
        code = code.strip()
        if code:
            names.append(label("phase", code))
    if not names:
        return label("phase", "NA")
    return " / ".join(names)


def phase_choice(code):
    """Give the text the phase filter shows for one option.

    Input: code is the raw phase code (text); the empty text means no filter.

    Output: the text to show.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    if code:
        return label("phase", code)
    return "Any phase"


def sponsor_choice(code):
    """Give the text the sponsor filter shows for one option.

    Input: code is the raw sponsor class (text); the empty text means no filter.

    Output: the text to show.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    if code:
        return label("sponsor_class", code)
    return "Any sponsor"


def chips_for(row, risk=None):
    """Build the small labels of one trial: risk (if known), phase and status.

    Input: row is the flat summary of one trial from summarize (dict); risk is the predicted risk (number, or None).

    Output: the labels as an HTML text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The risk label only exists when a risk is known.
    chips = ""
    if risk is not None:
        chips = risk_chip(risk)
    return chips + chip(pretty_phase(row["Phase"])) + chip(label("status", row["Status"]), "gray")


# ---------------------------------------------------------------------------------------------------------------
# What the buttons and inputs do. Streamlit calls these functions before it runs the page again.
# ---------------------------------------------------------------------------------------------------------------

def new_search():
    """Start again with the first ten results after any change of the search text or a filter.

    Input: none.

    Output: nothing. It sets the session value "shown".
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state["shown"] = PAGE


def reset_filters():
    """Go back to the empty search with no filters.

    Input: none.

    Output: nothing. It resets the values in the session state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state["q"] = ""
    st.session_state["phase_filter"] = ""
    st.session_state["sponsor_filter"] = ""
    st.session_state["shown"] = PAGE


def pick_example():
    """Turn a clicked example pill into the search text.

    Input: none.

    Output: nothing. It writes the clicked example into the session state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    example = st.session_state.get("example")
    if example:
        st.session_state["q"] = example
        st.session_state["example"] = None
        st.session_state["shown"] = PAGE


def show_more():
    """Show ten more results.

    Input: none.

    Output: nothing. It raises the session value "shown" by one page.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state["shown"] += PAGE


def open_trial(nct_id):
    """Show the detail view of one trial.

    Input: nct_id is the ID of the trial to open (text).

    Output: nothing. It stores the ID in the session state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state["selected"] = nct_id


def close_trial():
    """Go back from the detail view to the result list.

    Input: none.

    Output: nothing. It clears the selected trial in the session state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state["selected"] = None


# ---------------------------------------------------------------------------------------------------------------
# The detail view of one trial
# ---------------------------------------------------------------------------------------------------------------

def detail_view(nct_id):
    """Draw the full page for one trial: its risk on the left, key design facts on the right, the summary below.

    Input: nct_id is the ID of the trial (text).

    Output: nothing. It draws the page; it stops the script with an error if the record cannot be loaded.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Load the full record again (the API client remembers answers for an hour).
    st.button("← Back to results", on_click=close_trial)
    study = get_trial(nct_id)
    if study is None:
        st.error("Could not load this trial. Check your connection, then go back and open it again.")
        st.stop()

    # Header: title, chips, sponsor and a link to the record on ClinicalTrials.gov.
    row = summarize(study)
    protocol = study.get("protocolSection", {})
    design = protocol.get("designModule", {})
    st.markdown(
        '<div class="card-title" style="font-size:clamp(1.75rem,3.2vw,2.75rem);font-weight:300;'
        f'letter-spacing:-0.035em">{esc(row["Title"])}</div>'
        f'{chips_for(row)}<div class="card-sub">{esc(row["Sponsor"])} · '
        f'<a href="https://clinicaltrials.gov/study/{esc(nct_id)}" target="_blank">'
        f'{esc(nct_id)} on ClinicalTrials.gov</a></div>',
        unsafe_allow_html=True)
    st.write("")

    # Two columns: risk on the left, facts on the right.
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Risk of stopping early")
        # Risk card, comparison bar and drivers chart, or a plain message if the risk cannot be estimated.
        risk = None
        if get_model() is not None:
            risk = risk_for(nct_id, last_update_of(study), study)
        if get_model() is None:
            st.info("The risk estimate is not available here yet: the model file is missing.")
        elif risk is None:
            st.warning("This record has too little information to estimate a risk.")
        else:
            st.markdown(risk_card(risk), unsafe_allow_html=True)
            bar = risk_bar(risk)
            if bar is not None:
                chart(bar)
            st.caption("Rough estimate for comparing trials, not an exact chance. Not medical advice.")
            with st.expander("How to read this"):
                st.write(bar_caption())
                st.write(estimate_caption(risk, ongoing=True))
            drivers = drivers_for(nct_id, last_update_of(study), study)
            if drivers:
                chart(drivers_chart(drivers), key="drivers_chart")
                st.caption(drivers_caption())

    with right:
        # Key design facts from the record, in plain words. label("", None) is the text for a missing value.
        st.markdown("##### At a glance")
        names = []
        for intervention in protocol.get("armsInterventionsModule", {}).get("interventions", []):
            names.append(intervention.get("name", ""))
        interventions = ", ".join(names)
        # One line per fact.
        facts = {
            "Planned enrollment": design.get("enrollmentInfo", {}).get("count", label("", None)),
            "Allocation": label("allocation", design.get("designInfo", {}).get("allocation")),
            "Blinding": label("masking", design.get("designInfo", {}).get("maskingInfo", {}).get("masking")),
            "Interventions": interventions or label("", None),
        }
        for name, value in facts.items():
            st.markdown(f"**{name}**  \n{value}")

    # The brief summary of the trial, folded away.
    with st.expander("Summary"):
        st.write(protocol.get("descriptionModule", {}).get("briefSummary", "No summary available."))


# ---------------------------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------------------------

# ---- 1. Page state and detail view ----
# What the page remembers between clicks. At the start: empty search, no filters, the first ten results, no
# trial opened.
START = {"q": "", "phase_filter": "", "sponsor_filter": "", "shown": PAGE, "selected": None}
for key, value in START.items():
    if key not in st.session_state:
        st.session_state[key] = value

# A trial was opened: show only its detail view.
if st.session_state["selected"]:
    page_header("Find a trial")
    detail_view(st.session_state["selected"])
    st.stop()

# ---- 2. Header, search box and filters ----
# Search box, two filters and the example pills.
page_header("Find a trial", "Search ongoing interventional trials (treatments, devices, behaviour changes) on "
                            "ClinicalTrials.gov and see how likely each one is to stop early.")
st.text_input("Search", key="q", on_change=new_search, label_visibility="collapsed",
              placeholder="Condition, drug or keyword, for example breast cancer")
c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
c1.selectbox("Phase", [""] + PHASES, key="phase_filter", on_change=new_search, format_func=phase_choice)
c2.selectbox("Sponsor", [""] + SPONSOR_CLASSES, key="sponsor_filter", on_change=new_search,
             format_func=sponsor_choice)
nothing_to_clear = not (st.session_state["q"] or st.session_state["phase_filter"] or st.session_state["sponsor_filter"])
c3.button("Clear all", on_click=reset_filters, width="stretch", disabled=nothing_to_clear)
if not st.session_state["q"]:
    st.caption("Try an example:")
    st.pills("Examples", EXAMPLES, key="example", on_change=pick_example, label_visibility="collapsed")

# ---- 3. Search on ClinicalTrials.gov ----
# What to search for. An empty filter means "no filter", which the API client wants as None.
query = st.session_state["q"].strip()
phase = st.session_state["phase_filter"] or None
sponsor = st.session_state["sponsor_filter"] or None
shown = st.session_state["shown"]

# Ask the API. Answers are remembered for an hour. Errors come back as a value, the search never raises.
with st.spinner("Searching ClinicalTrials.gov..."):
    result = search_trials(query=query, phase=phase, sponsor_class=sponsor, page_size=shown)

if result["error"]:
    # Offline: fall back to the saved snapshot of real ongoing trials if there is one, so the page (and a demo)
    # keeps working. The banner makes clear these are not live results.
    offline = search_examples(query=query, phase=phase, sponsor_class=sponsor, page_size=shown)
    if offline is None:
        st.error(f"{result['error']} Check your internet connection, then try again. Every other page works "
                 "without it.")
        st.button("Try again")
        # The technical reason stays available for debugging, folded away so it does not alarm users.
        if result.get("detail"):
            with st.expander("Technical details"):
                st.code(result["detail"], language=None, wrap_lines=True)
        st.stop()
    st.warning(f"ClinicalTrials.gov cannot be reached right now, so you see saved example trials (downloaded "
               f"{offline['saved_on']}), not live results. Their status may have changed since.")
    st.button("Try the live search again")
    result = offline

# ---- 3b. No results ----
# Nothing found: say so and offer a way back.
if not result["studies"]:
    st.info("No ongoing trials match this search. Try a broader keyword or remove a filter.")
    st.button("Clear filters", on_click=reset_filters)
    st.stop()

# ---- 4. Result count and note ----
# How many trials match, in words that fit a search or the unfiltered list.
total = result["total"]
if query or phase or sponsor:
    st.markdown(f"**{total:,}** ongoing trials match")
else:
    st.markdown(f"**{total:,}** ongoing trials. Search or filter to narrow them down.")
model_ready = get_model() is not None
if model_ready:
    st.caption("The colour compares each trial with past trials. The percentage is a rough estimate; "
               "for trials that are still running it is less certain. Not medical advice.")

# ---- 5. Result cards ----
# Result list: one card per trial with its risk chip. The search results already hold the full records, so each
# trial is scored right here. Studies that cannot be scored get no risk chip.
for i, study in enumerate(result["studies"]):
    row = summarize(study)
    risk = None
    if model_ready:
        risk = risk_for(row["NCT ID"], last_update_of(study), study)
    with st.container(key=f"card_result_{i}"):
        st.markdown(f'<div class="card-title">{esc(row["Title"])}</div>{chips_for(row, risk)}'
                    f'<div class="card-sub">{esc(row["Sponsor"])}</div>', unsafe_allow_html=True)
        st.button("See details", key=f"open_{i}", on_click=open_trial, args=(row["NCT ID"],), type="primary",
                  icon=":material/arrow_forward:")

# ---- 6. Show more ----
# More results on the server than shown: offer the next ten (or what is left).
shown_now = len(result["studies"])
if total > shown_now:
    st.button(f"Show {min(PAGE, total - shown_now)} more", on_click=show_more)

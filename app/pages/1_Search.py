"""Find a recruiting trial, open it and see its predicted termination risk."""
import sys
from pathlib import Path

import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge and ui
from risk_gauge import bar_caption, estimate_caption, risk_bar, risk_card  # noqa: E402
from src.api_client import PHASES, SPONSOR_CLASSES, get_trial, search_trials, summarize  # noqa: E402
from ui import chip, esc, page_header  # noqa: E402

PAGE = 10
EXAMPLES = ["breast cancer", "diabetes", "asthma", "depression", "COVID-19"]
PHASE_NAMES = {"EARLY_PHASE1": "Early phase 1", "PHASE1": "Phase 1", "PHASE2": "Phase 2", "PHASE3": "Phase 3",
               "PHASE4": "Phase 4", "NA": "Not applicable", "N/A": "Not applicable"}
STATUS_NAMES = {"RECRUITING": "Recruiting", "NOT_YET_RECRUITING": "Not yet recruiting",
                "ACTIVE_NOT_RECRUITING": "Active, not recruiting", "ENROLLING_BY_INVITATION": "By invitation"}
SPONSOR_NAMES = {"INDUSTRY": "Industry", "NIH": "NIH", "FED": "Other US federal", "OTHER_GOV": "Other government",
                 "INDIV": "Individual", "NETWORK": "Research network", "OTHER": "Academic / other"}


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server; None while src/predict.py or models/gbm.joblib is missing."""
    try:
        from src.predict import load_model
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


def pretty_phase(raw):
    """'PHASE1, PHASE2' -> 'Phase 1 / Phase 2'; 'N/A' -> 'Not applicable'."""
    return " / ".join(PHASE_NAMES.get(p.strip(), p.strip()) for p in raw.split(",") if p.strip()) or "Not applicable"


def pretty(value):
    """'NON_RANDOMIZED' -> 'Non randomized'; missing values become 'Not stated'."""
    return value.replace("_", " ").capitalize() if value else "Not stated"


def new_search():
    """Any change of the query or a filter starts again at the first page."""
    st.session_state["shown"] = PAGE


def reset_filters():
    """Back to the empty search with no filters."""
    st.session_state.update(q="", phase_filter="", sponsor_filter="", shown=PAGE)


def pick_example():
    """Turn a clicked example pill into the search text."""
    if st.session_state.get("example"):
        st.session_state.update(q=st.session_state["example"], example=None, shown=PAGE)


def show_more():
    """Load the next page of results."""
    st.session_state["shown"] += PAGE


def open_trial(nct_id):
    """Show the detail view of one trial."""
    st.session_state["selected"] = nct_id


def close_trial():
    """Back from the detail view to the result list."""
    st.session_state["selected"] = None


def chips_for(row):
    """Phase and status as small labels."""
    return chip(pretty_phase(row["Phase"])) + chip(STATUS_NAMES.get(row["Status"], pretty(row["Status"])), "gray")


def detail_view(nct_id):
    """Full page for one trial: risk card first, key design facts beside it."""
    st.button("← Back to results", on_click=close_trial)
    study = get_trial(nct_id)
    if study is None:
        st.error("Could not load this trial. Please try again.")
        st.stop()
    row = summarize(study)
    p = study.get("protocolSection", {})
    design = p.get("designModule", {})
    st.markdown(f'<div class="card-title" style="font-size:1.5rem">{esc(row["Title"])}</div>'
                f'{chips_for(row)}<div class="card-sub">{esc(row["Sponsor"])} · '
                f'<a href="https://clinicaltrials.gov/study/{esc(nct_id)}" target="_blank">{esc(nct_id)} on ClinicalTrials.gov</a></div>',
                unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Termination risk")
        if get_model() is None:
            st.info("Model coming soon. The risk appears here once the model is trained.")
        else:
            from src.features import build_features  # noqa: E402
            from src.predict import score  # noqa: E402

            # score() reuses the model get_model() already loaded (predict.py caches it per process).
            risk = score(build_features(study))
            st.markdown(risk_card(risk), unsafe_allow_html=True)
            bar = risk_bar(risk)
            if bar is not None:
                st.plotly_chart(bar, width="stretch")
            st.caption("Rough estimate for comparing trials, not an exact chance. Not medical advice.")
            with st.expander("How to read this"):
                st.write(bar_caption())
                st.write(estimate_caption(risk, ongoing=True))
    with right:
        st.markdown("##### At a glance")
        interventions = ", ".join(i.get("name", "") for i in p.get("armsInterventionsModule", {}).get("interventions", []))
        facts = {
            "Planned enrollment": design.get("enrollmentInfo", {}).get("count", "Not stated"),
            "Allocation": pretty(design.get("designInfo", {}).get("allocation")),
            "Masking": pretty(design.get("designInfo", {}).get("maskingInfo", {}).get("masking")),
            "Interventions": interventions or "Not stated",
        }
        for label, value in facts.items():
            st.markdown(f"**{label}**  \n{value}")
    with st.expander("Summary"):
        st.write(p.get("descriptionModule", {}).get("briefSummary", "No summary available."))


st.set_page_config(page_title="Find a trial - Trial-Radar", page_icon="🔎", layout="wide")
for key, default in {"q": "", "phase_filter": "", "sponsor_filter": "", "shown": PAGE, "selected": None}.items():
    st.session_state.setdefault(key, default)

if st.session_state["selected"]:
    page_header("🔎 Find a trial")
    detail_view(st.session_state["selected"])
    st.stop()

page_header("🔎 Find a trial", "Search recruiting trials on ClinicalTrials.gov and see how likely each one is to end early.")
st.text_input("Search", key="q", on_change=new_search, label_visibility="collapsed",
              placeholder="Condition, drug or keyword, for example breast cancer")
c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
c1.selectbox("Phase", [""] + PHASES, key="phase_filter", on_change=new_search,
             format_func=lambda x: PHASE_NAMES.get(x, "Any phase"))
c2.selectbox("Sponsor", [""] + SPONSOR_CLASSES, key="sponsor_filter", on_change=new_search,
             format_func=lambda x: SPONSOR_NAMES.get(x, "Any sponsor"))
c3.button("Clear all", on_click=reset_filters, width="stretch",
          disabled=not (st.session_state.q or st.session_state.phase_filter or st.session_state.sponsor_filter))
if not st.session_state.q:
    st.pills("Examples", EXAMPLES, key="example", on_change=pick_example, label_visibility="collapsed")

with st.spinner("Searching ClinicalTrials.gov..."):
    result = search_trials(query=st.session_state.q.strip(), phase=st.session_state.phase_filter or None,
                           sponsor_class=st.session_state.sponsor_filter or None, page_size=st.session_state.shown)

if result["error"]:
    st.error(f"{result['error']} Check your connection, then try again.")
    st.button("Try again")
    st.stop()
if not result["studies"]:
    st.info("No recruiting trials match this search. Try a broader keyword or remove a filter.")
    st.button("Clear filters", on_click=reset_filters)
    st.stop()

rows = [summarize(s) for s in result["studies"]]
total = result["total"]
searching = st.session_state.q.strip() or st.session_state.phase_filter or st.session_state.sponsor_filter
st.markdown(f"**{total:,}** trials match" if searching else f"**{total:,}** trials. Search or filter to narrow them down.")
for i, row in enumerate(rows):
    with st.container(key=f"card_result_{i}"):
        text, action = st.columns([5, 1], vertical_alignment="center")
        text.markdown(f'<div class="card-title">{esc(row["Title"])}</div>{chips_for(row)}'
                      f'<div class="card-sub">{esc(row["Sponsor"])}</div>', unsafe_allow_html=True)
        action.button("See risk", key=f"open_{i}", on_click=open_trial, args=(row["NCT ID"],), width="stretch", type="primary")
if total > len(rows):
    st.button(f"Show {min(PAGE, total - len(rows))} more", on_click=show_more)

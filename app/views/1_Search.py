"""Find an ongoing trial, see its predicted risk of stopping early in the list and open its details."""
import sys
from pathlib import Path

import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge and ui
from risk_gauge import bar_caption, estimate_caption, risk_bar, risk_card, risk_chip  # noqa: E402
from src.api_client import PHASES, SPONSOR_CLASSES, get_trial, search_trials, summarize  # noqa: E402
from src.labels import label  # noqa: E402
from ui import chip, esc, page_header, require_router  # noqa: E402

require_router("views/1_Search.py")

PAGE = 10
EXAMPLES = ["breast cancer", "diabetes", "asthma", "depression", "COVID-19"]


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server; None while src/predict.py or models/gbm.joblib is missing."""
    try:
        from src.predict import load_model
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


@st.cache_data(show_spinner=False, max_entries=500)
def risk_for(nct_id, last_update, _study):
    """Predicted risk of one trial from its record, or None if the record cannot be scored.

    Cached per trial and record version (`last_update`); the record itself is not hashed, hence the underscore.
    """
    try:
        from src.features import build_features
        from src.predict import score
        if _study["protocolSection"]["designModule"].get("studyType") != "INTERVENTIONAL":
            return None  # the model only learned from interventional trials
        return score(build_features(_study))
    except (ImportError, FileNotFoundError, KeyError, ValueError, TypeError):
        return None


def last_update(study):
    """Date of the record's last update; part of the cache key so a changed record is scored again."""
    return study.get("protocolSection", {}).get("statusModule", {}).get("lastUpdatePostDateStruct", {}).get("date", "")


def pretty_phase(raw):
    """'PHASE1, PHASE2' -> 'Phase 1 / Phase 2'; 'NA' -> 'No phase'."""
    return " / ".join(label("phase", p.strip()) for p in raw.split(",") if p.strip()) or label("phase", "NA")


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


def chips_for(row, risk=None):
    """Risk (if known), phase and status as small labels."""
    risk_html = risk_chip(risk) if risk is not None else ""
    return risk_html + chip(pretty_phase(row["Phase"])) + chip(label("status", row["Status"]), "gray")


def detail_view(nct_id):
    """Full page for one trial: risk card first, key design facts beside it."""
    st.button("← Back to results", on_click=close_trial)
    study = get_trial(nct_id)
    if study is None:
        st.error("Could not load this trial. Check your connection, then go back and open it again.")
        st.stop()
    row = summarize(study)
    p = study.get("protocolSection", {})
    design = p.get("designModule", {})
    st.markdown(f'<div class="card-title" style="font-size:clamp(1.75rem,3.2vw,2.75rem);font-weight:300;letter-spacing:-0.035em">{esc(row["Title"])}</div>'
                f'{chips_for(row)}<div class="card-sub">{esc(row["Sponsor"])} · '
                f'<a href="https://clinicaltrials.gov/study/{esc(nct_id)}" target="_blank">{esc(nct_id)} on ClinicalTrials.gov</a></div>',
                unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Risk of stopping early")
        risk = risk_for(nct_id, last_update(study), study) if get_model() is not None else None
        if get_model() is None:
            st.info("The risk estimate is not available here yet: the model file is missing.")
        elif risk is None:
            st.warning("This record has too little information to estimate a risk.")
        else:
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
            "Planned enrollment": design.get("enrollmentInfo", {}).get("count", label("", None)),
            "Allocation": label("allocation", design.get("designInfo", {}).get("allocation")),
            "Blinding": label("masking", design.get("designInfo", {}).get("maskingInfo", {}).get("masking")),
            "Interventions": interventions or label("", None),
        }
        for name, value in facts.items():
            st.markdown(f"**{name}**  \n{value}")
    with st.expander("Summary"):
        st.write(p.get("descriptionModule", {}).get("briefSummary", "No summary available."))


for key, default in {"q": "", "phase_filter": "", "sponsor_filter": "", "shown": PAGE, "selected": None}.items():
    st.session_state.setdefault(key, default)

if st.session_state["selected"]:
    page_header("🔎 Find a trial")
    detail_view(st.session_state["selected"])
    st.stop()

page_header("🔎 Find a trial", "Search ongoing interventional trials (treatments, devices, behaviour changes) on ClinicalTrials.gov and see how likely each one is to stop early.")
st.text_input("Search", key="q", on_change=new_search, label_visibility="collapsed",
              placeholder="Condition, drug or keyword, for example breast cancer")
c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
c1.selectbox("Phase", [""] + PHASES, key="phase_filter", on_change=new_search,
             format_func=lambda x: label("phase", x) if x else "Any phase")
c2.selectbox("Sponsor", [""] + SPONSOR_CLASSES, key="sponsor_filter", on_change=new_search,
             format_func=lambda x: label("sponsor_class", x) if x else "Any sponsor")
c3.button("Clear all", on_click=reset_filters, width="stretch",
          disabled=not (st.session_state.q or st.session_state.phase_filter or st.session_state.sponsor_filter))
if not st.session_state.q:
    st.caption("Try an example:")
    st.pills("Examples", EXAMPLES, key="example", on_change=pick_example, label_visibility="collapsed")

with st.spinner("Searching ClinicalTrials.gov..."):
    result = search_trials(query=st.session_state.q.strip(), phase=st.session_state.phase_filter or None,
                           sponsor_class=st.session_state.sponsor_filter or None, page_size=st.session_state.shown)

if result["error"]:
    st.error(f"{result['error']} Check your connection, then try again.")
    st.button("Try again")
    st.stop()
if not result["studies"]:
    st.info("No ongoing trials match this search. Try a broader keyword or remove a filter.")
    st.button("Clear filters", on_click=reset_filters)
    st.stop()

total = result["total"]
searching = st.session_state.q.strip() or st.session_state.phase_filter or st.session_state.sponsor_filter
st.markdown(f"**{total:,}** ongoing trials match" if searching else f"**{total:,}** ongoing trials. Search or filter to narrow them down.")
model_ready = get_model() is not None
if model_ready:
    st.caption("The colour compares each trial with past trials. The percentage is a rough estimate; "
               "for trials that are still running it is less certain. Not medical advice.")
for i, study in enumerate(result["studies"]):
    row = summarize(study)
    risk = risk_for(row["NCT ID"], last_update(study), study) if model_ready else None
    with st.container(key=f"card_result_{i}"):
        st.markdown(f'<div class="card-title">{esc(row["Title"])}</div>{chips_for(row, risk)}'
                    f'<div class="card-sub">{esc(row["Sponsor"])}</div>', unsafe_allow_html=True)
        st.button("See details", key=f"open_{i}", on_click=open_trial, args=(row["NCT ID"],), type="primary",
                  icon=":material/arrow_forward:")
if total > len(result["studies"]):
    st.button(f"Show {min(PAGE, total - len(result['studies']))} more", on_click=show_more)

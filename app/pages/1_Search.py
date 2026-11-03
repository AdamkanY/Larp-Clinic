"""Find a recruiting trial, open it and see its predicted termination risk."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge and ui
from risk_gauge import bar_caption, estimate_caption, risk_bar  # noqa: E402
from src.api_client import PHASES, SPONSOR_CLASSES, get_trial, search_trials, summarize  # noqa: E402
from ui import page_header  # noqa: E402

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


st.set_page_config(page_title="Find a trial - Trial-Radar", page_icon="🔎", layout="wide")
for key, default in {"q": "", "phase_filter": "", "sponsor_filter": "", "shown": PAGE}.items():
    st.session_state.setdefault(key, default)

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
table = pd.DataFrame({
    "Trial": [r["Title"] for r in rows],
    "Phase": [pretty_phase(r["Phase"]) for r in rows],
    "Status": [STATUS_NAMES.get(r["Status"], pretty(r["Status"])) for r in rows],
})
selection = st.dataframe(
    table, hide_index=True, width="stretch", height="content", on_select="rerun", selection_mode="single-row",
    column_config={"Trial": st.column_config.TextColumn(width=380), "Phase": st.column_config.TextColumn(width=120),
                   "Status": st.column_config.TextColumn(width=170)},
)
if total > len(rows):
    st.button(f"Show {min(PAGE, total - len(rows))} more", on_click=show_more)

selected = selection.selection.rows
if not selected:
    st.caption("Select a trial in the list to see its risk and design.")
    st.stop()

row = rows[selected[0]]
nct_id = row["NCT ID"]
study = get_trial(nct_id)
if study is None:
    st.error("Could not load this trial. Please try again.")
    st.stop()

p = study.get("protocolSection", {})
design = p.get("designModule", {})
with st.container(border=True):
    st.subheader(p.get("identificationModule", {}).get("briefTitle", row["Title"]))
    st.caption(f"[{nct_id}](https://clinicaltrials.gov/study/{nct_id}) · "
               f"{STATUS_NAMES.get(row['Status'], pretty(row['Status']))} · {row['Sponsor']}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Termination risk")
        if get_model() is None:
            st.info("Model coming soon. The risk appears here once the model is trained.")
        else:
            from src.features import build_features  # noqa: E402
            from src.predict import score  # noqa: E402

            # score() reuses the model get_model() already loaded (predict.py caches it per process).
            risk = score(build_features(study))
            st.metric("Estimated risk", f"{risk:.1%}", label_visibility="collapsed")
            try:
                from src.predict import percentile

                st.write(f"Riskier than **{percentile(risk):.0f}%** of historical trials (started 2008-2016).")
            except (ImportError, FileNotFoundError):
                pass  # no score_reference.npy yet: show the probability alone
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
            "Phase": pretty_phase(row["Phase"]),
            "Planned enrollment": design.get("enrollmentInfo", {}).get("count", "Not stated"),
            "Allocation": pretty(design.get("designInfo", {}).get("allocation")),
            "Masking": pretty(design.get("designInfo", {}).get("maskingInfo", {}).get("masking")),
            "Interventions": interventions or "Not stated",
        }
        for label, value in facts.items():
            st.markdown(f"**{label}**  \n{value}")
    with st.expander("Summary"):
        st.write(p.get("descriptionModule", {}).get("briefSummary", "No summary available."))

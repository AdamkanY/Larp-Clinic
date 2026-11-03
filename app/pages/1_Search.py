"""Search ongoing trials, view details and (when a model exists) the termination risk."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for risk_gauge
from risk_gauge import estimate_caption, gauge_caption, risk_gauge  # noqa: E402
from src.api_client import PHASES, SPONSOR_CLASSES, get_trial, search_trials, summarize  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per server; None while src/predict.py or models/gbm.joblib is missing."""
    try:
        from src.predict import load_model
        return load_model()
    except (ImportError, FileNotFoundError):
        return None


st.set_page_config(page_title="Search - Trial-Radar", page_icon="🔎", layout="wide")
st.title("🔎 Search trials")

with st.form("search"):
    c1, c2 = st.columns(2)
    query = c1.text_input("Search term", placeholder="e.g. diabetes, semaglutide")
    condition = c2.text_input("Condition keyword", placeholder="e.g. breast cancer")
    c3, c4 = st.columns(2)
    phase = c3.selectbox("Phase", [""] + PHASES, format_func=lambda x: x or "Any")
    sponsor = c4.selectbox("Sponsor type", [""] + SPONSOR_CLASSES, format_func=lambda x: x or "Any")
    submitted = st.form_submit_button("Search")

if submitted or "results" not in st.session_state:
    with st.spinner("Searching ClinicalTrials.gov..."):
        st.session_state["results"] = search_trials(
            query=query, condition=condition, phase=phase or None, sponsor_class=sponsor or None
        )

result = st.session_state["results"]
if result["error"]:
    st.error(result["error"])
    st.stop()
if not result["studies"]:
    st.warning("No ongoing trials found. Try a different search term or fewer filters.")
    st.stop()

rows = [summarize(s) for s in result["studies"]]
st.caption(f"Showing {len(rows)} of {result['total']} matching trials")
table = st.dataframe(
    pd.DataFrame(rows), hide_index=True, width="stretch",
    on_select="rerun", selection_mode="single-row",
)

selected = table.selection.rows
if not selected:
    st.stop()

nct_id = rows[selected[0]]["NCT ID"]
study = get_trial(nct_id)
if study is None:
    st.error("Could not load this trial.")
    st.stop()

p = study.get("protocolSection", {})
design = p.get("designModule", {})
st.divider()
st.subheader(f"{p.get('identificationModule', {}).get('briefTitle', nct_id)}")
st.caption(nct_id)
d1, d2, d3 = st.columns(3)
d1.metric("Phase", ", ".join(design.get("phases", [])) or "N/A")
d2.metric("Study type", design.get("studyType", "N/A"))
d3.metric("Planned enrollment", design.get("enrollmentInfo", {}).get("count", "N/A"))
st.write("**Allocation:**", design.get("designInfo", {}).get("allocation", "N/A"))
st.write("**Masking:**", design.get("designInfo", {}).get("maskingInfo", {}).get("masking", "N/A"))
st.write("**Interventions:**", ", ".join(
    i.get("name", "") for i in p.get("armsInterventionsModule", {}).get("interventions", [])) or "N/A")
st.write("**Summary:**", p.get("descriptionModule", {}).get("briefSummary", "N/A"))

st.subheader("Termination risk")
if get_model() is None:
    st.info("Model coming soon. The risk gauge appears here once the model is trained.")
    st.stop()

from src.features import build_features  # noqa: E402
from src.predict import score  # noqa: E402

# score() reuses the model get_model() already loaded (predict.py caches it per process).
risk = score(build_features(study))

st.plotly_chart(risk_gauge(risk), width="stretch")

st.caption(gauge_caption())

try:
    from src.predict import percentile

    rank = percentile(risk)
except (ImportError, FileNotFoundError):
    # Older predict.py or no models/score_reference.npy yet: show the probability alone.
    rank = None
if rank is not None:
    st.write(f"Riskier than **{rank:.0f}%** of historical trials (started 2008-2016).")
st.caption(estimate_caption(risk, ongoing=True))

"""Trial-Radar entry point: sets up the top navigation (in the order people use the app) and the landing page."""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import band_copy, chip, hero, inject_css, tick_scale  # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "models" / "gbm_metrics.json"

# (headline, sentence, page, action label, band tone). The sentences are the ones the old cards carried.
BANDS = [
    ("Find a trial", "Search ongoing trials on ClinicalTrials.gov and see how likely each one is to stop early.",
     "views/1_Search.py", "Search trials", "paper"),
    ("Check your own trial", "Describe a design and watch the risk change as you adjust it.",
     "views/2_Check_Your_Trial.py", "Open the checker", "ink"),
    ("Explore the data", "Charts of more than 160,000 past trials: which kinds get stopped early?",
     "views/4_Explore.py", "Open the charts", "paper"),
    ("How the model works", "How well it works, what matters most and where it falls short.",
     "views/5_Explain.py", "Read the details", "ink"),
]


@st.cache_data(show_spinner=False)
def headline_numbers():
    """Share stopped early, number of past trials and test ROC-AUC from models/gbm_metrics.json; None if it is missing."""
    try:
        metrics = json.loads(METRICS.read_text())
        train, test = metrics["split"]["train"], metrics["split"]["test"]
        total = train["n_trials"] + test["n_trials"]
        stopped = train["n_terminated"] + test["n_terminated"]
        return {"rate": stopped / total, "total": total, "roc_auc": metrics["test"]["roc_auc"]}
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return None


def home():
    """Landing page: one giant fact, four ways in as alternating bands, and how to read the numbers."""
    inject_css()
    numbers = headline_numbers()
    with st.container(key="band_hero"):
        if numbers:
            ticks = tick_scale(round(numbers["rate"] * 100),
                               f"Each line is 1% of {numbers['total']:,} past trials. The pink ones were stopped early.")
            hero("How likely is a clinical trial to be stopped early?",
                 "Estimated from what is known when it is designed.",
                 number=f"{numbers['rate']:.1%}",
                 lead="of the past trials in our data ended before their planned end.", scale=ticks)
        else:
            hero("How likely is a clinical trial to be stopped early?", "Estimated from what is known when it is designed.")
        st.page_link("views/2_Check_Your_Trial.py", label="Check your own trial")
    for index, (headline, text, page, action, tone) in enumerate(BANDS):
        number = label = None
        if numbers and page == "views/4_Explore.py":
            number, label = f"{numbers['total']:,}", "past trials that ended, 2008 to 2019"
        if numbers and page == "views/5_Explain.py":
            number, label = f"{numbers['roc_auc']:.2f}", "ROC-AUC on newer trials the model never saw"
        with st.container(key=f"band_{tone}_{index}"):
            st.markdown(band_copy(headline, text, number, label), unsafe_allow_html=True)
            st.page_link(page, label=action)
    with st.container(key="band_paper_read"):
        st.markdown('<h2 class="band-title small">How to read a result</h2>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="legend">{chip("Lower than most past trials", "green")}{chip("Higher than most past trials", "yellow")}'
            f'{chip("Among the riskiest 10%", "red")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="note">The colour compares a trial with past trials. The percentage is a rough estimate for '
            'comparing trials, not an exact chance, and it tends to be too low. The model only sees facts known '
            'at registration, so it cannot know about funding, recruitment or safety problems.</div>',
            unsafe_allow_html=True,
        )
        st.caption("Student project (HSG Computer Science). Not medical advice: do not use it to decide about "
                   "treatment or participation.")


st.set_page_config(page_icon=":material/radar:", layout="wide")  # no page_title: each page then names its own browser tab
st.session_state["via_router"] = True
navigation = st.navigation([
    st.Page(home, title="Home", default=True),
    st.Page("views/1_Search.py", title="Find a trial"),
    st.Page("views/2_Check_Your_Trial.py", title="Check your own trial"),
    st.Page("views/4_Explore.py", title="Explore the data"),
    st.Page("views/5_Explain.py", title="How the model works"),
], position="top")
target = st.session_state.pop("target_page", None)  # a page that was opened directly comes back here
if target:
    st.switch_page(target)
navigation.run()

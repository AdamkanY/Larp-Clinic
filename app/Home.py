"""Trial-Radar entry point: sets up the top navigation (in the order people use the app) and the landing page."""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import band_copy, chip, hero, inject_css, tick_scale  # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "models" / "gbm_metrics.json"

# The five pages in the order people use them. Navigation and landing-page bands both come from this list, so
# they cannot drift apart: (title, file, sentence for the band, action label).
PAGES = [
    ("Find a trial", "pages/1_Find_a_Trial.py",
     "Search ongoing trials on ClinicalTrials.gov and see how likely each one is to stop early.", "Search trials"),
    ("Check your own trial", "pages/2_Check_Your_Trial.py",
     "Describe a design and watch the risk change as you adjust it.", "Open the checker"),
    ("Explore the data", "pages/3_Explore_Data.py",
     "Charts of more than 160,000 past trials: which kinds get stopped early?", "Open the charts"),
    ("How the model works", "pages/4_How_the_Model_Works.py",
     "How well it works, how to read a score, what matters most and where it falls short.", "Read the details"),
    ("Play the game", "pages/5_Game.py",
     "Two past trials, one was stopped early. Can you spot it more often than the model?", "Start a game"),
]
PAGE_FILE = {title: path for title, path, _, _ in PAGES}


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
    """Landing page: one giant fact, the five pages as alternating bands (same order as the navigation), and how
    to read the numbers."""
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
        st.page_link(PAGE_FILE["Check your own trial"], label="Check your own trial")
    for index, (headline, page, text, action) in enumerate(PAGES):
        tone = "paper" if index % 2 == 0 else "ink"  # bands alternate, starting light
        number = label = None
        if numbers and page == PAGE_FILE["Explore the data"]:
            number, label = f"{numbers['total']:,}", "past trials that ended, 2008 to 2019"
        if numbers and page == PAGE_FILE["How the model works"]:
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
# Each st.Page title is also the browser tab title of that page.
navigation = st.navigation([st.Page(home, title="Home", default=True)]
                           + [st.Page(path, title=title) for title, path, _, _ in PAGES], position="top")
target = st.session_state.pop("target_page", None)  # a page that was opened directly comes back here
if target:
    st.switch_page(target)
navigation.run()

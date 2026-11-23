# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Trial-Radar entry point: sets up the top navigation (in the order people use the app) and the landing page."""
import json
import sys
from pathlib import Path

import streamlit as st

# Add app/ to the import path, so ui.py is found wherever Streamlit was started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import band_copy, chip, esc, hero, inject_css, tick_scale

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
# The file of each page by its title, for the links on the landing page.
PAGE_FILE = {}
for page_title, page_file, page_sentence, page_action in PAGES:
    PAGE_FILE[page_title] = page_file

# The use case in one place: who decides what with the app. The video and the README tell the same story, so a
# change here should be mirrored there: (who, the decision they face, which page helps).
AUDIENCES = [
    ("Trial sponsors and study designers",
     "Before registering a trial, they can test the design and see which single change, such as blinding or a data "
     "monitoring committee, goes with a lower estimate in past trials.", "Check your own trial"),
    ("Investors and business analysts",
     "Screening a company's ongoing trials, they can see which ones look riskier than most and deserve a closer look "
     "before they weigh the pipeline.", "Find a trial"),
]
# Why an app and not just another paper: what the published models do not offer. Sources are in the README.
NOVELTY = ("Published models predict early termination well on paper, but the ones we found do not let you search "
           "live trials or change a design and watch the risk move. Trial-Radar does both, with a model that only "
           "sees what is known at registration and is tested on later trials than it learned from.")


@st.cache_data(show_spinner=False)
def headline_numbers():
    """The big numbers of the landing page: share stopped early, number of past trials and test ROC-AUC.

    They are read from models/gbm_metrics.json, so they always match the saved model. Returns None if the file is
    missing or incomplete.
    """
    try:
        metrics = json.loads(METRICS.read_text())
        train = metrics["split"]["train"]
        test = metrics["split"]["test"]
        total = train["n_trials"] + test["n_trials"]
        stopped = train["n_terminated"] + test["n_terminated"]
        return {"rate": stopped / total, "total": total, "roc_auc": metrics["test"]["roc_auc"]}
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return None


def home():
    """The landing page: one giant fact, who the app is for, one band per page and how to read a result."""
    inject_css()
    numbers = headline_numbers()

    # First band: the question, the share of trials stopped early as a giant number and the tick scale below it.
    # Without the metrics file only the question is shown.
    question = "How likely is a clinical trial to be stopped early?"
    answer = "Estimated from what is known when it is designed."
    with st.container(key="band_hero"):
        if numbers:
            ticks_label = f"Each line is 1% of {numbers['total']:,} past trials. The pink ones were stopped early."
            ticks = tick_scale(round(numbers["rate"] * 100), ticks_label)
            hero(question, answer, number=f"{numbers['rate']:.1%}",
                 lead="of the past trials in our data ended before their planned end.", scale=ticks)
        else:
            hero(question, answer)
        st.page_link(PAGE_FILE["Check your own trial"], label="Check your own trial")

    # Who the app is for and what it adds. "band-title small" on purpose: the Home test lists the page bands by
    # the plain "band-title" class, and this band is not a page.
    with st.container(key="band_paper_who"):
        st.markdown('<h2 class="band-title small">Who it is for</h2>', unsafe_allow_html=True)
        columns = st.columns(len(AUDIENCES), gap="large")
        for i in range(len(AUDIENCES)):
            who, decision, page_title = AUDIENCES[i]
            with columns[i]:
                st.markdown(f'<p class="band-sentence"><strong>{esc(who)}.</strong> {esc(decision)}</p>',
                            unsafe_allow_html=True)
                st.page_link(PAGE_FILE[page_title], label=page_title)
        st.markdown(f'<div class="note">{esc(NOVELTY)}</div>', unsafe_allow_html=True)

    # One band per page, in navigation order. The bands alternate between light and dark, starting light.
    for index in range(len(PAGES)):
        headline, page, text, action = PAGES[index]
        if index % 2 == 0:
            tone = "paper"
        else:
            tone = "ink"
        # Two of the bands carry a big number from the metrics file.
        number = None
        label = None
        if numbers and page == PAGE_FILE["Explore the data"]:
            number = f"{numbers['total']:,}"
            label = "past trials that ended, 2008 to 2019"
        if numbers and page == PAGE_FILE["How the model works"]:
            number = f"{numbers['roc_auc']:.2f}"
            label = "ROC-AUC on newer trials the model never saw"
        with st.container(key=f"band_{tone}_{index}"):
            st.markdown(band_copy(headline, text, number, label), unsafe_allow_html=True)
            st.page_link(page, label=action)

    # Last band: the three colours and a plain note on how to read a result.
    with st.container(key="band_paper_read"):
        st.markdown('<h2 class="band-title small">How to read a result</h2>', unsafe_allow_html=True)
        legend = (chip("Lower than most past trials", "green") + chip("Higher than most past trials", "yellow")
                  + chip("Among the riskiest 10%", "red"))
        st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="note">The colour compares a trial with past trials. The percentage is a rough estimate for '
            'comparing trials, not an exact chance, and it tends to be too low. The model only sees facts known '
            'at registration, so it cannot know about funding, recruitment or safety problems.</div>',
            unsafe_allow_html=True,
        )
        st.caption("Student project (HSG Computer Science). Not medical advice: do not use it to decide about "
                   "treatment or participation.")


# ---- Router ----
# Home.py is the entry point for every page: it sets the layout once, builds the top navigation from PAGES and runs
# the chosen page. There is no page_title here, so each page names its own browser tab.
st.set_page_config(page_icon=":material/radar:", layout="wide")

# Pages check this flag (ui.require_router) to know they were opened through the router.
st.session_state["via_router"] = True

# The landing page first, then the five pages. Each title is also the browser tab title of that page.
all_pages = [st.Page(home, title="Home", default=True)]
for page_title, page_file, page_sentence, page_action in PAGES:
    all_pages.append(st.Page(page_file, title=page_title))
navigation = st.navigation(all_pages, position="top")

# A page that was opened directly stored its name (ui.require_router) and comes back here.
target = st.session_state.pop("target_page", None)
if target:
    st.switch_page(target)
navigation.run()

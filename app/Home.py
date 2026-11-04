"""Trial-Radar entry point: sets up the sidebar navigation (in the order people use the app) and the landing page."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import chip, esc, hero  # noqa: E402

CARDS = [
    ("🔎", "Find a trial", "Search ongoing trials on ClinicalTrials.gov and see how likely each one is to stop early.",
     "pages/1_Search.py", "Search trials"),
    ("🧪", "Check your own trial", "Describe a design and watch the risk change as you adjust it.",
     "pages/5_Check_Your_Trial.py", "Open the checker"),
    ("📊", "Explore the data", "Charts of more than 160,000 past trials: which kinds get stopped early?",
     "pages/2_Explore.py", "Open the charts"),
    ("🧠", "How the model works", "How well it works, what matters most and where it falls short.",
     "pages/2_Explain.py", "Read the details"),
]


def home():
    """Landing page: what the app does, four ways in, and how to read the numbers."""
    hero("📡 Trial-Radar", "How likely is a clinical trial to be stopped early? Estimated from what is known when it is designed.")
    for start in (0, 2):
        for column, (icon, title, text, page, label) in zip(st.columns(2), CARDS[start:start + 2]):
            with column.container(key=f"card_home_{start}_{title[:3]}"):
                st.markdown(f'<div class="big-icon">{icon}</div><div class="card-title">{esc(title)}</div>'
                            f'<div class="card-sub">{esc(text)}</div>', unsafe_allow_html=True)
                st.page_link(page, label=label, icon=":material/arrow_forward:")

    st.subheader("How to read a result")
    st.markdown(
        f'{chip("Lower than most past trials", "green")}{chip("Higher than most past trials", "yellow")}'
        f'{chip("Among the riskiest 10%", "red")}',
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


st.set_page_config(page_icon="📡", layout="wide")  # no page_title: each page then names its own browser tab
st.session_state["via_router"] = True
navigation = st.navigation([
    st.Page(home, title="Home", icon="📡", default=True),
    st.Page("pages/1_Search.py", title="Find a trial", icon="🔎"),
    st.Page("pages/5_Check_Your_Trial.py", title="Check your own trial", icon="🧪"),
    st.Page("pages/2_Explore.py", title="Explore the data", icon="📊"),
    st.Page("pages/2_Explain.py", title="How the model works", icon="🧠"),
])
target = st.session_state.pop("target_page", None)  # a page that was opened directly comes back here
if target:
    st.switch_page(target)
navigation.run()

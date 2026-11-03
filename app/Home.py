"""Trial-Radar landing page."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Trial-Radar", page_icon="📡", layout="wide")
page_header(
    "📡 Trial-Radar",
    "How likely is a clinical trial to be stopped early? Estimated from what is known when it is designed.",
)

CARDS = [
    ("🔎 Find a trial", "Search recruiting trials on ClinicalTrials.gov and see each one's risk.", "pages/1_Search.py", "Search trials"),
    ("🧪 Check your own trial", "Describe a design and watch the risk change as you adjust it.", "pages/5_Check_Your_Trial.py", "Open the checker"),
    ("🧠 How the model works", "How accurate it is, what matters most and where it falls short.", "pages/2_Explain.py", "Read the details"),
    ("📊 Explore the data", "Charts of the 160,000 finished trials the model learned from.", "pages/2_Explore.py", "Open the charts"),
]
for row in (CARDS[:2], CARDS[2:]):
    for column, (title, text, page, label) in zip(st.columns(2), row):
        with column.container(border=True):
            st.subheader(title)
            st.write(text)
            st.page_link(page, label=label, icon=":material/arrow_forward:")

st.caption("Student project (HSG Computer Science). Not medical advice: do not use it to decide about treatment or participation.")

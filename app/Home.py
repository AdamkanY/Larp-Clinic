"""Trial-Radar landing page."""
import streamlit as st

st.set_page_config(page_title="Trial-Radar", page_icon="📡", layout="wide")

st.title("📡 Trial-Radar")
st.write(
    "Trial-Radar estimates how likely a clinical trial is to be terminated early, "
    "using only information known when the trial was designed. "
    "Search ongoing trials from ClinicalTrials.gov and see their predicted risk. "
    "Or describe your own trial design and see how its risk changes."
)

st.subheader("Where to go")
st.page_link("pages/1_Search.py", label="Search trials", icon="🔎")
st.page_link("pages/5_Check_Your_Trial.py", label="Check your own trial", icon="🧪")

st.info(
    "This is a student project (HSG Computer Science) and not medical advice. "
    "Do not use it to make decisions about treatment or trial participation."
)

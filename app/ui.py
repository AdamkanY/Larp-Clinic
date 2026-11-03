"""Shared look of the Trial-Radar pages: one stylesheet and one page header, so every page feels the same."""
import streamlit as st

CSS = """
.block-container { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 4rem; }
h1 { font-weight: 700; margin-bottom: 0.25rem; }
h2, h3 { margin-top: 1.5rem; }
[data-testid="stCaptionContainer"] { color: #4b5563; }
[data-testid="stMetricValue"] { font-weight: 700; }
"""


def page_header(title, subtitle=None):
    """Apply the shared stylesheet, then show the page title with an optional one-line description."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)

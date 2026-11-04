"""Shared look of the Trial-Radar pages: one stylesheet, a hero banner and a page header, so every page feels the same."""
import html

import streamlit as st

CSS = """
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebarNavLink"] { border-radius: 10px; }
[data-testid="stCaptionContainer"] { color: #475569; }
.page-head h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; color: #0f172a; margin: 0; padding: 0; }
.page-head p { color: #475569; margin: 0.35rem 0 1.4rem 0; font-size: 1.02rem; }
.page-head { margin-bottom: 0.6rem; }
.hero { background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 60%, #2563eb 100%); color: #fff;
        padding: 2.4rem 2.2rem; border-radius: 22px; margin-bottom: 1.6rem; }
.hero h1 { color: #fff; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.5rem 0; padding: 0; }
.hero p { color: #f0f9ff; margin: 0; font-size: 1.08rem; max-width: 38rem; }
[class*="st-key-card"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.1rem 1.3rem;
                          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); transition: box-shadow 0.15s, border-color 0.15s; }
[class*="st-key-card_home"] { min-height: 13rem; }
[class*="st-key-card"]:hover { box-shadow: 0 8px 22px rgba(15, 23, 42, 0.12); border-color: #93c5fd; }
.card-title { font-weight: 700; font-size: 1.05rem; color: #0f172a; line-height: 1.35; margin-bottom: 0.45rem; }
.card-sub { color: #475569; font-size: 0.9rem; margin-top: 0.4rem; margin-bottom: 0.5rem; }
.big-icon { font-size: 2rem; line-height: 1; margin-bottom: 0.4rem; }
.chip { display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        margin-right: 0.4rem; background: #eff6ff; color: #1e40af; }
.chip.gray { background: #f1f5f9; color: #334155; }
.chip.green { background: #dcfce7; color: #14532d; }
.chip.yellow { background: #fef9c3; color: #713f12; }
.chip.red { background: #fee2e2; color: #7f1d1d; }
[data-testid="stMetricValue"] { font-size: clamp(1.4rem, 2.6vw, 2rem); }
[data-testid="stMetricLabel"] p { white-space: normal; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 0.8rem 1rem; }
[class*="st-key-panel"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 0.9rem 1.1rem; }
.note { background: #f1f5f9; border-radius: 12px; padding: 0.8rem 1rem; color: #334155; font-size: 0.92rem; }
/* Columns inside a panel wrap onto their own line instead of squeezing a dropdown into 100 pixels. */
[class*="st-key-panel"] [data-testid="stColumn"] { min-width: 13.5rem; }
/* The live result stays in view while the form below it is edited. */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-sticky"]) { position: sticky; top: 3.75rem; z-index: 50; }
@media (max-width: 640px) {
  .hero { padding: 1.6rem 1.2rem; }
  .hero h1 { font-size: 2rem; }
  .page-head h1 { font-size: 1.8rem; }
}
.risk-card { border-radius: 18px; padding: 1.2rem 1.4rem; margin: 0.3rem 0 1rem 0; }
.risk-card .num { white-space: nowrap; font-size: clamp(2.4rem, 4.5vw, 3.2rem); font-weight: 800; line-height: 1; letter-spacing: -0.03em; }
.risk-card .lab { margin-top: 0.5rem; font-weight: 700; font-size: 1.05rem; }
.risk-card .sub { margin-top: 0.15rem; font-size: 0.92rem; opacity: 0.9; }
.risk-card.compact { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.2rem 1.1rem; padding: 0.7rem 1.1rem;
                     margin: 0 0 0.8rem 0; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15); }
.risk-card.compact .num { font-size: 2rem; }
.risk-card.compact .lab, .risk-card.compact .sub { margin: 0; }
.risk-card.green { background: #dcfce7; color: #14532d; }
.risk-card.yellow { background: #fef9c3; color: #713f12; }
.risk-card.red { background: #fee2e2; color: #7f1d1d; }
.risk-card.gray { background: #f1f5f9; color: #1e293b; }
"""


def esc(text):
    """Escape text for use inside HTML that goes through st.markdown (also keeps '$' from turning into math)."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def chip(text, tone="blue"):
    """A small rounded label, for example a phase or a status."""
    return f'<span class="chip {tone}">{esc(text)}</span>'


def require_router(page):
    """Send a page that was opened directly (browser refresh, bookmark) through Home.py first.

    Opened on its own, a page would miss the sidebar navigation and the wide layout that Home.py sets up. Home.py
    sets `via_router` and, if a target was stored here, switches straight back to that page.
    """
    if not st.session_state.get("via_router"):
        st.session_state["target_page"] = page
        st.switch_page("Home.py")


def inject_css():
    """Apply the shared stylesheet to the current page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hero(title, subtitle):
    """Large gradient banner for the landing page."""
    inject_css()
    st.markdown(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def page_header(title, subtitle=None):
    """Page title with an optional one-line description, styled the same on every page."""
    inject_css()
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="page-head"><h1>{esc(title)}</h1>{sub}</div>', unsafe_allow_html=True)

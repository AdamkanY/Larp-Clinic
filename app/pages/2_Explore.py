"""Explore: interactive dashboard of historical COMPLETED vs TERMINATED trials (2008-2019).

Descriptive only: no model is loaded here. Aggregations and figures live in `src/viz.py`.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs pages as scripts, so the repo root is not on sys.path by default.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import viz  # noqa: E402

st.set_page_config(page_title="Explore - Trial-Radar", layout="wide")


@st.cache_data(show_spinner="Loading historical trials...")
def get_trials():
    """Load trials.parquet once per session/server; Streamlit caches the DataFrame."""
    return viz.load_trials()


def chart(fig, caption):
    """Render a Plotly figure with its one-line caption underneath."""
    st.plotly_chart(fig, width="stretch")
    st.caption(caption)


df = get_trials()

st.title("Explore historical trials")
st.write("Interventional trials that started 2008-2019 and are now completed or terminated. "
         "Use the sidebar to filter; every chart updates.")

# ---- Sidebar filters ----
with st.sidebar:
    st.header("Filters")
    phase_options = [p for p in viz.PHASE_ORDER if p in set(df["phase_label"])]
    phases = st.multiselect("Phase", phase_options, placeholder="All phases")
    sponsor_options = df["sponsor_class"].value_counts().index.tolist()
    sponsors = st.multiselect("Sponsor class", sponsor_options, placeholder="All sponsor classes")
    y_min, y_max = int(df["start_year"].min()), int(df["start_year"].max())
    years = st.slider("Start year", y_min, y_max, (y_min, y_max))
    condition_kw = st.text_input("Condition keyword", placeholder="e.g. breast cancer")

filtered = viz.filter_trials(df, phases, sponsors, years, condition_kw)

# ---- KPI tiles ----
k1, k2, k3 = st.columns(3)
k1.metric("Trials in dataset", f"{len(df):,}")
if len(filtered):
    overall = df["label"].mean()
    rate = filtered["label"].mean()
    # Only compare against the full dataset once a filter actually narrows it.
    delta = f"{(rate - overall) * 100:+.1f} pp vs all trials" if len(filtered) < len(df) else None
    k2.metric("Terminated (filtered)", f"{rate:.1%}", delta=delta, delta_color="inverse")
else:
    k2.metric("Terminated (filtered)", "-")
k3.metric("Trials after filters", f"{len(filtered):,}")

if filtered.empty:
    st.warning("No trials match these filters. Widen the year range or clear a filter.")
    st.stop()

# ---- Design factors ----
st.subheader("Termination rate by design")
c1, c2 = st.columns(2)
with c1:
    chart(viz.rate_bar(viz.rate_by(filtered, "phase_label", viz.PHASE_ORDER), "phase_label", "Phase"),
          "Share of trials terminated early in each phase (faded bars: fewer than 30 trials).")
with c2:
    chart(viz.rate_bar(viz.rate_by(filtered, "sponsor_class"), "sponsor_class", "Lead sponsor class"),
          "Share of trials terminated early by type of lead sponsor (industry, NIH, other...).")
c3, c4 = st.columns(2)
with c3:
    chart(viz.rate_bar(viz.rate_by(filtered, "masking", viz.MASKING_ORDER), "masking", "Masking (blinding)"),
          "Share of trials terminated early by how many parties were blinded.")
with c4:
    chart(viz.rate_bar(viz.rate_by(filtered, "allocation"), "allocation", "Allocation"),
          "Share of trials terminated early for randomized vs non-randomized designs.")

# ---- Time ----
st.subheader("Over time")
chart(viz.rate_over_time(filtered),
      "Termination rate by start year; recent years have had less time to terminate, so read the trend with care.")

# ---- Conditions and eligibility ----
st.subheader("Conditions and eligibility")
c5, c6 = st.columns([3, 2])
with c5:
    chart(viz.conditions_bar(filtered),
          "The 15 most common conditions by trial count, labelled with their termination rate.")
with c6:
    chart(viz.eligibility_box(filtered),
          "Length of the eligibility criteria text for completed vs terminated trials (log scale).")

# ---- Trial table ----
st.subheader("Trials")
search = st.text_input("Search title, sponsor or NCT id", placeholder="e.g. NCT01234567 or Pfizer")
table = filtered
if search.strip():
    s = search.strip()
    table = table[table["brief_title"].str.contains(s, case=False, regex=False, na=False)
                  | table["lead_sponsor_name"].str.contains(s, case=False, regex=False, na=False)
                  | table["nct_id"].str.contains(s, case=False, regex=False, na=False)]
MAX_ROWS = 5000
st.caption(f"{len(table):,} trials match"
           + (f"; showing the first {MAX_ROWS:,} (narrow the filters to see more)." if len(table) > MAX_ROWS else "."))
st.dataframe(
    viz.trials_table(table.head(MAX_ROWS)),
    hide_index=True,
    width="stretch",
    column_config={
        "link": st.column_config.LinkColumn("NCT id", display_text=r"https://clinicaltrials\.gov/study/(.*)"),
        "brief_title": st.column_config.TextColumn("Title", width="large"),
        "phase_label": "Phase",
        "lead_sponsor_name": "Lead sponsor",
        "overall_status": "Status",
    },
)

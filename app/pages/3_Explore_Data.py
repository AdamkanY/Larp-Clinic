# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Explore: interactive charts of historical COMPLETED vs TERMINATED trials (2008-2019).

Descriptive only: no model is loaded here. The numbers come from the SQLite database `data/processed/trials.db`
through SQL queries (`src/database.py`: WHERE for the filters, COUNT / AVG for the key numbers, GROUP BY for the
bars, JOIN for the conditions); the figures live in `src/viz.py`. A folded box at the end shows every query.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs pages as scripts, so the repo root is not on sys.path by default.
ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "app"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from src import database, viz  # noqa: E402
from ui import chart as show_chart, page_header, require_router  # noqa: E402

require_router("pages/3_Explore_Data.py")

# The trial table shows at most this many rows (the browser gets slow beyond that); filters narrow the rest.
MAX_ROWS = 5000
FILTER_KEYS = {"explore_phases": [], "explore_sponsors": [], "explore_condition": ""}


@st.cache_resource(show_spinner="Preparing the trial database (first visit only, a few seconds)...")
def get_database():
    """Path of trials.db, built from trials.parquet on the first visit (or when the parquet is newer)."""
    return database.ensure_database()


@st.cache_data(show_spinner="Loading the matching trials...", max_entries=20)
def matching_trials(phases, sponsors, years, condition_kw):
    """The trials that pass the filters (one SQL SELECT ... WHERE), and that query for the SQL box.

    Cached per filter combination, so moving between charts does not re-read the database.
    """
    db = database.TrialDatabase(get_database(), phases=phases, sponsor_classes=sponsors, year_range=years,
                                condition_kw=condition_kw)
    return db.trials(), db.log[-1]


def reset_filters():
    """Back to all trials: clears every filter, the year range included."""
    st.session_state.update(FILTER_KEYS)
    st.session_state.pop("explore_years", None)


def chart(fig, caption):
    """Render a Plotly figure in the app look (ui.chart) with its short explanation underneath."""
    show_chart(fig)
    st.caption(caption)


# Open the database once; without the trial data the page explains what is missing instead of failing.
try:
    db_path = get_database()
except (FileNotFoundError, OSError):
    page_header("Explore the data")
    st.info("The trial data (data/processed/trials.parquet) is not available here, so there is nothing to chart yet.")
    st.stop()

# The overall numbers and the filter choices, straight from SQL (no filters yet).
overview = database.TrialDatabase(db_path)
choices = overview.options()
everything = overview.summary(filtered=False)
total, overall = everything["n"], everything["rate"]
y_min, y_max = choices["years"]
page_header("Explore the data",
            f"{total:,} past trials that started {y_min}-{y_max} and have ended. "
            f"{overall:.1%} of them were stopped early. Each bar shows how that share changes with one trait.")

# ---- Filters ----
# Filters live in st.session_state so 'Clear filters' can reset them and they survive reruns.
for key, default in FILTER_KEYS.items():
    st.session_state.setdefault(key, default)
with st.container(key="panel_filters"):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
    phase_options = [p for p in viz.PHASE_ORDER if p in choices["phases"]]
    phases = f1.multiselect("Phase", phase_options, key="explore_phases", placeholder="All phases")
    sponsors = f2.multiselect("Sponsor", choices["sponsors"], key="explore_sponsors", placeholder="All sponsors")
    years = f3.slider("Start year", y_min, y_max, (y_min, y_max), key="explore_years")
    condition_kw = f4.text_input("Condition keyword", key="explore_condition", placeholder="e.g. breast cancer")
    narrowed = bool(phases or sponsors or condition_kw.strip() or years != (y_min, y_max))
    st.button("Clear filters", on_click=reset_filters, disabled=not narrowed)

# Every query below applies the same filters: SQL does the counting, the WHERE clause the filtering. The full year
# range is no filter, so it adds nothing to the queries.
year_filter = tuple(years) if years != (y_min, y_max) else None
db = database.TrialDatabase(db_path, phases=phases, sponsor_classes=sponsors, year_range=year_filter,
                            condition_kw=condition_kw)
shown = db.summary()
filtered, trials_query = matching_trials(tuple(phases), tuple(sponsors), year_filter, condition_kw)

# ---- Key numbers ----
k1, k2, k3 = st.columns(3)
k1.metric("Trials shown", f"{shown['n']:,}", help=f"Out of {total:,} trials in total.")
if shown["n"]:
    rate = shown["rate"]
    # Only compare with all trials once a filter actually narrows the selection.
    delta = f"{(rate - overall) * 100:+.1f} points vs all trials" if shown["n"] < total else None
    k2.metric("Stopped early", f"{rate:.1%}", delta=delta, delta_color="inverse")
else:
    k2.metric("Stopped early", "-")
k3.metric("All trials", f"{overall:.1%}", help=f"Share stopped early across all {total:,} trials, for comparison.")

if not shown["n"]:
    st.warning("No trials match these filters. Widen the year range or clear a filter.")
    st.stop()
# The dashed average line in the bar charts is the average of the trials shown, not of all trials.
average = shown["rate"]

# ---- Design factors ----
st.subheader("Which kinds of trials get stopped more often?")
st.caption("The dashed line is the average of the trials shown. Faded bars rest on fewer than "
           f"{viz.MIN_GROUP_N} trials and are not reliable.")
c1, c2 = st.columns(2)
with c1:
    stats = db.rate_by("phase_label", viz.PHASE_ORDER)  # SQL: GROUP BY phase_label
    chart(viz.rate_bar(stats, "phase_label", "Phase", average),
          f"By phase. {viz.extremes(stats, 'phase_label')} \"No phase\" means trials without a drug-development "
          "phase, such as behavioural studies.")
with c2:
    stats = db.rate_by("sponsor_class")  # SQL: GROUP BY sponsor_class
    chart(viz.rate_bar(stats, "sponsor_class", "Lead sponsor", average),
          f"By who runs the trial. {viz.extremes(stats, 'sponsor_class')}")
c3, c4 = st.columns(2)
with c3:
    stats = db.rate_by("masking", viz.MASKING_ORDER)  # SQL: GROUP BY masking
    chart(viz.rate_bar(stats, "masking", "Blinding", average),
          f"By how many people were kept unaware of the treatment (blinding). {viz.extremes(stats, 'masking')}")
with c4:
    stats = db.rate_by("allocation", viz.ALLOCATION_ORDER)  # SQL: GROUP BY allocation
    chart(viz.rate_bar(stats, "allocation", "Allocation", average),
          f"By how participants were assigned to groups. {viz.extremes(stats, 'allocation')}")

# ---- Two traits at once ----
# A heatmap of two traits shows combinations that single bars hide. The user picks the pair; phase x sponsor is the
# default because it is the question a sponsor or investor asks first ("industry phase 2 trials: how risky?").
st.subheader("Which combinations are riskiest?")
PAIRS = {
    "Phase × lead sponsor": ("phase_label", "sponsor_class", viz.PHASE_ORDER, None, "Lead sponsor", "Phase"),
    "Phase × blinding": ("phase_label", "masking", viz.PHASE_ORDER, viz.MASKING_ORDER, "Blinding", "Phase"),
    "Lead sponsor × allocation": ("sponsor_class", "allocation", None, viz.ALLOCATION_ORDER, "Allocation",
                                  "Lead sponsor"),
}
pair = st.segmented_control("Compare", list(PAIRS), default=list(PAIRS)[0], key="explore_pair")
rows, cols, row_order, col_order, x_title, y_title = PAIRS[pair or list(PAIRS)[0]]  # None if the pill is unselected
heatmap = viz.rate_heatmap(filtered, rows, cols, row_order, col_order, x_title, y_title)
if heatmap is None:
    st.info(f"No combination has at least {viz.MIN_GROUP_N} trials with these filters. Widen them to see the heatmap.")
else:
    chart(heatmap, f"Each cell is the share stopped early for one combination; darker means stopped more often. "
                   f"Empty cells rest on fewer than {viz.MIN_GROUP_N} trials. Hover a cell for the number of trials. "
                   "The filters above apply here too.")

# ---- Time ----
st.subheader("Has it changed over the years?")
chart(viz.rate_over_time(filtered, average),
      "By the year the trial started. Only trials that have already ended are included, so the most recent "
      "years may look different from how they will end up.")

# ---- Conditions and eligibility ----
st.subheader("Conditions and eligibility")
c5, c6 = st.columns([3, 2])
with c5:
    st.markdown("**The 15 most common conditions**")
    chart(viz.conditions_bar(filtered, stats=db.top_conditions()),  # SQL: JOIN + GROUP BY + LIMIT
          "Each bar is the share stopped early; the label adds how many trials it rests on.")
with c6:
    st.markdown("**Length of the eligibility criteria**")
    chart(viz.rate_bar(viz.eligibility_stats(filtered), "eligibility_group", "Characters of eligibility text", average),
          "Longer criteria go with more stopped trials. A pattern, not proof of cause: long criteria often "
          "belong to larger, more complex studies. About 500 characters is a short paragraph.")

# ---- Trial table ----
st.subheader("Browse the trials")
search = st.text_input("Search title, sponsor or NCT id", placeholder="e.g. NCT01234567 or Pfizer")
# Free-text search on top of the filters, as plain substrings (no regex), so any input is safe.
table = filtered
if search.strip():
    s = search.strip()
    table = table[table["brief_title"].str.contains(s, case=False, regex=False, na=False)
                  | table["lead_sponsor_name"].str.contains(s, case=False, regex=False, na=False)
                  | table["nct_id"].str.contains(s, case=False, regex=False, na=False)]
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
        "overall_status": "Result",
    },
)

# ---- Under the hood ----
# Every SQL query this page just ran, with its parameters, so anyone can see (and copy) how the numbers are made.
with st.expander("Under the hood: the SQL queries behind this page"):
    st.write(
        "The numbers on this page come from a small SQLite database (`data/processed/trials.db`) with two tables: "
        "`trials` (one row per trial, key `nct_id`) and `trial_conditions` (one row per trial and condition, so "
        "one cell never holds a list). Your filters go into each query as parameters (the `?`), never into the SQL "
        "text itself. These are the queries for your current filters:"
    )
    for purpose, sql, params in overview.log + db.log + [trials_query]:
        st.markdown(f"**{purpose}**" + (f" (parameters: {params})" if params else ""))
        st.code(sql, language="sql", wrap_lines=True)

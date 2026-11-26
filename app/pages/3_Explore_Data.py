"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Explore: interactive charts of the past trials (start 2008-2019), completed against stopped early.

This page only describes the data, no model is loaded here. The numbers come from the SQLite database
data/processed/trials.db through SQL queries (src/database.py): WHERE for the filters, COUNT / AVG for the key
numbers, GROUP BY for the bars and a JOIN for the conditions. The charts are built in src/viz.py. A folded box at
the end of the page shows every query that was run.

Outline of the page code at the bottom of this file:
    1. Open the database and write the header
    2. Filters
    3. Key numbers
    4. One trait at a time
    5. Two traits at once
    6. Over the years
    7. Conditions and eligibility
    7b. More ways to look at it (four tabs)
    8. Trial table
    9. Under the hood
"""

import sys  # changes the import path
from pathlib import Path  # builds file paths that work on every computer

import streamlit as st  # the web app framework

# Streamlit runs a page as its own script, so we add the project folder (for src/) and app/ (for ui) by hand.
ROOT = Path(__file__).resolve().parents[2]
for folder in [ROOT, ROOT / "app"]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from src import database, viz  # SQLite database of the trials and the chart builders
from ui import chart, page_header, require_router  # shared look: charts, header, router

# Send the page through Home.py if it was opened directly.
require_router("pages/3_Explore_Data.py")

# The trial table shows at most this many rows, because the browser gets slow with more.
MAX_ROWS = 5000
# The filters that 'Clear filters' sets back, with their empty value. The year slider is handled separately.
FILTER_DEFAULTS = {"explore_phases": [], "explore_sponsors": [], "explore_condition": ""}

# The three pairs of traits the heatmap can compare. rows and cols are column names of the trials table, the
# orders say how the groups are sorted (None = as they come) and the titles go next to the axes.
PAIRS = {
    "Phase × lead sponsor": {
        "rows": "phase_label", "cols": "sponsor_class", "row_order": viz.PHASE_ORDER, "col_order": None,
        "x_title": "Lead sponsor", "y_title": "Phase",
    },
    "Phase × blinding": {
        "rows": "phase_label", "cols": "masking", "row_order": viz.PHASE_ORDER, "col_order": viz.MASKING_ORDER,
        "x_title": "Blinding", "y_title": "Phase",
    },
    "Lead sponsor × allocation": {
        "rows": "sponsor_class", "cols": "allocation", "row_order": None, "col_order": viz.ALLOCATION_ORDER,
        "x_title": "Allocation", "y_title": "Lead sponsor",
    },
}
PAIR_NAMES = list(PAIRS)


@st.cache_resource(show_spinner="Preparing the trial database (first visit only, a few seconds)...")
def get_database():
    """Get the path of trials.db. The file is built from trials.parquet on the first visit, or when that is newer.

    Input: none.

    Output: the path of the database file.
    """
    return database.ensure_database()


@st.cache_data(show_spinner="Loading the matching trials...", max_entries=20)
def matching_trials(phases, sponsors, years, condition_kw):
    """Load the trials that pass the filters (one SQL SELECT ... WHERE), and that query for the SQL box.

    Streamlit keeps the result per filter combination, so the database is not read again on every rerun.

    Input: phases is a tuple of phase names; sponsors is a tuple of sponsor classes; years is a tuple (first year,
    last year) or None for all years; condition_kw is a keyword searched in the conditions (text).

    Output: a tuple (trials, query). trials is a DataFrame with one row per matching trial; query is the logged SQL
    query (purpose, sql, parameters).
    """
    # Run the filtered query and remember it for the SQL box.
    db = database.TrialDatabase(get_database(), phases=phases, sponsor_classes=sponsors, year_range=years,
                                condition_kw=condition_kw)
    trials = db.trials()
    query = db.log[-1]  # the query that db.trials() just ran
    return trials, query


def reset_filters():
    """Set every filter back to "all trials". It runs when 'Clear filters' is clicked.

    Input: none.

    Output: nothing. It resets the values in the session state.
    """
    # The text and list filters get their empty value back.
    for key, default in FILTER_DEFAULTS.items():
        st.session_state[key] = default
    # Without its key in the session state the slider starts again at the full year range.
    st.session_state.pop("explore_years", None)


def chart_with_caption(fig, caption):
    """Show a Plotly figure in the app look with a short explanation under it.

    Input: fig is the Plotly figure; caption is the explanation below it (text).

    Output: nothing. It draws the chart and the caption on the page.
    """
    chart(fig)
    st.caption(caption)


# ---- 1. Open the database and write the header ----
# Open the database. Without the trial data the page says what is missing and stops (no error page).
try:
    db_path = get_database()
except (FileNotFoundError, OSError):
    page_header("Explore the data")
    st.info("The trial data (data/processed/trials.parquet) is not available here, so there is nothing to chart yet.")
    st.stop()

# The numbers of ALL trials and the choices for the filters, read with SQL (no filters yet).
overview = database.TrialDatabase(db_path)
choices = overview.options()
everything = overview.summary(filtered=False)
total = everything["n"]
overall = everything["rate"]
y_min, y_max = choices["years"]
page_header("Explore the data",
            f"{total:,} past trials that started {y_min}-{y_max} and have ended. "
            f"{overall:.1%} of them were stopped early. Each bar shows how that share changes with one trait.")

# ---- 2. Filters ----
# The filter values live in st.session_state, so they survive reruns and 'Clear filters' can set them back.
for key, default in FILTER_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container(key="panel_filters"):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
    # Only offer phases that occur in the data, in clinical order.
    phase_options = []
    for phase in viz.PHASE_ORDER:
        if phase in choices["phases"]:
            phase_options.append(phase)
    phases = f1.multiselect("Phase", phase_options, key="explore_phases", placeholder="All phases")
    sponsors = f2.multiselect("Sponsor", choices["sponsors"], key="explore_sponsors", placeholder="All sponsors")
    years = f3.slider("Start year", y_min, y_max, (y_min, y_max), key="explore_years")
    condition_kw = f4.text_input("Condition keyword", key="explore_condition", placeholder="e.g. breast cancer")

    # The button is only active when at least one filter is set.
    years_narrowed = years != (y_min, y_max)
    narrowed = bool(phases) or bool(sponsors) or condition_kw.strip() != "" or years_narrowed
    st.button("Clear filters", on_click=reset_filters, disabled=not narrowed)

# Every query below gets the same filters. The full year range is no filter, so it is left out of the SQL.
year_filter = None
if years_narrowed:
    year_filter = tuple(years)
db = database.TrialDatabase(db_path, phases=phases, sponsor_classes=sponsors, year_range=year_filter,
                            condition_kw=condition_kw)
# The numbers of the filtered trials, then the rows themselves for the pandas charts.
shown = db.summary()
# The cache needs arguments it can compare, so the lists are passed as tuples.
filtered, trials_query = matching_trials(tuple(phases), tuple(sponsors), year_filter, condition_kw)

# ---- 3. Key numbers ----
# Three tiles: trials shown, share stopped early (with the change against all trials) and the overall share.
k1, k2, k3 = st.columns(3)
k1.metric("Trials shown", f"{shown['n']:,}", help=f"Out of {total:,} trials in total.")
if shown["n"]:
    rate = shown["rate"]
    # Compare with all trials only when a filter really leaves trials out.
    delta = None
    if shown["n"] < total:
        delta = f"{(rate - overall) * 100:+.1f} points vs all trials"
    k2.metric("Stopped early", f"{rate:.1%}", delta=delta, delta_color="inverse")
else:
    k2.metric("Stopped early", "-")
k3.metric("All trials", f"{overall:.1%}", help=f"Share stopped early across all {total:,} trials, for comparison.")

# Nothing left: say so and stop, the charts below would be empty.
if not shown["n"]:
    st.warning("No trials match these filters. Widen the year range or clear a filter.")
    st.stop()
# The dashed average line in the bar charts is the average of the trials shown, not of all trials.
average = shown["rate"]

# ---- 4. One trait at a time ----
# Four bar charts in two rows. Each one is a GROUP BY query on one column.
st.subheader("Which kinds of trials get stopped more often?")
st.caption("The dashed line is the average of the trials shown. Faded bars rest on fewer than "
           f"{viz.MIN_GROUP_N} trials and are not reliable.")
c1, c2 = st.columns(2)
# Row 1: phase and sponsor.
with c1:
    stats = db.rate_by("phase_label", viz.PHASE_ORDER)
    chart_with_caption(
        viz.rate_bar(stats, "phase_label", "Phase", average),
        f"By phase. {viz.extremes(stats, 'phase_label')} \"No phase\" means trials without a drug-development "
        "phase, such as behavioural studies.")
with c2:
    stats = db.rate_by("sponsor_class")
    chart_with_caption(
        viz.rate_bar(stats, "sponsor_class", "Lead sponsor", average),
        f"By who runs the trial. {viz.extremes(stats, 'sponsor_class')}")
# Row 2: blinding and allocation.
c3, c4 = st.columns(2)
with c3:
    stats = db.rate_by("masking", viz.MASKING_ORDER)
    chart_with_caption(
        viz.rate_bar(stats, "masking", "Blinding", average),
        f"By how many people were kept unaware of the treatment (blinding). {viz.extremes(stats, 'masking')}")
with c4:
    stats = db.rate_by("allocation", viz.ALLOCATION_ORDER)
    chart_with_caption(
        viz.rate_bar(stats, "allocation", "Allocation", average),
        f"By how participants were assigned to groups. {viz.extremes(stats, 'allocation')}")

# ---- 5. Two traits at once ----
# A heatmap of two traits shows combinations that single bars hide. Phase x sponsor is the default, because it is
# the first question of a sponsor or investor ("industry phase 2 trials: how risky?").
st.subheader("Which combinations are riskiest?")
pair_name = st.segmented_control("Compare", PAIR_NAMES, default=PAIR_NAMES[0], key="explore_pair")
# A click on the selected pill unselects it. Then the control returns None and we show the first pair.
if pair_name is None:
    pair_name = PAIR_NAMES[0]
# Build the heatmap of the chosen pair from the filtered rows.
pair = PAIRS[pair_name]
heatmap = viz.rate_heatmap(filtered, pair["rows"], pair["cols"], pair["row_order"], pair["col_order"],
                           pair["x_title"], pair["y_title"])
if heatmap is None:
    st.info(f"No combination has at least {viz.MIN_GROUP_N} trials with these filters. Widen them to see the heatmap.")
else:
    chart_with_caption(
        heatmap,
        f"Each cell is the share stopped early for one combination; darker means stopped more often. "
        f"Empty cells rest on fewer than {viz.MIN_GROUP_N} trials. Hover a cell for the number of trials. "
        "The filters above apply here too.")

# ---- 6. Over the years ----
st.subheader("Has it changed over the years?")
chart_with_caption(
    viz.rate_over_time(filtered, average),
    "By the year the trial started. Only trials that have already ended are included, so the most recent "
    "years may look different from how they will end up.")

# ---- 7. Conditions and eligibility ----
st.subheader("Conditions and eligibility")
# Conditions on the left, eligibility length on the right.
c5, c6 = st.columns([3, 2])
with c5:
    st.markdown("**The 15 most common conditions**")
    # The counting is done by SQL (JOIN + GROUP BY + LIMIT), the chart only draws the result.
    condition_stats = db.top_conditions()
    chart_with_caption(
        viz.conditions_bar(filtered, stats=condition_stats),
        "Each bar is the share stopped early; the label adds how many trials it rests on.")
with c6:
    st.markdown("**Length of the eligibility criteria**")
    chart_with_caption(
        viz.rate_bar(viz.eligibility_stats(filtered), "eligibility_group", "Characters of eligibility text", average),
        "Longer criteria go with more stopped trials. A pattern, not proof of cause: long criteria often "
        "belong to larger, more complex studies. About 500 characters is a short paragraph.")

# ---- 7b. More ways to look at it ----
# Four other chart types on the same filtered trials: a treemap, a bubble chart, stacked areas and a Sankey flow.
st.subheader("More ways to look at it")
tab_map, tab_bubbles, tab_years, tab_flow = st.tabs(
    ["Conditions map", "Conditions: size vs risk", "Trials per year", "How trials flow"])
# Each tab shows its chart, or a sentence when the filters leave too little data.
with tab_map:
    # The counting is the same SQL query (JOIN + GROUP BY + LIMIT) as for the bars above, only with more conditions.
    treemap = viz.condition_treemap(filtered, stats=db.top_conditions(k=30))
    if treemap is None:
        st.info(f"No condition has at least {viz.MIN_GROUP_N} trials with these filters. Widen them to see the map.")
    else:
        chart_with_caption(
            treemap,
            "Every rectangle is a condition. The bigger it is, the more trials; the darker it is, the larger the "
            "share stopped early. Hover a rectangle for the numbers.")
with tab_bubbles:
    bubbles = viz.condition_bubbles(filtered, stats=db.top_conditions(k=40))
    if bubbles is None:
        st.info(f"No condition has at least {viz.MIN_GROUP_N} trials with these filters. Widen them to see the bubbles.")
    else:
        chart_with_caption(
            bubbles,
            "Each bubble is a condition. Further right means more trials, higher up means more of them stopped "
            "early, a bigger bubble means more stopped trials. The names mark the conditions with the most stopped trials.")
with tab_years:
    area = viz.trials_per_year_area(filtered)
    if area is None:
        st.info("Not enough trials over several years with these filters. Widen the year range to see this chart.")
    else:
        chart_with_caption(
            area,
            "How many trials started in each year, split by phase (bottom to top in clinical order). The height of "
            "a layer is the number of trials of that phase. It shows how the mix of trials changed, not their risk.")
with tab_flow:
    flow = viz.flow_sankey(filtered)
    if flow is None:
        st.info(f"No phase and blinding group has at least {viz.MIN_GROUP_N} trials with these filters.")
    else:
        chart_with_caption(
            flow,
            "Read from left to right: trials of each phase, how they were blinded, and how they ended. The wider "
            "a band, the more trials. Groups with few trials are left out.")

# ---- 8. Trial table ----
st.subheader("Browse the trials")
# A search box on top of the filters.
search = st.text_input("Search title, sponsor or NCT id", placeholder="e.g. NCT01234567 or Pfizer")
# The search works on top of the filters. regex=False means the text is searched as it is, so any input is safe.
table = filtered
if search.strip():
    word = search.strip()
    in_title = table["brief_title"].str.contains(word, case=False, regex=False, na=False)
    in_sponsor = table["lead_sponsor_name"].str.contains(word, case=False, regex=False, na=False)
    in_id = table["nct_id"].str.contains(word, case=False, regex=False, na=False)
    table = table[in_title | in_sponsor | in_id]

# Say how many trials match, and that the table is cut if there are too many.
if len(table) > MAX_ROWS:
    st.caption(f"{len(table):,} trials match; showing the first {MAX_ROWS:,} (narrow the filters to see more).")
else:
    st.caption(f"{len(table):,} trials match.")
# The table itself, with the NCT id as a link to clinicaltrials.gov.
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

# ---- 9. Under the hood ----
# Every SQL query this page just ran, with its parameters, so anyone can see (and copy) how the numbers are made.
with st.expander("Under the hood: the SQL queries behind this page"):
    # An explanation of the two tables, then every query.
    st.write(
        "The numbers on this page come from a small SQLite database (`data/processed/trials.db`) with two tables: "
        "`trials` (one row per trial, key `nct_id`) and `trial_conditions` (one row per trial and condition, so "
        "one cell never holds a list). Your filters go into each query as parameters (the `?`), never into the SQL "
        "text itself. These are the queries for your current filters:"
    )
    # The queries of the overview, of the filtered numbers and of the row load, in the order they ran.
    all_queries = overview.log + db.log + [trials_query]
    for purpose, sql, params in all_queries:
        title = f"**{purpose}**"
        if params:
            title += f" (parameters: {params})"
        st.markdown(title)
        st.code(sql, language="sql", wrap_lines=True)

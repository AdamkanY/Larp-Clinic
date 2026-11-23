# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""The numbers and the Plotly charts of the page "Explore the data".

Every function here takes a table of trials (a pandas DataFrame) and returns a table or a figure. There is no
Streamlit code and no model in this file, so the functions can also be used by other pages and by the tests.

"Share stopped early" is always the mean of the column "label" (TERMINATED = 1, COMPLETED = 0).

There are no colours and no fonts in this file. app/ui.py sets one Plotly template for the whole app before a
page draws a chart, and the figures built here take their look from it.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# The other files of src/ are imported by their plain name, so this works from the app and from the terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "trials.parquet"
STUDY_URL = "https://clinicaltrials.gov/study/"

NOT_STATED = labels.NOT_STATED

# The order of the bars. Phases go in their clinical order and not by the alphabet, so the chart can be read
# from left to right. "Not stated" always comes last.
PHASE_ORDER = labels.PHASE_ORDER + [NOT_STATED]
MASKING_ORDER = list(labels.MASKING.values()) + [NOT_STATED]
ALLOCATION_ORDER = [
    labels.ALLOCATION["RANDOMIZED"], labels.ALLOCATION["NON_RANDOMIZED"], labels.ALLOCATION["NA"], NOT_STATED,
]

# Groups for the length of the eligibility text in characters. About 500 characters is a short paragraph,
# several thousand are a long checklist.
ELIGIBILITY_EDGES = [-1, 500, 1000, 2000, 4000, float("inf")]
ELIGIBILITY_LABELS = ["Up to 500", "501-1,000", "1,001-2,000", "2,001-4,000", "Over 4,000"]

# A group with fewer trials than this has a share that jumps around by chance. Such bars are drawn pale and
# such heatmap cells stay empty.
MIN_GROUP_N = 30

# The text that appears when the mouse is over a bar or a point. customdata holds the number of trials (n) and
# the number of stopped trials of the group.
HOVER_TEXT = "<b>%{x}</b><br>Stopped early: %{y:.1%}<br>%{customdata[1]:,} of %{customdata[0]:,} trials<extra></extra>"


def plain_names(codes, kind):
    """Turn a column of API codes into plain names with labels.py, for example "DOUBLE" -> "Double blind"."""
    names = []
    for code in codes:
        names.append(labels.label(kind, code))
    return names


def load_trials(path=DATA_PATH):
    """Read the trials file and prepare it for the dashboard.

    Codes become plain English names. A missing value becomes "Not stated", so it shows up as its own bar and
    does not silently disappear. Three columns are added: start_year, outcome and eligibility_group.
    """
    df = pd.read_parquet(path)

    # Codes -> plain names.
    df["phase_label"] = plain_names(df["phase_label"], "phase")
    df["sponsor_class"] = plain_names(df["sponsor_class"], "sponsor_class")
    df["masking"] = plain_names(df["masking"], "masking")
    df["allocation"] = plain_names(df["allocation"], "allocation")
    df["overall_status"] = plain_names(df["overall_status"], "status")
    df["conditions"] = df["conditions"].fillna("")

    # The extra columns for the charts.
    df["start_year"] = df["start_date"].dt.year.astype(int)
    df["outcome"] = df["label"].map({1: "Stopped early", 0: "Completed"})
    df["eligibility_group"] = pd.cut(df["eligibility_text_len"].astype(float), ELIGIBILITY_EDGES,
                                     labels=ELIGIBILITY_LABELS)
    return df


def filter_trials(df, phases=None, sponsor_classes=None, year_range=None, condition_kw=""):
    """Keep the trials that match the filters. An empty filter means "all" for that filter."""
    # We start with "keep every row" and every filter that is set removes rows.
    mask = pd.Series(True, index=df.index)
    if phases:
        mask = mask & df["phase_label"].isin(phases)
    if sponsor_classes:
        mask = mask & df["sponsor_class"].isin(sponsor_classes)
    if year_range:
        mask = mask & df["start_year"].between(year_range[0], year_range[1])
    keyword = (condition_kw or "").strip()
    if keyword:
        # regex=False: the keyword is searched as plain text, so a keyword like "C++" or "(AML)" cannot break it.
        mask = mask & df["conditions"].str.contains(keyword, case=False, regex=False)
    return df[mask]


def rate_by(df, column, order=None):
    """Count the trials, the stopped trials and the share stopped early for every value of one column.

    With order (a list of values) the rows come in that order and values that are not in the list go last.
    Without order the biggest group comes first.
    """
    out = df.groupby(column, observed=True)["label"].agg(n="size", terminated="sum", rate="mean").reset_index()
    if order:
        # Give every value its place in the list and sort by this place.
        place_in_order = {}
        for i in range(len(order)):
            place_in_order[order[i]] = i
        out["place"] = out[column].astype(object).map(place_in_order).fillna(len(order))
        out = out.sort_values("place").drop(columns="place")
    else:
        out = out.sort_values("n", ascending=False)
    return out


def extremes(stats, column):
    """One sentence that names the group with the highest and the group with the lowest share stopped early.

    stats is the table from rate_by(). Returns "" if fewer than two groups are big enough to compare.
    """
    # "Not stated" is a gap in the data and not a kind of trial, so it can never be the highest or the lowest.
    big = stats[(stats["n"] >= MIN_GROUP_N) & (stats[column] != NOT_STATED)]
    if len(big) < 2:
        return ""
    high = big.loc[big["rate"].idxmax()]
    low = big.loc[big["rate"].idxmin()]
    return f"Highest: {high[column]} at {high['rate']:.0%}. Lowest: {low[column]} at {low['rate']:.0%}."


def top_conditions(df, k=15):
    """The k most common conditions with their number of trials and their share stopped early."""
    # A trial can have several conditions in one cell ("Asthma; COPD"). We split the cell into a list and
    # explode() then makes one row per condition.
    exploded = df[["conditions", "label"]].copy()
    exploded["condition"] = df["conditions"].str.split("; ")
    exploded = exploded.explode("condition")
    exploded["condition"] = exploded["condition"].str.strip()
    exploded = exploded[exploded["condition"] != ""]

    out = exploded.groupby("condition")["label"].agg(n="size", terminated="sum", rate="mean").reset_index()
    return out.nlargest(k, "n")


def finish_rate_chart(fig):
    """The layout that the bar chart and the line chart share: a percent axis that starts at 0 and no legend."""
    # The bottom margin (b) leaves room for the axis title under slanted labels.
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=40), showlegend=False, bargap=0.35)
    fig.update_yaxes(title="Share stopped early", tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return fig


def add_average_line(fig, average):
    """Draw a dashed line at the average of all trials on screen, so every bar can be read as above or below it.

    The line has no text, because text would cover the bars. The page explains what the line means.
    """
    if average is not None:
        fig.add_hline(y=average, line_dash="dash")


def rate_bar(stats, column, x_title, average=None):
    """Bar chart of the share stopped early, one bar per group. stats is the table from rate_by()."""
    # A very small group gets a pale bar without a number on it. Otherwise "1 of 2 trials" would look like a
    # finding of 50%.
    bar_texts = []
    bar_opacities = []
    for rate, n in zip(stats["rate"], stats["n"]):
        if n < MIN_GROUP_N:
            bar_texts.append("")
            bar_opacities.append(0.35)
        else:
            bar_texts.append(f"{rate:.0%}")
            bar_opacities.append(1.0)

    fig = px.bar(stats, x=column, y="rate", custom_data=["n", "terminated"], text=bar_texts)
    fig.update_traces(
        marker_opacity=bar_opacities,
        textposition="outside", cliponaxis=False, textfont_size=13,
        hovertemplate=HOVER_TEXT,
    )
    fig.update_xaxes(title=x_title, type="category")

    # The y axis ends a quarter above the highest bar (or above the average line if that is higher), so the
    # numbers on top of the bars have room.
    top = stats["rate"].max()
    if average is not None and average > top:
        top = average
    fig.update_yaxes(range=[0, top * 1.25])

    add_average_line(fig, average)
    return finish_rate_chart(fig)


def rate_over_time(df, average=None):
    """Line chart of the share stopped early for every start year."""
    stats = rate_by(df, "start_year").sort_values("start_year")
    fig = px.line(stats, x="start_year", y="rate", markers=True, custom_data=["n", "terminated"])
    fig.update_traces(line=dict(width=2), marker=dict(size=8), hovertemplate=HOVER_TEXT)
    fig.update_xaxes(title="Start year", dtick=1)
    fig.update_layout(hovermode="x unified")
    add_average_line(fig, average)
    return finish_rate_chart(fig)


def conditions_bar(df, k=15, name_width=34, stats=None):
    """Horizontal bar chart of the k most common conditions, the one with the highest share on top.

    stats (columns condition, n, terminated, rate) can be passed in ready-made, for example from the SQL query
    in database.py. Without stats the conditions are counted from df with top_conditions().
    """
    if stats is None:
        stats = top_conditions(df, k)
    # Plotly draws the first row at the bottom, so the highest share must come last.
    stats = stats.sort_values("rate").copy()

    # Long names are cut off so they do not run into the chart. The full name is shown on hover.
    short_names = []
    for name in stats["condition"]:
        if len(name) <= name_width:
            short_names.append(name)
        else:
            short_names.append(name[: name_width - 1] + "…")
    stats["short"] = short_names

    # The text next to each bar: the share and how many trials it is based on.
    bar_texts = []
    for rate, n in zip(stats["rate"], stats["n"]):
        bar_texts.append(f"{rate:.0%} of {n:,}")

    fig = px.bar(stats, x="rate", y="short", orientation="h", custom_data=["condition", "n", "terminated"],
                 text=bar_texts)
    fig.update_traces(
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Stopped early: %{x:.1%}<br>"
                      "%{customdata[2]:,} of %{customdata[1]:,} trials<extra></extra>",
    )
    # The chart gets taller with every bar.
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      height=max(320, 30 * len(stats) + 70), bargap=0.3)
    # The x axis is about twice as long as the longest bar, so the text to the right of it has room.
    fig.update_xaxes(title="Share stopped early", tickformat=".0%", rangemode="tozero",
                     range=[0, stats["rate"].max() * 2.1], zeroline=False)
    fig.update_yaxes(title=None)
    return fig


def values_in_order(sizes, order):
    """The row (or column) values of the heatmap in the order they are shown.

    sizes holds the number of trials per value. First come the values from the list order, in that order. All
    other values follow, the one with the most trials first.
    """
    present = list(sizes.index)
    values = []
    if order:
        for value in order:
            if value in present:
                values.append(value)
    # kind="stable": values with the same number of trials keep the order they had.
    for value in sizes.sort_values(ascending=False, kind="stable").index:
        if value not in values:
            values.append(value)
    return values


def rate_matrix(df, rows, cols, row_order=None, col_order=None):
    """Share stopped early and number of trials for every combination of two columns.

    Returns two tables (rate, n). The values of the column rows are the index and the values of the column cols
    are the columns, sorted like in values_in_order(). A combination without trials is NaN in rate and 0 in n.
    """
    stats = df.groupby([rows, cols], observed=True)["label"].agg(n="size", rate="mean").reset_index()
    # pivot() turns the long list of combinations into a grid.
    rate = stats.pivot(index=rows, columns=cols, values="rate")
    n = stats.pivot(index=rows, columns=cols, values="n").fillna(0).astype(int)

    row_values = values_in_order(n.sum(axis=1), row_order)
    col_values = values_in_order(n.sum(axis=0), col_order)
    return rate.loc[row_values, col_values], n.loc[row_values, col_values]


def rate_heatmap(df, rows, cols, row_order=None, col_order=None, x_title="", y_title=""):
    """Heatmap of the share stopped early for every combination of two columns, for example phase x sponsor.

    Two columns at once show what single bar charts hide: a combination can be risky even if neither of its two
    values is risky alone. A cell with fewer than MIN_GROUP_N trials stays empty, because its share would be
    mostly chance. Returns None if no cell has enough trials.
    """
    rate, n = rate_matrix(df, rows, cols, row_order, col_order)
    shown = rate.where(n >= MIN_GROUP_N)

    # A row or a column without a single filled cell would only add an empty stripe, so we drop it.
    keep_rows = shown.notna().any(axis=1)
    keep_cols = shown.notna().any(axis=0)
    shown = shown.loc[keep_rows, keep_cols]
    n = n.loc[keep_rows, keep_cols]
    # Nothing left with these filters. The page then shows a message.
    if shown.empty:
        return None

    # The number written in each cell, for example "12%". Empty cells get no text.
    text = []
    for row in shown.to_numpy():
        row_text = []
        for value in row:
            if pd.notna(value):
                row_text.append(f"{value:.0%}")
            else:
                row_text.append("")
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=shown.to_numpy(), x=[str(c) for c in shown.columns], y=[str(r) for r in shown.index],
        text=text, texttemplate="%{text}", customdata=n.to_numpy(), hoverongaps=False, xgap=3, ygap=3,
        colorbar={"title": {"text": "Stopped<br>early"}, "tickformat": ".0%", "thickness": 14},
        hovertemplate="<b>%{y} · %{x}</b><br>Stopped early: %{z:.1%}<br>%{customdata:,} trials<extra></extra>",
    ))
    fig.update_xaxes(title=x_title, type="category", showgrid=False, side="bottom")
    # autorange="reversed" puts the first row (for example Early phase 1) on top, like in a table.
    fig.update_yaxes(title=y_title, type="category", showgrid=False, autorange="reversed")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=40), height=max(320, 52 * len(shown.index) + 90))
    return fig


def gain_curve(labels, risks, points=101):
    """If we review the riskiest x% of the trials first, which share of all stopped trials do we find?

    labels holds 1 for stopped early and 0 for completed, risks holds the model's score for the same trials.
    The result has one row for each of points evenly spaced shares from 0% to 100%, with the columns:
    reviewed (share of the trials we looked at, riskiest first), caught (share of all stopped trials that are
    among them) and precision (share of the reviewed trials that were stopped). If we picked trials at random,
    caught would be equal to reviewed. That is the diagonal line in the chart.
    """
    # Sort the trials from the riskiest to the safest. kind="mergesort" keeps trials with the same score in
    # their old order, so the result does not depend on chance.
    frame = pd.DataFrame({"y": list(labels), "p": list(risks)}).sort_values("p", ascending=False, kind="mergesort")
    total = len(frame)
    stopped = frame["y"].sum()
    # hits[i] = how many stopped trials are among the first i + 1 trials.
    hits = frame["y"].cumsum().to_numpy()

    rows = []
    for i in range(points):
        share = i / (points - 1)
        k = int(round(share * total))  # how many trials we review at this share
        if k > 0:
            found = hits[k - 1]
        else:
            found = 0
        if stopped:
            caught = found / stopped
        else:
            caught = 0.0
        if k:
            precision = found / k
        else:
            precision = float("nan")
        rows.append({"reviewed": share, "caught": caught, "precision": precision})
    return pd.DataFrame(rows)


def eligibility_stats(df):
    """Share stopped early for every length group of the eligibility text, the shortest group first.

    Trials without an eligibility text are in no group and are left out.
    """
    return rate_by(df, "eligibility_group", ELIGIBILITY_LABELS)


def trials_table(df):
    """The columns for the list of trials. "link" is the address of the trial on ClinicalTrials.gov."""
    out = df[["nct_id", "brief_title", "phase_label", "lead_sponsor_name", "overall_status"]].copy()
    out.insert(0, "link", STUDY_URL + out["nct_id"])
    return out.drop(columns="nct_id")

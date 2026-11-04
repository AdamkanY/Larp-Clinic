"""Aggregations and Plotly figures for the Explore dashboard.

Pure functions on the `trials.parquet` DataFrame: no Streamlit and no model code, so they can be
reused by other pages (or tested) without a running app. Termination rate = mean of `label`
(TERMINATED = 1, COMPLETED = 0); the pages call it "stopped early".
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "trials.parquet"
STUDY_URL = "https://clinicaltrials.gov/study/"

# One hue for every "share stopped early" bar, so the charts read as one family.
BLUE = "#2a78d6"
GRID = "rgba(128,128,128,0.2)"

NOT_STATED = labels.NOT_STATED

# Clinical order rather than alphabetical, so the phase chart reads left to right.
PHASE_ORDER = labels.PHASE_ORDER + [NOT_STATED]
MASKING_ORDER = list(labels.MASKING.values()) + [NOT_STATED]
ALLOCATION_ORDER = [labels.ALLOCATION["RANDOMIZED"], labels.ALLOCATION["NON_RANDOMIZED"],
                    labels.ALLOCATION["NA"], NOT_STATED]

# Eligibility text length in characters: about 500 is a short paragraph, several thousand a long checklist.
ELIGIBILITY_EDGES = [-1, 500, 1000, 2000, 4000, float("inf")]
ELIGIBILITY_LABELS = ["Up to 500", "501-1,000", "1,001-2,000", "2,001-4,000", "Over 4,000"]

# Groups smaller than this give noisy rates; they are faded rather than hidden.
MIN_GROUP_N = 30


def load_trials(path=DATA_PATH) -> pd.DataFrame:
    """Read the processed trials file and add tidy display columns used by the dashboard.

    Codes become plain-English names (see `labels.py`); missing values become "Not stated" so they show
    up as their own bar instead of vanishing.
    """
    df = pd.read_parquet(path)
    df["phase_label"] = df["phase_label"].map(lambda v: labels.label("phase", v))
    df["sponsor_class"] = df["sponsor_class"].map(lambda v: labels.label("sponsor_class", v))
    df["masking"] = df["masking"].map(lambda v: labels.label("masking", v))
    df["allocation"] = df["allocation"].map(lambda v: labels.label("allocation", v))
    df["overall_status"] = df["overall_status"].map(lambda v: labels.label("status", v))
    df["conditions"] = df["conditions"].fillna("")
    df["start_year"] = df["start_date"].dt.year.astype(int)
    df["outcome"] = df["label"].map({1: "Stopped early", 0: "Completed"})
    df["eligibility_group"] = pd.cut(df["eligibility_text_len"].astype(float), ELIGIBILITY_EDGES,
                                     labels=ELIGIBILITY_LABELS)
    return df


def filter_trials(df, phases=None, sponsor_classes=None, year_range=None, condition_kw="") -> pd.DataFrame:
    """Apply the filters; an empty selection means "no filter" for that dimension."""
    mask = pd.Series(True, index=df.index)
    if phases:
        mask &= df["phase_label"].isin(phases)
    if sponsor_classes:
        mask &= df["sponsor_class"].isin(sponsor_classes)
    if year_range:
        mask &= df["start_year"].between(year_range[0], year_range[1])
    kw = (condition_kw or "").strip()
    if kw:
        # Plain substring match (regex=False) so keywords like "C++" or "(AML)" cannot break it.
        mask &= df["conditions"].str.contains(kw, case=False, regex=False)
    return df[mask]


def rate_by(df, column, order=None) -> pd.DataFrame:
    """Termination rate and trial count per category of `column`."""
    out = (df.groupby(column, observed=True)["label"]
           .agg(n="size", terminated="sum", rate="mean").reset_index())
    if order:
        rank = {v: i for i, v in enumerate(order)}
        out = out.sort_values(column, key=lambda s: s.astype(object).map(rank).fillna(len(order)))
    else:
        out = out.sort_values("n", ascending=False)
    return out


def extremes(stats, column) -> str:
    """One plain sentence naming the highest and lowest rate among groups with enough trials, or "" if fewer than two."""
    # "Not stated" is a gap in the data, not a kind of trial, so it never counts as highest or lowest.
    big = stats[(stats["n"] >= MIN_GROUP_N) & (stats[column] != NOT_STATED)]
    if len(big) < 2:
        return ""
    high, low = big.loc[big["rate"].idxmax()], big.loc[big["rate"].idxmin()]
    return f"Highest: {high[column]} at {high['rate']:.0%}. Lowest: {low[column]} at {low['rate']:.0%}."


def top_conditions(df, k=15) -> pd.DataFrame:
    """The k most frequent individual conditions (split on "; ") with their termination rate."""
    exploded = (df[["conditions", "label"]]
                .assign(condition=df["conditions"].str.split("; "))
                .explode("condition"))
    exploded["condition"] = exploded["condition"].str.strip()
    exploded = exploded[exploded["condition"] != ""]
    out = (exploded.groupby("condition")["label"]
           .agg(n="size", terminated="sum", rate="mean").reset_index())
    return out.nlargest(k, "n")


def _style(fig, y_title="Share stopped early") -> go.Figure:
    """Shared layout: recessive grid, percent axis, compact margins, no legend for one series."""
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      hoverlabel=dict(align="left"), bargap=0.35)
    fig.update_yaxes(title=y_title, tickformat=".0%", rangemode="tozero", gridcolor=GRID)
    return fig


def _average_line(fig, average) -> None:
    """Dashed reference line at the average rate of everything on screen, so each bar reads as above or below it.

    No text on the line: it would sit on top of the bars, and the page says what the line means.
    """
    if average is not None:
        fig.add_hline(y=average, line_dash="dash", line_color="#475569")


def rate_bar(stats, column, x_title, average=None) -> go.Figure:
    """Vertical bar chart of the share stopped early per category, with the value on each bar and n in the hover."""
    stats = stats.assign(small=stats["n"] < MIN_GROUP_N)
    # Very small groups are faded and carry no value label, so a 1-of-2 "50%" bar does not look like a finding.
    labels_on_bars = [("" if small else f"{rate:.0%}") for rate, small in zip(stats["rate"], stats["small"])]
    fig = px.bar(stats, x=column, y="rate", custom_data=["n", "terminated"], text=labels_on_bars)
    fig.update_traces(
        marker_color=BLUE,
        marker_opacity=[0.35 if s else 1.0 for s in stats["small"]],
        textposition="outside", cliponaxis=False, textfont_size=13,
        hovertemplate="<b>%{x}</b><br>Stopped early: %{y:.1%}<br>"
                      "%{customdata[1]:,} of %{customdata[0]:,} trials<extra></extra>",
    )
    fig.update_xaxes(title=x_title, type="category")
    fig.update_yaxes(range=[0, max(stats["rate"].max(), average or 0) * 1.25])
    _average_line(fig, average)
    return _style(fig)


def rate_over_time(df, average=None) -> go.Figure:
    """Line chart of the share stopped early by start year, with n per year in the hover."""
    stats = rate_by(df, "start_year").sort_values("start_year")
    fig = px.line(stats, x="start_year", y="rate", markers=True, custom_data=["n", "terminated"])
    fig.update_traces(
        line=dict(color=BLUE, width=2), marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Stopped early: %{y:.1%}<br>"
                      "%{customdata[1]:,} of %{customdata[0]:,} trials<extra></extra>",
    )
    fig.update_xaxes(title="Start year", dtick=1)
    fig.update_layout(hovermode="x unified")
    _average_line(fig, average)
    return _style(fig)


def conditions_bar(df, k=15, name_width=34) -> go.Figure:
    """Horizontal bars: the share stopped early for the k most common conditions, riskiest on top.

    The label on each bar gives the rate and how many trials it rests on; long condition names are cut
    (the hover shows them in full) so they never overlap the chart.
    """
    stats = top_conditions(df, k).sort_values("rate")  # ascending, so the highest rate sits on top
    short = stats["condition"].map(lambda c: c if len(c) <= name_width else c[: name_width - 1] + "…")
    fig = px.bar(stats.assign(short=short), x="rate", y="short", orientation="h",
                 custom_data=["condition", "n", "terminated"],
                 text=[f"{r:.0%} of {n:,}" for r, n in zip(stats["rate"], stats["n"])])
    fig.update_traces(
        marker_color=BLUE, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Stopped early: %{x:.1%}<br>"
                      "%{customdata[2]:,} of %{customdata[1]:,} trials<extra></extra>",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      height=max(320, 30 * len(stats) + 70), bargap=0.3)
    fig.update_xaxes(title="Share stopped early", tickformat=".0%", rangemode="tozero", gridcolor=GRID,
                     range=[0, stats["rate"].max() * 2.1])  # room for the label right of the longest bar
    fig.update_yaxes(title=None, automargin=True)
    return fig


def eligibility_stats(df) -> pd.DataFrame:
    """Share stopped early per eligibility-text length group, shortest first (trials without a length are skipped)."""
    return rate_by(df, "eligibility_group", ELIGIBILITY_LABELS)


def trials_table(df) -> pd.DataFrame:
    """Columns for the trial list, with a study URL the page renders as a link on the NCT id."""
    out = df[["nct_id", "brief_title", "phase_label", "lead_sponsor_name", "overall_status"]].copy()
    out.insert(0, "link", STUDY_URL + out["nct_id"])
    return out.drop(columns="nct_id")

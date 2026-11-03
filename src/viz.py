"""Aggregations and Plotly figures for the Explore dashboard.

Pure functions on the `trials.parquet` DataFrame: no Streamlit and no model code, so they can be
reused by other pages (or tested) without a running app. Termination rate = mean of `label`
(TERMINATED = 1, COMPLETED = 0).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "trials.parquet"
STUDY_URL = "https://clinicaltrials.gov/study/"

# Default categorical palette slots (validated for colour-vision deficiency); bars of a single
# measure share one hue, the two-outcome box plot uses slots 1 and 2.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
OUTCOME_COLORS = {"Completed": BLUE, "Terminated": ORANGE}

UNKNOWN = "Unknown"

# Clinical order rather than alphabetical, so the phase chart reads left to right.
PHASE_ORDER = [
    "Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3",
    "Phase 3", "Phase 4", "Not Applicable", UNKNOWN,
]
MASKING_ORDER = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", UNKNOWN]
ALLOCATION_LABELS = {"RANDOMIZED": "Randomized", "NON_RANDOMIZED": "Non-randomized",
                     "NA": "Not applicable (single arm)"}

# Groups smaller than this give noisy rates; they are greyed out in the hover rather than hidden.
MIN_GROUP_N = 30


def load_trials(path=DATA_PATH) -> pd.DataFrame:
    """Read the processed trials file and add tidy display columns used by the dashboard.

    Missing categories become "Unknown" so they show up as their own bar instead of vanishing.
    """
    df = pd.read_parquet(path)
    df["phase_label"] = df["phase_label"].fillna(UNKNOWN)
    df["sponsor_class"] = df["sponsor_class"].fillna(UNKNOWN)
    df["masking"] = df["masking"].fillna(UNKNOWN)
    df["allocation"] = df["allocation"].map(ALLOCATION_LABELS).fillna(UNKNOWN)
    df["conditions"] = df["conditions"].fillna("")
    df["start_year"] = df["start_date"].dt.year.astype(int)
    df["outcome"] = df["label"].map({1: "Terminated", 0: "Completed"})
    return df


def filter_trials(df, phases=None, sponsor_classes=None, year_range=None, condition_kw="") -> pd.DataFrame:
    """Apply the sidebar filters; an empty selection means "no filter" for that dimension."""
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
        out = out.sort_values(column, key=lambda s: s.map(rank).fillna(len(order)))
    else:
        out = out.sort_values("n", ascending=False)
    return out


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


def _style(fig, y_title="Termination rate") -> go.Figure:
    """Shared layout: recessive grid, percent axis, compact margins, no legend for one series."""
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      hoverlabel=dict(align="left"), bargap=0.35)
    fig.update_yaxes(title=y_title, tickformat=".0%", rangemode="tozero", gridcolor="rgba(128,128,128,0.2)")
    return fig


def rate_bar(stats, column, x_title) -> go.Figure:
    """Vertical bar chart of termination rate per category, with n in the hover."""
    stats = stats.assign(small=stats["n"] < MIN_GROUP_N)
    fig = px.bar(stats, x=column, y="rate", custom_data=["n", "terminated"])
    fig.update_traces(
        marker_color=BLUE,
        # Fade very small groups so a 1-of-2 "50%" bar does not look like a finding.
        marker_opacity=[0.35 if s else 1.0 for s in stats["small"]],
        hovertemplate="<b>%{x}</b><br>Termination rate: %{y:.1%}<br>"
                      "Terminated: %{customdata[1]:,} of %{customdata[0]:,} trials<extra></extra>",
    )
    fig.update_xaxes(title=x_title, type="category")
    return _style(fig)


def rate_over_time(df) -> go.Figure:
    """Line chart of termination rate by start year, with n per year in the hover."""
    stats = rate_by(df, "start_year").sort_values("start_year")
    fig = px.line(stats, x="start_year", y="rate", markers=True, custom_data=["n", "terminated"])
    fig.update_traces(
        line=dict(color=BLUE, width=2), marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Termination rate: %{y:.1%}<br>"
                      "Terminated: %{customdata[1]:,} of %{customdata[0]:,} trials<extra></extra>",
    )
    fig.update_xaxes(title="Start year", dtick=1)
    fig.update_layout(hovermode="x unified")
    return _style(fig)


def conditions_bar(df, k=15) -> go.Figure:
    """Horizontal bars: top-k conditions by trial count; bar length = count, label = rate."""
    stats = top_conditions(df, k).sort_values("n")  # ascending so the largest sits on top
    fig = px.bar(stats, x="n", y="condition", orientation="h", custom_data=["rate", "terminated"],
                 text=stats["rate"].map(lambda r: f"{r:.0%} terminated"))
    fig.update_traces(
        marker_color=BLUE, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Trials: %{x:,}<br>Terminated: %{customdata[1]:,} "
                      "(%{customdata[0]:.1%})<extra></extra>",
    )
    fig.update_layout(margin=dict(l=10, r=80, t=10, b=10), showlegend=False,
                      height=max(300, 28 * len(stats) + 60))
    fig.update_xaxes(title="Number of trials", gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title=None)
    return fig


def eligibility_box(df) -> go.Figure:
    """Box plot of eligibility-criteria text length for completed vs terminated trials (log axis)."""
    # Log scale because the length is heavily right-skewed; zero-length texts cannot be logged.
    data = df[df["eligibility_text_len"].fillna(0) > 0]
    fig = px.box(data, x="outcome", y="eligibility_text_len", color="outcome",
                 color_discrete_map=OUTCOME_COLORS, category_orders={"outcome": ["Completed", "Terminated"]},
                 points=False, log_y=True)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    fig.update_xaxes(title=None)
    # Explicit decade ticks: Plotly's default log ticks label 2 and 5 between decades, which is noisy.
    fig.update_yaxes(title="Eligibility text length (characters, log scale)",
                     tickvals=[10, 100, 1_000, 10_000, 100_000], tickformat=",",
                     gridcolor="rgba(128,128,128,0.2)")
    return fig


def trials_table(df) -> pd.DataFrame:
    """Columns for the trial list, with a study URL the page renders as a link on the NCT id."""
    out = df[["nct_id", "brief_title", "phase_label", "lead_sponsor_name", "overall_status"]].copy()
    out.insert(0, "link", STUDY_URL + out["nct_id"])
    return out.drop(columns="nct_id")

# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
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

# No colours or fonts here: app/ui.py registers the app's Plotly template (palette, fonts, grid, tooltips) as the
# default before any page builds a chart, so these figures pick it up. Series take the template's first colour and
# reference lines its default shape colour; outside the app (tests, notebooks) Plotly's own default applies.

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
    """Shared layout: percent axis, compact margins, no legend for one series (colours come from the template)."""
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=40), showlegend=False, bargap=0.35)  # b: axis title under slanted labels
    fig.update_yaxes(title=y_title, tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return fig


def _average_line(fig, average) -> None:
    """Dashed reference line at the average rate of everything on screen, so each bar reads as above or below it.

    No text on the line: it would sit on top of the bars, and the page says what the line means.
    """
    if average is not None:
        fig.add_hline(y=average, line_dash="dash")  # colour: the template's shape default


def rate_bar(stats, column, x_title, average=None) -> go.Figure:
    """Vertical bar chart of the share stopped early per category, with the value on each bar and n in the hover."""
    stats = stats.assign(small=stats["n"] < MIN_GROUP_N)
    # Very small groups are faded and carry no value label, so a 1-of-2 "50%" bar does not look like a finding.
    labels_on_bars = [("" if small else f"{rate:.0%}") for rate, small in zip(stats["rate"], stats["small"])]
    fig = px.bar(stats, x=column, y="rate", custom_data=["n", "terminated"], text=labels_on_bars)
    fig.update_traces(
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
        line=dict(width=2), marker=dict(size=8),
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
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Stopped early: %{x:.1%}<br>"
                      "%{customdata[2]:,} of %{customdata[1]:,} trials<extra></extra>",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      height=max(320, 30 * len(stats) + 70), bargap=0.3)
    fig.update_xaxes(title="Share stopped early", tickformat=".0%", rangemode="tozero",
                     range=[0, stats["rate"].max() * 2.1], zeroline=False)  # room for the label right of the longest bar
    fig.update_yaxes(title=None)
    return fig


def rate_matrix(df, rows, cols, row_order=None, col_order=None) -> tuple:
    """Share stopped early and trial count for every combination of two traits.

    Returns (rate, n): two DataFrames with the `rows` values as index and the `cols` values as columns, in the
    given orders (values missing from an order go last, in order of size). Combinations without trials are NaN
    in `rate` and 0 in `n`.
    """
    stats = df.groupby([rows, cols], observed=True)["label"].agg(n="size", rate="mean").reset_index()
    rate = stats.pivot(index=rows, columns=cols, values="rate")
    n = stats.pivot(index=rows, columns=cols, values="n").fillna(0).astype(int)

    def ordered(present, order, sizes):
        """Known values in the given order, then the rest by how many trials they hold."""
        known = [v for v in (order or []) if v in present]
        rest = sorted((v for v in present if v not in known), key=lambda v: -sizes[v])
        return known + rest

    row_keys = ordered(list(rate.index), row_order, n.sum(axis=1))
    col_keys = ordered(list(rate.columns), col_order, n.sum(axis=0))
    return rate.loc[row_keys, col_keys], n.loc[row_keys, col_keys]


def rate_heatmap(df, rows, cols, row_order=None, col_order=None, x_title="", y_title="") -> go.Figure:
    """Heatmap of the share stopped early for each combination of two traits (for example phase x sponsor).

    Two traits at once show what single bar charts hide: a combination can be risky even when neither trait is.
    Cells with fewer than MIN_GROUP_N trials stay empty instead of showing a noisy rate; the hover always gives the
    number of trials. Rows and columns without one readable cell are dropped. The colour scale comes from the app's
    Plotly template (no colours here). Returns None if no cell has enough trials.
    """
    rate, n = rate_matrix(df, rows, cols, row_order, col_order)
    shown = rate.where(n >= MIN_GROUP_N)  # too few trials: leave the cell blank
    # Rows and columns without a single readable cell (tiny sponsor classes, "Not stated") would only add empty stripes.
    keep_rows, keep_cols = shown.notna().any(axis=1), shown.notna().any(axis=0)
    shown, n = shown.loc[keep_rows, keep_cols], n.loc[keep_rows, keep_cols]
    if shown.empty:  # the filters left no combination with enough trials; the page says so instead
        return None
    text = [[f"{value:.0%}" if pd.notna(value) else "" for value in row] for row in shown.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=shown.to_numpy(), x=[str(c) for c in shown.columns], y=[str(r) for r in shown.index],
        text=text, texttemplate="%{text}", customdata=n.to_numpy(), hoverongaps=False, xgap=3, ygap=3,
        colorbar={"title": {"text": "Stopped<br>early"}, "tickformat": ".0%", "thickness": 14},
        hovertemplate="<b>%{y} · %{x}</b><br>Stopped early: %{z:.1%}<br>%{customdata:,} trials<extra></extra>",
    ))
    fig.update_xaxes(title=x_title, type="category", showgrid=False, side="bottom")
    # Reversed so the first row of the order (for example Early Phase 1) sits on top, as in a table.
    fig.update_yaxes(title=y_title, type="category", showgrid=False, autorange="reversed")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=40), height=max(320, 52 * len(shown.index) + 90))
    return fig


def gain_curve(labels, risks, points=101) -> pd.DataFrame:
    """Cumulative gain: if you review the riskiest x% of trials first, which share of all stopped trials do you find?

    Args:
        labels: 1 for stopped early, 0 for completed.
        risks: the model's score for each trial (higher = riskier).
        points: how many evenly spaced review shares (0% to 100%) to return; the chart needs no more.

    Returns:
        DataFrame with `reviewed` (share of trials looked at, riskiest first), `caught` (share of all stopped trials
        among them) and `precision` (share of the reviewed trials that were stopped). Picking trials at random would
        give caught = reviewed, the diagonal of the chart.
    """
    frame = pd.DataFrame({"y": list(labels), "p": list(risks)}).sort_values("p", ascending=False, kind="mergesort")
    total, stopped = len(frame), frame["y"].sum()
    hits = frame["y"].cumsum().to_numpy()
    rows = []
    for share in [i / (points - 1) for i in range(points)]:
        k = int(round(share * total))  # number of trials reviewed at this share
        found = hits[k - 1] if k > 0 else 0
        rows.append({"reviewed": share, "caught": found / stopped if stopped else 0.0,
                     "precision": found / k if k else float("nan")})
    return pd.DataFrame(rows)


def eligibility_stats(df) -> pd.DataFrame:
    """Share stopped early per eligibility-text length group, shortest first (trials without a length are skipped)."""
    return rate_by(df, "eligibility_group", ELIGIBILITY_LABELS)


def trials_table(df) -> pd.DataFrame:
    """Columns for the trial list, with a study URL the page renders as a link on the NCT id."""
    out = df[["nct_id", "brief_title", "phase_label", "lead_sponsor_name", "overall_status"]].copy()
    out.insert(0, "link", STUDY_URL + out["nct_id"])
    return out.drop(columns="nct_id")

"""Aggregations and Plotly figures for the Explore dashboard.

Pure functions on the `trials.parquet` DataFrame: no Streamlit and no model code, so they can be
reused by other pages (or tested) without a running app. Termination rate = mean of `label`
(TERMINATED = 1, COMPLETED = 0); the pages call it "stopped early".

The follow-up and COVID charts at the end read `outcomes.parquet` instead (every trial since 2008,
still-running ones included, with the stop date and stop reason). That table is for charts only; no
model ever trains on it.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "trials.parquet"
OUTCOMES_PATH = DATA_PATH.parent / "outcomes.parquet"
STUDY_URL = "https://clinicaltrials.gov/study/"

# One deep green (the app's "ink" family) for every "share stopped early" bar, so the charts read as one family.
BAR = "#17463A"
INK = "#0A1F1A"
MUTED = "#4B5A53"
GRID = "rgba(10,31,26,0.12)"
FONT = {"family": "IBM Plex Sans, sans-serif", "size": 13, "color": INK}
# Shared look of every chart: transparent so the page's paper shows through, hairline grid, ink tooltips.
CHART_LAYOUT = {
    "font": FONT, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "hoverlabel": {"bgcolor": INK, "font": {"family": FONT["family"], "color": "#EDEFE6", "size": 13}, "align": "left"},
    "colorway": [BAR, "#B8452E", "#6B7A72"],
}

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


def finish(fig) -> go.Figure:
    """Give both axes readable muted text (the page theme would otherwise pick a pale grey); returns the figure."""
    muted = {"family": FONT["family"], "color": MUTED, "size": 12}
    fig.update_xaxes(tickfont=muted, title_font=muted)
    fig.update_yaxes(tickfont=muted, title_font=muted)
    return fig


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
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False, bargap=0.35, **CHART_LAYOUT)
    fig.update_yaxes(title=y_title, tickformat=".0%", rangemode="tozero", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(linecolor=INK, showgrid=False)
    return finish(fig)


def _average_line(fig, average) -> None:
    """Dashed reference line at the average rate of everything on screen, so each bar reads as above or below it.

    No text on the line: it would sit on top of the bars, and the page says what the line means.
    """
    if average is not None:
        fig.add_hline(y=average, line_dash="dash", line_color=MUTED)


def rate_bar(stats, column, x_title, average=None) -> go.Figure:
    """Vertical bar chart of the share stopped early per category, with the value on each bar and n in the hover."""
    stats = stats.assign(small=stats["n"] < MIN_GROUP_N)
    # Very small groups are faded and carry no value label, so a 1-of-2 "50%" bar does not look like a finding.
    labels_on_bars = [("" if small else f"{rate:.0%}") for rate, small in zip(stats["rate"], stats["small"])]
    fig = px.bar(stats, x=column, y="rate", custom_data=["n", "terminated"], text=labels_on_bars)
    fig.update_traces(
        marker_color=BAR,
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
        line=dict(color=BAR, width=2), marker=dict(size=8),
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
        marker_color=BAR, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Stopped early: %{x:.1%}<br>"
                      "%{customdata[2]:,} of %{customdata[1]:,} trials<extra></extra>",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      height=max(320, 30 * len(stats) + 70), bargap=0.3, **CHART_LAYOUT)
    fig.update_xaxes(title="Share stopped early", tickformat=".0%", rangemode="tozero", gridcolor=GRID,
                     range=[0, stats["rate"].max() * 2.1], zeroline=False)  # room for the label right of the longest bar
    fig.update_yaxes(title=None, automargin=True, linecolor=INK)
    return finish(fig)


def eligibility_stats(df) -> pd.DataFrame:
    """Share stopped early per eligibility-text length group, shortest first (trials without a length are skipped)."""
    return rate_by(df, "eligibility_group", ELIGIBILITY_LABELS)


def trials_table(df) -> pd.DataFrame:
    """Columns for the trial list, with a study URL the page renders as a link on the NCT id."""
    out = df[["nct_id", "brief_title", "phase_label", "lead_sponsor_name", "overall_status"]].copy()
    out.insert(0, "link", STUDY_URL + out["nct_id"])
    return out.drop(columns="nct_id")


# ---- Follow-up and COVID (outcomes.parquet) ----

SECOND = "#B8452E"  # the second series colour of CHART_LAYOUT's colorway
STATUS_GROUPS = {
    "COMPLETED": "Completed", "TERMINATED": "Stopped early", "RECRUITING": "Still running",
    "ACTIVE_NOT_RECRUITING": "Still running", "ENROLLING_BY_INVITATION": "Still running",
    "NOT_YET_RECRUITING": "Not started yet",
}


def load_outcomes(path=None) -> pd.DataFrame:
    """Read `outcomes.parquet` (written by `src/build_dataset.py`) and add start year, stop year and status group.

    `path=None` means `OUTCOMES_PATH`, looked up at call time so tests can point it elsewhere.
    """
    df = pd.read_parquet(path or OUTCOMES_PATH)
    df["start_year"] = df["start_date"].dt.year.astype(int)
    df["stop_year"] = df["completion_date"].dt.year.astype("Int64")
    df["status_group"] = df["overall_status"].map(STATUS_GROUPS).fillna("Other")
    df["phase_label"] = df["phase_label"].map(lambda v: labels.label("phase", v))
    df["sponsor_class"] = df["sponsor_class"].map(lambda v: labels.label("sponsor_class", v))
    return df


def followup_by_year(df) -> pd.DataFrame:
    """Per start year: trials, share not ended yet, and the share stopped early counted two ways.

    `rate_status` is today's label (stopped early among the trials that have ended); `rate_3y` is the horizon
    label (stopped within 3 years, every trial counted, still-running ones as "not stopped"). Years without
    3 years of follow-up have no `rate_3y`.
    """
    groups = df.groupby("start_year")
    out = pd.DataFrame({
        "trials": groups.size(),
        "not_ended": groups["status_group"].agg(lambda s: s.isin(["Still running", "Not started yet"]).mean()),
        "ended": groups["label_status"].count(),
        "rate_status": groups["label_status"].mean(),
        "labelled_3y": groups["label_3y"].count(),
        "rate_3y": groups["label_3y"].mean(),
    }).reset_index()
    for column in ("rate_status", "rate_3y", "not_ended"):
        out[column] = out[column].astype(float)
    return out


def last_followed_year(stats) -> int:
    """The latest start year in which some trials already have the 3-year label (None if none do)."""
    years = stats.loc[stats["labelled_3y"] > 0, "start_year"]
    return int(years.max()) if len(years) else None


def followup_chart(stats) -> go.Figure:
    """Two lines by start year: stopped early among ended trials vs stopped within 3 years among all trials."""
    fig = go.Figure()
    series = [("rate_status", "Stopped early, among trials that have ended", BAR, "ended"),
              ("rate_3y", "Stopped within 3 years, all trials", SECOND, "labelled_3y")]
    for column, name, colour, count in series:
        fig.add_trace(go.Scatter(
            x=stats["start_year"], y=stats[column], name=name, mode="lines+markers",
            line={"color": colour, "width": 2}, marker={"size": 7}, customdata=stats[[count]],
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1%}}<br>%{{customdata[0]:,}} trials<extra></extra>"))
    last = last_followed_year(stats)
    if last is not None and last < stats["start_year"].max():
        # Shade the years whose trials have not had 3 years yet; the page explains what that means.
        fig.add_vrect(x0=last + 0.5, x1=stats["start_year"].max() + 0.5, fillcolor=MUTED, opacity=0.08,
                      line_width=0)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend={"orientation": "h", "y": 1.12, "x": 0}, **CHART_LAYOUT)
    fig.update_yaxes(title="Share stopped early", tickformat=".0%", rangemode="tozero", gridcolor=GRID,
                     zeroline=False)
    _year_axis(fig, stats["start_year"])
    return finish(fig)


def _year_axis(fig, years) -> None:
    """Start-year axis over many years: every second year, so the labels stay level and readable."""
    first, last = int(years.min()), int(years.max())
    fig.update_xaxes(title="Start year", dtick=2 if last - first > 12 else 1, tick0=first,
                     range=[first - 0.6, last + 0.6], linecolor=INK, showgrid=False)


def not_ended_chart(stats) -> go.Figure:
    """Bars: the share of each start year's trials that has not ended yet (still running or not started)."""
    # No labels on the bars: with one bar per year they shrink to unreadable; the hover gives the numbers.
    fig = px.bar(stats, x="start_year", y="not_ended", custom_data=["trials"])
    fig.update_traces(marker_color=BAR,
                      hovertemplate="<b>%{x}</b><br>Not ended yet: %{y:.1%}<br>of %{customdata[0]:,} trials"
                                    "<extra></extra>")
    _year_axis(fig, stats["start_year"])
    return _style(fig, y_title="Share not ended yet")  # last, so the axis titles get the shared font


def covid_by_stop_year(df, first_year=2015) -> pd.DataFrame:
    """Per year a trial stopped: terminated trials and the share whose stated reason mentions COVID-19."""
    stopped = df[df["overall_status"].eq("TERMINATED") & df["stop_year"].notna()]
    stopped = stopped[stopped["stop_year"] >= first_year]
    out = (stopped.groupby("stop_year")["mentions_covid"]
           .agg(n="size", covid="sum", rate="mean").reset_index())
    out["stop_year"] = out["stop_year"].astype(int)
    return out


def covid_chart(stats) -> go.Figure:
    """Bars: share of the trials stopped in each year that name COVID-19 as a reason."""
    fig = px.bar(stats, x="stop_year", y="rate", custom_data=["n", "covid"],
                 text=[f"{r:.0%}" if r >= 0.005 else "" for r in stats["rate"]])
    fig.update_traces(marker_color=SECOND, textposition="outside", cliponaxis=False, textfont_size=12,
                      hovertemplate="<b>Stopped in %{x}</b><br>Mention COVID-19: %{y:.1%}<br>"
                                    "%{customdata[1]:,} of %{customdata[0]:,} trials<extra></extra>")
    fig.update_xaxes(title="Year the trial stopped", dtick=1)
    fig.update_yaxes(range=[0, max(stats["rate"].max(), 0.01) * 1.3])
    return _style(fig, y_title="Share naming COVID-19")


def time_to_stop(df, start_years=(2008, 2015), max_years=10.0, step=0.25) -> pd.DataFrame:
    """Share of terminations that happened within t years of the start, for t = 0 .. max_years.

    Only trials that started in `start_years` count, so every one of them has been followed for years
    and late terminations are not missing.
    """
    first, last = start_years
    years = df.loc[df["overall_status"].eq("TERMINATED") & df["start_year"].between(first, last),
                   "years_to_stop"].dropna()
    grid = [round(i * step, 2) for i in range(int(max_years / step) + 1)]
    return pd.DataFrame({"years": grid, "share": [(years <= t).mean() if len(years) else float("nan")
                                                  for t in grid], "n": len(years)})


def time_to_stop_chart(curve, marks=(3, 5)) -> go.Figure:
    """Cumulative line: share of terminations within t years of the start, with the 3- and 5-year points named."""
    fig = px.line(curve, x="years", y="share")
    fig.update_traces(line=dict(color=BAR, width=2),
                      hovertemplate="Within %{x} years: %{y:.0%} of terminations<extra></extra>")
    for mark in marks:
        point = curve.loc[curve["years"] == mark, "share"]
        if len(point) and point.notna().all():
            share = float(point.iloc[0])
            fig.add_scatter(x=[mark], y=[share], mode="markers+text", text=[f"{share:.0%} within {mark} years"],
                            textposition="bottom right", marker={"color": SECOND, "size": 9},
                            hoverinfo="skip", showlegend=False)
    fig.update_xaxes(title="Years since the start", dtick=1, range=[0, curve["years"].max()])
    fig.update_yaxes(range=[0, 1.05])
    return _style(fig, y_title="Share of stopped trials")


def stop_reasons(df) -> pd.DataFrame:
    """Share of terminated trials per rough stop-reason group (from `src/horizon.py`), commonest first."""
    stopped = df[df["overall_status"].eq("TERMINATED")]
    out = stopped["stop_reason"].fillna("Other or not stated").value_counts().rename_axis("reason")
    return out.reset_index(name="n").assign(share=lambda t: t["n"] / t["n"].sum())


def stop_reasons_chart(stats) -> go.Figure:
    """Horizontal bars of the stop-reason groups, the commonest on top."""
    ordered = stats.iloc[::-1]
    fig = px.bar(ordered, x="share", y="reason", orientation="h", custom_data=["n"],
                 text=[f"{s:.0%}" for s in ordered["share"]])
    fig.update_traces(marker_color=BAR, textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{y}</b><br>%{x:.1%} of stopped trials<br>%{customdata[0]:,} trials"
                                    "<extra></extra>")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      height=max(300, 34 * len(stats) + 70), bargap=0.3, **CHART_LAYOUT)
    fig.update_xaxes(title="Share of stopped trials", tickformat=".0%", rangemode="tozero", gridcolor=GRID,
                     range=[0, stats["share"].max() * 1.3], zeroline=False)
    fig.update_yaxes(title=None, automargin=True, linecolor=INK)
    return finish(fig)

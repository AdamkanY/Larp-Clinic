"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Shared risk display: a comparison bar that ranks a trial among the historical trials the model learned from,
and the chart of what pushes one trial's risk up or down.
"""
import json  # reads the saved metrics file
from pathlib import Path  # builds file paths that work on every computer

import plotly.graph_objects as go  # draws the comparison bar and the drivers chart
import streamlit as st  # the web app framework (here only for caching)

from src.predict import percentile, reference_size  # rank of a score among past trials, size of that reference set
from ui import PALETTE  # the one colour palette of the app

# The metrics file gives the under-prediction gap quoted in the captions.
METRICS = Path(__file__).resolve().parents[1] / "models" / "gbm_metrics.json"
INK = PALETTE["ink"]  # the text and line colour of both charts


def percentile_of(risk):
    """Find how many percent of the historical trials have a lower score than risk (from src.predict.percentile).

    Input: risk is the predicted risk of stopping early, a number between 0 and 1.

    Output: the percent rank (0 to 100), or None while models/score_reference.npy is missing, so the page can still
    show the risk without a rank.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Ask the model code for the rank; a missing reference file means no rank.
    try:
        return percentile(risk)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def underestimate():
    """Find how much higher the real share stopped early was than the model's average prediction, on the test trials.

    Input: none.

    Output: a number such as 0.18, which means 18% higher, or None if the metrics file is missing or incomplete.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Compare the real test share with the mean predicted risk, both stored in the metrics file.
    try:
        metrics = json.loads(METRICS.read_text())
        real_share = metrics["split"]["test"]["terminated_rate"]
        predicted_share = metrics["test"]["mean_predicted"]
        return real_share / predicted_share - 1
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return None


def estimate_caption(risk, ongoing=False):
    """Write an honest paragraph on how to read a risk score.

    Input: risk is the predicted risk, a number between 0 and 1; ongoing is True for trials that are still running
    (adds a warning), otherwise False.

    Output: the paragraph as a text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The caption is built from parts: the basic reading, the measured gap and the warning for running trials.
    text = (f"Rough estimate from design information only: read {risk:.0%} as a lower-end figure and mainly use "
            "it to compare trials. ")
    gap = underestimate()
    if gap is not None and gap > 0:
        text += (f"On newer trials (the test set) the real share stopped early was about {gap:.0%} higher "
                 "than the model's average prediction. ")
    if ongoing:
        text += ("The model learned from finished trials of 2008-2019; ongoing trials can look different "
                 "(for example longer eligibility texts), so their scores are less certain. ")
    return text + "Not medical advice."


def risk_tone(risk):
    """Choose the colour and the words for a risk.

    The colour follows the trial's rank among past trials, not the raw percentage: green below the median trial,
    yellow up to the riskiest 10%, red above. Without the reference file the tone is "gray" and the rank is None.

    Input: risk is the predicted risk, a number between 0 and 1.

    Output: a tuple (tone, label, percent rank). The tone is "green", "yellow", "red" or "gray", the label is a
    sentence and the rank is a number from 0 to 100 or None.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Rank first, then pick the tone by rank.
    pct = percentile_of(risk)
    if pct is None:
        return "gray", "Estimated risk of stopping early", None
    if pct < 50:
        return "green", "Lower than most past trials", pct
    if pct < 90:
        return "yellow", "Higher than most past trials", pct
    return "red", "Among the riskiest 10% of past trials", pct


def risk_chip(risk):
    """Build a small coloured label with the risk and its rank in words, for result lists.

    Input: risk is the predicted risk, a number between 0 and 1.

    Output: the label as an HTML text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The chip only needs the tone, the label and the rank is not shown.
    tone, label, pct = risk_tone(risk)
    return f'<span class="chip {tone}">{risk:.1%} risk · {label.lower()}</span>'


def risk_card(risk, delta=None, compact=False):
    """Build a coloured card: the risk number, how it ranks among past trials and an optional change.

    compact=True gives a one-line strip, which keeps the result in view while a form is edited.

    Input: risk is the predicted risk, a number between 0 and 1; delta is the change against the previous input
    (a number, or None for no change line); compact is True for the strip and False for the big card.

    Output: the card as an HTML text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Tone, label and rank decide the colour and the words of the card.
    tone, label, pct = risk_tone(risk)

    # The rank sentence needs the reference file, the change is only shown when there is one.
    sub = ""
    if pct is not None:
        sub = f"Riskier than {pct:.0f}% of historical trials (started 2009-2016)"
    change = ""
    if delta is not None:
        change = f'<div class="sub">{delta:+.1%} compared with your previous input</div>'
    size = ""
    if compact:
        size = " compact"

    # The strip also carries a small scale of 100 ticks. The one pink tick ("me") is this trial.
    meter = ""
    if compact and pct is not None:
        my_tick = min(int(pct), 99)
        ticks = ""
        for i in range(100):
            if i == my_tick:
                ticks += '<i class="me"></i>'
            else:
                ticks += '<i class=""></i>'
        meter = f'<div class="meter" aria-hidden="true">{ticks}</div>'

    # Put the parts together as one HTML block.
    return (f'<div class="risk-card {tone}{size}"><div class="num">{risk:.1%}</div><div class="lab">{label}</div>'
            f'<div class="sub">{sub}</div>{change}{meter}</div>')


def risk_bar(risk):
    """Draw a horizontal bar from the least risky to the riskiest historical trials, with a marker for this trial.

    Input: risk is the predicted risk, a number between 0 and 1.

    Output: a Plotly figure, or None if the reference scores are missing, so the pages can simply skip the bar.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The bar is drawn in percent of past trials (0-100), so its bands match the colours of the chips.
    pct = percentile_of(risk)
    if pct is None:
        return None
    # An empty figure that gets the bands, the marker and the label below.
    fig = go.Figure()

    # Three pale bands, so the ink text and the pink marker stay readable on them.
    bands = [
        (0, 50, PALETTE["low_band"], "Less risky than half"),
        (50, 90, PALETTE["mid_band"], "Higher"),
        (90, 100, PALETTE["high_band"], ""),  # too narrow for a label
    ]
    for start, end, colour, name in bands:
        fig.add_trace(go.Bar(x=[end - start], y=[""], base=[start], orientation="h", marker_color=colour,
                             width=0.7, text=name, textposition="inside", insidetextanchor="middle",
                             textfont={"size": 13, "color": INK}, hoverinfo="skip"))

    # One pink line through the bar marks this trial.
    fig.add_shape(type="line", x0=pct, x1=pct, y0=-0.45, y1=0.45, line={"color": PALETTE["accent_deep"], "width": 4})

    # Its label sits above the bar, so it never covers a band label. Near the edges it is turned inwards, so it
    # stays inside the chart.
    if pct > 80:
        anchor = "right"
    elif pct < 20:
        anchor = "left"
    else:
        anchor = "center"
    fig.add_annotation(x=pct, y=0.45, text=f"<b>This trial</b> ({pct:.0f}%)", showarrow=False, yanchor="bottom",
                       xanchor=anchor, yshift=6, bgcolor=INK, font={"color": PALETTE["paper"], "size": 13}, borderpad=5)

    # Fixed x range 0-100% with ticks only at the band borders.
    fig.update_layout(
        barmode="overlay", showlegend=False, height=190, margin={"l": 10, "r": 10, "t": 55, "b": 60},
        xaxis={"range": [0, 100], "tickvals": [0, 50, 90, 100], "ticktext": ["0%", "50%", "90%", "100%"],
               "title": "Past trials that were less risky", "showgrid": False, "linecolor": INK},
        yaxis={"visible": False, "range": [-0.5, 0.5]},  # fixed, so the label does not stretch the axis
    )
    return fig


def bar_caption():
    """Write one sentence explaining how to read the bar.

    Input: none.

    Output: the sentence as a text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Without the reference file the sentence still works: "among the past trials".
    try:
        count = f"{reference_size():,}"
    except (OSError, ValueError):
        count = "the"
    # Name the number of past trials and explain the three colours.
    return (f"The bar ranks this trial among {count} past trials (started 2009-2016): green = less risky than "
            "half of them, yellow = up to the riskiest 10%, red = the riskiest 10%.")


def drivers_chart(drivers):
    """Draw horizontal bars of what pushes one trial's risk up (to the right, orange) or down (to the left, green).

    Input: drivers is the list from src.predict.explain, with one tuple (name, value, direction, size) per factor,
    largest first.

    Output: a Plotly figure, or None for an empty list, so the pages can show a sentence instead.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Nothing to draw without drivers.
    if not drivers:
        return None

    # One bar per factor. Plotly draws the first bar at the bottom, so we go through the list backwards and the
    # strongest factor ends up on top. Factors that raise the risk point right, the others left.
    lengths = []
    names = []
    colours = []
    texts = []
    for name, value, direction, size in drivers[::-1]:
        if direction == "up":
            length = size
        else:
            length = -size
        lengths.append(length)
        names.append(f"{name}: {value}")
        if length > 0:
            colours.append(PALETTE["up"])
            texts.append("raises the risk")
        else:
            colours.append(PALETTE["down"])
            texts.append("lowers the risk")

    # The bar chart itself, with the text next to each bar.
    fig = go.Figure(go.Bar(
        x=lengths, y=names, orientation="h", marker_color=colours, text=texts, textposition="outside",
        cliponaxis=False, hovertemplate="%{y}<br>%{text}<extra></extra>",
    ))
    # Zero line: the typical past trial.
    fig.add_vline(x=0, line={"color": INK, "width": 1})

    # The axis is the same width on both sides of zero, with room for the text next to the longest bar.
    widest = max(abs(length) for length in lengths)
    fig.update_layout(
        title={"text": "What pushes this trial's risk up / down", "x": 0, "xanchor": "left"},
        showlegend=False, height=140 + 46 * len(drivers), margin={"l": 0, "r": 10, "t": 50, "b": 60},
        xaxis={"range": [-widest * 1.9, widest * 1.9], "showticklabels": False, "showgrid": False, "zeroline": False,
               "title": {"text": "← lowers the risk · compared with a typical past trial · raises the risk →",
                         "font": {"color": PALETTE["muted"], "size": 12}}},
        yaxis={"automargin": True},
    )
    return fig


def drivers_caption():
    """Write how to read the drivers chart, in plain words.

    Input: none.

    Output: the explanation as a text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # One fixed explanation text.
    return ("Each bar takes one factor of this trial and swaps it for the value a typical past trial has "
            "(for example the most common blinding, or the median length of the eligibility criteria). The longer the "
            "bar, the more the model's estimate moves. These are patterns in past trials, not causes, and the "
            "factors are measured one at a time, so the bars do not add up to the total.")

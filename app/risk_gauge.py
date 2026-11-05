"""Shared risk display: a comparison bar that ranks a trial among the historical trials the model learned from."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_metrics
from src.viz import CHART_LAYOUT, INK, finish

REFERENCE = Path(__file__).resolve().parents[1] / "models" / "score_reference.npy"
GREEN, YELLOW, RED = "#C5DDD0", "#EBDDAE", "#E3B7B0"  # pale bands; ink text and the pink marker stay readable on them
MARKER = "#C80F5A"


@st.cache_data(show_spinner=False)
def reference_scores():
    """Sorted out-of-fold risk scores of the historical training trials, or None if the file is missing."""
    try:
        return np.sort(np.load(REFERENCE))
    except (OSError, ValueError):
        return None


def percentile_of(risk):
    """Share of historical trials (in percent) with a lower score than risk; None without the reference file."""
    reference = reference_scores()
    if reference is None:
        return None
    return float(100.0 * np.searchsorted(reference, risk, side="left") / len(reference))


def underestimate():
    """How much higher the real share stopped early on the newer test trials was than the model's average prediction.

    0.18 means 18% higher. None if the metrics file is missing or incomplete.
    """
    metrics = load_metrics()
    if metrics is None:
        return None
    try:
        return metrics["split"]["test"]["terminated_rate"] / metrics["test"]["mean_predicted"] - 1
    except (KeyError, ZeroDivisionError):
        return None


def estimate_caption(risk, ongoing=False):
    """Honest one-paragraph reading of a risk score; ongoing=True adds the caveat for live (unfinished) trials."""
    text = f"Rough estimate from design information only: read {risk:.0%} as a lower-end figure and mainly use it to compare trials. "
    gap = underestimate()
    if gap is not None and gap > 0:
        text += (f"On newer trials (the test set) the real share stopped early was about {gap:.0%} higher "
                 "than the model's average prediction. ")
    if ongoing:
        text += ("The model learned from finished trials of 2008-2019; ongoing trials can look different "
                 "(for example longer eligibility texts), so their scores are less certain. ")
    return text + "Not medical advice."


def risk_tone(risk):
    """Colour and words for a risk, chosen by its rank among past trials: (tone, label, percent rank).

    Green below the median trial, yellow up to the riskiest 10%, red above. Without the reference file the
    tone is "gray" and the rank is None.
    """
    pct = percentile_of(risk)
    if pct is None:
        return "gray", "Estimated risk of stopping early", None
    if pct < 50:
        return "green", "Lower than most past trials", pct
    if pct < 90:
        return "yellow", "Higher than most past trials", pct
    return "red", "Among the riskiest 10% of past trials", pct


def risk_chip(risk):
    """HTML for a small coloured label with the risk and its rank wording, for result lists."""
    tone, label, _ = risk_tone(risk)
    return f'<span class="chip {tone}">{risk:.1%} risk · {label.lower()}</span>'


def risk_card(risk, delta=None, compact=False):
    """HTML for a coloured card: the risk number, how it ranks among past trials, and an optional change.

    compact=True gives a one-line strip, for keeping the result in view while a form is edited.
    """
    tone, label, pct = risk_tone(risk)
    sub = f"Riskier than {pct:.0f}% of historical trials (started 2008-2016)" if pct is not None else ""
    change = f'<div class="sub">{delta:+.1%} compared with your previous input</div>' if delta is not None else ""
    size = " compact" if compact else ""
    meter = ""
    if compact and pct is not None:  # the strip carries a small instrument scale: one pink tick = this trial
        ticks = "".join(f'<i class="{"me" if i == min(int(pct), 99) else ""}"></i>' for i in range(100))
        meter = f'<div class="meter" aria-hidden="true">{ticks}</div>'
    return (f'<div class="risk-card {tone}{size}"><div class="num">{risk:.1%}</div><div class="lab">{label}</div>'
            f'<div class="sub">{sub}</div>{change}{meter}</div>')


def risk_bar(risk):
    """Horizontal bar from the least to the riskiest historical trials with a marker for this trial.

    Returns None if the reference scores are missing, so callers can simply skip the bar.
    """
    pct = percentile_of(risk)
    if pct is None:
        return None
    fig = go.Figure()
    for start, end, colour, name in [(0, 50, GREEN, "Less risky than half"), (50, 90, YELLOW, "Higher"),
                                     (90, 100, RED, "")]:  # too narrow for a label
        fig.add_trace(go.Bar(x=[end - start], y=[""], base=[start], orientation="h", marker_color=colour,
                             width=0.7, text=name, textposition="inside", insidetextanchor="middle",
                             textfont={"size": 13, "color": INK}, hoverinfo="skip"))
    # One pink line through the bar marks this trial; its label sits above the bar so it never covers a segment label.
    fig.add_shape(type="line", x0=pct, x1=pct, y0=-0.45, y1=0.45, line={"color": MARKER, "width": 4})
    anchor = "right" if pct > 80 else ("left" if pct < 20 else "center")
    fig.add_annotation(x=pct, y=0.45, text=f"<b>This trial</b> ({pct:.0f}%)", showarrow=False, yanchor="bottom",
                       xanchor=anchor, yshift=6, bgcolor=INK, font={"color": "#EDEFE6", "size": 13}, borderpad=5)
    fig.update_layout(
        barmode="overlay", showlegend=False, height=190, margin={"l": 10, "r": 10, "t": 55, "b": 60},
        xaxis={"range": [0, 100], "tickvals": [0, 50, 90, 100], "ticktext": ["0%", "50%", "90%", "100%"],
               "title": "Past trials that were less risky", "showgrid": False, "linecolor": INK},
        yaxis={"visible": False, "range": [-0.5, 0.5]},  # fixed, so the label does not stretch the axis
        **CHART_LAYOUT,
    )
    return finish(fig)


def bar_caption():
    """One sentence explaining how to read the bar."""
    reference = reference_scores()
    count = f"{len(reference):,}" if reference is not None else "the"
    return (f"The bar ranks this trial among {count} past trials (started 2008-2016): green = less risky than "
            "half of them, yellow = up to the riskiest 10%, red = the riskiest 10%.")

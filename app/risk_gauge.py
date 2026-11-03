"""Shared risk display: a comparison bar that ranks a trial among the historical trials the model learned from."""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

REFERENCE = Path(__file__).resolve().parents[1] / "models" / "score_reference.npy"
METRICS = Path(__file__).resolve().parents[1] / "models" / "gbm_metrics.json"
GREEN, YELLOW, RED = "#9be7a1", "#ffe08a", "#f5a3a3"


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


@st.cache_data(show_spinner=False)
def underestimate():
    """How much higher the real termination rate on the newer test trials was than the model's average prediction.

    0.18 means 18% higher. None if the metrics file is missing or incomplete.
    """
    try:
        metrics = json.loads(METRICS.read_text())
        return metrics["split"]["test"]["terminated_rate"] / metrics["test"]["mean_predicted"] - 1
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return None


def estimate_caption(risk, ongoing=False):
    """Honest one-paragraph reading of a risk score; ongoing=True adds the caveat for live (unfinished) trials."""
    text = f"Rough estimate from design information only: read {risk:.0%} as a lower-end figure and mainly use it to compare trials. "
    gap = underestimate()
    if gap is not None and gap > 0:
        text += (f"On newer trials (the test set) the real termination rate was about {gap:.0%} higher "
                 "than the model's average prediction. ")
    if ongoing:
        text += ("The model learned from finished trials of 2008-2019; ongoing trials can look different "
                 "(for example longer eligibility texts), so their scores are less certain. ")
    return text + "Not medical advice."


def risk_bar(risk):
    """Horizontal bar from the least to the riskiest historical trials with a marker for this trial.

    Returns None if the reference scores are missing, so callers can simply skip the bar.
    """
    pct = percentile_of(risk)
    if pct is None:
        return None
    fig = go.Figure()
    for start, end, colour, name in [(0, 50, GREEN, "Less risky than half"), (50, 90, YELLOW, "Higher"),
                                     (90, 100, RED, "Top 10%")]:
        fig.add_trace(go.Bar(x=[end - start], y=[""], base=[start], orientation="h", marker_color=colour,
                             width=0.7, text=name, textposition="inside", insidetextanchor="middle",
                             textfont={"size": 13, "color": "#111827"}, hoverinfo="skip"))
    # An arrow above the bar instead of a line through it, so it never covers the segment labels.
    fig.add_annotation(x=pct, y=0.4, text=f"<b>This trial</b> ({pct:.0f}%)", showarrow=True, arrowhead=2,
                       arrowwidth=3, arrowcolor="#222", ax=0, ay=-38, bgcolor="white", bordercolor="#222", borderpad=3)
    fig.update_layout(
        barmode="overlay", showlegend=False, height=170, margin={"l": 10, "r": 10, "t": 55, "b": 50},
        xaxis={"range": [0, 100], "tickvals": [0, 50, 90, 100], "ticktext": ["0%", "50%", "90%", "100%"],
               "title": "Share of past trials that were less risky than this one"},
        yaxis={"visible": False, "range": [-0.5, 0.5]},  # fixed, so the arrow does not stretch the axis
    )
    return fig


def bar_caption():
    """One sentence explaining how to read the bar."""
    reference = reference_scores()
    count = f"{len(reference):,}" if reference is not None else "the"
    return (f"The bar ranks this trial among {count} past trials (started 2008-2016): green = less risky than "
            "half of them, yellow = up to the riskiest 10%, red = the riskiest 10%.")

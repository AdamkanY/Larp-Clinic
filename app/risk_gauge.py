"""Shared risk gauge: colour bands come from the historical score distribution, not from fixed guesses."""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

REFERENCE = Path(__file__).resolve().parents[1] / "models" / "score_reference.npy"
METRICS = Path(__file__).resolve().parents[1] / "models" / "gbm_metrics.json"
# Rough median and 90th percentile of the 2008-2016 training trials, used if the reference file is missing.
FALLBACK_EDGES = (0.10, 0.22)
# Percent. Historical scores top out around 35%, so a 0-100 axis would squash every trial into the left corner.
AXIS_MAX = 40


@st.cache_data(show_spinner=False)
def band_edges():
    """Median and 90th percentile of the historical risk scores, as probabilities."""
    try:
        median, p90 = np.percentile(np.load(REFERENCE), [50, 90])
        return float(median), float(p90)
    except (OSError, ValueError):
        return FALLBACK_EDGES


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


def risk_gauge(risk):
    """Plotly gauge for a risk in [0, 1]: green below the median trial, yellow up to the riskiest 10%, red above."""
    low, high = (edge * 100 for edge in band_edges())
    return go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, AXIS_MAX]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, low], "color": "#9be7a1"},
                         {"range": [low, high], "color": "#ffe08a"},
                         {"range": [high, AXIS_MAX], "color": "#f5a3a3"}]},
    ))


def gauge_caption():
    """One sentence explaining what the gauge colours mean."""
    low, high = band_edges()
    return (f"Colours compare with historical trials: green = below the median trial ({low:.0%}), "
            f"yellow = up to the riskiest 10% ({high:.0%}), red = the riskiest 10%.")

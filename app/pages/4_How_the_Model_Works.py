# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""How the model works: the page that explains the model to a reader without a statistics background.

On top there are two boxes ("In short" and "How to read a risk score"), then six tabs with one question each:
what is predicted, how good it is, which model won the race, what matters most, can you trust the percentages,
and which versions and checks were run.

The file is ordered like this: loading the files, small tables, the charts, one function per box and per tab,
and at the very end the page itself. All numbers are read from the files in models/ or computed from the test
trials, so the texts follow a retrained model.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

# Streamlit runs a page as its own script, so we add the project folder (for src/) and app/ (for ui) by hand.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.labels import PHASE_ORDER, feature_name, label
from src.predict import load_model, score_frame
from src.viz import gain_curve
from ui import PALETTE, alpha, chart, chip, page_header, require_router

require_router("pages/4_How_the_Model_Works.py")

# Settings of this page (not of the model).
MODELS = ROOT / "models"
TRIALS_FILE = ROOT / "data" / "processed" / "trials.parquet"
TOP_FEATURES = 12  # how many factors the importance chart shows
MIN_GROUP = 1000  # a phase with fewer test trials is left out, its ROC-AUC would be too noisy
MIN_DIRECTION = 0.03  # a rank correlation weaker than this counts as "no clear direction"
# Rule that was fixed before model v2 was trained (models/CLAUDE.md): the app only switches to v2 if the test
# ROC-AUC of the calibrated GBM rises by at least this much and cross-validation agrees.
PROMOTION_BAR = 0.005
# Where the cost slider starts: a missed stopped trial costs 5 false alarms (line at 1/6, the DSF deck example).
DEFAULT_COST_RATIO = 5


# ---------------------------------------------------------------------------------------------------------------
# Loading the files (cached, so each file is read only once)
# ---------------------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_metrics(name="gbm"):
    """The metrics of a saved model from models/<name>_metrics.json ("gbm" is the app model).

    Returns None if the file is missing or cannot be read.
    """
    try:
        text = (MODELS / f"{name}_metrics.json").read_text()
        return json.loads(text)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_horse_race():
    """The result of the model race (models/horse_race.json, written by src/horse_race.py). None if it is missing."""
    try:
        text = (MODELS / "horse_race.json").read_text()
        return json.loads(text)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_directions(features):
    """For each feature: does a higher value (or a "yes") go with more or with fewer stopped trials?

    Looks only at the training trials and returns a dictionary {feature: text}, or None if the data file is
    missing. This describes the data on its own, not what the model does with it.
    """
    # Read only the training trials and only the columns we need.
    try:
        data = pd.read_parquet(TRIALS_FILE, columns=list(features) + ["split", "label"],
                               filters=[("split", "==", "train")])
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None

    directions = {}
    for feature in features:
        if feature in CATEGORICAL_FEATURES:
            # A category such as phase or blinding has no "higher", so there is no single direction.
            directions[feature] = "depends on the category"
        else:
            # Spearman works with ranks, so very skewed numbers (the eligibility length) do not distort it.
            values = pd.to_numeric(data[feature], errors="coerce")
            rho = values.corr(data["label"].astype(float), method="spearman")
            if feature in BINARY_FEATURES:
                word = "yes"
            else:
                word = "higher"
            if rho >= MIN_DIRECTION:
                directions[feature] = f"{word} → stopped more often"
            elif rho <= -MIN_DIRECTION:
                directions[feature] = f"{word} → stopped less often"
            else:
                directions[feature] = "no clear direction on its own"
    return directions


@st.cache_data(show_spinner=False)
def load_importance():
    """The most important factors of the app model, strongest first, each with a plain name and a direction text.

    The numbers are the permutation importance that src/evaluate.py saved. Returns None if the file is missing.
    """
    try:
        table = pd.read_csv(MODELS / "feature_importance.csv")
    except (OSError, ValueError):
        return None

    # Keep the rows of the app model and of them the strongest factors.
    table = table[table["model"] == "gbm"].sort_values("rank").head(TOP_FEATURES).copy()

    # Add the name people read and the direction in words.
    directions = load_directions(tuple(table["feature"]))
    names = []
    direction_texts = []
    for feature in table["feature"]:
        names.append(feature_name(feature))
        if directions:
            direction_texts.append(directions[feature])
        else:
            direction_texts.append("direction not available")
    table["Factor"] = names
    table["Direction"] = direction_texts
    return table


@st.cache_data(show_spinner="Scoring the test trials...")
def load_test_scores():
    """Score the test trials (start 2017-2019) with the app model. Returns None if the data or the model is missing.

    We score them here instead of saving a table, so the numbers always belong to the model that is in models/.
    The result has one row per test trial: label, risk, raw (the score before calibration), year and phase.
    """
    # The calibrated risk of every test trial.
    try:
        test = pd.read_parquet(TRIALS_FILE, columns=FEATURE_COLUMNS + ["split", "label", "start_date", "phase_label"],
                               filters=[("split", "==", "test")])
        risk = score_frame(test)
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None

    # The same trials scored by the GBM before calibration. Only the before/after calibration plot needs this,
    # so the page also works without that model file.
    base = "gbm_uncalibrated"
    metrics = load_metrics()
    if metrics and "base_model" in metrics.get("calibration", {}):
        base = metrics["calibration"]["base_model"]
    try:
        raw = score_frame(test, load_model(f"models/{base}.joblib"))
    except (FileNotFoundError, OSError, ValueError, KeyError):
        raw = float("nan")

    # The phase in plain words, for the breakdown by phase.
    phases = []
    for value in test["phase_label"]:
        phases.append(label("phase", value))

    return pd.DataFrame({
        "label": test["label"].astype(int).to_numpy(),
        "risk": risk,
        "raw": raw,
        "year": pd.to_datetime(test["start_date"]).dt.year.to_numpy(),
        "phase": phases,
    })


# ---------------------------------------------------------------------------------------------------------------
# Small tables that are computed from the scored test trials or from the metrics files
# ---------------------------------------------------------------------------------------------------------------

def phase_table(scores):
    """One row per phase: number of test trials, mean prediction, real share stopped and the ROC-AUC inside it.

    Phases with fewer than MIN_GROUP test trials are left out.
    """
    rows = []
    for phase, group in scores.groupby("phase"):
        # ROC-AUC can only be computed if the group has both outcomes.
        if len(group) >= MIN_GROUP and group["label"].nunique() == 2:
            rows.append({
                "Phase": phase,
                "Test trials": len(group),
                "Model predicted": group["risk"].mean(),
                "Really stopped": group["label"].mean(),
                "ROC-AUC inside the phase": roc_auc_score(group["label"], group["risk"]),
            })
    table = pd.DataFrame(rows)

    # Sort in clinical order (Early phase 1 ... Phase 4) instead of by alphabet. A name that is not in
    # PHASE_ORDER goes to the end.
    positions = []
    for phase in table["Phase"]:
        if phase in PHASE_ORDER:
            positions.append(PHASE_ORDER.index(phase))
        else:
            positions.append(len(PHASE_ORDER))
    table["position"] = positions
    table = table.sort_values("position", kind="stable").drop(columns="position")
    return table.reset_index(drop=True)


def year_table(scores):
    """One row per start year of the test trials: how many, real share stopped, mean prediction and the gap.

    too_low says how much higher the real share is than the prediction (0.25 = 25% higher).
    """
    table = scores.groupby("year").agg(
        trials=("label", "size"), real=("label", "mean"), predicted=("risk", "mean")
    ).reset_index()
    table["too_low"] = table["real"] / table["predicted"] - 1
    return table


def calibration_curve(predicted, observed):
    """Cut the trials into ten groups of the same size, lowest predictions first.

    Returns one row per group with the mean prediction and the real share stopped early.
    """
    frame = pd.DataFrame({"p": predicted, "y": observed}).dropna()
    # Many trials have exactly the same calibrated score. Ranking first gives every trial its own place, so the
    # ten groups really get the same number of trials.
    frame["group"] = pd.qcut(frame["p"].rank(method="first"), 10, labels=False)
    curve = frame.groupby("group").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return curve.reset_index(drop=True)


def confusion_counts(labels, risks, threshold):
    """Count the four cells of the confusion matrix at one threshold ("positive" means stopped early)."""
    flagged = risks >= threshold
    stopped = labels == 1
    return {
        "tp": int((flagged & stopped).sum()),  # stopped and flagged
        "fp": int((flagged & ~stopped).sum()),  # completed, but flagged (false alarm)
        "fn": int((~flagged & stopped).sum()),  # stopped, but not flagged (missed)
        "tn": int((~flagged & ~stopped).sum()),  # completed and not flagged
    }


def version_table(v1, v2):
    """Compare model v1 (the app model) with v2 (plus condition areas): inputs, CV and test numbers, status."""
    versions = [
        ("v1: design features", v1, "Used in the app"),
        ("v2: plus condition areas", v2, "Tested, not used"),
    ]
    rows = []
    for name, metrics, status in versions:
        rows.append({
            "Model": name,
            "Inputs": len(metrics["features"]["all"]),
            "CV ROC-AUC (training trials)": f"{metrics['cv']['best_roc_auc_mean']:.3f}",
            "Test ROC-AUC": f"{metrics['test']['roc_auc']:.4f}",
            "Test PR-AUC": f"{metrics['test']['pr_auc']:.4f}",
            "Test Brier": f"{metrics['test']['brier']:.4f}",
            "Status": status,
        })
    return pd.DataFrame(rows)


def ablation_table(ablations):
    """One row per ablation run: which inputs were left out and what that cost on the test trials."""
    rows = []
    for run in ablations["runs"]:
        dropped = run["dropped"]
        # The first run leaves nothing out, so it has no change to show.
        if not dropped:
            variant = "All inputs"
            change = "–"
        else:
            dropped_names = []
            for feature in dropped:
                dropped_names.append(feature_name(feature).lower())
            variant = "Without " + ", ".join(dropped_names)
            change = f"{run['delta_test_roc_auc']:+.3f}"
        rows.append({
            "Variant": variant,
            "Inputs": run["n_features"],
            "CV ROC-AUC (training trials)": f"{run['cv_roc_auc']:.3f}",
            "Test ROC-AUC": f"{run['test_roc_auc']:.3f}",
            "Change in test ROC-AUC": change,
            "Test PR-AUC": f"{run['test_pr_auc']:.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# The charts. Each function builds one Plotly figure; the colours come from PALETTE in ui.py.
# ---------------------------------------------------------------------------------------------------------------

def risk_group_chart(bins, average):
    """Two bars per risk group (10% of the test trials each): what the model predicted and what really happened."""
    # The ten groups are already computed in gbm_metrics.json.
    groups = [str(number) for number in range(1, len(bins) + 1)]
    predicted = [group["mean_predicted"] for group in bins]
    observed = [group["observed_rate"] for group in bins]

    fig = go.Figure()
    # The predicted bars are pale with a dark outline, so the two bars differ in lightness and not only in colour.
    fig.add_trace(go.Bar(x=groups, y=predicted, name="Model predicted",
                         marker={"color": PALETTE["pale"], "line": {"color": PALETTE["neutral"], "width": 2}}))
    fig.add_trace(go.Bar(x=groups, y=observed, name="Really stopped early", marker_color=PALETTE["bar"]))
    # Dashed line: the average of all test trials, so every pair can be read as above or below average.
    fig.add_hline(y=average, line_dash="dash",
                  annotation_text=f"Average of all test trials: {average:.1%}", annotation_position="top left")
    fig.update_layout(
        barmode="group", height=420,
        margin={"l": 10, "r": 0, "t": 30, "b": 70},  # b: room for the axis title
        xaxis_title="Risk group: 10% of the test trials each, from lowest (1) to highest (10) predicted risk",
        xaxis_type="category",  # otherwise Plotly reads "1".."10" as numbers and labels only every second group
        yaxis={"tickformat": ".0%", "title": None},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def calibration_chart(after, before):
    """Calibration plot: predicted risk (x) against the real share stopped early (y), one point per tenth.

    A model whose percentages can be taken literally lies on the diagonal. after is the app model, before the
    same GBM before calibration (tables with mean_predicted and observed_rate). before can be None, then only
    the app model is drawn.
    """
    # Both axes get the same range with a little room on top, so the diagonal is a real 45-degree line.
    top = max(after["mean_predicted"].max(), after["observed_rate"].max())
    if before is not None:
        top = max(top, before["mean_predicted"].max(), before["observed_rate"].max())
    top = min(1.0, top * 1.08)

    fig = go.Figure()
    # Dotted diagonal: where a perfectly calibrated model would lie.
    fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line={"dash": "dot", "width": 1.5})
    fig.add_annotation(x=top * 0.97, y=top * 0.97, text="Percentages exactly right", showarrow=False,
                       xanchor="right", yanchor="bottom", textangle=0)
    # Before calibration: dashed and grey, so the app model stays the main line.
    if before is not None:
        fig.add_trace(go.Scatter(
            x=before["mean_predicted"], y=before["observed_rate"], mode="lines+markers",
            name="Before calibration", line={"color": PALETTE["neutral"], "width": 2, "dash": "dash"},
            marker={"size": 8, "symbol": "circle-open"},
            hovertemplate="Predicted %{x:.1%}<br>Really stopped %{y:.1%}<extra>Before calibration</extra>"))
    # After calibration: the app model, in the main data colour.
    fig.add_trace(go.Scatter(
        x=after["mean_predicted"], y=after["observed_rate"], mode="lines+markers",
        name="After calibration (the app model)", line={"color": PALETTE["bar"], "width": 3}, marker={"size": 9},
        hovertemplate="Predicted %{x:.1%}<br>Really stopped %{y:.1%}<extra>App model</extra>"))
    fig.update_layout(height=460, margin={"l": 70, "r": 10, "t": 30, "b": 70},  # room for both axis titles
                      legend={"orientation": "h", "y": 1.1, "x": 0})
    fig.update_xaxes(title="Predicted risk (average of each tenth of the test trials)", tickformat=".0%",
                     range=[0, top])
    fig.update_yaxes(title="Really stopped early", tickformat=".0%", range=[0, top])
    return fig


def gain_chart(curve, share):
    """Gain curve: which share of all stopped test trials is found if the riskiest part of the trials is reviewed.

    Shows the model's curve against the diagonal of picking at random. The point the user chose on the slider
    (share) is marked in the pink accent colour.
    """
    # The row of the curve that is closest to the slider value.
    point = curve.iloc[(curve["reviewed"] - share).abs().argmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["reviewed"], y=curve["reviewed"], mode="lines", name="Picking trials at random",
                             line={"color": PALETTE["neutral"], "dash": "dash", "width": 1.5}, hoverinfo="skip"))
    # The area between the model and the diagonal is the model's head start. Shading it (fill="tonexty") makes
    # the gap easy to see.
    fig.add_trace(go.Scatter(
        x=curve["reviewed"], y=curve["caught"], mode="lines", name="Riskiest first (the model)", fill="tonexty",
        fillcolor=alpha(PALETTE["bar"], 0.12), line={"color": PALETTE["bar"], "width": 3},
        hovertemplate="Review the riskiest %{x:.0%}<br>Find %{y:.0%} of the stopped trials<extra></extra>"))
    # The user's choice: one marker with a small label next to it.
    fig.add_trace(go.Scatter(
        x=[point["reviewed"]], y=[point["caught"]], mode="markers", name="Your choice",
        marker={"color": PALETTE["accent"], "size": 14, "line": {"color": PALETTE["ink"], "width": 2}},
        hoverinfo="skip"))
    fig.add_annotation(x=point["reviewed"], y=point["caught"], text=f"<b>{point['caught']:.0%}</b> found",
                       showarrow=False, xanchor="left", xshift=12, yanchor="top", bgcolor=PALETTE["ink"],
                       font={"color": PALETTE["paper"], "size": 13}, borderpad=5)
    fig.update_layout(height=420, margin={"l": 70, "r": 10, "t": 30, "b": 70},
                      legend={"orientation": "h", "y": 1.1, "x": 0})
    fig.update_xaxes(title="Share of test trials reviewed, riskiest first", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Share of all stopped trials found", tickformat=".0%", range=[0, 1.02])
    return fig


def score_violin(scores):
    """How the predicted risk is spread for test trials that completed and for those that were stopped early.

    A violin shows the whole spread (the box inside marks the middle half), so the reader sees how much the two
    groups overlap. That overlap is the honest picture behind a single ROC-AUC number.
    """
    # One half-violin per outcome. Completed comes first, so the stopped trials are drawn above it.
    outcomes = [("Completed", 0, PALETTE["neutral"]), ("Stopped early", 1, PALETTE["up"])]
    fig = go.Figure()
    for outcome, value, colour in outcomes:
        risks = scores.loc[scores["label"] == value, "risk"]
        fig.add_trace(go.Violin(
            x=risks, y=[outcome] * len(risks), name=outcome, orientation="h", side="positive", width=1.2,
            line={"color": colour}, fillcolor=alpha(colour, 0.35), points=False, spanmode="hard",
            box={"visible": True, "fillcolor": PALETTE["paper_2"], "line": {"color": colour}},
            meanline={"visible": True, "color": PALETTE["ink"]},
            hoveron="kde", hovertemplate="%{y}: risk %{x:.1%}<extra></extra>"))
    fig.update_layout(height=360, showlegend=False, margin={"l": 10, "r": 10, "t": 20, "b": 70},
                      violingap=0.1)
    # Only few trials have a very high risk. The axis ends at the 99th percentile, so the rest stays readable.
    fig.update_xaxes(title="Predicted risk of stopping early", tickformat=".0%",
                     range=[0, float(scores["risk"].quantile(0.99))])
    fig.update_yaxes(title=None)
    return fig


def importance_chart(importance):
    """Horizontal bars: how much the model relies on each factor.

    The colour of a bar and the words next to it say which way the factor goes in the training trials.
    """
    # Plotly draws the first row at the bottom. Turned around, the strongest factor ends up on top.
    ordered = importance.iloc[::-1]

    # Sort the direction texts into three colour groups.
    kinds = []
    for direction in ordered["Direction"]:
        if "more often" in direction:
            kinds.append("More stopped")
        elif "less often" in direction:
            kinds.append("Fewer stopped")
        else:
            kinds.append("No single direction")
    colours = {"More stopped": PALETTE["up"], "Fewer stopped": PALETTE["down"],
               "No single direction": PALETTE["neutral"]}

    fig = px.bar(
        ordered, x="roc_auc_drop_mean", y="Factor", orientation="h",
        color=kinds, color_discrete_map=colours,
        text=ordered["Direction"],
        labels={"roc_auc_drop_mean": "How much the model relies on it", "color": ""},
    )
    # The texts stand right of the bars. The x-axis is made longer, so the longest text still fits.
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=540,
                      legend={"orientation": "h", "y": -0.18, "x": 0})
    fig.update_xaxes(range=[0, importance["roc_auc_drop_mean"].max() * 2.6])
    return fig


def confusion_chart(counts):
    """The 2x2 confusion matrix as coloured cells: rows are what really happened, columns what the model said.

    The layout is the one of scikit-learn (CS course 6.4.2). Correct cells are green, false alarms yellow and
    missed stopped trials red, so the two kinds of error look different.
    """
    # z only picks the colour of a cell: 0 = correct, 1 = false alarm, 2 = missed.
    z = [[0, 1],
         [2, 0]]
    # The text in each cell: the number of trials in bold and what the cell means.
    text = [
        [f"<b>{counts['tn']:,}</b><br>Completed, not flagged",
         f"<b>{counts['fp']:,}</b><br>Completed, but flagged<br>(false alarm)"],
        [f"<b>{counts['fn']:,}</b><br>Stopped, but not flagged<br>(missed)",
         f"<b>{counts['tp']:,}</b><br>Stopped and flagged"],
    ]
    scale = [[0.0, PALETTE["low_tint"]], [0.5, PALETTE["mid_tint"]], [1.0, PALETTE["high_tint"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Model: not flagged", "Model: flagged"], y=["Really completed", "Really stopped early"],
        text=text, texttemplate="%{text}", textfont={"size": 14, "color": PALETTE["ink"]}, colorscale=scale,
        zmin=0, zmax=2, showscale=False, xgap=4, ygap=4, hoverinfo="skip"))
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    fig.update_yaxes(autorange="reversed")  # "Really completed" on top, as in scikit-learn's confusion_matrix
    return fig


def horse_race_chart(race):
    """Horizontal bars: the validation PR-AUC of every model after tuning, with the spread across the folds.

    The open circle is the same model with its standard settings and the dashed line is the baseline. So the
    reader sees how much tuning helped and how far each model is ahead of guessing. The winner is the dark bar.
    """
    horses = race["horses"]
    winner = race["verdict"]["winner"]
    # Plotly draws the first row at the bottom, so we turn the ranking around: the winner ends up on top.
    order = list(reversed(race["verdict"]["ranking"]))

    # One entry per model for every list the chart needs.
    names = []
    tuned_scores = []
    tuned_spreads = []
    default_scores = []
    bar_colours = []
    for name in order:
        horse = horses[name]
        names.append(horse["label"])
        tuned_scores.append(horse["tuned"]["cv_pr_auc"])
        tuned_spreads.append(horse["tuned"]["cv_pr_auc_std"])
        default_scores.append(horse["default"]["cv_pr_auc"])
        if name == winner:
            bar_colours.append(PALETTE["bar"])
        else:
            bar_colours.append(PALETTE["pale"])

    fig = go.Figure()
    # The bars: tuned score, with the spread across the folds as an error bar.
    fig.add_trace(go.Bar(
        y=names, x=tuned_scores, orientation="h", name="Tuned", marker_color=bar_colours,
        marker_line={"color": PALETTE["neutral"], "width": 1},
        error_x={"type": "data", "array": tuned_spreads, "color": PALETTE["ink"], "thickness": 1.5, "width": 4},
        hovertemplate="%{y}<br>Tuned PR-AUC %{x:.3f}<extra></extra>"))
    # The open circles: the score with the standard settings.
    fig.add_trace(go.Scatter(
        y=names, x=default_scores, mode="markers", name="Standard settings",
        marker={"symbol": "circle-open", "size": 11, "color": PALETTE["up"], "line": {"width": 2}},
        hovertemplate="%{y}<br>Standard settings PR-AUC %{x:.3f}<extra></extra>"))
    # The dashed line: what guessing scores.
    base = horses["baseline"]["tuned"]["cv_pr_auc"]
    fig.add_vline(x=base, line_dash="dash", annotation_text=f"Guessing: {base:.2f}", annotation_position="top")
    fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 60},
                      legend={"orientation": "h", "y": -0.2, "x": 0})
    fig.update_xaxes(title="PR-AUC on the later validation blocks (higher is better)", rangemode="tozero")
    fig.update_yaxes(title=None)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# The two boxes above the tabs
# ---------------------------------------------------------------------------------------------------------------

def in_short_box(metrics, phases):
    """The "In short" box: the three or four findings that the rest of the page explains.

    phases is the table from phase_table(), or None if the test trials could not be scored. Then the line about
    the phases is left out.
    """
    test = metrics["split"]["test"]
    top_group = metrics["test"]["calibration_bins"][-1]["observed_rate"]  # real rate in the riskiest tenth
    predicted = metrics["calibration"]["test_after"]["mean_predicted"]

    # In how many phases the ranking is weaker than over all trials, and which phase is the weakest.
    phase_line = ""
    if phases is not None:
        aucs = phases["ROC-AUC inside the phase"]
        weaker = int((aucs < metrics["test"]["roc_auc"]).sum())
        phase_line = (
            f"- **Inside most phases it ranks less well** than the overall {metrics['test']['roc_auc']:.2f}: "
            f"{weaker} of {len(phases)} phases are lower, the weakest is {phases.loc[aucs.idxmin(), 'Phase']} "
            f"with {aucs.min():.2f} (see \"How good is it?\").\n"
        )

    with st.container(border=True, key="short_box"):
        st.markdown(
            "**In short**\n\n"
            f"- **The ranking works.** The 10% of trials rated riskiest were stopped "
            f"{top_group / test['terminated_rate']:.1f}x as often as the average trial.\n"
            f"- **The percentages are too low.** On the test trials the model predicts {predicted:.1%} on average; "
            f"in reality {test['terminated_rate']:.1%} were stopped, about "
            f"{test['terminated_rate'] / predicted - 1:.0%} more.\n"
            f"{phase_line}"
            "- **It cannot see why trials stop**, such as funding, recruitment or safety problems."
        )


def read_a_score_box(metrics):
    """The "How to read a risk score" box: what the percentage, the colour and the rank mean, and what not."""
    # All numbers come from the metrics file, so the text follows a retrained model.
    test = metrics["split"]["test"]
    predicted = metrics["calibration"]["test_after"]["mean_predicted"]
    gap = test["terminated_rate"] / predicted - 1
    riskiest = metrics["test"]["calibration_bins"][-1]["observed_rate"]

    # How many past trials a score is compared with. This is fewer than all training trials: the earliest block
    # of the time-ordered folds is only trained on, so its trials never get a score (src/train.py).
    if "n" in metrics.get("score_reference", {}):
        reference_n = metrics["score_reference"]["n"]
    else:
        reference_n = metrics["split"]["train"]["n_trials"]

    legend = (chip("Lower than most past trials", "green") + chip("Higher than most past trials", "yellow")
              + chip("Among the riskiest 10%", "red"))
    with st.container(border=True, key="read_box"):
        st.markdown("**How to read a risk score**")
        st.markdown(
            "- **The percentage** is the model's estimate that a trial with this design ends up stopped early. "
            "\"15%\" means: of 100 past trials that looked like this on paper, about 15 were stopped.\n"
            f"- **It runs a little low.** On newer trials about {gap:.0%} more were really stopped than the model "
            "predicted, so read the number as a lower-end figure.\n"
            f"- **The colour and \"riskier than X%\"** compare the trial with {reference_n:,} "
            "past trials (started 2009-2016): green is below the "
            "middle trial, yellow above it, red the riskiest tenth. Use this to compare trials; it is the part the "
            "model is best at.\n"
            f"- **Red is not a verdict.** Even in the riskiest tenth only about {riskiest:.0%} of trials were "
            f"stopped; most of them still finished.\n"
            "- **It only knows the design**, not funding, recruitment or safety findings. Not medical advice."
        )
        st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------------------
# The six tabs, one function each
# ---------------------------------------------------------------------------------------------------------------

def tab_predicted(metrics):
    """Tab "What is predicted": what "stopped early" means and which trials are in the data."""
    # Add up both splits: the size of the whole data set.
    train = metrics["split"]["train"]
    test = metrics["split"]["test"]
    total = train["n_trials"] + test["n_trials"]
    terminated = train["n_terminated"] + test["n_terminated"]
    st.markdown(
        f"- **Stopped early** means the trial's ClinicalTrials.gov status is *Terminated*: it was stopped before "
        f"its planned end and will not resume. We compare it with trials whose status is *Completed*.\n"
        f"- The data holds **{total:,}** such trials that started {train['start_from'][:4]}-{test['start_to'][:4]}; "
        f"**{terminated / total:.1%}** of them ({terminated:,}) were stopped early.\n"
        "- The reason a trial stopped (money, too few patients, safety, business decisions) is not used and not "
        "predicted. Every reason counts the same.\n"
        "- Trials with other statuses are not in the data, for example withdrawn trials (nobody was enrolled) and "
        "trials that are still running. So the model has never seen a running trial; scoring one in the search is "
        "an extrapolation. Observational studies are not in the data either, so the search lists only "
        "interventional ones.\n"
        "- Trials from 2017-2019 that are still running today are missing as well, and that tilts the test trials "
        "towards stopped ones: a stopped trial ends early by definition, so it is already in the data, while a "
        "long trial that will complete is not yet. The later the start year, the stronger this pull (see "
        "\"Can you trust it?\")."
    )


def quality_numbers(metrics):
    """First part of the tab "How good is it?": the three test numbers, each next to what guessing would score."""
    train = metrics["split"]["train"]
    test = metrics["split"]["test"]
    result = metrics["test"]  # the numbers of the model on the test trials
    baseline = metrics["majority_baseline"]  # what guessing would score

    st.write(
        f"The model learned from trials that started {train['start_from'][:4]}-{train['start_to'][:4]} "
        f"({train['n_trials']:,} trials) and was then tested on **newer trials it never saw**, started "
        f"{test['start_from'][:4]}-{test['start_to'][:4]} ({test['n_trials']:,} trials). "
        f"{test['terminated_rate']:.1%} of them were stopped early."
    )
    # Three metric tiles, each with a sentence that says what the number means.
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{result['roc_auc']:.2f}", help="0.5 = guessing, 1.0 = perfect.")
    c1.caption(
        f"Pick one stopped and one completed trial at random: the model scores the stopped one higher "
        f"about {result['roc_auc']:.0%} of the time (guessing: {baseline['roc_auc']:.0%}). "
        "This is not an accuracy."
    )
    c2.metric("PR-AUC", f"{result['pr_auc']:.2f}",
              help="Higher is better; compare it with the share of stopped trials.")
    c2.caption(
        f"Guessing scores {baseline['pr_auc']:.2f} (the share of stopped trials). The model reaches "
        f"{result['pr_auc']:.2f}, about {result['pr_auc'] / baseline['pr_auc']:.1f}x that."
    )
    c3.metric("Brier score", f"{result['brier']:.3f}",
              help="Lower is better; measures how honest the percentages are.")
    # How much better the Brier score is than always predicting the average rate.
    gain = 1 - result["brier"] / baseline["brier_constant_rate"]
    c3.caption(
        f"Always guessing the average rate scores {baseline['brier_constant_rate']:.3f}. The model is only about "
        f"{gain:.0%} better, so a single percentage is a rough guide."
    )
    st.caption(
        f"We do not report accuracy: saying \"completed\" for every trial would already be right "
        f"{baseline['accuracy']:.0%} of the time."
    )

    # In plain words: the riskiest and the safest tenth of the test trials, and what happens at the threshold
    # with the best F1 (that threshold was chosen on the training trials).
    bins = result["calibration_bins"]
    f1_line = result["at_threshold"]
    st.markdown(
        f"**In plain words**\n\n"
        f"- The 10% of test trials the model rated riskiest were really stopped {bins[-1]['observed_rate']:.1%} "
        f"of the time, {bins[-1]['observed_rate'] / test['terminated_rate']:.1f}x the average. "
        f"The 10% rated safest: {bins[0]['observed_rate']:.1%}. The ranking works.\n"
        f"- But it is far from certain. If you flag every trial with a score of {f1_line['threshold']:.0%} or more "
        f"({f1_line['flagged_share']:.0%} of all trials), you catch {f1_line['recall']:.0%} of the trials that "
        f"really get stopped, and about {1 - f1_line['precision']:.0%} of the flagged trials finish anyway."
    )
    st.warning(
        "Honest summary: the model is clearly better than guessing, but it is not a crystal ball. "
        "Whether a trial gets stopped also depends on things nobody knows at registration, "
        "such as funding, recruitment problems or safety findings."
    )


def cost_slider_part(metrics, scores):
    """The cost slider: the user says what a missed stopped trial costs, and sees where the yes/no line then is.

    Where the line belongs depends on what each kind of mistake costs (DSF deck 11). The confusion matrix shows
    what that line does on the test trials (CS course 6.4.2, DSF deck 12).
    """
    ratio = st.slider("A missed stopped trial costs how many times as much as a false alarm?", 1, 20,
                      DEFAULT_COST_RATIO, key="cost_ratio")
    # A false alarm costs 1 and a missed stopped trial costs `ratio`. Then it pays to flag a trial if
    # P(stopped) > cost of a false alarm / (both costs together) = 1 / (1 + ratio).
    line = 1 / (1 + ratio)
    counts = confusion_counts(scores["label"].to_numpy(), scores["risk"].to_numpy(), line)
    n_flagged = counts["tp"] + counts["fp"]
    # Precision: share of the flagged trials that were really stopped (0 if nothing is flagged).
    precision = 0.0
    if n_flagged:
        precision = counts["tp"] / n_flagged
    # Recall: share of the really stopped trials that were flagged.
    recall = counts["tp"] / (counts["tp"] + counts["fn"])
    st.write(
        f"If a missed stopped trial costs {ratio}x a false alarm, it pays to flag every trial whose estimated "
        f"risk is above **1 / (1 + {ratio}) = {line:.0%}**. On the {len(scores):,} test trials that flags "
        f"**{n_flagged:,}** ({n_flagged / len(scores):.0%}). It finds **{recall:.0%}** of the trials that were "
        f"really stopped (recall), and **{precision:.0%}** of the flagged trials were really stopped (precision)."
    )
    chart(confusion_chart(counts), key="confusion_chart")
    f1_threshold = metrics["test"]["at_threshold"]["threshold"]
    st.caption(
        "Rows: what really happened; columns: what the model said at this line. The line comes from the costs, "
        "not from the model, so two users with different costs draw it in different places. It only works "
        "because the percentages are calibrated (see \"Can you trust it?\"); since they run a little low, the "
        f"real line sits slightly lower. For comparison: the line that maximises F1 on the training trials is "
        f"{f1_threshold:.0%}."
    )


def review_slider_part(scores):
    """The review slider: if the riskiest part of the trials is checked first, how many stopped trials are found?

    This is the decision an investor or sponsor really has: "I can look closely at only so many trials."
    """
    # The slider shows whole percent, the chart functions work with shares (0.2 = 20%).
    share = st.slider("Share of trials you have time to review", 5, 50, 20, step=5, format="%d%%",
                      key="gain_share") / 100
    # The gain curve over all test trials, and the row of it that is closest to the slider value.
    curve = gain_curve(scores["label"], scores["risk"])
    point = curve.iloc[(curve["reviewed"] - share).abs().argmin()]
    reviewed = int(round(share * len(scores)))
    st.write(
        f"Reviewing the riskiest **{share:.0%}** of the {len(scores):,} test trials ({reviewed:,} trials) finds "
        f"**{point['caught']:.0%}** of all trials that were really stopped. Picking {share:.0%} at random would "
        f"find {share:.0%}. Of the trials you would review, {point['precision']:.0%} were really stopped "
        f"(the average is {scores['label'].mean():.0%})."
    )
    chart(gain_chart(curve, share), key="gain_chart")
    st.caption("Move the slider to set your review budget. The further the green line runs above the dashed "
               "one, the more the ranking saves you work. The curve uses the 2017-2019 test trials only.")


def phase_part(metrics, phases):
    """The ranking quality inside each phase, as a sentence and a table.

    Ranking trials of the same phase is harder than ranking across phases, because the phase itself already says
    a lot about the risk.
    """
    # The best and the worst phase, and the phase where the predictions are too low by the most.
    aucs = phases["ROC-AUC inside the phase"]
    best = phases.loc[aucs.idxmax()]
    worst = phases.loc[aucs.idxmin()]
    gap = phases.loc[(phases["Really stopped"] / phases["Model predicted"]).idxmax()]
    st.write(
        f"No. Compared only with trials of the same phase, the ranking quality ranges from "
        f"{worst['ROC-AUC inside the phase']:.2f} ({worst['Phase']}) to {best['ROC-AUC inside the phase']:.2f} "
        f"({best['Phase']}), while the overall value is {metrics['test']['roc_auc']:.2f} and guessing gives 0.50. "
        f"So if you compare two trials of the same phase, treat a small difference with caution. "
        f"The predictions are also too low in every phase, most clearly in {gap['Phase']}: "
        f"{gap['Model predicted']:.1%} predicted against {gap['Really stopped']:.1%} really stopped."
    )
    # Turn the numbers into text for the table (thousands separator, percent). The row order stays.
    shown = phases.copy()
    shown["Test trials"] = [f"{value:,}" for value in phases["Test trials"]]
    shown["Model predicted"] = [f"{value:.1%}" for value in phases["Model predicted"]]
    shown["Really stopped"] = [f"{value:.1%}" for value in phases["Really stopped"]]
    shown["ROC-AUC inside the phase"] = [f"{value:.2f}" for value in aucs]
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(f"Groups with fewer than {MIN_GROUP:,} test trials are left out because their numbers are too noisy.")


def tab_quality(metrics, phases, scores):
    """Tab "How good is it?": the test numbers in plain words, two sliders, the score spread and the phases.

    scores are the scored test trials and phases the table from phase_table(). Both are None if the data or the
    model is missing, and the parts that need them then say so.
    """
    quality_numbers(metrics)

    st.markdown("##### From a score to a yes/no: where should the line be?")
    if scores is None:
        st.info("This part needs the trial data and the model file, which are not available here.")
    else:
        cost_slider_part(metrics, scores)

    st.markdown("##### If you can only check some trials, which ones?")
    if scores is None:
        st.info("This part needs the trial data and the model file, which are not available here.")
    else:
        review_slider_part(scores)

        # The spread of the scores, as two half-violins.
        st.markdown("##### How far apart are stopped and completed trials?")
        st.write("Each shape shows how the predicted risks of one group are spread; the box marks the middle half "
                 "and the dark line the average. Stopped trials sit further right, but the two shapes overlap a "
                 "lot. That overlap is why a single score is never a verdict.")
        chart(score_violin(scores), key="score_violin")

    st.markdown("##### Is it equally good in every phase?")
    if phases is None:
        st.info("The breakdown by phase needs the trial data and the model file, which are not available here.")
    else:
        phase_part(metrics, phases)


def tab_race():
    """Tab "Which model won?": six models on the same time-ordered folds, standard against tuned, and the winner.

    This is the horse race of DSF deck 16. The numbers are read from models/horse_race.json.
    """
    race = load_horse_race()
    if race is None:
        st.info("The model race is not available yet (run python src/horse_race.py to create models/horse_race.json).")
        return
    horses = race["horses"]
    verdict = race["verdict"]

    st.write(
        "We did not just pick a model. Six candidates ran a fair race: the same training trials (2008-2016), the "
        "same five checks and the same score. Each check trains on the earlier trials and tests on the next, later "
        "block, the way the app meets new trials. Every model ran with its standard settings and with a few tuned "
        "ones. The 2017-2019 test trials were not used here; they are kept for one final check of the winner."
    )
    chart(horse_race_chart(race), key="race_chart")

    # The table: one row per model, best first.
    rows = []
    for name in verdict["ranking"]:
        horse = horses[name]
        default = horse["default"]
        tuned = horse["tuned"]
        # The tuned settings as text, for example "max_depth=5, min_samples_leaf=50".
        settings = []
        for setting, value in tuned["params"].items():
            settings.append(f"{setting}={value}")
        settings_text = ", ".join(settings)
        if not settings_text:
            settings_text = "none"
        rows.append({
            "Model": horse["label"],
            "From the course": horse["course"],
            "PR-AUC, standard settings": f"{default['cv_pr_auc']:.3f} ± {default['cv_pr_auc_std']:.3f}",
            "PR-AUC, tuned": f"{tuned['cv_pr_auc']:.3f} ± {tuned['cv_pr_auc_std']:.3f}",
            "ROC-AUC, tuned": f"{tuned['cv_roc_auc']:.3f}",
            "Tuned settings": settings_text,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # The verdict in words. Rule of DSF deck 16: a lead only counts if it is larger than the spread across the
    # checks. Otherwise the two best models are practically tied.
    win = horses[verdict["winner"]]
    second = horses[verdict["runner_up"]]
    if verdict["lead_larger_than_spread"]:
        lead_text = (f"Its lead over {second['label'].lower()} ({verdict['lead']:.3f}) is larger than the spread "
                     f"across the five checks ({verdict['winner_spread']:.3f}), so the win is clear.")
    else:
        simplest = horses[verdict["simplest_tied"]]["label"].lower()
        lead_text = (f"Its lead over {second['label'].lower()} ({verdict['lead']:.3f}) is smaller than the spread "
                     f"across the five checks ({verdict['winner_spread']:.3f}), so the two are practically tied. "
                     f"By the race rules either may be chosen (the best score, or {simplest} as the one that needs "
                     "less tuning).")
        if verdict["winner"] == "gbm":
            lead_text += (" The app keeps gradient boosting: it scored best, handles missing values and categories "
                          "without extra steps, and its calibration and per-trial explanations are built on it.")
    st.write(f"**Winner: {win['label']}.** {lead_text}")
    st.caption(
        "± is the spread (standard deviation) across the five checks: how much a score depends on which later trials "
        "it is checked on. Guessing scores the share of stopped trials. The open circles show that tuning helps some "
        "models a lot (a single tree without limits learns the training trials by heart) and others hardly at all."
    )


def tab_factors():
    """Tab "What matters most": which factors the model relies on, and which way each one goes."""
    # The importance numbers come from models/feature_importance.csv. Without the file the tab says so.
    importance = load_importance()
    if importance is None or importance.empty:
        st.info("Feature importance is not available yet (models/feature_importance.csv is missing).")
        return
    st.write(
        "For each factor we shuffled its values in the test trials and measured how much worse the model "
        "ranks stopped trials. The longer the bar, the more the model relies on that factor. "
        "Colour and the words next to a bar say which way the factor goes in past trials."
    )
    chart(importance_chart(importance))
    st.caption(
        "The direction describes the 2008-2016 training trials for this factor on its own; the model combines "
        "factors, so a single case can differ. Important does not mean causal: changing the eligibility text of a "
        "real trial will not by itself make it more or less likely to finish."
    )


def tab_trust(metrics, scores):
    """Tab "Can you trust it?": how far the predicted percentages are from what really happened.

    Shows the gap over all test trials, per risk group, as a calibration plot and per start year. scores are the
    scored test trials or None.
    """
    # ---- The average prediction before and after calibration against the real rate ----
    mean_before = metrics["calibration"]["test_before"]["mean_predicted"]
    mean_after = metrics["calibration"]["test_after"]["mean_predicted"]
    real = metrics["split"]["test"]["terminated_rate"]
    st.write(
        f"Only partly. The raw model overestimated risk (average prediction {mean_before:.0%} when the real rate "
        f"was {real:.1%}), so it was recalibrated. That fixed most of the problem but not all of it: on the test "
        f"trials the average prediction is {mean_after:.1%}, while {real:.1%} were really stopped, about "
        f"{real / mean_after - 1:.0%} more. The chart shows that this is not just an average effect."
    )
    chart(risk_group_chart(metrics["test"]["calibration_bins"], real))
    st.caption(
        "In every one of the ten risk groups more trials were really stopped than the model predicted. "
        "The ranking is more reliable than the exact percentage."
    )

    # ---- The calibration plot ----
    # The curve of the app model is saved in the metrics file. The curve before calibration is computed here
    # from fresh scores, and only if the uncalibrated model could be loaded.
    st.markdown("##### Calibration plot: before and after")
    after_curve = pd.DataFrame(metrics["test"]["calibration_bins"])
    before_curve = None
    if scores is not None and scores["raw"].notna().all():
        before_curve = calibration_curve(scores["raw"], scores["label"])
    chart(calibration_chart(after_curve, before_curve), key="calibration_chart")
    caption = (
        "Each point is one tenth of the 2017-2019 test trials. On the dotted diagonal a predicted 20% would mean "
        f"20% really stopped. Before calibration the model said {mean_before:.0%} on average, because it "
        "was trained to pay extra attention to the rare stopped trials. Isotonic calibration (fitted on the training "
        "trials only) pulled the percentages down to the right scale; they now sit just under the diagonal, "
        "which is the \"slightly too low\" described above."
    )
    if before_curve is None:
        caption += " (The curve before calibration needs the uncalibrated model file.)"
    st.caption(caption)

    # ---- Per start year: the gap grows towards 2019 ----
    if scores is not None:
        years = year_table(scores)
        st.write("The gap also grew from year to year, so trials from today may be underestimated even more:")
        st.dataframe(pd.DataFrame({
            "Start year": years["year"].astype(str),
            "Test trials": [f"{value:,}" for value in years["trials"]],
            "Model predicted": [f"{value:.1%}" for value in years["predicted"]],
            "Really stopped": [f"{value:.1%}" for value in years["real"]],
            "Real rate higher by": [f"{value:+.0%}" for value in years["too_low"]],
        }), hide_index=True, width="stretch")
    st.write(
        "Why the gap? On paper the 2017-2019 trials look lower-risk (more trials without a phase, behavioral studies "
        "and studies with healthy volunteers than in 2008-2016), yet more of them were stopped. The most likely "
        "reason is in the data, not the model: only trials that have already ended are included. A stopped trial "
        "ends early by definition, so it is already in the data; a long trial that will complete is often still "
        "running and missing. The later the start year, the more completions are still missing, which matches the "
        "gap growing from 2017 to 2019. So part of the gap is a tilt in the test data. "
        "We have not yet counted the 2017-2019 trials that are still running, "
        "so it is a well-supported explanation, not a proof. Either way: read a percentage as a lower estimate and "
        "use it mainly to compare trials."
    )


def tab_versions(metrics):
    """Tab "Versions and checks": model v1 against v2, and the runs without the fields that can be edited later.

    It explains why the app uses the model it uses, and shows that the model does not depend on edited fields.
    """
    # ---- Version 1 or version 2 ----
    st.markdown("##### Version 1 or version 2?")
    v2 = load_metrics("gbm_v2")
    if v2 is None:
        st.info("The comparison needs models/gbm_v2_metrics.json, which is not available here.")
    else:
        # What v2 gains over v1 on the test trials and in cross-validation.
        gain = v2["test"]["roc_auc"] - metrics["test"]["roc_auc"]
        cv_gain = v2["cv"]["best_roc_auc_mean"] - metrics["cv"]["best_roc_auc_mean"]
        extra_inputs = len(v2["features"]["all"]) - len(metrics["features"]["all"])
        st.write(
            f"Version 2 adds {extra_inputs} inputs that sort the trial's "
            "conditions into areas (cancer, heart, brain and nerves, infections, metabolism, mental health, lungs, "
            "healthy volunteers) plus how many areas a trial touches. Everything else was kept identical: same "
            "split, same cross-validation, same calibration. Before training we fixed the rule: switch only if the "
            f"test ROC-AUC rises by at least {PROMOTION_BAR:.3f} and cross-validation agrees."
        )
        st.dataframe(version_table(metrics, v2), hide_index=True, width="stretch")
        # Check the gain against the rule that was fixed before training.
        if gain >= PROMOTION_BAR and cv_gain > 0:
            verdict = "meets"
            decision = "should switch to version 2"
        else:
            verdict = "misses"
            decision = "keeps version 1"
        st.caption(
            f"Version 2 gains {gain:+.4f} test ROC-AUC (cross-validation {cv_gain:+.4f}), which {verdict} the bar of "
            f"+{PROMOTION_BAR:.3f}, so the app {decision}. "
            "The PR-AUC gain is a little larger, mostly from cancer trials, "
            "which are stopped about twice as often as average."
        )

    # ---- Ablations: does the model lean on fields a sponsor can edit after the start? ----
    st.markdown("##### Does it rely on fields that can be edited later?")
    ablations = metrics.get("ablations")
    if not ablations:
        st.info("The ablation results are not in models/gbm_metrics.json.")
        return
    st.write(
        "ClinicalTrials.gov shows the latest version of each record, and a sponsor can change some fields after the "
        "trial has started: the eligibility text and the number of outcomes. If the model leaned on later edits, "
        "its scores would carry information from after the start. So we refitted it without those fields (same "
        "settings, uncalibrated) and scored the test trials once."
    )
    # st.table instead of st.dataframe, because in a plain table the long variant names wrap.
    st.table(ablation_table(ablations), hide_index=True)
    # The biggest loss of all runs (the most negative change).
    changes = [run["delta_test_roc_auc"] for run in ablations["runs"]]
    worst = min(changes)
    st.caption(
        f"Leaving all of them out costs {abs(worst):.3f} test ROC-AUC at most, so the model does not depend on "
        "them. These numbers are for the GBM before calibration, which is why the first row differs slightly from "
        "the app model's."
    )


# ---------------------------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------------------------

# The header. The first sentence uses the real numbers of trials if the metrics file exists.
metrics = load_metrics()
if metrics:
    split = metrics["split"]
    intro = (f"Trial-Radar looks only at facts known when a trial is registered. It learned from "
             f"{split['train']['n_trials']:,} past trials from ClinicalTrials.gov that had ended (completed or stopped "
             f"early) and was checked on {split['test']['n_trials']:,} newer ones.")
else:
    intro = ("Trial-Radar looks only at facts known when a trial is registered and compares them with past trials "
             "that had ended.")
page_header("How the model works", intro)
st.page_link("pages/2_Check_Your_Trial.py", label="Try it yourself: check your own trial")

if metrics:
    # Score the test trials (cached) and compute the phase table from them. Both are None if the trial data or
    # the model file is missing, and the parts of the page that need them then say so.
    scores = load_test_scores()
    phases = None
    if scores is not None:
        phases = phase_table(scores)

    in_short_box(metrics, phases)
    read_a_score_box(metrics)
    # Six tabs with one question each, in the order a reader asks them.
    predicted_tab, quality_tab, race_tab, factors_tab, trust_tab, versions_tab = st.tabs(
        ["What is predicted", "How good is it?", "Which model won?", "What matters most", "Can you trust it?",
         "Versions and checks"])
    with predicted_tab:
        tab_predicted(metrics)
    with quality_tab:
        tab_quality(metrics, phases, scores)
    with race_tab:
        tab_race()
    with factors_tab:
        tab_factors()
    with trust_tab:
        tab_trust(metrics, scores)
    with versions_tab:
        tab_versions(metrics)
else:
    # Without the metrics file only the factor chart can be shown, because it reads its own file.
    st.info("Model metrics are not available yet (models/gbm_metrics.json is missing).")
    tab_factors()

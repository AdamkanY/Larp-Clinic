"""Explain the model: what it predicts, how well it works on unseen trials, which factors matter, how far to trust it."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # repo root, for src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for ui
from src.labels import PHASE_ORDER, label  # noqa: E402
from ui import page_header, require_router  # noqa: E402

require_router("pages/2_Explain.py")

MODELS = ROOT / "models"
TOP_FEATURES = 12
MIN_GROUP = 1000  # smaller groups give ROC-AUC values too noisy to read
MIN_DIRECTION = 0.03  # rank correlations weaker than this count as "no clear direction"
ORANGE, BLUE, GREY = "#eb6834", "#2a78d6", "#64748b"  # each at least 3:1 against white

# Plain-English names for the model's input columns; anything missing falls back to a cleaned-up column name.
FRIENDLY = {
    "healthy_volunteers": "Accepts healthy volunteers",
    "eligibility_text_len": "Length of eligibility criteria",
    "max_age_years": "Maximum age",
    "min_age_years": "Minimum age",
    "phase": "Phase",
    "intervention_drug": "Tests a drug",
    "masking": "Blinding",
    "has_dmc": "Has a data monitoring committee",
    "sponsor_class": "Sponsor type",
    "intervention_behavioral": "Behavioral intervention",
    "intervention_model": "Intervention model",
    "n_primary_outcomes": "Number of primary outcomes",
    "n_secondary_outcomes": "Number of secondary outcomes",
    "intervention_device": "Tests a device",
    "primary_purpose": "Primary purpose",
    "n_conditions": "Number of conditions",
    "allocation": "Allocation (randomized or not)",
    "has_max_age": "Has a maximum age",
    "has_collaborators": "Has collaborators",
    "n_arms": "Number of arms",
}


@st.cache_data(show_spinner=False)
def load_metrics():
    """Metrics of the app model (models/gbm_metrics.json), or None if the file is missing."""
    try:
        return json.loads((MODELS / "gbm_metrics.json").read_text())
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_directions(features):
    """For each feature: does a higher value (or "yes") go with more or fewer stopped trials, in the training trials?

    Returns {feature: text}; None if the data file is missing. Numbers and yes/no flags use a rank correlation with
    the outcome; categories have no single direction. This describes the data on its own, not the model's reasoning.
    """
    try:
        from src.features import BINARY_FEATURES, CATEGORICAL_FEATURES
        data = pd.read_parquet(ROOT / "data" / "processed" / "trials.parquet",
                               columns=list(features) + ["split", "label"], filters=[("split", "==", "train")])
    except (ImportError, FileNotFoundError, OSError, ValueError, KeyError):
        return None
    out = {}
    for feature in features:
        if feature in CATEGORICAL_FEATURES:
            out[feature] = "depends on the category"
            continue
        rho = pd.to_numeric(data[feature], errors="coerce").corr(data["label"].astype(float), method="spearman")
        word = "yes" if feature in BINARY_FEATURES else "higher"
        if rho >= MIN_DIRECTION:
            out[feature] = f"{word} → stopped more often"
        elif rho <= -MIN_DIRECTION:
            out[feature] = f"{word} → stopped less often"
        else:
            out[feature] = "no clear direction on its own"
    return out


@st.cache_data(show_spinner=False)
def load_importance():
    """Permutation importance of the app model, strongest first, with a direction text, or None if the file is missing."""
    try:
        table = pd.read_csv(MODELS / "feature_importance.csv")
    except (OSError, ValueError):
        return None
    table = table[table["model"] == "gbm"].sort_values("rank").head(TOP_FEATURES).copy()
    table["Factor"] = [FRIENDLY.get(f, f.replace("_", " ").capitalize()) for f in table["feature"]]
    directions = load_directions(tuple(table["feature"]))
    table["Direction"] = [directions[f] if directions else "direction not available" for f in table["feature"]]
    return table


@st.cache_data(show_spinner="Scoring the test trials...")
def load_test_scores():
    """Score the held-out test trials (start 2017-2019) with the app model; None if data or model is missing.

    Scoring them here instead of storing a table keeps the phase and year numbers in step with whatever model
    is deployed.
    """
    try:
        from src.features import FEATURE_COLUMNS
        from src.predict import load_model, to_model_frame
        test = pd.read_parquet(
            ROOT / "data" / "processed" / "trials.parquet",
            columns=FEATURE_COLUMNS + ["split", "label", "start_date", "phase_label"],
            filters=[("split", "==", "test")],
        )
        risk = load_model().predict_proba(to_model_frame(test))[:, 1]
    except (ImportError, FileNotFoundError, OSError, ValueError, KeyError):
        return None
    return pd.DataFrame({
        "label": test["label"].astype(int).to_numpy(),
        "risk": risk,
        "year": pd.to_datetime(test["start_date"]).dt.year.to_numpy(),
        "phase": test["phase_label"].map(lambda v: label("phase", v)).to_numpy(),
    })


def phase_table(scores):
    """One row per phase with at least MIN_GROUP test trials: size, real rate, mean prediction, ROC-AUC inside the phase."""
    rows = []
    for phase, group in scores.groupby("phase"):
        if len(group) >= MIN_GROUP and group["label"].nunique() == 2:
            rows.append({
                "Phase": phase, "Test trials": len(group),
                "Model predicted": group["risk"].mean(), "Really stopped": group["label"].mean(),
                "ROC-AUC inside the phase": roc_auc_score(group["label"], group["risk"]),
            })
    table = pd.DataFrame(rows)
    rank = {name: i for i, name in enumerate(PHASE_ORDER)}
    return table.sort_values("Phase", key=lambda names: names.map(rank)).reset_index(drop=True)


def year_table(scores):
    """One row per start year of the test trials: size, real rate, mean prediction and how far the prediction is too low."""
    table = scores.groupby("year").agg(
        trials=("label", "size"), real=("label", "mean"), predicted=("risk", "mean")
    ).reset_index()
    table["too_low"] = table["real"] / table["predicted"] - 1
    return table


def risk_group_chart(bins, average):
    """Bars per risk group (10% of the test trials each): what the model predicted vs what really happened."""
    groups = [str(i) for i in range(1, len(bins) + 1)]
    fig = go.Figure([
        # Pale with a dark outline, so predicted and real bars differ in lightness and not only in hue.
        go.Bar(x=groups, y=[b["mean_predicted"] for b in bins], name="Model predicted",
               marker={"color": "#cbd5e1", "line": {"color": GREY, "width": 2}}),
        go.Bar(x=groups, y=[b["observed_rate"] for b in bins], name="Really stopped early", marker_color=BLUE),
    ])
    fig.add_hline(y=average, line_dash="dash", line_color="#475569",
                  annotation_text=f"Average of all test trials: {average:.1%}", annotation_position="top left")
    fig.update_layout(
        barmode="group", height=380, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        xaxis_title="Risk group: 10% of the test trials each, from lowest (1) to highest (10) predicted risk",
        xaxis_type="category",  # otherwise Plotly reads "1".."10" as numbers and labels only every second group
        yaxis={"tickformat": ".0%", "title": None},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def importance_chart(importance):
    """Horizontal bars of how much the model relies on each factor; colour and the words on the bar say which way it goes."""
    ordered = importance.iloc[::-1]
    kind = ordered["Direction"].map(
        lambda d: "More stopped" if "more often" in d else ("Fewer stopped" if "less often" in d else "No single direction"))
    fig = px.bar(
        ordered, x="roc_auc_drop_mean", y="Factor", orientation="h",
        color=kind, color_discrete_map={"More stopped": ORANGE, "Fewer stopped": BLUE, "No single direction": GREY},
        text=ordered["Direction"],
        labels={"roc_auc_drop_mean": "How much the model relies on it", "color": ""},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=540,
                      legend={"orientation": "h", "y": -0.18, "x": 0})
    fig.update_xaxes(range=[0, importance["roc_auc_drop_mean"].max() * 2.6])
    return fig


def tab_predicted(metrics):
    """What the model predicts and what is in the data."""
    train, test = metrics["split"]["train"], metrics["split"]["test"]
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
        "an extrapolation. Observational studies are not in the data either, so the search lists only interventional ones.\n"
        "- Trials from 2017-2019 that are still running today are missing as well. We have not checked whether "
        "that makes the recent years look worse than they are."
    )


def tab_quality(metrics, phases):
    """Headline numbers in plain words, then the phase breakdown."""
    train, test = metrics["split"]["train"], metrics["split"]["test"]
    result, baseline = metrics["test"], metrics["majority_baseline"]
    st.write(
        f"The model learned from trials that started {train['start_from'][:4]}-{train['start_to'][:4]} "
        f"({train['n_trials']:,} trials) and was then tested on **newer trials it never saw**, started "
        f"{test['start_from'][:4]}-{test['start_to'][:4]} ({test['n_trials']:,} trials). "
        f"{test['terminated_rate']:.1%} of them were stopped early."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{result['roc_auc']:.2f}", help="0.5 = guessing, 1.0 = perfect.")
    c1.caption(
        f"Pick one stopped and one completed trial at random: the model scores the stopped one higher "
        f"about {result['roc_auc']:.0%} of the time (guessing: {baseline['roc_auc']:.0%}). "
        "This is not an accuracy."
    )
    c2.metric("PR-AUC", f"{result['pr_auc']:.2f}", help="Higher is better; compare it with the share of stopped trials.")
    c2.caption(
        f"Guessing scores {baseline['pr_auc']:.2f} (the share of stopped trials). The model reaches "
        f"{result['pr_auc']:.2f}, about {result['pr_auc'] / baseline['pr_auc']:.1f}x that."
    )
    c3.metric("Brier score", f"{result['brier']:.3f}", help="Lower is better; measures how honest the percentages are.")
    gain = 1 - result["brier"] / baseline["brier_constant_rate"]
    c3.caption(
        f"Always guessing the average rate scores {baseline['brier_constant_rate']:.3f}. The model is only about "
        f"{gain:.0%} better, so a single percentage is a rough guide."
    )
    st.caption(
        f"We do not report accuracy: saying \"completed\" for every trial would already be right "
        f"{baseline['accuracy']:.0%} of the time."
    )

    bins = result["calibration_bins"]
    flagged = result["at_threshold"]
    st.markdown(
        f"**In plain words**\n\n"
        f"- The 10% of test trials the model rated riskiest were really stopped {bins[-1]['observed_rate']:.1%} "
        f"of the time, {bins[-1]['observed_rate'] / test['terminated_rate']:.1f}x the average. "
        f"The 10% rated safest: {bins[0]['observed_rate']:.1%}. The ranking works.\n"
        f"- But it is far from certain. If you flag every trial with a score of {flagged['threshold']:.0%} or more "
        f"({flagged['flagged_share']:.0%} of all trials), you catch {flagged['recall']:.0%} of the trials that "
        f"really get stopped, and about {1 - flagged['precision']:.0%} of the flagged trials finish anyway."
    )
    st.warning(
        "Honest summary: the model is clearly better than guessing, but it is not a crystal ball. "
        "Whether a trial gets stopped also depends on things nobody knows at registration, "
        "such as funding, recruitment problems or safety findings."
    )

    st.markdown("##### Is it equally good in every phase?")
    if phases is None:
        st.info("The breakdown by phase needs the trial data and the model file, which are not available here.")
        return
    aucs = phases["ROC-AUC inside the phase"]
    best, worst = phases.loc[aucs.idxmax()], phases.loc[aucs.idxmin()]
    gap = phases.loc[(phases["Really stopped"] / phases["Model predicted"]).idxmax()]
    st.write(
        f"No. Compared only with trials of the same phase, the ranking quality ranges from "
        f"{worst['ROC-AUC inside the phase']:.2f} ({worst['Phase']}) to {best['ROC-AUC inside the phase']:.2f} "
        f"({best['Phase']}), while the overall value is {result['roc_auc']:.2f} and guessing gives 0.50. "
        f"So if you compare two trials of the same phase, treat a small difference with caution. "
        f"The predictions are also too low in every phase, most clearly in {gap['Phase']}: "
        f"{gap['Model predicted']:.1%} predicted against {gap['Really stopped']:.1%} really stopped."
    )
    shown = phases.assign(**{
        "Test trials": phases["Test trials"].map("{:,}".format),
        "Model predicted": phases["Model predicted"].map("{:.1%}".format),
        "Really stopped": phases["Really stopped"].map("{:.1%}".format),
        "ROC-AUC inside the phase": aucs.map("{:.2f}".format),
    })
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(f"Groups with fewer than {MIN_GROUP:,} test trials are left out because their numbers are too noisy.")


def tab_factors():
    """Which factors the model relies on, and which way each one goes."""
    importance = load_importance()
    if importance is None or importance.empty:
        st.info("Feature importance is not available yet (models/feature_importance.csv is missing).")
        return
    st.write(
        "For each factor we shuffled its values in the test trials and measured how much worse the model "
        "ranks stopped trials. The longer the bar, the more the model relies on that factor. "
        "Colour and the words next to a bar say which way the factor goes in past trials."
    )
    st.plotly_chart(importance_chart(importance), width="stretch")
    st.caption(
        "The direction describes the 2008-2016 training trials for this factor on its own; the model combines "
        "factors, so a single case can differ. Important does not mean causal: changing the eligibility text of a "
        "real trial will not by itself make it more or less likely to finish."
    )


def tab_trust(metrics, scores):
    """How far the percentages can be trusted: the gap between predicted and real, overall, per risk group and per year."""
    test = metrics["split"]["test"]
    before = metrics["calibration"]["test_before"]["mean_predicted"]
    after = metrics["calibration"]["test_after"]["mean_predicted"]
    real = test["terminated_rate"]
    st.write(
        f"Only partly. The raw model overestimated risk (average prediction {before:.0%} when the real rate was "
        f"{real:.1%}), so it was recalibrated. That fixed most of the problem but not all of it: on the test "
        f"trials the average prediction is {after:.1%}, while {real:.1%} were really stopped, about "
        f"{real / after - 1:.0%} more. The chart shows that this is not just an average effect."
    )
    st.plotly_chart(risk_group_chart(metrics["test"]["calibration_bins"], real), width="stretch")
    st.caption(
        "In every one of the ten risk groups more trials were really stopped than the model predicted. "
        "The ranking is more reliable than the exact percentage."
    )
    if scores is not None:
        years = year_table(scores)
        st.write("The gap also grew from year to year, so trials from today may be underestimated even more:")
        st.dataframe(pd.DataFrame({
            "Start year": years["year"].astype(str),
            "Test trials": years["trials"].map("{:,}".format),
            "Model predicted": years["predicted"].map("{:.1%}".format),
            "Really stopped": years["real"].map("{:.1%}".format),
            "Real rate higher by": years["too_low"].map("{:+.0%}".format),
        }), hide_index=True, width="stretch")
    st.write(
        "One hint about the cause: on paper the 2017-2019 trials look lower-risk (more trials without a phase, "
        "behavioral studies and studies with healthy volunteers than in 2008-2016), yet they were stopped slightly "
        "more often. We have not found out why. Until then, read a percentage as a lower estimate and use it "
        "mainly to compare trials."
    )


metrics = load_metrics()
if metrics:
    split = metrics["split"]
    intro = (f"Trial-Radar looks only at facts known when a trial is registered. It learned from "
             f"{split['train']['n_trials']:,} past trials from ClinicalTrials.gov that had ended (completed or stopped "
             f"early) and was checked on {split['test']['n_trials']:,} newer ones.")
else:
    intro = "Trial-Radar looks only at facts known when a trial is registered and compares them with past trials that had ended."
page_header("🧠 How the model works", intro)
st.page_link("pages/5_Check_Your_Trial.py", label="Try it yourself: check your own trial", icon="🧪")

scores = load_test_scores() if metrics else None
phases = phase_table(scores) if scores is not None else None

if metrics:
    test = metrics["split"]["test"]
    top_group = metrics["test"]["calibration_bins"][-1]["observed_rate"]
    predicted = metrics["calibration"]["test_after"]["mean_predicted"]
    phase_line = ""
    if phases is not None:
        aucs = phases["ROC-AUC inside the phase"]
        weaker = int((aucs < metrics["test"]["roc_auc"]).sum())
        phase_line = (
            f"- **Inside most phases it ranks less well** than the overall {metrics['test']['roc_auc']:.2f}: "
            f"{weaker} of {len(phases)} phases are lower, the weakest is {phases.loc[aucs.idxmin(), 'Phase']} "
            f"with {aucs.min():.2f} (see \"How good is it?\").\n"
        )
    with st.container(border=True):
        st.markdown(
            "**In short**\n\n"
            f"- **The ranking works.** The 10% of trials rated riskiest were stopped "
            f"{top_group / test['terminated_rate']:.1f}x as often as the average trial.\n"
            f"- **The percentages are too low.** On the test trials the model predicts {predicted:.1%} on average; "
            f"in reality {test['terminated_rate']:.1%} were stopped, about {test['terminated_rate'] / predicted - 1:.0%} more.\n"
            f"{phase_line}"
            "- **It cannot see why trials stop**, such as funding, recruitment or safety problems."
        )
    predicted_tab, quality_tab, factors_tab, trust_tab = st.tabs(
        ["What is predicted", "How good is it?", "What matters most", "Can you trust it?"])
    with predicted_tab:
        tab_predicted(metrics)
    with quality_tab:
        tab_quality(metrics, phases)
    with factors_tab:
        tab_factors()
    with trust_tab:
        tab_trust(metrics, scores)
else:
    st.info("Model metrics are not available yet (models/gbm_metrics.json is missing).")
    tab_factors()

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
from ui import page_header  # noqa: E402

MODELS = ROOT / "models"
TOP_FEATURES = 12
MIN_GROUP = 1000  # smaller groups give ROC-AUC values too noisy to read
PHASE_ORDER = ["Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4", "Not Applicable"]

# Plain-English names for the model's input columns; anything missing falls back to a cleaned-up column name.
FRIENDLY = {
    "healthy_volunteers": "Accepts healthy volunteers",
    "eligibility_text_len": "Length of eligibility criteria",
    "max_age_years": "Maximum age",
    "min_age_years": "Minimum age",
    "phase": "Phase",
    "intervention_drug": "Tests a drug",
    "masking": "Masking (blinding)",
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
def load_importance():
    """Permutation importance of the app model, strongest first, or None if the file is missing."""
    try:
        table = pd.read_csv(MODELS / "feature_importance.csv")
    except (OSError, ValueError):
        return None
    table = table[table["model"] == "gbm"].sort_values("rank").head(TOP_FEATURES).copy()
    table["Factor"] = [FRIENDLY.get(f, f.replace("_", " ").capitalize()) for f in table["feature"]]
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
        "phase": test["phase_label"].fillna("Unknown").to_numpy(),
    })


def phase_table(scores):
    """One row per phase with at least MIN_GROUP test trials: size, real rate, mean prediction, ROC-AUC inside the phase."""
    rows = []
    for phase, group in scores.groupby("phase"):
        if len(group) >= MIN_GROUP and group["label"].nunique() == 2:
            rows.append({
                "Phase": phase, "Test trials": len(group),
                "Model predicted": group["risk"].mean(), "Really terminated": group["label"].mean(),
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
        go.Bar(x=groups, y=[b["mean_predicted"] for b in bins], name="Model predicted", marker_color="#94a3b8"),
        go.Bar(x=groups, y=[b["observed_rate"] for b in bins], name="Really terminated", marker_color="#1d4ed8"),
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


st.set_page_config(page_title="Why this prediction - Trial-Radar", page_icon="🧠", layout="wide")
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

if metrics:
    train, test = split["train"], split["test"]
    top_group = metrics["test"]["calibration_bins"][-1]["observed_rate"]
    predicted = metrics["calibration"]["test_after"]["mean_predicted"]
    scores = load_test_scores()
    phases = phase_table(scores) if scores is not None else None
    phase_line = ""
    if phases is not None:
        aucs = phases["ROC-AUC inside the phase"]
        weaker = int((aucs < metrics["test"]["roc_auc"]).sum())
        phase_line = (
            f"- **Inside most phases it ranks less well** than the overall {metrics['test']['roc_auc']:.2f}: "
            f"{weaker} of {len(phases)} phases are lower, the weakest is {phases.loc[aucs.idxmin(), 'Phase']} "
            f"with {aucs.min():.2f} (see the table below).\n"
        )
    with st.container(border=True):
        st.markdown(
            "**In short**\n\n"
            f"- **The ranking works.** The 10% of trials rated riskiest were terminated "
            f"{top_group / test['terminated_rate']:.1f}x as often as the average trial.\n"
            f"- **The percentages are too low.** On the test trials the model predicts {predicted:.1%} on average; "
            f"in reality {test['terminated_rate']:.1%} were terminated, about {test['terminated_rate'] / predicted - 1:.0%} more.\n"
            f"{phase_line}"
            "- **It cannot see why trials stop**, such as funding, recruitment or safety problems."
        )
    st.subheader("What exactly is predicted?")
    total = train["n_trials"] + test["n_trials"]
    terminated = train["n_terminated"] + test["n_terminated"]
    st.markdown(
        f"- **Terminated early** means the trial's ClinicalTrials.gov status is *Terminated*: it was stopped before "
        f"its planned end and will not resume. We compare it with trials whose status is *Completed*.\n"
        f"- The data holds **{total:,}** such trials that started {train['start_from'][:4]}-{test['start_to'][:4]}; "
        f"**{terminated / total:.1%}** of them ({terminated:,}) were terminated.\n"
        "- The reason a trial stopped (money, too few patients, safety, business decisions) is not used and not "
        "predicted. Every reason counts the same.\n"
        "- Trials with other statuses are not in the data, for example withdrawn trials (nobody was enrolled) and "
        "trials that are still running. So the model has never seen a running trial; scoring one in the search is "
        "an extrapolation.\n"
        "- Trials from 2017-2019 that are still running today are missing as well. We have not checked whether "
        "that makes the recent years look worse than they are."
    )

    result, baseline = metrics["test"], metrics["majority_baseline"]
    st.subheader("How good is it?")
    st.write(
        f"The model learned from trials that started {train['start_from'][:4]}-{train['start_to'][:4]} "
        f"({train['n_trials']:,} trials) and was then tested on **newer trials it never saw**, started "
        f"{test['start_from'][:4]}-{test['start_to'][:4]} ({test['n_trials']:,} trials). "
        f"{test['terminated_rate']:.1%} of them were terminated early."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{result['roc_auc']:.2f}", help="0.5 = guessing, 1.0 = perfect.")
    c1.caption(
        f"Pick one terminated and one completed trial at random: the model scores the terminated one higher "
        f"about {result['roc_auc']:.0%} of the time (guessing: {baseline['roc_auc']:.0%}). "
        "This is not an accuracy."
    )
    c2.metric("PR-AUC", f"{result['pr_auc']:.2f}", help="Higher is better; compare it with the share of terminated trials.")
    c2.caption(
        f"Guessing scores {baseline['pr_auc']:.2f} (the share of terminated trials). The model reaches "
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
        f"- The 10% of test trials the model rated riskiest were really terminated {bins[-1]['observed_rate']:.1%} "
        f"of the time, {bins[-1]['observed_rate'] / test['terminated_rate']:.1f}x the average. "
        f"The 10% rated safest: {bins[0]['observed_rate']:.1%}. The ranking works.\n"
        f"- But it is far from certain. If you flag every trial with a score of {flagged['threshold']:.0%} or more "
        f"({flagged['flagged_share']:.0%} of all trials), you catch {flagged['recall']:.0%} of the trials that "
        f"really get terminated, and about {1 - flagged['precision']:.0%} of the flagged trials finish anyway."
    )
    st.warning(
        "Honest summary: the model is clearly better than guessing, but it is not a crystal ball. "
        "Whether a trial gets terminated also depends on things nobody knows at registration, "
        "such as funding, recruitment problems or safety findings."
    )

    st.subheader("Is it equally good in every phase?")
    if scores is None:
        st.info("The breakdown by phase needs the trial data and the model file, which are not available here.")
    else:
        gap = phases.loc[(phases["Really terminated"] / phases["Model predicted"]).idxmax()]
        best, worst = phases.loc[phases["ROC-AUC inside the phase"].idxmax()], phases.loc[phases["ROC-AUC inside the phase"].idxmin()]
        st.write(
            f"No. Compared only with trials of the same phase, the ranking quality ranges from "
            f"{worst['ROC-AUC inside the phase']:.2f} ({worst['Phase']}) to {best['ROC-AUC inside the phase']:.2f} "
            f"({best['Phase']}), while the overall value is {result['roc_auc']:.2f} and guessing gives 0.50. "
            f"So if you compare two trials of the same phase, treat a small difference with caution. "
            f"The predictions are also too low in every phase, most clearly in {gap['Phase']}: "
            f"{gap['Model predicted']:.1%} predicted against {gap['Really terminated']:.1%} really terminated."
        )
        shown = phases.assign(**{
            "Test trials": phases["Test trials"].map("{:,}".format),
            "Model predicted": phases["Model predicted"].map("{:.1%}".format),
            "Really terminated": phases["Really terminated"].map("{:.1%}".format),
            "ROC-AUC inside the phase": phases["ROC-AUC inside the phase"].map("{:.2f}".format),
            "Phase": phases["Phase"].replace({"Not Applicable": "No phase (not applicable)"}),
        })
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption(f"Groups with fewer than {MIN_GROUP:,} test trials are left out because their numbers are too noisy.")
else:
    scores = None
    st.info("Model metrics are not available yet (models/gbm_metrics.json is missing).")

importance = load_importance()
st.subheader("Which factors matter most?")
if importance is None or importance.empty:
    st.info("Feature importance is not available yet (models/feature_importance.csv is missing).")
else:
    st.write(
        "For each factor we shuffled its values in the test trials and measured how much worse the model "
        "ranks terminated trials (drop in ROC-AUC). The bigger the drop, the more the model relies on that factor."
    )
    fig = px.bar(
        importance.iloc[::-1], x="roc_auc_drop_mean", y="Factor", orientation="h",
        error_x="roc_auc_drop_std", labels={"roc_auc_drop_mean": "Drop in ROC-AUC when shuffled"},
    )
    fig.update_layout(yaxis_title=None, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Important does not mean causal. These are patterns in past trials: for example, changing the "
        "eligibility text of a real trial will not by itself make it more or less likely to finish."
    )

st.subheader("Can you trust the percentages?")
if metrics:
    before = metrics["calibration"]["test_before"]["mean_predicted"]
    after = metrics["calibration"]["test_after"]["mean_predicted"]
    real = test["terminated_rate"]
    st.write(
        f"Only partly. The raw model overestimated risk (average prediction {before:.0%} when the real rate was "
        f"{real:.1%}), so it was recalibrated. That fixed most of the problem but not all of it: on the test "
        f"trials the average prediction is {after:.1%}, while {real:.1%} were really terminated, about "
        f"{real / after - 1:.0%} more. The chart shows that this is not just an average effect."
    )
    st.plotly_chart(risk_group_chart(bins, real), width="stretch")
    st.caption(
        "In every one of the ten risk groups more trials were really terminated than the model predicted. "
        "The ranking is more reliable than the exact percentage."
    )
    if scores is not None:
        years = year_table(scores)
        st.write("The gap also grew from year to year, so trials from today may be underestimated even more:")
        st.dataframe(pd.DataFrame({
            "Start year": years["year"].astype(str),
            "Test trials": years["trials"].map("{:,}".format),
            "Model predicted": years["predicted"].map("{:.1%}".format),
            "Really terminated": years["real"].map("{:.1%}".format),
            "Real rate higher by": years["too_low"].map("{:+.0%}".format),
        }), hide_index=True, width="stretch")
    st.write(
        "One hint about the cause: on paper the 2017-2019 trials look lower-risk (more trials without a phase, "
        "behavioral studies and studies with healthy volunteers than in 2008-2016), yet they were terminated slightly more often. We have not found "
        "out why. Until then, read a percentage as a lower estimate and use it mainly to compare trials."
    )
else:
    st.write("Calibration details appear here once the metrics file is available.")

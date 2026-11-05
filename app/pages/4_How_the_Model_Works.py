"""How the model works: what it predicts, how to read a score, how well it works on unseen trials, which factors
matter, how far to trust the percentages (calibration), and which model versions and checks were run."""
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
from src.labels import PHASE_ORDER, feature_name, label  # noqa: E402
from ui import PALETTE, chart, chip, page_header, require_router  # noqa: E402

require_router("pages/4_How_the_Model_Works.py")

MODELS = ROOT / "models"
TOP_FEATURES = 12
MIN_GROUP = 1000  # smaller groups give ROC-AUC values too noisy to read
MIN_DIRECTION = 0.03  # rank correlations weaker than this count as "no clear direction"
# Rule fixed before model v2 was trained (models/CLAUDE.md): replace the app model only if the calibrated GBM's test
# ROC-AUC improves by at least this much and cross-validation agrees.
PROMOTION_BAR = 0.005

@st.cache_data(show_spinner=False)
def load_metrics(name="gbm"):
    """Metrics of a saved model (models/<name>_metrics.json; "gbm" is the app model), or None if the file is missing."""
    try:
        return json.loads((MODELS / f"{name}_metrics.json").read_text())
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
    table["Factor"] = [feature_name(f) for f in table["feature"]]
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
        from src.predict import load_model, score_frame
        test = pd.read_parquet(
            ROOT / "data" / "processed" / "trials.parquet",
            columns=FEATURE_COLUMNS + ["split", "label", "start_date", "phase_label"],
            filters=[("split", "==", "test")],
        )
        risk = score_frame(test)
    except (ImportError, FileNotFoundError, OSError, ValueError, KeyError):
        return None
    # The GBM before calibration, only for the before/after calibration plot; the page works without it.
    try:
        base = (load_metrics() or {}).get("calibration", {}).get("base_model", "gbm_uncalibrated")
        raw = score_frame(test, load_model(f"models/{base}.joblib"))
    except (FileNotFoundError, OSError, ValueError, KeyError):
        raw = None
    return pd.DataFrame({
        "label": test["label"].astype(int).to_numpy(),
        "risk": risk,
        "raw": raw if raw is not None else float("nan"),
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
               marker={"color": PALETTE["pale"], "line": {"color": PALETTE["neutral"], "width": 2}}),
        go.Bar(x=groups, y=[b["observed_rate"] for b in bins], name="Really stopped early",
               marker_color=PALETTE["bar"]),
    ])
    fig.add_hline(y=average, line_dash="dash",
                  annotation_text=f"Average of all test trials: {average:.1%}", annotation_position="top left")
    fig.update_layout(
        barmode="group", height=420, margin={"l": 10, "r": 0, "t": 30, "b": 70},  # b: room for the axis title
        xaxis_title="Risk group: 10% of the test trials each, from lowest (1) to highest (10) predicted risk",
        xaxis_type="category",  # otherwise Plotly reads "1".."10" as numbers and labels only every second group
        yaxis={"tickformat": ".0%", "title": None},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def calibration_curve(predicted, observed, groups=10):
    """Mean prediction and real share stopped early per group of equally many trials, lowest predictions first."""
    frame = pd.DataFrame({"p": predicted, "y": observed}).dropna()
    frame["group"] = pd.qcut(frame["p"].rank(method="first"), groups, labels=False)
    return frame.groupby("group").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index(drop=True)


def calibration_chart(after, before=None):
    """Calibration plot: predicted risk (x) against the real share stopped early (y) per tenth of the test trials.

    A model whose percentages can be taken literally sits on the diagonal. `after` is the app model, `before` the
    same GBM before isotonic calibration (both DataFrames with mean_predicted and observed_rate).
    """
    top = max(after["mean_predicted"].max(), after["observed_rate"].max())
    if before is not None:
        top = max(top, before["mean_predicted"].max(), before["observed_rate"].max())
    top = min(1.0, top * 1.08)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line={"dash": "dot", "width": 1.5})
    fig.add_annotation(x=top * 0.97, y=top * 0.97, text="Percentages exactly right", showarrow=False,
                       xanchor="right", yanchor="bottom", textangle=0)
    if before is not None:
        fig.add_trace(go.Scatter(
            x=before["mean_predicted"], y=before["observed_rate"], mode="lines+markers",
            name="Before calibration", line={"color": PALETTE["neutral"], "width": 2, "dash": "dash"},
            marker={"size": 8, "symbol": "circle-open"},
            hovertemplate="Predicted %{x:.1%}<br>Really stopped %{y:.1%}<extra>Before calibration</extra>"))
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


def importance_chart(importance):
    """Horizontal bars of how much the model relies on each factor; colour and the words on the bar say which way it goes."""
    ordered = importance.iloc[::-1]
    kind = ordered["Direction"].map(
        lambda d: "More stopped" if "more often" in d else ("Fewer stopped" if "less often" in d else "No single direction"))
    fig = px.bar(
        ordered, x="roc_auc_drop_mean", y="Factor", orientation="h",
        color=kind, color_discrete_map={"More stopped": PALETTE["up"], "Fewer stopped": PALETTE["down"],
                                        "No single direction": PALETTE["neutral"]},
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
    chart(importance_chart(importance))
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
    chart(risk_group_chart(metrics["test"]["calibration_bins"], real))
    st.caption(
        "In every one of the ten risk groups more trials were really stopped than the model predicted. "
        "The ranking is more reliable than the exact percentage."
    )

    st.markdown("##### Calibration plot: before and after")
    after = pd.DataFrame(metrics["test"]["calibration_bins"])
    before = None
    if scores is not None and scores["raw"].notna().all():
        before = calibration_curve(scores["raw"], scores["label"])
    chart(calibration_chart(after, before), key="calibration_chart")
    st.caption(
        "Each point is one tenth of the 2017-2019 test trials. On the dotted diagonal a predicted 20% would mean "
        f"20% really stopped. Before calibration the model said {before_mean_text(metrics)} on average, because it "
        "was trained to pay extra attention to the rare stopped trials. Isotonic calibration (fitted on the training "
        "trials only) pulled the percentages down to the right scale; they now sit just under the diagonal, "
        "which is the \"slightly too low\" described above."
        + ("" if before is not None else " (The curve before calibration needs the uncalibrated model file.)")
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


def before_mean_text(metrics):
    """The raw model's average test prediction in words, for the calibration caption."""
    return f"{metrics['calibration']['test_before']['mean_predicted']:.0%}"


def read_a_score_box(metrics):
    """Plain-English box: what the percentage, the colour and the rank of a risk score mean, and what they do not."""
    test = metrics["split"]["test"]
    predicted = metrics["calibration"]["test_after"]["mean_predicted"]
    gap = test["terminated_rate"] / predicted - 1
    riskiest = metrics["test"]["calibration_bins"][-1]["observed_rate"]
    legend = (chip("Lower than most past trials", "green") + chip("Higher than most past trials", "yellow")
              + chip("Among the riskiest 10%", "red"))
    with st.container(border=True, key="read_box"):
        st.markdown("**How to read a risk score**")
        st.markdown(
            "- **The percentage** is the model's estimate that a trial with this design ends up stopped early. "
            "\"15%\" means: of 100 past trials that looked like this on paper, about 15 were stopped.\n"
            f"- **It runs a little low.** On newer trials about {gap:.0%} more were really stopped than the model "
            "predicted, so read the number as a lower-end figure.\n"
            f"- **The colour and \"riskier than X%\"** compare the trial with {metrics['split']['train']['n_trials']:,} "
            "past trials (started 2008-2016): green is below the "
            "middle trial, yellow above it, red the riskiest tenth. Use this to compare trials; it is the part the "
            "model is best at.\n"
            f"- **Red is not a verdict.** Even in the riskiest tenth only about {riskiest:.0%} of trials were "
            f"stopped; most of them still finished.\n"
            "- **It only knows the design**, not funding, recruitment or safety findings. Not medical advice."
        )
        st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)


def version_table(v1, v2):
    """v1 (the app model) against v2 (plus condition categories): features, CV and test numbers, which one is used."""
    rows = []
    for name, metrics, status in [("v1: design features", v1, "Used in the app"),
                                  ("v2: plus condition areas", v2, "Tested, not used")]:
        if metrics is None:
            continue
        rows.append({
            "Model": name, "Inputs": len(metrics["features"]["all"]),
            "CV ROC-AUC (training trials)": f"{metrics['cv']['best_roc_auc_mean']:.3f}",
            "Test ROC-AUC": f"{metrics['test']['roc_auc']:.4f}", "Test PR-AUC": f"{metrics['test']['pr_auc']:.4f}",
            "Test Brier": f"{metrics['test']['brier']:.4f}", "Status": status,
        })
    return pd.DataFrame(rows)


def ablation_table(ablations):
    """One row per ablation run: which inputs were left out and what that cost on the test trials."""
    rows = []
    for run in ablations["runs"]:
        dropped = run["dropped"]
        rows.append({
            "Variant": "All inputs" if not dropped else "Without " + ", ".join(feature_name(f).lower() for f in dropped),
            "Inputs": run["n_features"],
            "CV ROC-AUC (training trials)": f"{run['cv_roc_auc']:.3f}",
            "Test ROC-AUC": f"{run['test_roc_auc']:.3f}",
            "Change in test ROC-AUC": "–" if not dropped else f"{run['delta_test_roc_auc']:+.3f}",
            "Test PR-AUC": f"{run['test_pr_auc']:.3f}",
        })
    return pd.DataFrame(rows)


def tab_versions(metrics):
    """Model v1 vs v2 and the ablations: why the app uses the model it uses, and that it does not hang on edited fields."""
    v2 = load_metrics("gbm_v2")
    st.markdown("##### Version 1 or version 2?")
    if v2 is None:
        st.info("The comparison needs models/gbm_v2_metrics.json, which is not available here.")
    else:
        gain = v2["test"]["roc_auc"] - metrics["test"]["roc_auc"]
        cv_gain = v2["cv"]["best_roc_auc_mean"] - metrics["cv"]["best_roc_auc_mean"]
        st.write(
            f"Version 2 adds {len(v2['features']['all']) - len(metrics['features']['all'])} inputs that sort the trial's "
            "conditions into areas (cancer, heart, brain and nerves, infections, metabolism, mental health, lungs, "
            "healthy volunteers) plus how many areas a trial touches. Everything else was kept identical: same "
            "split, same cross-validation, same calibration. Before training we fixed the rule: switch only if the "
            f"test ROC-AUC rises by at least {PROMOTION_BAR:.3f} and cross-validation agrees."
        )
        st.dataframe(version_table(metrics, v2), hide_index=True, width="stretch")
        verdict = "meets" if gain >= PROMOTION_BAR and cv_gain > 0 else "misses"
        st.caption(
            f"Version 2 gains {gain:+.4f} test ROC-AUC (cross-validation {cv_gain:+.4f}), which {verdict} the bar of "
            f"+{PROMOTION_BAR:.3f}, so the app {'should switch to version 2' if verdict == 'meets' else 'keeps version 1'}. "
            "The PR-AUC gain is a little larger, mostly from cancer trials, "
            "which are stopped about twice as often as average."
        )
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
    st.table(ablation_table(ablations), hide_index=True)  # a table, not a grid: the long variant names wrap
    worst = min(run["delta_test_roc_auc"] for run in ablations["runs"])
    st.caption(
        f"Leaving all of them out costs {abs(worst):.3f} test ROC-AUC at most, so the model does not depend on "
        "them. These numbers are for the GBM before calibration, which is why the first row differs slightly from "
        "the app model's."
    )


metrics = load_metrics()
if metrics:
    split = metrics["split"]
    intro = (f"Trial-Radar looks only at facts known when a trial is registered. It learned from "
             f"{split['train']['n_trials']:,} past trials from ClinicalTrials.gov that had ended (completed or stopped "
             f"early) and was checked on {split['test']['n_trials']:,} newer ones.")
else:
    intro = "Trial-Radar looks only at facts known when a trial is registered and compares them with past trials that had ended."
page_header("How the model works", intro)
st.page_link("pages/2_Check_Your_Trial.py", label="Try it yourself: check your own trial")

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
    with st.container(border=True, key="short_box"):
        st.markdown(
            "**In short**\n\n"
            f"- **The ranking works.** The 10% of trials rated riskiest were stopped "
            f"{top_group / test['terminated_rate']:.1f}x as often as the average trial.\n"
            f"- **The percentages are too low.** On the test trials the model predicts {predicted:.1%} on average; "
            f"in reality {test['terminated_rate']:.1%} were stopped, about {test['terminated_rate'] / predicted - 1:.0%} more.\n"
            f"{phase_line}"
            "- **It cannot see why trials stop**, such as funding, recruitment or safety problems."
        )
    read_a_score_box(metrics)
    predicted_tab, quality_tab, factors_tab, trust_tab, versions_tab = st.tabs(
        ["What is predicted", "How good is it?", "What matters most", "Can you trust it?", "Versions and checks"])
    with predicted_tab:
        tab_predicted(metrics)
    with quality_tab:
        tab_quality(metrics, phases)
    with factors_tab:
        tab_factors()
    with trust_tab:
        tab_trust(metrics, scores)
    with versions_tab:
        tab_versions(metrics)
else:
    st.info("Model metrics are not available yet (models/gbm_metrics.json is missing).")
    tab_factors()

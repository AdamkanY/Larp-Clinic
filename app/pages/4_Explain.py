"""Explain the model: how well it works on unseen trials and which design factors matter most."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for ui
from ui import page_header  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "models"
TOP_FEATURES = 12

# Plain-English names for every model input column (src.features.FEATURE_COLUMNS); anything missing falls back
# to a cleaned-up column name.
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
    "sex": "Sex eligible",
    "intervention_biological": "Tests a biological (e.g. vaccine)",
    "intervention_procedure": "Tests a procedure",
    "intervention_radiation": "Tests radiation",
    "intervention_genetic": "Tests a genetic intervention",
    "intervention_dietary_supplement": "Tests a dietary supplement",
    "intervention_combination_product": "Tests a combination product",
    "intervention_diagnostic_test": "Tests a diagnostic test",
    "intervention_other": "Other intervention type",
    "n_intervention_types": "Number of intervention types",
    "cond_oncology": "Condition: cancer",
    "cond_cardio": "Condition: heart and blood vessels",
    "cond_neuro": "Condition: brain and nerves",
    "cond_infectious": "Condition: infection",
    "cond_metabolic": "Condition: diabetes and metabolism",
    "cond_psych": "Condition: mental health",
    "cond_respiratory": "Condition: lungs and breathing",
    "cond_healthy": "Condition: healthy volunteers",
    "n_condition_categories": "Number of condition areas",
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


st.set_page_config(page_title="Why this prediction - Trial-Radar", page_icon="🧠", layout="wide")
page_header("🧠 How the model works",
            "Trial-Radar looks only at facts known when a trial is registered and compares them with about 160,000 finished trials from ClinicalTrials.gov.")
st.page_link("pages/5_Check_Your_Trial.py", label="Try it yourself: check your own trial", icon="🧪")

metrics = load_metrics()
if metrics is None:
    st.info("Model metrics are not available yet (models/gbm_metrics.json is missing).")
else:
    train, test = metrics["split"]["train"], metrics["split"]["test"]
    result, baseline = metrics["test"], metrics["majority_baseline"]

    st.subheader("How good is it?")
    st.write(
        f"The model learned from trials that started {train['start_from'][:4]}-{train['start_to'][:4]} "
        f"({train['n_trials']:,} trials) and was then tested on **newer trials it never saw**, started "
        f"{test['start_from'][:4]}-{test['start_to'][:4]} ({test['n_trials']:,} trials). "
        f"About {test['terminated_rate']:.0%} of them were terminated early."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{result['roc_auc']:.2f}", help="0.5 = guessing, 1.0 = perfect.")
    c1.caption(
        f"Pick one terminated and one completed trial at random: the model scores the terminated one higher "
        f"about {result['roc_auc']:.0%} of the time (guessing: {baseline['roc_auc']:.0%})."
    )
    c2.metric("PR-AUC", f"{result['pr_auc']:.2f}", help="Compared with the share of terminated trials.")
    c2.caption(
        f"Among the trials it ranks as riskiest, about {result['pr_auc'] / baseline['pr_auc']:.1f}x as many "
        f"are really terminated as by chance (guessing gives {baseline['pr_auc']:.2f})."
    )
    c3.metric("Brier score", f"{result['brier']:.3f}", help="Lower is better; measures how honest the percentages are.")
    c3.caption(f"Always guessing the average rate would score {baseline['brier_constant_rate']:.3f}.")
    st.warning(
        "Honest summary: the model is clearly better than guessing, but it is not a crystal ball. "
        "Whether a trial gets terminated also depends on things nobody knows at registration, "
        "such as funding, recruitment problems or safety findings. It is also less sharp between trials of the "
        "same phase than between different kinds of trials, so it ranks trials better than it predicts single outcomes."
    )

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
    real = metrics["split"]["test"]["terminated_rate"]
    st.write(
        f"Only partly. The raw model overestimated risk (average prediction {before:.0%} when the real rate was "
        f"{real:.0%}), so it was recalibrated. That fixed most of the problem but not all of it: on the newer test "
        f"trials the average prediction is {after:.1%}, while {real:.1%} were really terminated, about "
        f"{real / after - 1:.0%} more. Read the percentages as a lower-end figure and use them mainly to compare "
        "trials with each other."
    )
else:
    st.write("Calibration details appear here once the metrics file is available.")
if (MODELS / "calibration_before_after.png").exists():
    st.image(str(MODELS / "calibration_before_after.png"), caption="Calibration on the test trials, before and after recalibration.")

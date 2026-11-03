"""Test-set evaluation of the saved models: metrics, calibration, importance.

Everything that touches the test split (start 2017-2019) lives here, so there is
one code path for it. ``train.py`` calls ``evaluate_saved`` once after saving the
models; running this file re-evaluates the saved models without retraining.

For each model in ``models/`` this:
  * computes test ROC-AUC, PR-AUC and Brier score, plus precision / recall / F1
    and the confusion matrix at the threshold stored in its metrics JSON (chosen
    on train CV, never on test), and the majority-class baseline;
  * writes those numbers into ``models/<name>_metrics.json`` under ``test``;
  * draws ``models/calibration_curve.png`` for all models;
  * computes permutation importance on test -> ``models/feature_importance.csv``;
  * prints a metrics table and the logistic regression coefficients.

Usage:
    python src/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_COLUMNS, NUMERIC_FEATURES  # noqa: E402
from predict import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CALIBRATION_PATH = MODELS_DIR / "calibration_curve.png"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

MODEL_NAMES = ["baseline_lr", "gbm"]
DISPLAY_NAMES = {"baseline_lr": "Logistic regression", "gbm": "Gradient boosting"}
RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 10
N_CALIBRATION_BINS = 10

# Chart tokens (validated categorical slots 1-2 plus neutral chrome), light mode
# because the PNG goes into reports and slides.
SERIES_COLORS = {"baseline_lr": "#2a78d6", "gbm": "#eb6834"}
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def metrics_path(name: str) -> Path:
    """Path of the metrics JSON that belongs to model ``name``."""
    return MODELS_DIR / f"{name}_metrics.json"


def best_f1_threshold(y_true, proba) -> tuple:
    """Threshold on ``proba`` that maximises F1, and that F1.

    Meant for out-of-fold *train* predictions; applying the result to test
    keeps the test split out of every decision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall pair has no threshold (it is the "predict
    # nothing positive" end of the curve), so drop it before aligning.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def _confusion(y_true, y_pred) -> dict:
    """Confusion matrix as named counts (positive class = TERMINATED)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Ranking metrics plus thresholded metrics for one model on one split."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted": float(np.mean(proba)),
        "at_threshold": {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "flagged_share": float(y_pred.mean()),
            "confusion_matrix": _confusion(y_true, y_pred),
        },
    }


def majority_baseline(y_true) -> dict:
    """Scores of a model that calls every trial COMPLETED.

    A constant score ranks nothing, so ROC-AUC is 0.5 and PR-AUC equals the
    terminated rate. Any real model has to beat both.
    """
    y_true = np.asarray(y_true)
    rate = float(y_true.mean())
    return {
        "roc_auc": 0.5,
        "pr_auc": rate,
        "terminated_rate": rate,
        "accuracy": 1 - rate,
        "confusion_matrix": _confusion(y_true, np.zeros_like(y_true)),
    }


def calibration_bins(y_true, proba, n_bins: int = N_CALIBRATION_BINS) -> list:
    """Quantile-binned calibration points (the table behind the plot).

    Quantile bins hold equal numbers of trials, so every point rests on about
    the same evidence even though most predictions are small.
    """
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return [{"mean_predicted": float(p), "observed_rate": float(o)}
            for p, o in zip(predicted, observed)]


def plot_calibration(y_true, probas: dict, path: Path = CALIBRATION_PATH) -> None:
    """Save a calibration curve (top) and prediction histogram (bottom) per model.

    Args:
        y_true: test labels.
        probas: {model name: test probabilities}.
        path: output PNG.
    """
    # Imported here so the app can use the metric helpers without matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    base_rate = float(np.mean(y_true))

    ax_cal.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax_cal.text(0.97, 0.93, "perfectly calibrated", color=MUTED, fontsize=8.5,
                ha="right", va="bottom", rotation=0)
    ax_cal.axhline(base_rate, color=AXIS, linewidth=1)
    ax_cal.text(0.01, base_rate + 0.012, f"test terminated rate {base_rate:.1%}",
                color=INK_2, fontsize=8.5, va="bottom")

    bins = np.linspace(0, 1, 41)
    curves = {}
    for name, proba in probas.items():
        points = calibration_bins(y_true, proba)
        xs = [p["mean_predicted"] for p in points]
        ys = [p["observed_rate"] for p in points]
        curves[name] = (xs[-1], ys[-1])
        color = SERIES_COLORS.get(name, INK_2)
        ax_cal.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=DISPLAY_NAMES.get(name, name), zorder=3)
        ax_hist.hist(proba, bins=bins, histtype="step", linewidth=2, color=color,
                     weights=np.full(len(proba), 100 / len(proba)))
    # Direct labels at the curve ends, in ink (the line colour carries identity).
    # The ends sit close together, so the higher one is nudged up, the rest down.
    top = max(curves, key=lambda n: curves[n][1])
    for name, end in curves.items():
        ax_cal.annotate(DISPLAY_NAMES.get(name, name), end,
                        xytext=(8, 9 if name == top else -9), textcoords="offset points",
                        color=INK, fontsize=9, va="center")

    ax_cal.set_ylim(0, 1)
    ax_cal.set_ylabel("Observed share terminated", color=INK_2)
    ax_cal.set_title("Calibration on test trials (start 2017-2019)", color=INK,
                     loc="left", fontsize=12, pad=22)
    ax_cal.text(0, 1.02, f"{N_CALIBRATION_BINS} equal-count bins. Below the diagonal = risk "
                "over-stated (class_weight='balanced')",
                transform=ax_cal.transAxes, color=INK_2, fontsize=8.5)
    ax_cal.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92),
                  labelcolor=INK, fontsize=9)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Predicted probability of TERMINATED", color=INK_2)
    ax_hist.set_ylabel("% of trials", color=INK_2)

    for ax in (ax_cal, ax_hist):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def permutation_importance_table(models: dict, X, y) -> pd.DataFrame:
    """Drop in test ROC-AUC / PR-AUC when one raw feature column is shuffled.

    The whole pipeline is scored, so each categorical counts as one feature
    (not one per one-hot column) and both models are directly comparable.
    """
    frames = []
    for name, model in models.items():
        result = permutation_importance(
            model, X, y, scoring=["roc_auc", "average_precision"],
            n_repeats=N_PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=-1)
        frame = pd.DataFrame({
            "model": name,
            "feature": X.columns,
            "roc_auc_drop_mean": result["roc_auc"].importances_mean,
            "roc_auc_drop_std": result["roc_auc"].importances_std,
            "pr_auc_drop_mean": result["average_precision"].importances_mean,
            "pr_auc_drop_std": result["average_precision"].importances_std,
        }).sort_values("roc_auc_drop_mean", ascending=False)
        frame.insert(1, "rank", range(1, len(frame) + 1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def lr_coefficients(pipeline) -> pd.DataFrame:
    """Logistic regression coefficients by transformed column, largest |coef| first.

    Numeric columns are standardised, so their coefficient is the change in
    log-odds per one training standard deviation (``unit`` shows it in raw
    units). Flags and one-hot columns are 0/1, so theirs is the change when the
    flag switches on.
    """
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    sd = dict(zip(NUMERIC_FEATURES, scaler.scale_))
    names = prep.get_feature_names_out()
    coef = clf.coef_[0]
    table = pd.DataFrame({
        "feature": names,
        "coef": coef,
        "odds_ratio": np.exp(coef),
        "unit": [f"per SD ({sd[n]:.3g})" if n in sd else "0 -> 1" for n in names],
    })
    order = table["coef"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def print_metrics_table(reports: dict) -> None:
    """Print CV and test metrics of every model plus the majority baseline."""
    rows = []
    for name, rep in reports.items():
        test, at = rep["test"], rep["test"]["at_threshold"]
        cm = at["confusion_matrix"]
        rows.append({
            "model": name,
            "cv_roc_auc": rep["cv"]["best_roc_auc_mean"],
            "cv_pr_auc": rep["cv"]["best_pr_auc_mean"],
            "test_roc_auc": test["roc_auc"],
            "test_pr_auc": test["pr_auc"],
            "brier": test["brier"],
            "threshold": at["threshold"],
            "precision": at["precision"],
            "recall": at["recall"],
            "f1": at["f1"],
            "tn/fp/fn/tp": f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}",
        })
    base = next(iter(reports.values()))["majority_baseline"]
    cm = base["confusion_matrix"]
    rows.append({"model": "majority (all COMPLETED)", "test_roc_auc": base["roc_auc"],
                 "test_pr_auc": base["pr_auc"],
                 "tn/fp/fn/tp": f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}"})
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).set_index("model").round(4).to_string(na_rep="-"))


def evaluate_saved(X_test=None, y_test=None, names=MODEL_NAMES) -> dict:
    """Evaluate the saved models on test and write every test-time output.

    Args:
        X_test, y_test: the test split; loaded from the parquet when omitted.
        names: model names; each needs ``models/<name>.joblib`` and the
            ``models/<name>_metrics.json`` written by ``train.py``.

    Returns:
        {model name: full metrics dict}, as written to the JSON files.
    """
    if X_test is None:
        # Imported here, not at the top, because train.py imports this module.
        from train import load_split
        X_test, y_test = load_split("test")
    y_test = np.asarray(y_test)

    models, reports, probas = {}, {}, {}
    for name in names:
        models[name] = load_model(MODELS_DIR / f"{name}.joblib")
        reports[name] = json.loads(metrics_path(name).read_text(encoding="utf-8"))
        probas[name] = models[name].predict_proba(X_test)[:, 1]
        threshold = reports[name]["threshold"]["value"]
        reports[name]["test"] = classification_metrics(y_test, probas[name], threshold)
        reports[name]["test"]["calibration_bins"] = calibration_bins(y_test, probas[name])
        reports[name]["majority_baseline"] = majority_baseline(y_test)
        metrics_path(name).write_text(json.dumps(reports[name], indent=2) + "\n",
                                      encoding="utf-8")

    print("\n=== Metrics (CV = 5-fold stratified on train; test = start 2017-2019) ===")
    print_metrics_table(reports)

    plot_calibration(y_test, probas)
    print(f"\nSaved {CALIBRATION_PATH.relative_to(ROOT)}")

    print(f"\nPermutation importance on test ({N_PERMUTATION_REPEATS} repeats per feature)...")
    importance = permutation_importance_table(models, X_test[FEATURE_COLUMNS], y_test)
    importance.to_csv(IMPORTANCE_PATH, index=False, float_format="%.6f")
    print(f"Saved {IMPORTANCE_PATH.relative_to(ROOT)}")
    for name in names:
        top = importance[importance["model"] == name].head(10)
        print(f"\nTop 10 features by test ROC-AUC drop, {name}:")
        print(top[["rank", "feature", "roc_auc_drop_mean", "roc_auc_drop_std",
                   "pr_auc_drop_mean"]].round(4).to_string(index=False))

    if "baseline_lr" in models:
        print("\nLogistic regression coefficients (log-odds of TERMINATED), largest |coef| first:")
        with pd.option_context("display.max_rows", 500, "display.width", 200):
            print(lr_coefficients(models["baseline_lr"]).round(4).to_string(index=False))
    return reports


if __name__ == "__main__":
    evaluate_saved()

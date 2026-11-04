"""Test-set evaluation of the saved models: metrics, calibration, importance, ablations.

Everything that scores the test split (start 2017-2019) lives here, so there is
one code path for it. ``train.py`` calls ``evaluate_saved`` once after saving the
models; running this file re-evaluates the saved models without retraining them.

For each saved model (``baseline_lr``, ``gbm_uncalibrated``, ``gbm`` and the
challengers of ``train.CHALLENGERS``, i.e. ``gbm_v2``) this:
  * computes test ROC-AUC, PR-AUC and Brier score, plus precision / recall / F1
    and the confusion matrix at the threshold stored in its metrics JSON (chosen
    on train CV, never on test), and the majority-class baseline;
  * writes those numbers into ``models/<name>_metrics.json`` under ``test``;
  * compares each challenger with the app model (CV, out-of-fold and test
    metrics, per-fold CV gains, a paired bootstrap of the test ROC-AUC gain) and
    checks the decision rule -> ``gbm_metrics.json`` under ``version_comparison``;
  * adds the before/after calibration comparison to ``gbm_metrics.json`` and
    draws ``models/calibration_before_after.png``;
  * computes permutation importance on test -> ``models/feature_importance.csv``;
  * runs the feature ablations (GBM refitted on train without features that can
    be edited after registration) -> ``gbm_metrics.json`` under ``ablations``;
  * prints the metrics, comparison, ablation and logistic regression tables.

Every model is scored on exactly the columns it was trained on
(``predict.model_columns``).

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
from sklearn.model_selection import cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_model, model_columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CALIBRATION_PATH = MODELS_DIR / "calibration_before_after.png"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

# The app model and its baselines; train.CHALLENGERS (gbm_v2) are added at run
# time, because importing train here would be circular.
BASE_MODEL_NAMES = ["baseline_lr", "gbm_uncalibrated", "gbm"]
# gbm_uncalibrated ranks trials almost exactly like gbm, so importance is only
# computed for the models people will look at (plus the challengers).
IMPORTANCE_MODELS = ["baseline_lr", "gbm"]
CALIBRATION_PLOT_MODELS = ["gbm_uncalibrated", "gbm"]
DISPLAY_NAMES = {
    "baseline_lr": "Logistic regression",
    "gbm_uncalibrated": "Before: GBM, class_weight='balanced'",
    "gbm": "After: + isotonic calibration (cv=5)",
}
# Decision rule, fixed before v2 was trained: a challenger replaces the app
# model only if its ROC-AUC is at least this much higher on test AND in 5-fold
# CV on train (best grid cell, same folds).
MIN_ROC_AUC_GAIN = 0.005
N_BOOTSTRAP = 1000
SHORT_NAMES = {"gbm_uncalibrated": "Before", "gbm": "After"}

# Features a sponsor can amend after the trial has started (protocol
# amendments), so the latest record may not show the design-time value.
ABLATIONS = {
    "without_eligibility_text_len": ["eligibility_text_len"],
    "without_outcome_counts": ["n_primary_outcomes", "n_secondary_outcomes"],
    "without_both": ["eligibility_text_len", "n_primary_outcomes", "n_secondary_outcomes"],
}

RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 10
N_CALIBRATION_BINS = 10

# Chart tokens (validated categorical slots 1-2 plus neutral chrome), light mode
# because the PNG goes into reports and slides. "After" is the result, so it
# takes slot 1.
SERIES_COLORS = {"gbm": "#2a78d6", "gbm_uncalibrated": "#eb6834"}
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


def ranking_metrics(y_true, proba) -> dict:
    """Threshold-free metrics: ranking (ROC-AUC, PR-AUC) and calibration (Brier)."""
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted": float(np.mean(proba)),
    }


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Ranking metrics plus thresholded metrics for one model on one split."""
    y_pred = (proba >= threshold).astype(int)
    return {
        **ranking_metrics(y_true, proba),
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
    terminated rate. Predicting that rate for everyone gives the Brier score a
    calibrated model has to beat. Any real model has to beat all three.
    """
    y_true = np.asarray(y_true)
    rate = float(y_true.mean())
    return {
        "roc_auc": 0.5,
        "pr_auc": rate,
        "brier_constant_rate": float(rate * (1 - rate)),
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
    """Save calibration curves (top) and prediction histograms (bottom).

    Args:
        y_true: test labels.
        probas: {model name: test probabilities}, drawn in this order.
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
                ha="right", va="bottom")
    ax_cal.axhline(base_rate, color=AXIS, linewidth=1)
    ax_cal.text(0.99, base_rate + 0.012, f"test terminated rate {base_rate:.1%}",
                color=INK_2, fontsize=8.5, va="bottom", ha="right")

    bins = np.linspace(0, 1, 41)
    ends = {}
    for name, proba in probas.items():
        points = calibration_bins(y_true, proba)
        xs = [p["mean_predicted"] for p in points]
        ys = [p["observed_rate"] for p in points]
        ends[name] = (xs[-1], ys[-1])
        color = SERIES_COLORS.get(name, INK_2)
        label = (f"{DISPLAY_NAMES.get(name, name)}  "
                 f"(Brier {brier_score_loss(y_true, proba):.3f})")
        ax_cal.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax_hist.hist(proba, bins=bins, histtype="step", linewidth=2, color=color,
                     weights=np.full(len(proba), 100 / len(proba)))
    # Direct labels at the curve ends, in ink (the line colour carries identity).
    # If two ends sit close together, the higher one is nudged up, the rest down.
    top = max(ends, key=lambda n: ends[n][1])
    for name, end in ends.items():
        ax_cal.annotate(SHORT_NAMES.get(name, DISPLAY_NAMES.get(name, name)), end,
                        xytext=(8, 9 if name == top else -9), textcoords="offset points",
                        color=INK, fontsize=9, va="center")

    ax_cal.set_ylim(0, 1)
    ax_cal.set_ylabel("Observed share terminated", color=INK_2)
    ax_cal.set_title("Gradient boosting calibration on test trials (start 2017-2019)",
                     color=INK, loc="left", fontsize=12, pad=22)
    ax_cal.text(0, 1.02, f"{N_CALIBRATION_BINS} equal-count bins. On the diagonal = "
                "predicted risk matches observed", transform=ax_cal.transAxes,
                color=INK_2, fontsize=8.5)
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

    The whole model is scored, so each categorical counts as one feature
    (not one per one-hot column) and the models are directly comparable.
    Each model is permuted over its own input columns only.
    """
    frames = []
    for name, model in models.items():
        X_model = X[model_columns(model)]
        result = permutation_importance(
            model, X_model, y, scoring=["roc_auc", "average_precision"],
            n_repeats=N_PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=-1)
        frame = pd.DataFrame({
            "model": name,
            "feature": X_model.columns,
            "roc_auc_drop_mean": result["roc_auc"].importances_mean,
            "roc_auc_drop_std": result["roc_auc"].importances_std,
            "pr_auc_drop_mean": result["average_precision"].importances_mean,
            "pr_auc_drop_std": result["average_precision"].importances_std,
        }).sort_values("roc_auc_drop_mean", ascending=False)
        frame.insert(1, "rank", range(1, len(frame) + 1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_ablations(X_train, y_train, X_test, y_test, params: dict, reference: dict) -> dict:
    """Refit the GBM without features that can change after registration.

    ClinicalTrials.gov returns the latest version of each record, so eligibility
    criteria and outcome lists may have been amended after the trial started.
    If the model leaned on such edits, dropping those features would cost much
    more than their honest share. Each variant keeps the tuned hyperparameters
    (no re-tuning; the grid was flat) and stays uncalibrated, so it is compared
    with ``gbm_uncalibrated``. Fitting uses train only; test is scored once.

    Args:
        X_train, y_train, X_test, y_test: the two splits.
        params: tuned HistGradientBoosting hyperparameters.
        reference: ``{"cv_roc_auc", "cv_pr_auc", "test_roc_auc", "test_pr_auc"}``
            of the full-feature ``gbm_uncalibrated``.

    Returns:
        The ``ablations`` section for ``gbm_metrics.json``.
    """
    # Imported here, not at the top, because train.py imports this module.
    from train import APP_FEATURES, build_gbm_pipeline, make_cv

    runs = [{"name": "all_features", "dropped": [], "n_features": len(APP_FEATURES),
             **reference, "delta_test_roc_auc": 0.0, "delta_test_pr_auc": 0.0}]
    for name, dropped in ABLATIONS.items():
        model = build_gbm_pipeline(drop=dropped).set_params(
            **{f"clf__{k}": v for k, v in params.items()})
        cv = cross_validate(model, X_train, y_train, cv=make_cv(),
                            scoring=["roc_auc", "average_precision"], n_jobs=-1)
        proba = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        test_roc = float(roc_auc_score(y_test, proba))
        test_pr = float(average_precision_score(y_test, proba))
        runs.append({
            "name": name, "dropped": dropped, "n_features": len(APP_FEATURES) - len(dropped),
            "cv_roc_auc": float(cv["test_roc_auc"].mean()),
            "cv_pr_auc": float(cv["test_average_precision"].mean()),
            "test_roc_auc": test_roc, "test_pr_auc": test_pr,
            "delta_test_roc_auc": test_roc - reference["test_roc_auc"],
            "delta_test_pr_auc": test_pr - reference["test_pr_auc"],
        })
    return {
        "purpose": "robustness to post-registration edits: gbm_uncalibrated refitted on "
                   "train without features a sponsor can amend after the start",
        "model": "gbm_uncalibrated pipeline, same hyperparameters, not re-tuned, uncalibrated",
        "hyperparameters": params,
        "runs": runs,
    }


def paired_bootstrap_gain(y_true, proba_a, proba_b, n: int = N_BOOTSTRAP) -> dict:
    """Bootstrap of ROC-AUC(b) - ROC-AUC(a) on the same resampled trials.

    Both models are scored on each resample, so the interval reflects only
    how much the gain depends on which test trials happened to be drawn.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    gains = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        gains.append(roc_auc_score(y_true[idx], proba_b[idx]) - roc_auc_score(y_true[idx], proba_a[idx]))
    return {"n_resamples": n, "mean": float(np.mean(gains)),
            "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
            "share_at_least_min_gain": float(np.mean(np.asarray(gains) >= MIN_ROC_AUC_GAIN))}


def compare_versions(app: dict, challenger: dict, y_test, proba_app, proba_challenger) -> dict:
    """Challenger vs app model, and whether the challenger passes the decision rule.

    Args:
        app, challenger: full metrics dicts (with ``cv``, ``threshold`` and
            ``test``) of the two calibrated models.
        y_test, proba_app, proba_challenger: test labels and probabilities.

    Returns:
        The ``version_comparison`` entry for ``gbm_metrics.json``.
    """
    def summary(rep):
        cv, oof, test = rep["cv"], rep["threshold"], rep["test"]
        return {"model": rep["model"], "feature_version": rep["features"].get("version"),
                "n_features": len(rep["features"]["all"]),
                "best_params": cv["best_params"],
                "cv_roc_auc": cv["best_roc_auc_mean"], "cv_roc_auc_std": cv["best_roc_auc_std"],
                "cv_pr_auc": cv["best_pr_auc_mean"], "cv_pr_auc_std": cv["best_pr_auc_std"],
                "oof_roc_auc": oof["oof_roc_auc"], "oof_pr_auc": oof["oof_pr_auc"],
                "oof_brier": oof["oof_brier"],
                "test_roc_auc": test["roc_auc"], "test_pr_auc": test["pr_auc"],
                "test_brier": test["brier"]}

    a, b = summary(app), summary(challenger)
    gain = {key: b[key] - a[key] for key in a if key.startswith(("cv_", "oof_", "test_"))
            and not key.endswith("_std")}
    passes = (gain["test_roc_auc"] >= MIN_ROC_AUC_GAIN and gain["cv_roc_auc"] >= MIN_ROC_AUC_GAIN)
    return {
        "rule": f"the challenger replaces the app model only if its ROC-AUC beats the app "
                f"model's by >= {MIN_ROC_AUC_GAIN} on test AND in 5-fold CV on train "
                f"(best grid cell, same folds); fixed before the challenger was trained",
        "app_model": a,
        "challenger": b,
        "gain": gain,
        "cv_fold_roc_auc_gains": [c - p for p, c in zip(app["cv"]["best_roc_auc_folds"],
                                                       challenger["cv"]["best_roc_auc_folds"])],
        "test_roc_auc_gain_bootstrap": paired_bootstrap_gain(y_test, proba_app, proba_challenger),
        "passes_rule": bool(passes),
        "decision": (f"{b['model']} passes the rule and should replace {a['model']}" if passes
                     else f"{a['model']} stays the app model; {b['model']} is kept for the report"),
    }


def print_comparison(comparison: dict) -> None:
    """Print a challenger-vs-app table and the decision."""
    a, b, gain = comparison["app_model"], comparison["challenger"], comparison["gain"]
    rows = [{"metric": key, a["model"]: a[key], b["model"]: b[key], "gain": gain[key]}
            for key in gain]
    print(pd.DataFrame(rows).set_index("metric").round(4).to_string())
    boot = comparison["test_roc_auc_gain_bootstrap"]
    folds = ", ".join(f"{g:+.4f}" for g in comparison["cv_fold_roc_auc_gains"])
    print(f"CV ROC-AUC gain per fold: {folds}")
    print(f"Test ROC-AUC gain, paired bootstrap 95% CI: [{boot['ci95'][0]:+.4f}, "
          f"{boot['ci95'][1]:+.4f}]; resamples with gain >= {MIN_ROC_AUC_GAIN}: "
          f"{boot['share_at_least_min_gain']:.0%}")
    print(f"Decision: {comparison['decision']}")


def lr_coefficients(pipeline) -> pd.DataFrame:
    """Logistic regression coefficients by transformed column, largest |coef| first.

    Numeric columns are standardised, so their coefficient is the change in
    log-odds per one training standard deviation (``unit`` shows it in raw
    units). Flags and one-hot columns are 0/1, so theirs is the change when the
    flag switches on.
    """
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    # The model's own numeric columns, which may be a subset of NUMERIC_FEATURES.
    numeric = next(cols for name, _, cols in prep.transformers_ if name == "num")
    sd = dict(zip(numeric, scaler.scale_))
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
    """Print OOF train and test metrics of every model plus the majority baseline."""
    rows = []
    for name, rep in reports.items():
        test, at = rep["test"], rep["test"]["at_threshold"]
        cm = at["confusion_matrix"]
        rows.append({
            "model": name,
            "oof_roc_auc": rep["threshold"]["oof_roc_auc"],
            "oof_pr_auc": rep["threshold"]["oof_pr_auc"],
            "test_roc_auc": test["roc_auc"],
            "test_pr_auc": test["pr_auc"],
            "brier": test["brier"],
            "mean_pred": test["mean_predicted"],
            "threshold": at["threshold"],
            "precision": at["precision"],
            "recall": at["recall"],
            "f1": at["f1"],
            "tn/fp/fn/tp": f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}",
        })
    base = next(iter(reports.values()))["majority_baseline"]
    cm = base["confusion_matrix"]
    rows.append({"model": "majority (all COMPLETED)", "test_roc_auc": base["roc_auc"],
                 "test_pr_auc": base["pr_auc"], "brier": base["brier_constant_rate"],
                 "mean_pred": base["terminated_rate"],
                 "tn/fp/fn/tp": f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}"})
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(rows).set_index("model").round(4).to_string(na_rep="-"))


def evaluate_saved(X_test=None, y_test=None, names=None,
                   X_train=None, y_train=None) -> dict:
    """Evaluate the saved models on test and write every test-time output.

    Args:
        X_test, y_test: the test split (all ``FEATURE_COLUMNS``); loaded from
            the parquet when omitted.
        names: model names; each needs ``models/<name>.joblib`` and the
            ``models/<name>_metrics.json`` written by ``train.py``. Defaults
            to ``BASE_MODEL_NAMES`` plus ``train.CHALLENGERS``.
        X_train, y_train: the train split, needed to refit the ablations;
            loaded from the parquet when omitted.

    Returns:
        {model name: full metrics dict}, as written to the JSON files.
    """
    # Imported here, not at the top, because train.py imports this module.
    from train import CHALLENGERS, load_split
    if names is None:
        names = BASE_MODEL_NAMES + list(CHALLENGERS)
    if X_test is None:
        X_test, y_test = load_split("test")
    if X_train is None:
        X_train, y_train = load_split("train")
    y_test, y_train = np.asarray(y_test), np.asarray(y_train)

    models, reports, probas = {}, {}, {}
    for name in names:
        models[name] = load_model(MODELS_DIR / f"{name}.joblib")
        reports[name] = json.loads(metrics_path(name).read_text(encoding="utf-8"))
        probas[name] = models[name].predict_proba(X_test[model_columns(models[name])])[:, 1]
        threshold = reports[name]["threshold"]["value"]
        reports[name]["test"] = classification_metrics(y_test, probas[name], threshold)
        reports[name]["test"]["calibration_bins"] = calibration_bins(y_test, probas[name])
        reports[name]["majority_baseline"] = majority_baseline(y_test)

    print("\n=== Metrics (OOF = out-of-fold on train; test = start 2017-2019) ===")
    print_metrics_table(reports)

    challengers = [n for n in names if n in CHALLENGERS]
    if "gbm" in reports and challengers:
        comparisons = {}
        for name in challengers:
            print(f"\n=== {name} vs gbm (app model) ===")
            comparisons[name] = compare_versions(reports["gbm"], reports[name], y_test,
                                                 probas["gbm"], probas[name])
            print_comparison(comparisons[name])
        reports["gbm"]["version_comparison"] = comparisons

    if {"gbm", "gbm_uncalibrated"} <= set(reports):
        reports["gbm"].setdefault("calibration", {}).update({
            "test_before": ranking_metrics(y_test, probas["gbm_uncalibrated"]),
            "test_after": ranking_metrics(y_test, probas["gbm"]),
        })
        cal = reports["gbm"]["calibration"]
        print(f"\nGBM test Brier: before {cal['test_before']['brier']:.4f} -> after "
              f"{cal['test_after']['brier']:.4f} (constant base rate: "
              f"{reports['gbm']['majority_baseline']['brier_constant_rate']:.4f}); "
              f"ROC-AUC {cal['test_before']['roc_auc']:.4f} -> {cal['test_after']['roc_auc']:.4f}")
        plot_calibration(y_test, {n: probas[n] for n in CALIBRATION_PLOT_MODELS})
        print(f"Saved {CALIBRATION_PATH.relative_to(ROOT)}")

        base = reports["gbm_uncalibrated"]
        print("\nAblations (GBM refitted on train without post-registration-editable features)...")
        reports["gbm"]["ablations"] = run_ablations(
            X_train, y_train, X_test, y_test, params=base["cv"]["best_params"],
            reference={"cv_roc_auc": base["cv"]["best_roc_auc_mean"],
                       "cv_pr_auc": base["cv"]["best_pr_auc_mean"],
                       "test_roc_auc": base["test"]["roc_auc"],
                       "test_pr_auc": base["test"]["pr_auc"]})
        table = pd.DataFrame(reports["gbm"]["ablations"]["runs"])
        table["dropped"] = table["dropped"].map(lambda d: ", ".join(d) or "-")
        print(table.drop(columns="name").round(4).to_string(index=False))

    for name in names:
        metrics_path(name).write_text(json.dumps(reports[name], indent=2) + "\n",
                                      encoding="utf-8")

    importance_models = {n: models[n] for n in IMPORTANCE_MODELS + challengers if n in models}
    print(f"\nPermutation importance on test ({N_PERMUTATION_REPEATS} repeats per feature)...")
    importance = permutation_importance_table(importance_models, X_test, y_test)
    importance.to_csv(IMPORTANCE_PATH, index=False, float_format="%.6f")
    print(f"Saved {IMPORTANCE_PATH.relative_to(ROOT)}")
    for name in importance_models:
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

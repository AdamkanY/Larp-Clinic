# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Evaluate the saved models on the test trials (start 2017-2019).

This is the only file that scores the test trials, so there is exactly one place where it happens. train.py
calls evaluate_saved() once after it saved the models. Running this file evaluates the saved models again
without training them.

For every saved model (baseline_lr, gbm_uncalibrated, gbm) it:
  * computes ROC-AUC, PR-AUC and the Brier score on the test trials, plus precision, recall, F1 and the
    confusion matrix at the threshold stored in the metrics file (the threshold was chosen on training
    trials, never on test trials), and the numbers of a "model" that says "completed" for everyone;
  * writes these numbers into models/<name>_metrics.json under "test";
  * adds the comparison before/after calibration to gbm_metrics.json and draws
    models/calibration_before_after.png;
  * measures which features matter (permutation importance) -> models/feature_importance.csv;
  * trains the gradient boosting again without the features that can be edited after registration
    ("ablations") -> gbm_metrics.json under "ablations";
  * prints the tables.

Usage:
    python src/evaluate.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw into a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate

# The other files of src/ are imported by their plain name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import NUMERIC_FEATURES
from predict import load_model, model_features

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CALIBRATION_PATH = MODELS_DIR / "calibration_before_after.png"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

MODEL_NAMES = ["baseline_lr", "gbm_uncalibrated", "gbm"]
# gbm_uncalibrated ranks the trials almost exactly like gbm, so the importance is only computed for the two
# models people look at.
IMPORTANCE_MODELS = ["baseline_lr", "gbm"]
CALIBRATION_PLOT_MODELS = ["gbm_uncalibrated", "gbm"]
DISPLAY_NAMES = {
    "baseline_lr": "Logistic regression",
    "gbm_uncalibrated": "Before: GBM, class_weight='balanced'",
    "gbm": "After: + isotonic calibration (cv=5)",
}
SHORT_NAMES = {"gbm_uncalibrated": "Before", "gbm": "After"}

# Features a sponsor can change after the trial has started. ClinicalTrials.gov gives us the latest version of
# a record, so these may not show the value from the day of registration.
ABLATIONS = {
    "without_eligibility_text_len": ["eligibility_text_len"],
    "without_outcome_counts": ["n_primary_outcomes", "n_secondary_outcomes"],
    "without_both": ["eligibility_text_len", "n_primary_outcomes", "n_secondary_outcomes"],
}

RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 10
N_CALIBRATION_BINS = 10

# Colours of the calibration picture (light, because the PNG goes into reports and slides).
SERIES_COLORS = {"gbm": "#2a78d6", "gbm_uncalibrated": "#eb6834"}
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------

def metrics_path(name):
    """Path of the metrics file that belongs to the model called name."""
    return MODELS_DIR / f"{name}_metrics.json"


def best_f1_threshold(y_true, proba):
    """The threshold on proba that gives the best F1, and that F1.

    Meant for out-of-fold scores of TRAINING trials. Using the result on the test trials keeps the test trials
    out of every decision.
    """
    # Precision and recall for every possible threshold. The last pair has no threshold (it stands for
    # "flag nobody"), so we drop it.
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision = precision[:-1]
    recall = recall[:-1]

    # F1 = 2 * precision * recall / (precision + recall). Where both are 0, F1 stays 0.
    f1 = np.zeros(len(precision))
    has_hits = (precision + recall) > 0
    f1[has_hits] = 2 * precision[has_hits] * recall[has_hits] / (precision[has_hits] + recall[has_hits])

    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def confusion_counts(y_true, y_pred):
    """The confusion matrix as four named counts ("positive" means stopped early)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def ranking_metrics(y_true, proba):
    """The metrics that need no threshold: ROC-AUC and PR-AUC (ranking) and the Brier score (calibration)."""
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted": float(np.mean(proba)),
    }


def classification_metrics(y_true, proba, threshold):
    """The ranking metrics plus the yes/no metrics at one threshold, for one model."""
    # Everything at or above the threshold counts as "predicted to stop early".
    y_pred = (proba >= threshold).astype(int)
    metrics = ranking_metrics(y_true, proba)
    metrics["at_threshold"] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flagged_share": float(y_pred.mean()),
        "confusion_matrix": confusion_counts(y_true, y_pred),
    }
    return metrics


def majority_baseline(y_true):
    """The numbers of a "model" that says "completed" for every trial.

    The same score for everyone ranks nothing, so ROC-AUC is 0.5 and PR-AUC equals the share of stopped
    trials. Predicting that share for everyone gives the Brier score a calibrated model has to beat. A real
    model has to beat all three.
    """
    y_true = np.asarray(y_true)
    rate = float(y_true.mean())
    return {
        "roc_auc": 0.5,
        "pr_auc": rate,
        "brier_constant_rate": float(rate * (1 - rate)),
        "terminated_rate": rate,
        "accuracy": 1 - rate,
        "confusion_matrix": confusion_counts(y_true, np.zeros_like(y_true)),
    }


def calibration_bins(y_true, proba, n_bins=N_CALIBRATION_BINS):
    """The points of the calibration curve: mean predicted risk and real share stopped, per group of trials.

    The groups hold equal numbers of trials, so every point rests on about the same amount of data even though
    most predictions are small.
    """
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    points = []
    for p, o in zip(predicted, observed):
        points.append({"mean_predicted": float(p), "observed_rate": float(o)})
    return points


# ---------------------------------------------------------------------------------------------------------------
# The calibration picture
# ---------------------------------------------------------------------------------------------------------------

def plot_calibration(y_true, probas, path=CALIBRATION_PATH):
    """Save the picture with the calibration curves (top) and the histograms of the predictions (bottom).

    probas is a dictionary {model name: test scores}. The models are drawn in this order.
    """
    # Two panels that share the x axis: the curves on top, the histograms below.
    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    base_rate = float(np.mean(y_true))

    # Two helper lines: the diagonal (perfect calibration) and the real share of stopped test trials.
    ax_cal.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax_cal.text(0.97, 0.93, "perfectly calibrated", color=MUTED, fontsize=8.5,
                ha="right", va="bottom")
    ax_cal.axhline(base_rate, color=AXIS, linewidth=1)
    ax_cal.text(0.99, base_rate + 0.012, f"test terminated rate {base_rate:.1%}",
                color=INK_2, fontsize=8.5, va="bottom", ha="right")

    # One curve and one histogram per model. We remember where each curve ends, for the labels.
    bins = np.linspace(0, 1, 41)
    ends = {}
    for name, proba in probas.items():
        xs = []
        ys = []
        for point in calibration_bins(y_true, proba):
            xs.append(point["mean_predicted"])
            ys.append(point["observed_rate"])
        ends[name] = (xs[-1], ys[-1])
        color = SERIES_COLORS.get(name, INK_2)
        label = (f"{DISPLAY_NAMES.get(name, name)}  "
                 f"(Brier {brier_score_loss(y_true, proba):.3f})")
        ax_cal.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax_hist.hist(proba, bins=bins, histtype="step", linewidth=2, color=color,
                     weights=np.full(len(proba), 100 / len(proba)))

    # A short label at the end of each curve. The curve that ends highest gets its label above the point, the
    # others below, so two labels close together do not overlap.
    top = None
    for name in ends:
        if top is None or ends[name][1] > ends[top][1]:
            top = name
    for name, end in ends.items():
        offset = -9
        if name == top:
            offset = 9
        ax_cal.annotate(SHORT_NAMES.get(name, DISPLAY_NAMES.get(name, name)), end,
                        xytext=(8, offset), textcoords="offset points",
                        color=INK, fontsize=9, va="center")

    # Axis titles, title and legend.
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

    # The same look for both panels.
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


# ---------------------------------------------------------------------------------------------------------------
# Which features matter
# ---------------------------------------------------------------------------------------------------------------

def permutation_importance_table(name, model, X, y):
    """How much ROC-AUC and PR-AUC drop on the test trials when ONE feature column is shuffled.

    The whole model is scored, so a category counts as one feature (not one per one-hot column) and the models
    can be compared directly. Returns one row per feature, the most important first.
    """
    # Shuffle each column 10 times and look how much worse the model gets.
    result = permutation_importance(
        model, X, y, scoring=["roc_auc", "average_precision"],
        n_repeats=N_PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=-1)
    table = pd.DataFrame({
        "model": name,
        "feature": X.columns,
        "roc_auc_drop_mean": result["roc_auc"].importances_mean,
        "roc_auc_drop_std": result["roc_auc"].importances_std,
        "pr_auc_drop_mean": result["average_precision"].importances_mean,
        "pr_auc_drop_std": result["average_precision"].importances_std,
    })
    table = table.sort_values("roc_auc_drop_mean", ascending=False)
    table.insert(1, "rank", range(1, len(table) + 1))
    return table


def run_ablations(X_train, y_train, X_test, y_test, params, reference):
    """Train the gradient boosting again without the features that can change after registration.

    If the model leaned on such later edits, leaving these features out would cost much more than their fair
    share. Each variant keeps the tuned settings (no new tuning, the grid was flat) and has no calibration, so
    it is compared with gbm_uncalibrated. Only training trials are used for fitting, the test trials are
    scored once.

    params are the tuned settings. reference holds cv_roc_auc, cv_pr_auc, test_roc_auc and test_pr_auc of the
    model with all features. Returns the "ablations" part of gbm_metrics.json.
    """
    # Imported here and not at the top, because train.py imports this file.
    from train import build_gbm_pipeline, make_cv

    # The settings need the "clf__" in front again, so the pipeline knows they belong to the classifier.
    settings = {}
    for key, value in params.items():
        settings["clf__" + key] = value

    # First row: the model with all features. Every other row is compared with it.
    n_features = X_train.shape[1]
    runs = [{
        "name": "all_features", "dropped": [], "n_features": n_features,
        "cv_roc_auc": reference["cv_roc_auc"], "cv_pr_auc": reference["cv_pr_auc"],
        "test_roc_auc": reference["test_roc_auc"], "test_pr_auc": reference["test_pr_auc"],
        "delta_test_roc_auc": 0.0, "delta_test_pr_auc": 0.0,
    }]

    # One row per variant: cross-validation on the training trials, then one score on the test trials.
    for name, dropped in ABLATIONS.items():
        model = build_gbm_pipeline(drop=dropped, features=list(X_train.columns))
        model.set_params(**settings)
        cv = cross_validate(model, X_train, y_train, cv=make_cv(),
                            scoring=["roc_auc", "average_precision"], n_jobs=-1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        test_roc = float(roc_auc_score(y_test, proba))
        test_pr = float(average_precision_score(y_test, proba))
        runs.append({
            "name": name, "dropped": dropped, "n_features": n_features - len(dropped),
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


def lr_coefficients(pipeline):
    """The coefficients of the logistic regression, the largest (plus or minus) first.

    Numbers were scaled, so their coefficient is the change in log-odds per one standard deviation (the column
    "unit" shows it in real units). Flags and one-hot columns are 0/1, so theirs is the change when the flag
    switches on.
    """
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    # The standard deviation of every numeric column, from the scaler inside the pipeline.
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    numeric = []
    for column in NUMERIC_FEATURES:
        if column in prep.feature_names_in_:
            numeric.append(column)
    standard_deviation = dict(zip(numeric, scaler.scale_))

    names = prep.get_feature_names_out()
    coef = clf.coef_[0]
    units = []
    for name in names:
        if name in standard_deviation:
            units.append(f"per SD ({standard_deviation[name]:.3g})")
        else:
            units.append("0 -> 1")

    table = pd.DataFrame({
        "feature": names,
        "coef": coef,
        "odds_ratio": np.exp(coef),
        "unit": units,
    })
    order = table["coef"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
# The whole evaluation
# ---------------------------------------------------------------------------------------------------------------

def print_metrics_table(reports):
    """Print the out-of-fold and the test metrics of every model, plus the "everyone completes" baseline."""
    # One row per model.
    rows = []
    for name, report in reports.items():
        test = report["test"]
        at = test["at_threshold"]
        cm = at["confusion_matrix"]
        rows.append({
            "model": name,
            "oof_roc_auc": report["threshold"]["oof_roc_auc"],
            "oof_pr_auc": report["threshold"]["oof_pr_auc"],
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

    # The last row: the baseline (it is the same for every model, so we take it from the first report).
    first_report = list(reports.values())[0]
    base = first_report["majority_baseline"]
    cm = base["confusion_matrix"]
    rows.append({"model": "majority (all COMPLETED)", "test_roc_auc": base["roc_auc"],
                 "test_pr_auc": base["pr_auc"], "brier": base["brier_constant_rate"],
                 "mean_pred": base["terminated_rate"],
                 "tn/fp/fn/tp": f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}"})

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(rows).set_index("model").round(4).to_string(na_rep="-"))


def evaluate_saved(X_test=None, y_test=None, names=MODEL_NAMES, X_train=None, y_train=None, suffix=""):
    """Evaluate the saved models on the test trials and write everything that comes out of it.

    Without X_test / X_train the splits are loaded from the data file (train.py passes them, running this file
    does not). names are the models to evaluate; each needs its .joblib and its metrics file from train.py.
    suffix is "" for the app models. With a suffix such as "_v2" (a comparison run) the calibration picture
    and the ablations are skipped, because both describe the app model, and only the own rows of
    feature_importance.csv are replaced. Returns {model name: the full content of its metrics file}.
    """
    # Load the splits only if the caller did not pass them.
    if X_test is None or X_train is None:
        # Imported here and not at the top, because train.py imports this file.
        from train import load_split
        if X_test is None:
            X_test, y_test = load_split("test")
        if X_train is None:
            X_train, y_train = load_split("train")
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)

    # Score every saved model once on the test trials. Each model only gets its own input columns.
    models = {}
    reports = {}
    probas = {}
    for name in names:
        model = load_model(MODELS_DIR / f"{name}.joblib")
        report = json.loads(metrics_path(name).read_text(encoding="utf-8"))
        proba = model.predict_proba(X_test[model_features(model)])[:, 1]
        threshold = report["threshold"]["value"]
        report["test"] = classification_metrics(y_test, proba, threshold)
        report["test"]["calibration_bins"] = calibration_bins(y_test, proba)
        report["majority_baseline"] = majority_baseline(y_test)
        models[name] = model
        reports[name] = report
        probas[name] = proba

    print("\n=== Metrics (OOF = out-of-fold on train; test = start 2017-2019) ===")
    print_metrics_table(reports)

    # The Brier score before and after calibration, for the README and the model page.
    gbm = f"gbm{suffix}"
    uncalibrated = f"gbm_uncalibrated{suffix}"
    if gbm in reports and uncalibrated in reports:
        if "calibration" not in reports[gbm]:
            reports[gbm]["calibration"] = {}
        cal = reports[gbm]["calibration"]
        cal["test_before"] = ranking_metrics(y_test, probas[uncalibrated])
        cal["test_after"] = ranking_metrics(y_test, probas[gbm])
        print(f"\nGBM test Brier: before {cal['test_before']['brier']:.4f} -> after "
              f"{cal['test_after']['brier']:.4f} (constant base rate: "
              f"{reports[gbm]['majority_baseline']['brier_constant_rate']:.4f}); "
              f"ROC-AUC {cal['test_before']['roc_auc']:.4f} -> {cal['test_after']['roc_auc']:.4f}")

    # Only for the app model: the calibration picture and the ablations.
    if suffix == "" and "gbm" in reports and "gbm_uncalibrated" in reports:
        plot_probas = {}
        for name in CALIBRATION_PLOT_MODELS:
            plot_probas[name] = probas[name]
        plot_calibration(y_test, plot_probas)
        print(f"Saved {CALIBRATION_PATH.relative_to(ROOT)}")

        base = reports["gbm_uncalibrated"]
        print("\nAblations (GBM refitted on train without post-registration-editable features)...")
        reference = {"cv_roc_auc": base["cv"]["best_roc_auc_mean"],
                     "cv_pr_auc": base["cv"]["best_pr_auc_mean"],
                     "test_roc_auc": base["test"]["roc_auc"],
                     "test_pr_auc": base["test"]["pr_auc"]}
        reports["gbm"]["ablations"] = run_ablations(X_train, y_train, X_test, y_test,
                                                    params=base["cv"]["best_params"], reference=reference)
        table = pd.DataFrame(reports["gbm"]["ablations"]["runs"])
        dropped_text = []
        for dropped in table["dropped"]:
            dropped_text.append(", ".join(dropped) or "-")
        table["dropped"] = dropped_text
        print(table.drop(columns="name").round(4).to_string(index=False))

    # Write the test results back into each model's metrics file (train.py wrote the training part).
    for name in names:
        text = json.dumps(reports[name], indent=2) + "\n"
        metrics_path(name).write_text(text, encoding="utf-8")

    # Permutation importance on the test trials, for the models people look at.
    print(f"\nPermutation importance on test ({N_PERMUTATION_REPEATS} repeats per feature)...")
    importance_names = []
    tables = []
    for name in IMPORTANCE_MODELS:
        full_name = name + suffix
        if full_name in models:
            model = models[full_name]
            importance_names.append(full_name)
            tables.append(permutation_importance_table(full_name, model, X_test[model_features(model)], y_test))
    importance = pd.concat(tables, ignore_index=True)
    if IMPORTANCE_PATH.exists():
        # Keep the rows of models that were not evaluated in this run (for example gbm_v2 next to gbm).
        old = pd.read_csv(IMPORTANCE_PATH)
        old = old[~old["model"].isin(importance_names)]
        importance = pd.concat([old, importance], ignore_index=True)
    importance.to_csv(IMPORTANCE_PATH, index=False, float_format="%.6f")
    print(f"Saved {IMPORTANCE_PATH.relative_to(ROOT)}")
    for name in importance_names:
        top = importance[importance["model"] == name].head(10)
        print(f"\nTop 10 features by test ROC-AUC drop, {name}:")
        print(top[["rank", "feature", "roc_auc_drop_mean", "roc_auc_drop_std",
                   "pr_auc_drop_mean"]].round(4).to_string(index=False))

    # The coefficients of the logistic regression, printed for the report.
    if "baseline_lr" in models:
        print("\nLogistic regression coefficients (log-odds of TERMINATED), largest |coef| first:")
        with pd.option_context("display.max_rows", 500, "display.width", 200):
            print(lr_coefficients(models["baseline_lr"]).round(4).to_string(index=False))
    return reports


if __name__ == "__main__":
    evaluate_saved()

# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Train the models on the train split and save them to ``models/``.

Models (every preprocessing step is fitted inside a Pipeline, so on train folds
only):

* ``baseline_lr``: LogisticRegression(class_weight='balanced').
  numeric -> median impute + standardise; 0/1 flags -> most-frequent impute;
  categorical -> most-frequent impute + one-hot.
* ``gbm_uncalibrated``: HistGradientBoostingClassifier(class_weight='balanced')
  with native categorical splits. Categories are ordinal-coded (unseen ones
  become missing) and missing values are left for the trees to route.
* ``gbm`` (the app model): the tuned ``gbm_uncalibrated`` pipeline wrapped in
  ``CalibratedClassifierCV(method='isotonic', cv=5)``, fitted on train. Each of
  the 5 folds fits the pipeline on 4/5 of train and an isotonic map on the
  remaining 1/5; predictions average the 5 calibrated copies.

Inputs are one of the ``FEATURE_SETS`` below (default ``APP_FEATURE_SET``, the
set of the model the app loads); meta and display columns
(``NON_FEATURE_COLUMNS``: nct_id, status, label, start_date, split, titles,
conditions, sponsor name, phase label) never reach a model.

Protocol: hyperparameters are picked by 5-fold stratified CV on train (refit on
PR-AUC); calibration adds no tuned parameters. Every model also gets
out-of-fold (OOF) train predictions: the decision threshold is the one that
maximises F1 on them, and the calibrated model's OOF scores are saved as
``models/score_reference.npy`` for ``predict.percentile``. The typical train
trial (median of every number and flag, most common value of every category)
is saved as ``models/explain_reference.json`` for ``predict.explain``. The test split
(start 2017-2019) is scored only by ``evaluate.evaluate_saved``, after every
model is saved.

Usage:
    python src/train.py                 # app models (feature set v1), ~5-10 min on 4 cores
    python src/train.py --features v2   # GBM with condition categories, saved as
                                        # gbm_v2 / gbm_uncalibrated_v2, app files untouched
"""

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES,  # noqa: E402
                      CONDITION_FEATURES, FEATURE_COLUMNS, LABELS,
                      NON_FEATURE_COLUMNS, NUMERIC_FEATURES)
from predict import to_model_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "trials.parquet"
MODELS_DIR = ROOT / "models"
SCORE_REFERENCE_PATH = MODELS_DIR / "score_reference.npy"
EXPLAIN_REFERENCE_PATH = MODELS_DIR / "explain_reference.json"

# Must match src/build_dataset.py; check_dataset() fails loudly if they drift.
SPLITS = {
    "train": (pd.Timestamp("2008-01-01"), pd.Timestamp("2016-12-31")),
    "test": (pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")),
}
# v1: the 29 original design features. v2: plus the keyword condition categories
# of features.condition_features. v2 was trained with this exact protocol on
# 2026-11-05 and raised the calibrated GBM's test ROC-AUC by only 0.004 (CV
# +0.004), below the +0.005 bar set for replacing the app model, so the app
# keeps v1. FEATURE_COLUMNS (what build_features returns and the parquet
# holds) is the superset; each model records its own inputs, see
# predict.model_features.
FEATURE_SETS = {
    "v1": [c for c in FEATURE_COLUMNS if c not in CONDITION_FEATURES],
    "v2": list(FEATURE_COLUMNS),
}
APP_FEATURE_SET = "v1"

RANDOM_STATE = 42
CV_FOLDS = 5
REFIT_METRIC = "average_precision"  # PR-AUC: the metric that matters at ~11% positives
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 5  # plain int: unshuffled stratified folds; train rows are in random order

# Small grids on purpose: this is a baseline, and every extra cell is another
# chance to overfit the CV folds. Both grids turned out flat (all cells within
# one CV standard deviation), so widening them buys nothing.
PARAM_GRIDS = {
    "baseline_lr": {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "gbm": {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__l2_regularization": [0.0, 1.0],
    },
}


def check_dataset(df: pd.DataFrame) -> None:
    """Fail if the parquet breaks the label, date-window or split rules.

    Cheap insurance against training on a stale or hand-edited
    ``trials.parquet``, or against a display column slipping into the
    feature list.
    """
    overlap = set(FEATURE_COLUMNS) & set(NON_FEATURE_COLUMNS)
    if overlap:
        raise ValueError(f"Meta/display columns listed as features: {sorted(overlap)}")
    missing = [c for c in FEATURE_COLUMNS + ["label", "start_date", "split"] if c not in df]
    if missing:
        raise ValueError(f"trials.parquet lacks columns: {missing}")
    if not set(df["overall_status"]) <= set(LABELS):
        raise ValueError(f"Unexpected statuses: {set(df['overall_status']) - set(LABELS)}")
    if not (df["overall_status"].map(LABELS) == df["label"]).all():
        raise ValueError("label does not match overall_status (TERMINATED=1, COMPLETED=0)")
    for split, (start, end) in SPLITS.items():
        in_window = df["start_date"].between(start, end)
        if not (in_window == (df["split"] == split)).all():
            raise ValueError(f"split='{split}' rows do not match start {start:%Y-%m-%d}"
                             f"..{end:%Y-%m-%d}; rebuild with src/build_dataset.py")


def load_split(split: str, df: pd.DataFrame = None, features=None) -> tuple:
    """Load one split as (X, y), with X holding only model input columns.

    Args:
        split: "train" or "test".
        df: the already loaded parquet; read (and checked) from disk if omitted.
        features: the input columns; defaults to the app's feature set.
    """
    # Every load goes through check_dataset, so a stale or hand-edited parquet stops the training run.
    if df is None:
        df = pd.read_parquet(DATA_PATH)
    check_dataset(df)
    part = df[df["split"] == split]
    features = FEATURE_SETS[APP_FEATURE_SET] if features is None else features
    return to_model_frame(part, features), part["label"].to_numpy(dtype=int)


def split_features(features) -> dict:
    """The categorical / binary / numeric parts of a feature list, in FEATURE_COLUMNS order."""
    return {"categorical": [c for c in CATEGORICAL_FEATURES if c in features],
            "binary": [c for c in BINARY_FEATURES if c in features],
            "numeric": [c for c in NUMERIC_FEATURES if c in features]}


def build_lr_pipeline(features=None) -> Pipeline:
    """Logistic regression with imputation, scaling and one-hot encoding.

    Args:
        features: input columns; defaults to the app's feature set.
    """
    # Logistic regression needs numbers only: impute and scale numbers, impute flags, one-hot encode categories.
    parts = split_features(FEATURE_SETS[APP_FEATURE_SET] if features is None else features)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    # Flags stay 0/1 (no scaling) so their coefficients read as "flag on vs off".
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, parts["numeric"]),
                              ("bin", binary, parts["binary"]),
                              ("cat", categorical, parts["categorical"])],
                             verbose_feature_names_out=False)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    return Pipeline([("prep", prep), ("clf", clf)])


def build_gbm_pipeline(drop=(), features=None) -> Pipeline:
    """Histogram gradient boosting with native categorical splits.

    Args:
        drop: feature names to leave out (used by the ablations in
            ``evaluate.py``); the default uses every feature of the set.
        features: input columns; defaults to the app's feature set.
    """
    parts = split_features(FEATURE_SETS[APP_FEATURE_SET] if features is None else features)
    # Gradient boosting reads categories natively, so they are only ordinal-coded; numbers pass through unchanged.
    categorical = [c for c in parts["categorical"] if c not in drop]
    other = [c for c in parts["binary"] + parts["numeric"] if c not in drop]
    # Unseen categories at scoring time (e.g. a new API enum value) become
    # missing instead of crashing the app.
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan)
    prep = ColumnTransformer([("cat", encoder, categorical), ("num", "passthrough", other)],
                             verbose_feature_names_out=False).set_output(transform="pandas")
    # Early stopping holds out 10% of each *training* fold, never test data.
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", categorical_features=categorical,
        max_iter=1000, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])


def explain_reference(X_train: pd.DataFrame) -> dict:
    """The "typical training trial" that ``predict.explain`` compares a trial with.

    Numbers and 0/1 flags take their train median (for a flag: the more common
    value), categories their most common value. Missing values are ignored, so
    the reference is always a complete, ordinary design.
    """
    # The typical training trial, used by predict.explain as the point every factor is compared with.
    values = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_FEATURES:
            values[col] = str(X_train[col].mode(dropna=True).iloc[0])
        else:
            values[col] = float(X_train[col].median(skipna=True))
    return {
        "contents": "median (numbers, 0/1 flags) or most common value (categories) of "
                    "every feature on the train split (start 2008-2016)",
        "n_train": int(len(X_train)),
        "values": values,
    }


def build_calibrated(base: Pipeline) -> CalibratedClassifierCV:
    """Isotonic calibration around an (unfitted copy of a) tuned pipeline.

    Isotonic regression learns a monotone map from the model's score to the
    observed terminated rate on held-out folds, which undoes the upward push
    of ``class_weight='balanced'`` without changing how trials are ranked.
    """
    return CalibratedClassifierCV(clone(base), method=CALIBRATION_METHOD, cv=CALIBRATION_CV)


BUILDERS = {"baseline_lr": build_lr_pipeline, "gbm": build_gbm_pipeline}
# Tuned name -> saved name. The tuned GBM is saved as gbm_uncalibrated; "gbm"
# is reserved for its calibrated version, the model the app loads.
SAVED_NAMES = {"baseline_lr": "baseline_lr", "gbm": "gbm_uncalibrated"}


def make_cv() -> StratifiedKFold:
    """The one CV splitter for tuning, OOF predictions and ablations."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def tune(name: str, X, y, cv) -> GridSearchCV:
    """Grid-search one model with stratified CV on train and refit on all of train.

    The model's inputs are the columns of ``X``.
    """
    # Grid search with 5-fold CV on the training trials only; both metrics are recorded, PR-AUC picks the winner.
    search = GridSearchCV(BUILDERS[name](features=list(X.columns)), PARAM_GRIDS[name], cv=cv,
                          scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
                          refit=REFIT_METRIC, n_jobs=-1)
    search.fit(X, y)
    return search


def cv_summary(search: GridSearchCV) -> dict:
    """Best parameters and the CV scores of every grid cell, for the metrics JSON."""
    res, best = search.cv_results_, search.best_index_
    grid = [{"params": {k.removeprefix("clf__"): v for k, v in params.items()},
             "cv_roc_auc": float(res["mean_test_roc_auc"][i]),
             "cv_pr_auc": float(res["mean_test_average_precision"][i])}
            for i, params in enumerate(res["params"])]
    return {
        "scheme": f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, "
                  f"random_state={RANDOM_STATE}) on the train split only",
        "refit_metric": "pr_auc",
        "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        "best_roc_auc_mean": float(res["mean_test_roc_auc"][best]),
        "best_roc_auc_std": float(res["std_test_roc_auc"][best]),
        "best_pr_auc_mean": float(res["mean_test_average_precision"][best]),
        "best_pr_auc_std": float(res["std_test_average_precision"][best]),
        "grid": grid,
    }


def out_of_fold(estimator, X, y, cv) -> tuple:
    """OOF train probabilities and the F1-max threshold chosen on them.

    Every train trial is scored by a copy of ``estimator`` that never saw it,
    so the threshold (and the percentile reference) are not fitted to their
    own data. For the calibrated model this is nested: 5 outer folds, each
    running its own 5-fold calibration inside.
    """
    # Out-of-fold scores on the training trials give an honest threshold without touching the test years.
    oof = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba",
                            n_jobs=-1)[:, 1]
    threshold, oof_f1 = evaluate.best_f1_threshold(y, oof)
    info = {
        "value": threshold,
        "rule": "maximises F1 on out-of-fold train predictions (same CV folds)",
        "oof_f1": oof_f1,
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_pr_auc": float(average_precision_score(y, oof)),
        "oof_brier": float(brier_score_loss(y, oof)),
    }
    return oof, info


def split_summary(df: pd.DataFrame) -> dict:
    """Split rule and observed size / base rate per split, for the metrics JSON."""
    out = {"rule": "by start_date, never random"}
    for split, (start, end) in SPLITS.items():
        part = df[df["split"] == split]
        out[split] = {
            "start_from": f"{start:%Y-%m-%d}", "start_to": f"{end:%Y-%m-%d}",
            "observed_first_start": f"{part['start_date'].min():%Y-%m-%d}",
            "observed_last_start": f"{part['start_date'].max():%Y-%m-%d}",
            "n_trials": int(len(part)),
            "n_terminated": int(part["label"].sum()),
            "terminated_rate": float(part["label"].mean()),
        }
    return out


def model_details(model) -> dict:
    """Estimator facts worth keeping next to the model file."""
    if isinstance(model, CalibratedClassifierCV):
        folds = [c.estimator.named_steps["clf"] for c in model.calibrated_classifiers_]
        return {"estimator": "CalibratedClassifierCV(HistGradientBoostingClassifier)",
                "class_weight": "balanced (inside each fold's pipeline)",
                "calibration_method": model.method,
                "n_calibration_folds": len(folds),
                "n_iter_per_fold": [int(f.n_iter_) for f in folds]}
    clf = model.named_steps["clf"]
    details = {"estimator": type(clf).__name__, "class_weight": "balanced"}
    if isinstance(clf, HistGradientBoostingClassifier):
        details["n_iter"] = int(clf.n_iter_)
    else:
        details["n_model_columns"] = int(clf.coef_.shape[1])
    return details


def save(name: str, model, df: pd.DataFrame, cv_info: dict, threshold: dict,
         feature_set: str, extra: dict = None) -> None:
    """Write ``models/<name>.joblib`` and the training part of its metrics JSON."""
    # Save the fitted model and a metrics file that records how it was made (split, features, CV, threshold).
    features = FEATURE_SETS[feature_set]
    joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)
    report = {
        "model": name,
        "model_file": f"models/{name}.joblib",
        "trained_on": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "label": {"TERMINATED": 1, "COMPLETED": 0},
        "split": split_summary(df),
        "feature_set": feature_set,
        "features": {"all": features, **split_features(features)},
        **model_details(model),
        "cv": cv_info,
        "threshold": threshold,
        **(extra or {}),
    }
    evaluate.metrics_path(name).write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")


def main(feature_set: str = APP_FEATURE_SET) -> None:
    """Tune, fit and save the models of one feature set, then evaluate them once on test.

    The app's feature set writes the usual files (``gbm.joblib`` and friends) and
    trains the logistic regression baseline too; any other set trains only the
    GBM and saves it with the set as suffix (``gbm_v2.joblib``,
    ``gbm_uncalibrated_v2.joblib``, ``score_reference_v2.npy``), so a comparison
    run never replaces the app model.
    """
    # Which models this run trains, and under which file names (a non-app feature set gets a suffix).
    features = FEATURE_SETS[feature_set]
    suffix = "" if feature_set == APP_FEATURE_SET else f"_{feature_set}"
    builders = list(BUILDERS) if not suffix else ["gbm"]
    df = pd.read_parquet(DATA_PATH)
    check_dataset(df)
    # The test split is loaded here but only passed to evaluate.evaluate_saved at the very end, after every model is
    # saved.
    X_train, y_train = load_split("train", df, features)
    X_test, y_test = load_split("test", df, features)
    print(f"Train: {len(y_train)} trials ({y_train.mean():.2%} terminated); "
          f"test: {len(y_test)} trials ({y_test.mean():.2%} terminated)")
    print(f"Feature set {feature_set} ({len(features)}): {', '.join(features)}")

    # The same CV folds for every model, so their CV numbers are comparable.
    cv = make_cv()
    MODELS_DIR.mkdir(exist_ok=True)
    # Over every column build_features makes, so it serves any feature set.
    reference = explain_reference(load_split("train", df, FEATURE_COLUMNS)[0])
    EXPLAIN_REFERENCE_PATH.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    # Step 1: tune and save the uncalibrated models (baseline LR and GBM).
    tuned = {}
    for name in builders:
        t0 = time.time()
        print(f"\n--- {name}: grid search ({len(ParameterGrid(PARAM_GRIDS[name]))} "
              f"settings x {CV_FOLDS}-fold CV on train) ---")
        search = tune(name, X_train, y_train, cv)
        cv_info = cv_summary(search)
        print(f"best {cv_info['best_params']}  CV ROC-AUC {cv_info['best_roc_auc_mean']:.4f}"
              f" ± {cv_info['best_roc_auc_std']:.4f}  CV PR-AUC "
              f"{cv_info['best_pr_auc_mean']:.4f} ± {cv_info['best_pr_auc_std']:.4f}")
        _, threshold = out_of_fold(search.best_estimator_, X_train, y_train, cv)
        print(f"F1-max threshold on out-of-fold train predictions: {threshold['value']:.4f} "
              f"(OOF F1 {threshold['oof_f1']:.4f})")
        save(SAVED_NAMES[name] + suffix, search.best_estimator_, df, cv_info, threshold,
             feature_set)
        tuned[name] = (search.best_estimator_, cv_info)
        print(f"Saved models/{SAVED_NAMES[name]}{suffix}.joblib ({time.time() - t0:.0f}s)")

    # Step 2: wrap the tuned GBM in isotonic calibration, get honest out-of-fold scores for the threshold and the
    # percentile reference, then fit it on all training trials.
    t0 = time.time()
    print(f"\n--- gbm: {CALIBRATION_METHOD} calibration (cv={CALIBRATION_CV}) of the tuned GBM ---")
    base, base_cv_info = tuned["gbm"]
    calibrated = build_calibrated(base)
    oof, threshold = out_of_fold(calibrated, X_train, y_train, cv)
    print(f"OOF Brier {threshold['oof_brier']:.4f}, OOF ROC-AUC {threshold['oof_roc_auc']:.4f}; "
          f"F1-max threshold {threshold['value']:.4f} (OOF F1 {threshold['oof_f1']:.4f})")
    calibrated.fit(X_train, y_train)
    # Sorted, so predict.percentile is a single binary search.
    reference_path = SCORE_REFERENCE_PATH.with_name(f"score_reference{suffix}.npy")
    np.save(reference_path, np.sort(oof))
    save(f"gbm{suffix}", calibrated, df,
         {**base_cv_info, "note": f"hyperparameters tuned on gbm_uncalibrated{suffix}; "
                                  "calibration adds no tuned parameters"},
         threshold, feature_set,
         extra={"calibration": {
                    "method": CALIBRATION_METHOD,
                    "cv": f"CalibratedClassifierCV(cv={CALIBRATION_CV}) on the train split",
                    "base_model": f"gbm_uncalibrated{suffix}"},
                "score_reference": {
                    "file": f"models/{reference_path.name}",
                    "contents": "sorted out-of-fold calibrated P(TERMINATED) of every "
                                "train trial (start 2008-2016)",
                    "n": int(len(oof)),
                    "quantiles": {f"p{q}": float(np.percentile(oof, q))
                                  for q in (10, 25, 50, 75, 90, 99)}}})
    print(f"Saved models/gbm{suffix}.joblib and models/{reference_path.name} "
          f"({time.time() - t0:.0f}s)")

    # Step 3: score the saved models once on the test years (the only place the test split is used).
    names = [f"{name}{suffix}" for name in evaluate.MODEL_NAMES
             if name.startswith("gbm") or not suffix]
    evaluate.evaluate_saved(X_test, y_test, names=names, X_train=X_train, y_train=y_train,
                            suffix=suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the models.")
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default=APP_FEATURE_SET,
                        help=f"feature set (default {APP_FEATURE_SET}, the app model's)")
    main(parser.parse_args().features)

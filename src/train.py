"""Train the two baseline models on the train split and save them to ``models/``.

Models (one sklearn Pipeline each; every preprocessing step is fitted inside the
pipeline, so on train folds only):

* ``baseline_lr``: LogisticRegression(class_weight='balanced').
  numeric -> median impute + standardise; 0/1 flags -> most-frequent impute;
  categorical -> most-frequent impute + one-hot.
* ``gbm``: HistGradientBoostingClassifier(class_weight='balanced') with native
  categorical splits. Categories are ordinal-coded (unseen ones become missing)
  and missing values are left for the trees to route.

Inputs are exactly ``FEATURE_COLUMNS`` from ``src/features.py``; meta and
display columns (``NON_FEATURE_COLUMNS``: nct_id, status, label, start_date,
split, titles, conditions, sponsor name, phase label) never reach a model.

Protocol: hyperparameters are picked by 5-fold stratified CV on train (refit on
PR-AUC). The decision threshold is the one that maximises F1 on out-of-fold
train predictions. The test split (start 2017-2019) is used once, by
``evaluate.evaluate_saved``, after the models are saved.

Usage:
    python src/train.py        # about 5-10 minutes on 4 cores
"""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES,  # noqa: E402
                      FEATURE_COLUMNS, LABELS, NON_FEATURE_COLUMNS,
                      NUMERIC_FEATURES)
from predict import to_model_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "trials.parquet"
MODELS_DIR = ROOT / "models"

# Must match src/build_dataset.py; check_dataset() fails loudly if they drift.
SPLITS = {
    "train": (pd.Timestamp("2008-01-01"), pd.Timestamp("2016-12-31")),
    "test": (pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")),
}
RANDOM_STATE = 42
CV_FOLDS = 5
REFIT_METRIC = "average_precision"  # PR-AUC: the metric that matters at ~11% positives

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


def load_split(split: str, df: pd.DataFrame = None) -> tuple:
    """Load one split as (X, y), with X holding only FEATURE_COLUMNS.

    Args:
        split: "train" or "test".
        df: the already loaded parquet; read (and checked) from disk if omitted.
    """
    if df is None:
        df = pd.read_parquet(DATA_PATH)
    check_dataset(df)
    part = df[df["split"] == split]
    return to_model_frame(part), part["label"].to_numpy(dtype=int)


def build_lr_pipeline() -> Pipeline:
    """Logistic regression with imputation, scaling and one-hot encoding."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    # Flags stay 0/1 (no scaling) so their coefficients read as "flag on vs off".
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("bin", binary, BINARY_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)],
                             verbose_feature_names_out=False)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    return Pipeline([("prep", prep), ("clf", clf)])


def build_gbm_pipeline() -> Pipeline:
    """Histogram gradient boosting with native categorical splits."""
    # Unseen categories at scoring time (e.g. a new API enum value) become
    # missing instead of crashing the app.
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan)
    prep = ColumnTransformer([("cat", encoder, CATEGORICAL_FEATURES),
                              ("num", "passthrough", BINARY_FEATURES + NUMERIC_FEATURES)],
                             verbose_feature_names_out=False).set_output(transform="pandas")
    # Early stopping holds out 10% of each *training* fold, never test data.
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", categorical_features=CATEGORICAL_FEATURES,
        max_iter=1000, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])


BUILDERS = {"baseline_lr": build_lr_pipeline, "gbm": build_gbm_pipeline}


def tune(name: str, X, y, cv) -> GridSearchCV:
    """Grid-search one model with stratified CV on train and refit on all of train."""
    search = GridSearchCV(BUILDERS[name](), PARAM_GRIDS[name], cv=cv,
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


def model_details(name: str, model: Pipeline) -> dict:
    """Estimator facts worth keeping next to the model file."""
    clf = model.named_steps["clf"]
    details = {"estimator": type(clf).__name__, "class_weight": "balanced"}
    if name == "gbm":
        details["n_iter"] = int(clf.n_iter_)
    else:
        details["n_model_columns"] = int(clf.coef_.shape[1])
    return details


def main() -> None:
    """Tune, fit and save both models, then evaluate them once on test."""
    df = pd.read_parquet(DATA_PATH)
    check_dataset(df)
    X_train, y_train = load_split("train", df)
    X_test, y_test = load_split("test", df)
    print(f"Train: {len(y_train)} trials ({y_train.mean():.2%} terminated); "
          f"test: {len(y_test)} trials ({y_test.mean():.2%} terminated)")
    print(f"Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    MODELS_DIR.mkdir(exist_ok=True)

    for name in BUILDERS:
        t0 = time.time()
        print(f"\n--- {name}: grid search ({len(ParameterGrid(PARAM_GRIDS[name]))} "
              f"settings x {CV_FOLDS}-fold CV on train) ---")
        search = tune(name, X_train, y_train, cv)
        cv_info = cv_summary(search)
        print(f"best {cv_info['best_params']}  CV ROC-AUC {cv_info['best_roc_auc_mean']:.4f}"
              f" ± {cv_info['best_roc_auc_std']:.4f}  CV PR-AUC "
              f"{cv_info['best_pr_auc_mean']:.4f} ± {cv_info['best_pr_auc_std']:.4f}")

        # Out-of-fold probabilities: every train trial is scored by a model
        # that did not see it, so the F1 threshold is not fitted to its own data.
        oof = cross_val_predict(clone(search.best_estimator_), X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        threshold, oof_f1 = evaluate.best_f1_threshold(y_train, oof)
        print(f"F1-max threshold on out-of-fold train predictions: {threshold:.4f} "
              f"(OOF F1 {oof_f1:.4f})")

        model = search.best_estimator_
        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)
        report = {
            "model": name,
            "model_file": f"models/{name}.joblib",
            "trained_on": date.today().isoformat(),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "label": {"TERMINATED": 1, "COMPLETED": 0},
            "split": split_summary(df),
            "features": {"all": FEATURE_COLUMNS, "categorical": CATEGORICAL_FEATURES,
                         "binary": BINARY_FEATURES, "numeric": NUMERIC_FEATURES},
            **model_details(name, model),
            "cv": cv_info,
            "threshold": {
                "value": threshold,
                "rule": "maximises F1 on out-of-fold train predictions (same CV folds)",
                "oof_f1": oof_f1,
                "oof_roc_auc": float(roc_auc_score(y_train, oof)),
                "oof_pr_auc": float(average_precision_score(y_train, oof)),
            },
        }
        evaluate.metrics_path(name).write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
        print(f"Saved models/{name}.joblib ({time.time() - t0:.0f}s)")

    evaluate.evaluate_saved(X_test, y_test)


if __name__ == "__main__":
    main()

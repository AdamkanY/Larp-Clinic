"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Train the models on the training trials and save them in models/.

The three models:

* baseline_lr: logistic regression. Numbers get the median for gaps and are scaled, 0/1 flags get the most
  common value for gaps, categories are one-hot encoded.
* gbm_uncalibrated: gradient boosting (HistGradientBoostingClassifier). It can read categories and gaps
  directly, so categories only get a number code.
* gbm (the model of the app): the tuned gbm_uncalibrated with isotonic calibration on top, so that its
  percentages match what really happened.

All preparation steps sit inside a scikit-learn Pipeline. That way they are learned from the training part of
every fold only. The models never see ids, titles, dates or the label (NON_FEATURE_COLUMNS in features.py).

How we keep the evaluation honest:

* The training trials are sorted by start date. Every choice uses time-ordered cross-validation
  (TimeSeriesSplit): train on the earlier trials, check on the next block of later ones, never the other way
  round (DSF decks 10 and 16).
* The settings of each model are picked by that cross-validation (best PR-AUC).
* Every model also scores the blocks it was checked on ("out-of-fold" scores). From them we pick the yes/no
  threshold (best F1), and the calibrated model's scores are saved as models/score_reference.npy for
  predict.percentile(). The first block is only ever used for training, so it has no such score.
* The typical training trial is saved as models/explain_reference.json for predict.explain().
* The test trials (start 2017-2019) are scored only at the very end, by evaluate.evaluate_saved().

Usage:
    python src/train.py                 # the app models (feature set v1), about 5-10 minutes
    python src/train.py --features v2   # gradient boosting with the condition categories, saved as
                                        # gbm_v2 / gbm_uncalibrated_v2; the app files stay untouched
"""

# Standard library: json files, Python version, command-line arguments, timing, today's date, file paths.
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

# Saving and loading models (joblib), numbers (numpy) and tables (pandas).
import joblib
import numpy as np
import pandas as pd
# scikit-learn: version number, copying models, calibration, column preparation, the gradient boosting and
# logistic regression models, gap filling, metrics, cross-validation, pipelines and encoders/scalers.
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# The other files of src/ are imported by their plain name: evaluate (metrics and test scoring), features
# (feature lists and labels) and predict (turns trials into the model's input table).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES, CONDITION_FEATURES, FEATURE_COLUMNS, LABELS,
                      NON_FEATURE_COLUMNS, NUMERIC_FEATURES)
from predict import to_model_frame

# Paths: the project folder, the data file and the places where models and references are saved.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "trials.parquet"
MODELS_DIR = ROOT / "models"
SCORE_REFERENCE_PATH = MODELS_DIR / "score_reference.npy"
EXPLAIN_REFERENCE_PATH = MODELS_DIR / "explain_reference.json"

# First and last start date of each split. Must be the same as in build_dataset.py; check_dataset() stops the
# run if the data file does not fit.
SPLITS = {
    "train": (pd.Timestamp("2008-01-01"), pd.Timestamp("2016-12-31")),
    "test": (pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")),
}

# v1: the 29 original design features. v2: v1 plus the condition categories. Both are trained the same way.
# Rule we fixed before the first v2 run: the app only switches to v2 if it raises the test ROC-AUC of the
# calibrated model by at least 0.005 and the cross-validation agrees (numbers in models/CLAUDE.md).
FEATURES_V1 = []
for column in FEATURE_COLUMNS:
    if column not in CONDITION_FEATURES:
        FEATURES_V1.append(column)
FEATURE_SETS = {"v1": FEATURES_V1, "v2": list(FEATURE_COLUMNS)}
APP_FEATURE_SET = "v1"

RANDOM_STATE = 42
CV_FOLDS = 5
REFIT_METRIC = "average_precision"  # PR-AUC, the metric that matters when only about 11% are positive
CALIBRATION_METHOD = "isotonic"
# Calibration only maps scores to observed rates and tunes nothing, so it uses shuffled folds. The training
# rows are sorted by date, so folds without shuffling would each calibrate on one era only.
CALIBRATION_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# The settings we try. Small on purpose: every extra setting is one more chance to fit the folds by luck. Both
# grids turned out flat (all settings within one standard deviation), so bigger grids would not help.
PARAM_GRIDS = {
    "baseline_lr": {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "gbm": {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__l2_regularization": [0.0, 1.0],
    },
}


# ---------------------------------------------------------------------------------------------------------------
# Loading the data
# ---------------------------------------------------------------------------------------------------------------

def check_dataset(df):
    """Stop with an error if the data file breaks the label, date or split rules.

    This protects us from training on an old or hand-edited trials.parquet, and from a display column that
    ends up in the feature list by mistake.

    Input: df is the trials table (pandas DataFrame) read from data/processed/trials.parquet.

    Output: nothing. It raises a ValueError if a rule is broken.
    """
    # No column may be a feature and a meta/display column at the same time.
    overlap = set(FEATURE_COLUMNS) & set(NON_FEATURE_COLUMNS)
    if overlap:
        raise ValueError(f"Meta/display columns listed as features: {sorted(overlap)}")

    # All columns we need must be there.
    missing = []
    for column in FEATURE_COLUMNS + ["label", "start_date", "split"]:
        if column not in df:
            missing.append(column)
    if missing:
        raise ValueError(f"trials.parquet lacks columns: {missing}")

    # Only finished trials, and the label must fit the status.
    if not set(df["overall_status"]) <= set(LABELS):
        raise ValueError(f"Unexpected statuses: {set(df['overall_status']) - set(LABELS)}")
    if not (df["overall_status"].map(LABELS) == df["label"]).all():
        raise ValueError("label does not match overall_status (TERMINATED=1, COMPLETED=0)")

    # The split column must fit the start dates.
    for split, (start, end) in SPLITS.items():
        in_window = df["start_date"].between(start, end)
        if not (in_window == (df["split"] == split)).all():
            raise ValueError(f"split='{split}' rows do not match start {start:%Y-%m-%d}"
                             f"..{end:%Y-%m-%d}; rebuild with src/build_dataset.py")


def load_split(split, df=None, features=None):
    """Load "train" or "test" as (X, y). X only has model input columns, y is the label (1 = stopped early).

    Without df the data file is read from disk. Without features the app's feature set is used.

    Input: split is "train" or "test" (text); df is the trials table (DataFrame) or None to read it from disk;
    features is a list of feature names or None for the app's feature set.

    Output: a tuple (X, y). X is a DataFrame of the model input columns sorted by start date, y is a numpy
    array of 0/1 labels in the same order.
    """
    # Every load is checked, so a wrong data file stops the run.
    if df is None:
        df = pd.read_parquet(DATA_PATH)
    check_dataset(df)
    if features is None:
        features = FEATURE_SETS[APP_FEATURE_SET]

    # Sorted by start date (the trial number decides ties), so the folds of make_cv() run from old to new.
    part = df[df["split"] == split].sort_values(["start_date", "nct_id"], kind="stable")
    X = to_model_frame(part, features)
    y = part["label"].to_numpy(dtype=int)
    return X, y


def split_features(features):
    """Sort a list of feature names into its categorical, binary and numeric columns.

    Input: features is a list of feature names (text).

    Output: a dict with the keys "categorical", "binary" and "numeric", each holding a list of feature names.
    """
    parts = {"categorical": [], "binary": [], "numeric": []}
    for column in CATEGORICAL_FEATURES:
        if column in features:
            parts["categorical"].append(column)
    for column in BINARY_FEATURES:
        if column in features:
            parts["binary"].append(column)
    for column in NUMERIC_FEATURES:
        if column in features:
            parts["numeric"].append(column)
    return parts


# ---------------------------------------------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------------------------------------------

def build_lr_pipeline(features=None):
    """Logistic regression with the preparation it needs (fill gaps, scale, one-hot). Not fitted yet.

    Input: features is a list of feature names, or None for the app's feature set.

    Output: a scikit-learn Pipeline (preparation step "prep" and model step "clf") that is not fitted yet.
    """
    if features is None:
        features = FEATURE_SETS[APP_FEATURE_SET]
    parts = split_features(features)

    # Numbers: fill gaps with the median, then scale to mean 0 and standard deviation 1.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    # Flags stay 0/1 (no scaling), so a coefficient reads as "flag on compared with flag off".
    binary = SimpleImputer(strategy="most_frequent")
    # Categories: fill gaps with the most common value, then one 0/1 column per value.
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    prep = ColumnTransformer([("num", numeric, parts["numeric"]),
                              ("bin", binary, parts["binary"]),
                              ("cat", categorical, parts["categorical"])],
                             verbose_feature_names_out=False)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    return Pipeline([("prep", prep), ("clf", clf)])


def build_gbm_pipeline(drop=(), features=None):
    """Gradient boosting with the little preparation it needs. Not fitted yet.

    drop is a list of feature names to leave out. Only the ablations in evaluate.py use it.

    Input: drop is a list (or tuple) of feature names to leave out; features is a list of feature names, or
    None for the app's feature set.

    Output: a scikit-learn Pipeline (preparation step "prep" and model step "clf") that is not fitted yet.
    """
    if features is None:
        features = FEATURE_SETS[APP_FEATURE_SET]
    parts = split_features(features)

    categorical = []
    for column in parts["categorical"]:
        if column not in drop:
            categorical.append(column)
    other = []
    for column in parts["binary"] + parts["numeric"]:
        if column not in drop:
            other.append(column)

    # Categories only get a number code. A category the model never saw (for example a new API value) becomes
    # "missing" instead of crashing the app. Numbers and flags pass through unchanged.
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan)
    prep = ColumnTransformer([("cat", encoder, categorical), ("num", "passthrough", other)],
                             verbose_feature_names_out=False).set_output(transform="pandas")
    # Early stopping holds back 10% of the trials it is TRAINED on to decide when to stop, never test trials.
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", categorical_features=categorical,
        max_iter=1000, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])


def build_calibrated(base):
    """Put isotonic calibration around a fresh copy of a tuned pipeline. Not fitted yet.

    Calibration learns, on held-out folds, which real share of stopped trials belongs to which model score.
    This undoes the upward push of class_weight="balanced" and does not change the order of the trials.

    Input: base is a tuned scikit-learn pipeline (fitted or not); it is copied, not changed.

    Output: a CalibratedClassifierCV around a fresh copy of base, not fitted yet.
    """
    return CalibratedClassifierCV(clone(base), method=CALIBRATION_METHOD, cv=CALIBRATION_CV)


def explain_reference(X_train):
    """The "typical training trial" that predict.explain() compares a trial with.

    Numbers and 0/1 flags take the median of the training trials, categories their most common value. Gaps are
    ignored, so the typical trial is always a complete, ordinary design.

    Input: X_train is a DataFrame of the training trials with a column for every feature in FEATURE_COLUMNS.

    Output: a dict with a description ("contents"), the number of training trials ("n_train") and the typical
    value of every feature ("values"). It is saved as models/explain_reference.json.
    """
    values = {}
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            values[column] = str(X_train[column].mode(dropna=True).iloc[0])
        else:
            values[column] = float(X_train[column].median(skipna=True))
    return {
        "contents": "median (numbers, 0/1 flags) or most common value (categories) of "
                    "every feature on the train split (start 2008-2016)",
        "n_train": int(len(X_train)),
        "values": values,
    }


# ---------------------------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------------------------

def make_cv():
    """The one way of cutting the training trials into folds, used for everything (tuning, thresholds, race).

    TimeSeriesSplit cuts the date-sorted training trials into 6 blocks of equal size. Fold k trains on blocks
    1 to k and checks on block k+1. So the model is always checked on trials that started later than the ones
    it learned from, like in the app (DSF deck 10). A shuffled K-fold would let it learn from 2016 trials to
    predict 2009 ones, which DSF deck 16 rules out for data over time.

    Input: none.

    Output: a TimeSeriesSplit object with CV_FOLDS splits.
    """
    return TimeSeriesSplit(n_splits=CV_FOLDS)


def tune(pipeline, grid, X, y, cv):
    """Try every setting of the grid with cross-validation, then fit the best one on all training trials.

    Both metrics are recorded, PR-AUC picks the winner.

    Input: pipeline is the unfitted scikit-learn pipeline; grid is a dict of settings to try (names like
    "clf__C" with a list of values); X is the training DataFrame; y is the 0/1 label array; cv is the
    cross-validation splitter from make_cv().

    Output: the fitted GridSearchCV object. Its best_estimator_ is the best pipeline fitted on all of X.
    """
    search = GridSearchCV(pipeline, grid, cv=cv,
                          scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
                          refit=REFIT_METRIC, n_jobs=-1)
    search.fit(X, y)
    return search


def without_prefix(params):
    """Remove the "clf__" start from every setting name, e.g. {"clf__C": 1.0} becomes {"C": 1.0}.

    The "clf__" only tells the pipeline which step a setting belongs to.

    Input: params is a dict of settings whose names may start with "clf__".

    Output: a new dict with the same values and the shortened names.
    """
    plain = {}
    for key, value in params.items():
        plain[key.removeprefix("clf__")] = value
    return plain


def cv_summary(search):
    """Collect the best settings and the cross-validation scores of every setting, for the metrics file.

    Input: search is a fitted GridSearchCV object (from tune()).

    Output: a dict with the scheme, the best settings, their mean and standard deviation of ROC-AUC and PR-AUC,
    and a "grid" list with the scores of every setting.
    """
    results = search.cv_results_
    best = search.best_index_

    # One entry per setting. The spread across the folds is kept too, the horse race compares gaps with it.
    grid = []
    for i, params in enumerate(results["params"]):
        grid.append({
            "params": without_prefix(params),
            "cv_roc_auc": float(results["mean_test_roc_auc"][i]),
            "cv_roc_auc_std": float(results["std_test_roc_auc"][i]),
            "cv_pr_auc": float(results["mean_test_average_precision"][i]),
            "cv_pr_auc_std": float(results["std_test_average_precision"][i]),
        })

    return {
        "scheme": f"TimeSeriesSplit(n_splits={CV_FOLDS}) on the train split sorted by start "
                  "date (expanding window: train on earlier trials, validate on the next block)",
        "refit_metric": "pr_auc",
        "best_params": without_prefix(search.best_params_),
        "best_roc_auc_mean": float(results["mean_test_roc_auc"][best]),
        "best_roc_auc_std": float(results["std_test_roc_auc"][best]),
        "best_pr_auc_mean": float(results["mean_test_average_precision"][best]),
        "best_pr_auc_std": float(results["std_test_average_precision"][best]),
        "grid": grid,
    }


def time_ordered_predictions(estimator, X, y, cv):
    """Scores for every block the model is checked on, each from a copy that only saw earlier trials.

    scikit-learn's cross_val_predict cannot do this, because it wants every row to be checked exactly once and
    our first block is only used for training. So we run the folds by hand.

    Input: estimator is an unfitted (or fitted, it is copied) model with predict_proba; X is the training
    DataFrame sorted by date; y is the 0/1 label array; cv is the splitter from make_cv().

    Output: a tuple (scores, labels) of two numpy arrays for the checked rows, in date order. The scores are
    the predicted probabilities of TERMINATED.
    """
    scores = []
    labels = []
    for train_rows, check_rows in cv.split(X, y):
        model = clone(estimator)
        model.fit(X.iloc[train_rows], y[train_rows])
        scores.append(model.predict_proba(X.iloc[check_rows])[:, 1])
        labels.append(y[check_rows])
    return np.concatenate(scores), np.concatenate(labels)


def out_of_fold(estimator, X, y, cv):
    """Out-of-fold scores of the training trials and the yes/no threshold with the best F1 on them.

    Every scored trial gets its score from a copy of the model that never saw it and only learned from earlier
    trials. So the threshold (and the reference for the percentile) is not fitted to its own data, and the
    test years stay untouched.

    Input: estimator is the model to check (it is copied for every fold); X is the training DataFrame sorted by
    date; y is the 0/1 label array; cv is the splitter from make_cv().

    Output: a tuple (oof, info). oof is a numpy array of out-of-fold scores; info is a dict with the threshold
    ("value"), the rule, the number of scored trials and the out-of-fold F1, ROC-AUC, PR-AUC, Brier score and
    share of terminated trials.
    """
    oof, y_oof = time_ordered_predictions(estimator, X, y, cv)
    threshold, oof_f1 = evaluate.best_f1_threshold(y_oof, oof)
    info = {
        "value": threshold,
        "rule": "maximises F1 on out-of-fold predictions of the validation blocks (same CV folds)",
        "n_scored": int(len(oof)),
        "oof_f1": oof_f1,
        "oof_roc_auc": float(roc_auc_score(y_oof, oof)),
        "oof_pr_auc": float(average_precision_score(y_oof, oof)),
        "oof_brier": float(brier_score_loss(y_oof, oof)),
        "oof_terminated_rate": float(y_oof.mean()),
    }
    return oof, info


# ---------------------------------------------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------------------------------------------

def split_summary(df):
    """Describe the split rule and the size and share of stopped trials per split, for the metrics file.

    Input: df is the trials table (DataFrame) with the columns "split", "start_date" and "label".

    Output: a dict with the rule and, for "train" and "test", the start window, the observed first and last
    start date, the number of trials, the number terminated and the terminated rate.
    """
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


def model_details(model):
    """Collect a few facts about a fitted model that are worth keeping next to the model file.

    Input: model is a fitted CalibratedClassifierCV or a fitted Pipeline (gradient boosting or logistic
    regression).

    Output: a dict with the estimator name, the class weight and the number of iterations (boosting) or model
    columns (logistic regression); for the calibrated model also the method and the iterations per fold.
    """
    # The calibrated model holds one fitted pipeline per calibration fold.
    if isinstance(model, CalibratedClassifierCV):
        iterations = []
        for calibrated in model.calibrated_classifiers_:
            iterations.append(int(calibrated.estimator.named_steps["clf"].n_iter_))
        return {"estimator": "CalibratedClassifierCV(HistGradientBoostingClassifier)",
                "class_weight": "balanced (inside each fold's pipeline)",
                "calibration_method": model.method,
                "n_calibration_folds": len(iterations),
                "n_iter_per_fold": iterations}

    # A plain pipeline: gradient boosting or logistic regression.
    clf = model.named_steps["clf"]
    details = {"estimator": type(clf).__name__, "class_weight": "balanced"}
    if isinstance(clf, HistGradientBoostingClassifier):
        details["n_iter"] = int(clf.n_iter_)
    else:
        details["n_model_columns"] = int(clf.coef_.shape[1])
    return details


def save(name, model, df, cv_info, threshold, feature_set, extra=None):
    """Write models/<name>.joblib and a metrics file that records how the model was made.

    evaluate.py later adds the test results to the same metrics file.

    Input: name is the file name without ending (text); model is the fitted model; df is the trials table
    (DataFrame); cv_info is the dict from cv_summary(); threshold is the dict from out_of_fold(); feature_set
    is "v1" or "v2"; extra is an optional dict of more entries for the report.

    Output: nothing. It writes models/<name>.joblib and the metrics json file.
    """
    features = FEATURE_SETS[feature_set]
    joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)

    feature_info = {"all": features}
    feature_info.update(split_features(features))

    report = {
        "model": name,
        "model_file": f"models/{name}.joblib",
        "trained_on": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "label": {"TERMINATED": 1, "COMPLETED": 0},
        "split": split_summary(df),
        "feature_set": feature_set,
        "features": feature_info,
    }
    report.update(model_details(model))
    report["cv"] = cv_info
    report["threshold"] = threshold
    if extra:
        report.update(extra)

    text = json.dumps(report, indent=2) + "\n"
    evaluate.metrics_path(name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------
# The whole training run
# ---------------------------------------------------------------------------------------------------------------

def tune_and_save(name, saved_name, pipeline, X_train, y_train, cv, df, feature_set):
    """Step 1 for one model: tune it, pick its threshold and save it.

    Input: name is the key in PARAM_GRIDS ("baseline_lr" or "gbm"); saved_name is the file name to save under;
    pipeline is the unfitted pipeline; X_train and y_train are the training table and labels; cv is the
    splitter from make_cv(); df is the trials table; feature_set is "v1" or "v2".

    Output: a tuple (fitted model, CV numbers). The CV numbers are the dict from cv_summary(). The model and its
    metrics file are also saved in models/.
    """
    start_time = time.time()
    grid = PARAM_GRIDS[name]
    print(f"\n--- {name}: grid search ({len(ParameterGrid(grid))} "
          f"settings x {CV_FOLDS} time-ordered folds on train) ---")
    search = tune(pipeline, grid, X_train, y_train, cv)
    cv_info = cv_summary(search)
    print(f"best {cv_info['best_params']}  CV ROC-AUC {cv_info['best_roc_auc_mean']:.4f}"
          f" ± {cv_info['best_roc_auc_std']:.4f}  CV PR-AUC "
          f"{cv_info['best_pr_auc_mean']:.4f} ± {cv_info['best_pr_auc_std']:.4f}")

    model = search.best_estimator_
    oof, threshold = out_of_fold(model, X_train, y_train, cv)
    print(f"F1-max threshold on out-of-fold train predictions: {threshold['value']:.4f} "
          f"(OOF F1 {threshold['oof_f1']:.4f})")

    save(saved_name, model, df, cv_info, threshold, feature_set)
    print(f"Saved models/{saved_name}.joblib ({time.time() - start_time:.0f}s)")
    return model, cv_info


def main(feature_set=APP_FEATURE_SET):
    """Tune, fit and save the models of one feature set, then evaluate them once on the test trials.

    The app's feature set writes the usual files (gbm.joblib and so on) and also trains the logistic
    regression. Any other feature set only trains the gradient boosting and saves it with the set as an ending
    (gbm_v2.joblib, gbm_uncalibrated_v2.joblib, score_reference_v2.npy). So a comparison run never replaces
    the app model.

    Input: feature_set is "v1" or "v2" (text); the default is the app's feature set.

    Output: nothing. It writes the model, metrics and reference files to models/ and prints progress.
    """
    features = FEATURE_SETS[feature_set]
    is_app_set = feature_set == APP_FEATURE_SET
    suffix = ""
    if not is_app_set:
        suffix = f"_{feature_set}"

    # The test split is loaded here, but it is only handed to evaluate.evaluate_saved() at the very end.
    df = pd.read_parquet(DATA_PATH)
    check_dataset(df)
    X_train, y_train = load_split("train", df, features)
    X_test, y_test = load_split("test", df, features)
    print(f"Train: {len(y_train)} trials ({y_train.mean():.2%} terminated); "
          f"test: {len(y_test)} trials ({y_test.mean():.2%} terminated)")
    print(f"Feature set {feature_set} ({len(features)}): {', '.join(features)}")

    # The same folds for every model, so their cross-validation numbers can be compared.
    cv = make_cv()
    MODELS_DIR.mkdir(exist_ok=True)

    # The typical training trial, over every column build_features makes, so it works for any feature set.
    X_all_columns, _ = load_split("train", df, FEATURE_COLUMNS)
    reference = explain_reference(X_all_columns)
    EXPLAIN_REFERENCE_PATH.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    # Step 1: tune and save the models without calibration. The tuned gradient boosting is saved as
    # "gbm_uncalibrated", because the name "gbm" belongs to its calibrated version (the model of the app).
    if is_app_set:
        tune_and_save("baseline_lr", "baseline_lr", build_lr_pipeline(features),
                      X_train, y_train, cv, df, feature_set)
    gbm, gbm_cv_info = tune_and_save("gbm", "gbm_uncalibrated" + suffix, build_gbm_pipeline(features=features),
                                     X_train, y_train, cv, df, feature_set)

    # Step 2: put calibration around the tuned gradient boosting. First get honest out-of-fold scores (for the
    # threshold and the percentile reference), then fit it on all training trials.
    start_time = time.time()
    print(f"\n--- gbm: {CALIBRATION_METHOD} calibration ({CALIBRATION_CV}) of the tuned GBM ---")
    calibrated = build_calibrated(gbm)
    oof, threshold = out_of_fold(calibrated, X_train, y_train, cv)
    print(f"OOF Brier {threshold['oof_brier']:.4f}, OOF ROC-AUC {threshold['oof_roc_auc']:.4f}; "
          f"F1-max threshold {threshold['value']:.4f} (OOF F1 {threshold['oof_f1']:.4f})")
    calibrated.fit(X_train, y_train)

    # The out-of-fold scores are saved sorted, so predict.percentile() only needs a quick search.
    reference_path = SCORE_REFERENCE_PATH.with_name(f"score_reference{suffix}.npy")
    np.save(reference_path, np.sort(oof))

    # The calibrated model keeps the cross-validation numbers of the model it wraps, with a note.
    cv_info = dict(gbm_cv_info)
    cv_info["note"] = (f"hyperparameters tuned on gbm_uncalibrated{suffix}; "
                       "calibration adds no tuned parameters")
    quantiles = {}
    for q in (10, 25, 50, 75, 90, 99):
        quantiles[f"p{q}"] = float(np.percentile(oof, q))
    extra = {
        "calibration": {
            "method": CALIBRATION_METHOD,
            "cv": f"CalibratedClassifierCV(cv={CALIBRATION_CV}) on the train split "
                  "(shuffled folds: calibration tunes nothing)",
            "base_model": f"gbm_uncalibrated{suffix}",
        },
        "score_reference": {
            "file": f"models/{reference_path.name}",
            "contents": "sorted out-of-fold calibrated P(TERMINATED) of the train "
                        "trials in the validation blocks of the time-ordered CV "
                        "(start about 2009-2016; the earliest block is never scored)",
            "n": int(len(oof)),
            "quantiles": quantiles,
        },
    }
    save(f"gbm{suffix}", calibrated, df, cv_info, threshold, feature_set, extra=extra)
    print(f"Saved models/gbm{suffix}.joblib and models/{reference_path.name} "
          f"({time.time() - start_time:.0f}s)")

    # Step 3: score the saved models once on the test years. This is the only place the test split is used.
    names = []
    for name in evaluate.MODEL_NAMES:
        if name.startswith("gbm") or is_app_set:
            names.append(name + suffix)
    evaluate.evaluate_saved(X_test, y_test, names=names, X_train=X_train, y_train=y_train,
                            suffix=suffix)


if __name__ == "__main__":
    # "python src/train.py --features v2" picks another feature set; without it the app's set is trained.
    chosen = APP_FEATURE_SET
    if "--features" in sys.argv:
        chosen = sys.argv[sys.argv.index("--features") + 1]
    if chosen not in FEATURE_SETS:
        sys.exit(f"Unknown feature set '{chosen}'. Choose one of: {', '.join(sorted(FEATURE_SETS))}")
    main(chosen)

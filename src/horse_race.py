# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Horse race: every candidate model on the same time-ordered folds, with the same metric.

The idea comes from the Data Science Fundamentals course (deck 16, "Tuning and a
horse race between models"): the models are the horses, they all run on the same
data, the same cross-validation folds and the same error measure, each with its
default settings and with a small tuned grid. The best validation score wins, but
only if its lead is larger than the spread across the folds; otherwise the
simpler model is just as good. The test split is not touched here: the chosen
model (the app's GBM) is scored once on test by ``evaluate.py``.

Horses (all fitted inside a Pipeline on the training folds only):

* ``baseline``: always predicts the training share of stopped trials (DSF deck 08:
  "a baseline first").
* ``logistic_regression`` and ``gbm``: the two models of ``train.py``. Their grid
  results are read from ``models/baseline_lr_metrics.json`` and
  ``models/gbm_uncalibrated_metrics.json``, which ``train.py`` computed on exactly
  these folds, so they are not fitted twice.
* ``tree``: one decision tree (DSF deck 17).
* ``forest``: a random forest (DSF deck 17).
* ``knn``: k-nearest neighbours with min-max scaling, the classifier of the CS
  course (chapter 6.4.4): k odd so that votes cannot tie.

Usage (after ``python src/train.py``, which writes the two metrics files it reads):
    python src/horse_race.py          # about 30-60 min on 2 cores; writes models/horse_race.json
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (CV_FOLDS, MODELS_DIR, RANDOM_STATE, load_split, make_cv,  # noqa: E402
                   split_features)

OUTPUT_PATH = MODELS_DIR / "horse_race.json"
METRIC = "pr_auc"  # the metric that picks the winner, as in train.py (refit on PR-AUC)

# Each new horse: label for the app, where it comes from in the courses, the estimator with its scikit-learn
# defaults, and a small grid. The defaults are always the first cell, so "default" and "tuned" come from the same
# run. Small grids on purpose (DSF deck 16: "keep tuning cheap, a modest search, then move on").
NEW_HORSES = {
    "baseline": {
        "label": "Baseline: always the average rate",
        "course": "DSF deck 08 (a baseline first)",
        "estimator": DummyClassifier(strategy="prior"),
        "grid": {},
    },
    "tree": {
        "label": "Decision tree",
        "course": "DSF deck 17 (one tree)",
        "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        # Default first: no depth limit, which memorises the training trials (deck 17: "a full tree memorizes").
        "grid": {"clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 200]},
    },
    "forest": {
        "label": "Random forest",
        "course": "DSF deck 17 (many trees)",
        "estimator": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "grid": {"clf__min_samples_leaf": [1, 20]},
    },
    "knn": {
        "label": "k-nearest neighbours",
        "course": "CS course 6.4.4 (KNN, min-max scaling)",
        "estimator": KNeighborsClassifier(n_jobs=-1),
        # 5 is the scikit-learn default; all odd, so a vote between the two classes cannot tie (CS course 6.4.4).
        "grid": {"clf__n_neighbors": [5, 51, 201]},
    },
}
# The two horses train.py already raced on these folds: where their results are and which cell is the default.
TRAINED_HORSES = {
    "logistic_regression": {
        "label": "Logistic regression",
        "course": "DSF deck 11 (logistic regression)",
        "metrics_file": "baseline_lr_metrics.json",
        "default": {"C": 1.0},
    },
    "gbm": {
        "label": "Gradient boosting (the app model)",
        "course": "DSF deck 17 (boosting)",
        "metrics_file": "gbm_uncalibrated_metrics.json",
        "default": {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.0},
    },
}
# Simplest first: used when two horses are closer than the spread across the folds (DSF deck 16, rule 4).
SIMPLICITY = ["baseline", "logistic_regression", "knn", "tree", "forest", "gbm"]


def build_onehot_pipeline(estimator, features) -> Pipeline:
    """Impute, scale numbers to [0, 1] and one-hot encode categories, then the estimator.

    Min-max scaling is the one the CS course uses for KNN, where distances need every input on the same scale.
    Trees do not care about scaling (DSF deck 06), so the same pipeline is fair to all new horses.
    """
    parts = split_features(features)
    # Numbers: median for gaps, then [0, 1]. Flags: most common value for gaps, they are 0/1 already.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    # Categories: one 0/1 column per value, so they are on the same [0, 1] scale as everything else.
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", numeric, parts["numeric"]),
                              ("bin", binary, parts["binary"]),
                              ("cat", categorical, parts["categorical"])])
    return Pipeline([("prep", prep), ("clf", estimator)])


def cell(params, roc, roc_std, pr, pr_std) -> dict:
    """One grid cell in the shape every horse uses in the JSON."""
    return {"params": params, "cv_roc_auc": float(roc), "cv_roc_auc_std": float(roc_std),
            "cv_pr_auc": float(pr), "cv_pr_auc_std": float(pr_std)}


def run_new_horse(name, spec, X, y, cv) -> dict:
    """Grid-search one new horse on the shared folds; return its default and best cell."""
    # No refit: the race only compares validation scores, and the winner is already the saved app model.
    search = GridSearchCV(build_onehot_pipeline(spec["estimator"], list(X.columns)), spec["grid"] or [{}],
                          cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                          refit=False, n_jobs=1)
    search.fit(X, y)
    res = search.cv_results_
    grid = [cell({k.removeprefix("clf__"): v for k, v in p.items()},
                 res["mean_test_roc_auc"][i], res["std_test_roc_auc"][i],
                 res["mean_test_pr_auc"][i], res["std_test_pr_auc"][i])
            for i, p in enumerate(res["params"])]
    return {"grid": grid, "default": grid[0]}


def read_trained_horse(spec) -> dict:
    """The grid train.py computed for one model, with the cell that holds scikit-learn's defaults."""
    metrics = json.loads((MODELS_DIR / spec["metrics_file"]).read_text(encoding="utf-8"))
    if "TimeSeriesSplit" not in metrics["cv"]["scheme"]:
        raise ValueError(f"{spec['metrics_file']} was not made with the time-ordered folds; "
                         "run python src/train.py first")
    grid = [cell(c["params"], c["cv_roc_auc"], c["cv_roc_auc_std"], c["cv_pr_auc"], c["cv_pr_auc_std"])
            for c in metrics["cv"]["grid"]]
    default = next(c for c in grid if c["params"] == spec["default"])
    return {"grid": grid, "default": default}


def verdict(horses: dict) -> dict:
    """The winner by validation PR-AUC, and whether its lead is larger than the spread across the folds."""
    ranked = sorted(horses, key=lambda n: horses[n]["tuned"][f"cv_{METRIC}"], reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    lead = horses[winner]["tuned"][f"cv_{METRIC}"] - horses[runner_up]["tuned"][f"cv_{METRIC}"]
    spread = horses[winner]["tuned"][f"cv_{METRIC}_std"]
    # Every horse whose score is within one spread of the winner counts as a tie; the simplest of them is named too.
    tied = [n for n in ranked if horses[winner]["tuned"][f"cv_{METRIC}"] - horses[n]["tuned"][f"cv_{METRIC}"] <= spread]
    return {"ranking": ranked, "winner": winner, "runner_up": runner_up,
            "lead": float(lead), "winner_spread": float(spread), "lead_larger_than_spread": bool(lead > spread),
            "tied_with_winner": tied, "simplest_tied": min(tied, key=SIMPLICITY.index)}


def main() -> None:
    """Run the new horses, read the two trained ones, pick the winner and write models/horse_race.json."""
    X, y = load_split("train")
    cv = make_cv()
    print(f"Horse race on {len(y):,} train trials, {CV_FOLDS} time-ordered folds, winner by {METRIC}")
    horses = {}
    # The new horses: fitted here on the shared folds, defaults first.
    for name, spec in NEW_HORSES.items():
        t0 = time.time()
        result = run_new_horse(name, spec, X, y, cv)
        horses[name] = {"label": spec["label"], "course": spec["course"], **result}
        print(f"{name}: {len(result['grid'])} settings ({time.time() - t0:.0f}s)")
    # The two models of train.py: same folds, results already in their metrics files.
    for name, spec in TRAINED_HORSES.items():
        horses[name] = {"label": spec["label"], "course": spec["course"], **read_trained_horse(spec)}
    # Tuned = the best cell of each grid by the race metric.
    for horse in horses.values():
        horse["tuned"] = max(horse["grid"], key=lambda c: c[f"cv_{METRIC}"])

    result = {
        "made_on": date.today().isoformat(),
        "folds": f"TimeSeriesSplit(n_splits={CV_FOLDS}) on the train split sorted by start date "
                 "(train on earlier trials, validate on the next block)",
        "metric": "PR-AUC (average precision) picks the winner; ROC-AUC is shown too",
        "test_split": "not used here; the chosen model is scored once on test by evaluate.py",
        "horses": horses,
        "verdict": verdict(horses),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # A short table for the terminal and the report.
    print(f"\n{'horse':22} {'default PR-AUC':>16} {'tuned PR-AUC':>16} {'tuned ROC-AUC':>14}  tuned settings")
    for name in result["verdict"]["ranking"]:
        d, t = horses[name]["default"], horses[name]["tuned"]
        print(f"{name:22} {d['cv_pr_auc']:.4f} ± {d['cv_pr_auc_std']:.4f} {t['cv_pr_auc']:.4f} ± "
              f"{t['cv_pr_auc_std']:.4f} {t['cv_roc_auc']:14.4f}  {t['params']}")
    v = result["verdict"]
    print(f"\nWinner: {v['winner']}, lead {v['lead']:.4f} over {v['runner_up']} "
          f"({'larger' if v['lead_larger_than_spread'] else 'not larger'} than the spread {v['winner_spread']:.4f}).")
    print(f"Saved {OUTPUT_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()

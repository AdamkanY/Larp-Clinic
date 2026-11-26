"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Horse race: every candidate model on the same time-ordered folds, with the same metric.

The idea is from the Data Science Fundamentals course (deck 16, "Tuning and a horse race between models"): the
models are the horses. They all run on the same data, the same cross-validation folds and the same metric, once
with their default settings and once with a small grid of settings. The best validation score wins, but only
if its lead is larger than the spread across the folds. Otherwise the simpler model is just as good. The test
trials are not touched here: the chosen model (the app's gradient boosting) is scored once on them by
evaluate.py.

The horses (all fitted inside a Pipeline, on the training part of each fold only):

* baseline: always predicts the share of stopped trials in the training data (DSF deck 08, "a baseline first").
* logistic_regression and gbm: the two models of train.py. Their results are read from
  models/baseline_lr_metrics.json and models/gbm_uncalibrated_metrics.json. train.py computed them on exactly
  these folds, so we do not fit them twice.
* tree: one decision tree (DSF deck 17).
* forest: a random forest (DSF deck 17).
* knn: k-nearest neighbours with min-max scaling, the classifier of the CS course (chapter 6.4.4), with an
  odd k so that a vote cannot end in a tie.

Usage (after "python src/train.py", which writes the two metrics files this script reads):
    python src/horse_race.py          # about 30-60 minutes on 2 cores; writes models/horse_race.json
"""

# Standard library: json files, the import path, timing, today's date and file paths.
import json
import sys
import time
from datetime import date
from pathlib import Path

# scikit-learn: column preparation, the baseline, forest, tree and KNN models, gap filling, grid search,
# pipelines and the scalers/encoders.
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# The other files of src/ are imported by their plain name: train.py gives the shared folds, data loading and
# settings, so every horse runs on exactly the same setup.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CV_FOLDS, MODELS_DIR, RANDOM_STATE, load_split, make_cv, split_features, without_prefix

# Output file and the metric that picks the winner.
OUTPUT_PATH = MODELS_DIR / "horse_race.json"
METRIC = "pr_auc"  # the metric that picks the winner, like in train.py

# The new horses. For each one: the name the app shows, where it comes from in the courses, the model with
# scikit-learn's default settings, and a small grid. The default settings are always the FIRST setting of the
# grid, so "default" and "tuned" come from the same run. The grids are small on purpose (DSF deck 16: "keep
# tuning cheap, a modest search, then move on").
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
        # Default first: no depth limit. Such a tree learns the training trials by heart (deck 17).
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
        # 5 is scikit-learn's default. All values are odd, so a vote between two classes cannot end in a tie.
        "grid": {"clf__n_neighbors": [5, 51, 201]},
    },
}

# The two horses that train.py already ran on these folds: the file with their results, and which setting is
# scikit-learn's default.
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

# From the simplest model to the most complex one. Used when two horses are closer than the spread across the
# folds (DSF deck 16, rule 4): then the simpler one is named.
SIMPLICITY = ["baseline", "logistic_regression", "knn", "tree", "forest", "gbm"]


def build_onehot_pipeline(estimator, features):
    """A pipeline for the new horses: fill gaps, scale numbers to 0-1, one-hot encode categories, then the model.

    Min-max scaling is what the CS course uses for KNN, where distances need every input on the same scale.
    Trees do not care about scaling (DSF deck 06), so the same pipeline is fair to all new horses.

    Input: estimator is the unfitted scikit-learn model to put at the end; features is a list of feature names.

    Output: a scikit-learn Pipeline (steps "prep" and "clf") that is not fitted yet.
    """
    parts = split_features(features)
    # Numbers: the median for gaps, then scaled to 0-1. Flags: the most common value for gaps (they are 0/1).
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    # Categories: one 0/1 column per value, so they are on the same 0-1 scale as everything else.
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", numeric, parts["numeric"]),
                              ("bin", binary, parts["binary"]),
                              ("cat", categorical, parts["categorical"])])
    return Pipeline([("prep", prep), ("clf", estimator)])


def cell(params, roc, roc_std, pr, pr_std):
    """Pack the result of one setting into the form that every horse uses in the JSON file.

    Input: params is a dict of the settings; roc and roc_std are the mean and standard deviation of the
    ROC-AUC over the folds; pr and pr_std are the same for the PR-AUC (numbers).

    Output: a dict with "params" and the four scores as plain floats.
    """
    return {"params": params, "cv_roc_auc": float(roc), "cv_roc_auc_std": float(roc_std),
            "cv_pr_auc": float(pr), "cv_pr_auc_std": float(pr_std)}


def run_new_horse(name, spec, X, y, cv):
    """Try every setting of one new horse on the shared folds.

    Input: name is the horse's key in NEW_HORSES (text); spec is its entry (a dict with the estimator and the
    grid); X is the training DataFrame; y is the 0/1 label array; cv is the splitter from make_cv().

    Output: a dict with "grid" (a list of result dicts, one per setting) and "default" (the result of the
    first setting, which is scikit-learn's default).
    """
    pipeline = build_onehot_pipeline(spec["estimator"], list(X.columns))
    # A horse without a grid still runs once, with its default settings.
    grid = spec["grid"]
    if not grid:
        grid = [{}]
    # refit=False: the race only compares validation scores, and the winner is already saved as the app model.
    search = GridSearchCV(pipeline, grid, cv=cv,
                          scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                          refit=False, n_jobs=1)
    search.fit(X, y)

    results = search.cv_results_
    cells = []
    for i, params in enumerate(results["params"]):
        cells.append(cell(without_prefix(params),
                          results["mean_test_roc_auc"][i], results["std_test_roc_auc"][i],
                          results["mean_test_pr_auc"][i], results["std_test_pr_auc"][i]))
    return {"grid": cells, "default": cells[0]}


def read_trained_horse(spec):
    """Read the results train.py stored for one model, and find the one with scikit-learn's default settings.

    Input: spec is the model's entry in TRAINED_HORSES (a dict with the metrics file name and the default
    settings).

    Output: a dict with "grid" (a list of result dicts, one per setting) and "default" (the result of the
    default settings). It raises a ValueError if the file is old or has no default result.
    """
    metrics = json.loads((MODELS_DIR / spec["metrics_file"]).read_text(encoding="utf-8"))
    if "TimeSeriesSplit" not in metrics["cv"]["scheme"]:
        raise ValueError(f"{spec['metrics_file']} was not made with the time-ordered folds; "
                         "run python src/train.py first")

    cells = []
    default = None
    for c in metrics["cv"]["grid"]:
        new_cell = cell(c["params"], c["cv_roc_auc"], c["cv_roc_auc_std"], c["cv_pr_auc"], c["cv_pr_auc_std"])
        cells.append(new_cell)
        if default is None and new_cell["params"] == spec["default"]:
            default = new_cell
    if default is None:
        raise ValueError(f"{spec['metrics_file']} has no result for the default settings {spec['default']}")
    return {"grid": cells, "default": default}


def verdict(horses):
    """Decide who wins by validation PR-AUC, and whether the lead is larger than the spread across the folds.

    Input: horses is a dict {horse name: dict} where every entry has a "tuned" result.

    Output: a dict with the ranking (best first), the winner, the runner-up, the lead, the winner's spread,
    whether the lead is larger than the spread, the horses tied with the winner and the simplest of them.
    """
    def tuned_score(name):
        """Look up the tuned validation score of one horse.

        Input: name is the horse's name (text), a key of horses.

        Output: the tuned cross-validation score of the race metric (float).
        """
        return horses[name]["tuned"][f"cv_{METRIC}"]

    # Best score first.
    ranked = sorted(horses, key=tuned_score, reverse=True)
    winner = ranked[0]
    runner_up = ranked[1]
    lead = tuned_score(winner) - tuned_score(runner_up)
    spread = horses[winner]["tuned"][f"cv_{METRIC}_std"]

    # Every horse within one spread of the winner counts as a tie. The simplest of them is named too.
    tied = []
    for name in ranked:
        if tuned_score(winner) - tuned_score(name) <= spread:
            tied.append(name)
    simplest_tied = min(tied, key=SIMPLICITY.index)

    return {"ranking": ranked, "winner": winner, "runner_up": runner_up,
            "lead": float(lead), "winner_spread": float(spread), "lead_larger_than_spread": bool(lead > spread),
            "tied_with_winner": tied, "simplest_tied": simplest_tied}


def main():
    """Run the new horses, read the two trained ones, pick the winner and write models/horse_race.json.

    Input: none.

    Output: nothing. It writes models/horse_race.json and prints a short table and the winner.
    """
    X, y = load_split("train")
    cv = make_cv()
    print(f"Horse race on {len(y):,} train trials, {CV_FOLDS} time-ordered folds, winner by {METRIC}")
    horses = {}

    # The new horses are fitted here on the shared folds.
    for name, spec in NEW_HORSES.items():
        start_time = time.time()
        result = run_new_horse(name, spec, X, y, cv)
        horses[name] = {"label": spec["label"], "course": spec["course"],
                        "grid": result["grid"], "default": result["default"]}
        print(f"{name}: {len(result['grid'])} settings ({time.time() - start_time:.0f}s)")

    # The two models of train.py ran on the same folds, their results are already in their metrics files.
    for name, spec in TRAINED_HORSES.items():
        result = read_trained_horse(spec)
        horses[name] = {"label": spec["label"], "course": spec["course"],
                        "grid": result["grid"], "default": result["default"]}

    # "Tuned" is the best setting of each horse by the race metric (the first one if two are equal).
    for horse in horses.values():
        best = horse["grid"][0]
        for c in horse["grid"]:
            if c[f"cv_{METRIC}"] > best[f"cv_{METRIC}"]:
                best = c
        horse["tuned"] = best

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
        d = horses[name]["default"]
        t = horses[name]["tuned"]
        print(f"{name:22} {d['cv_pr_auc']:.4f} ± {d['cv_pr_auc_std']:.4f} {t['cv_pr_auc']:.4f} ± "
              f"{t['cv_pr_auc_std']:.4f} {t['cv_roc_auc']:14.4f}  {t['params']}")
    v = result["verdict"]
    if v["lead_larger_than_spread"]:
        comparison = "larger"
    else:
        comparison = "not larger"
    print(f"\nWinner: {v['winner']}, lead {v['lead']:.4f} over {v['runner_up']} "
          f"({comparison} than the spread {v['winner_spread']:.4f}).")
    print(f"Saved {OUTPUT_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    # ---- 1. Run the whole horse race ----
    main()

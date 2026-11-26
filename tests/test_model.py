"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Offline checks for the model protocol: time-ordered cross-validation and the horse race verdict.

Run with ``python tests/test_model.py`` (or pytest). Uses small invented tables; one check reads the real
``data/processed/trials.parquet`` and is skipped when it is missing.
"""

import sys  # edit the module search path
from pathlib import Path  # file paths that work on every system

import numpy as np  # arrays and numeric helpers
import pandas as pd  # tables (DataFrame)
from sklearn.dummy import DummyClassifier  # a model that ignores its inputs, to test the scoring code

# Make src/ importable, then load the two modules under test (the model comparison and the training code).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import horse_race  # verdict of the model comparison
import train  # time-ordered cross-validation and data loading


def test_time_ordered_folds():
    """Check that every cross-validation fold trains only on earlier rows than it validates on.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # 120 rows in date order, split into the time-ordered folds.
    X = pd.DataFrame({"x": np.arange(120)})
    folds = list(train.make_cv().split(X))
    assert len(folds) == train.CV_FOLDS
    # In every fold all training rows come before all validation rows.
    for train_idx, valid_idx in folds:
        assert train_idx.max() < valid_idx.min()
    # Together the folds validate 100 rows; the first block of 20 only ever trains.
    validated = np.concatenate([valid for _, valid in folds])
    assert len(validated) == 100 and validated.min() == 20  # 6 blocks of 20 rows, block 1 only trains


def test_time_ordered_predictions():
    """Check that time_ordered_predictions scores only the validated rows, in date order.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Invented data and a model that always predicts the share of ones it saw in training.
    X = pd.DataFrame({"x": np.arange(120)})
    y = np.array([0, 1] * 60)
    proba, labels = train.time_ordered_predictions(DummyClassifier(strategy="prior"), X, y, train.make_cv())
    # Only the 100 validated rows get a score, with their own labels.
    assert len(proba) == len(labels) == 100
    assert np.array_equal(labels, y[20:])
    assert np.allclose(proba, 0.5)  # the prior of every training window is exactly one half


def test_train_split_sorted_by_start_date():
    """Check that load_split returns the train trials from oldest to newest start date (needs the real parquet).

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    if not train.DATA_PATH.exists():
        print("skipped: data/processed/trials.parquet is missing")
        return
    # Load the real table and the train split, then look at the start dates of the rows in that order.
    df = pd.read_parquet(train.DATA_PATH)
    X, _ = train.load_split("train", df)
    dates = df.loc[X.index, "start_date"]
    assert dates.is_monotonic_increasing


def race(scores):
    """Build invented model results in the form that horse_race.json uses.

    Input: scores is a dict {model name: (tuned PR-AUC as float, spread of the PR-AUC across folds as float)}.

    Output: a dict {model name: {"tuned": {"cv_pr_auc": ..., "cv_pr_auc_std": ...}}}.
    """
    # One entry per model, with the two numbers the verdict reads.
    horses = {}
    for name, (pr_auc, spread) in scores.items():
        horses[name] = {"tuned": {"cv_pr_auc": pr_auc, "cv_pr_auc_std": spread}}
    return horses


def test_verdict_clear_winner():
    """Check that a lead larger than the winner's spread across the folds counts as a clear win.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # The winner leads by 0.05 with a spread of only 0.01.
    v = horse_race.verdict(race({"gbm": (0.25, 0.01), "forest": (0.20, 0.01), "baseline": (0.11, 0.0)}))
    assert v["winner"] == "gbm" and v["runner_up"] == "forest"
    assert v["lead_larger_than_spread"] and v["tied_with_winner"] == ["gbm"]


def test_verdict_tie_names_the_simplest():
    """Check that a model within one spread of the winner is a tie and that the simplest tied model is named.

    Within one spread of the winner it is a tie, and the simplest tied model is named (DSF deck 16, rule 4).

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # The two best models differ by less than the spread, so they tie.
    v = horse_race.verdict(race({"gbm": (0.250, 0.02), "logistic_regression": (0.240, 0.01),
                                 "baseline": (0.11, 0.0)}))
    assert not v["lead_larger_than_spread"]
    assert set(v["tied_with_winner"]) == {"gbm", "logistic_regression"}
    assert v["simplest_tied"] == "logistic_regression"


# ---- Run all tests when the file is started directly ----
if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

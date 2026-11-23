# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Offline checks for the model protocol: time-ordered cross-validation and the horse race verdict.

Run with ``python tests/test_model.py`` (or pytest). Uses small invented tables; one check reads the real
``data/processed/trials.parquet`` and is skipped when it is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import horse_race
import train


def test_time_ordered_folds():
    """Every fold trains only on earlier rows than it validates on, and the first block is never validated."""
    X = pd.DataFrame({"x": np.arange(120)})
    folds = list(train.make_cv().split(X))
    assert len(folds) == train.CV_FOLDS
    for train_idx, valid_idx in folds:
        assert train_idx.max() < valid_idx.min()
    validated = np.concatenate([valid for _, valid in folds])
    assert len(validated) == 100 and validated.min() == 20  # 6 blocks of 20 rows, block 1 only trains


def test_time_ordered_predictions():
    """Scores come only for the validated rows, in date order, from a model fitted on earlier rows."""
    X = pd.DataFrame({"x": np.arange(120)})
    y = np.array([0, 1] * 60)
    proba, labels = train.time_ordered_predictions(DummyClassifier(strategy="prior"), X, y, train.make_cv())
    assert len(proba) == len(labels) == 100
    assert np.array_equal(labels, y[20:])
    assert np.allclose(proba, 0.5)  # the prior of every training window is exactly one half


def test_train_split_sorted_by_start_date():
    """load_split hands the folds the train trials from oldest to newest (needs the real parquet)."""
    if not train.DATA_PATH.exists():
        print("skipped: data/processed/trials.parquet is missing")
        return
    df = pd.read_parquet(train.DATA_PATH)
    X, _ = train.load_split("train", df)
    dates = df.loc[X.index, "start_date"]
    assert dates.is_monotonic_increasing


def race(scores):
    """Invented horses: {name: (tuned PR-AUC, spread)} becomes the form horse_race.json uses."""
    horses = {}
    for name, (pr_auc, spread) in scores.items():
        horses[name] = {"tuned": {"cv_pr_auc": pr_auc, "cv_pr_auc_std": spread}}
    return horses


def test_verdict_clear_winner():
    """A lead larger than the winner's spread across the folds counts as a clear win."""
    v = horse_race.verdict(race({"gbm": (0.25, 0.01), "forest": (0.20, 0.01), "baseline": (0.11, 0.0)}))
    assert v["winner"] == "gbm" and v["runner_up"] == "forest"
    assert v["lead_larger_than_spread"] and v["tied_with_winner"] == ["gbm"]


def test_verdict_tie_names_the_simplest():
    """Within one spread of the winner it is a tie, and the simplest tied model is named (DSF deck 16, rule 4)."""
    v = horse_race.verdict(race({"gbm": (0.250, 0.02), "logistic_regression": (0.240, 0.01),
                                 "baseline": (0.11, 0.0)}))
    assert not v["lead_larger_than_spread"]
    assert set(v["tied_with_winner"]) == {"gbm", "logistic_regression"}
    assert v["simplest_tied"] == "logistic_regression"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

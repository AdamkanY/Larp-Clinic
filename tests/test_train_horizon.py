"""Smoke test: src/train_horizon.py runs end to end on invented trials and never touches the app model.

Run with ``python tests/test_train_horizon.py`` (or pytest). No network. About 900 made-up records go through
the real build_dataset.py functions into a temporary trials_3y.parquet; the grids are cut to one cell so the
run takes seconds. The numbers it produces mean nothing: the point is that every step works and that
check_dataset refuses files that break the horizon rules.
"""

import hashlib
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import build_dataset  # noqa: E402
import horizon  # noqa: E402
import train  # noqa: E402
import train_horizon  # noqa: E402
from features import build_features  # noqa: E402
from test_horizon import SNAPSHOT, study  # noqa: E402

TINY_GRIDS = {
    "baseline_lr": {"clf__C": [1.0]},
    "gbm": {"clf__learning_rate": [0.1], "clf__max_leaf_nodes": [15], "clf__l2_regularization": [0.0]},
}


def invented_studies(n=900, seed=7):
    """Made-up records spread over 2008 to early 2023; phase 2 is stopped early more often."""
    rng = random.Random(seed)
    studies = []
    for i in range(n):
        year = rng.randint(2008, 2023)
        month = rng.randint(1, 12 if year < 2023 else 4)
        phase = rng.choice(["PHASE1", "PHASE2", "PHASE3", "NA"])
        if rng.random() < (0.35 if phase == "PHASE2" else 0.1):
            status, months = "TERMINATED", rng.randint(2, 30)
        else:
            status, months = rng.choice(["COMPLETED"] * 8 + ["RECRUITING", "TERMINATED"]), rng.randint(40, 70)
        stop_year, stop_month = divmod(year * 12 + month - 1 + months, 12)
        stop = None if status == "RECRUITING" else f"{stop_year}-{stop_month + 1:02d}"
        studies.append(study(f"NCT{i:08d}", status, f"{year}-{month:02d}", stop, phases=(phase,)))
    return studies


def horizon_file(folder):
    """Write the invented trials_3y.parquet into `folder` and return its path."""
    studies = invented_studies()
    rows = [build_features(s) for s in studies]
    outcomes = [horizon.outcome_fields(s) for s in studies]
    df = build_dataset.build_horizon_dataset(build_dataset.merge_outcomes(rows, outcomes), SNAPSHOT)
    path = Path(folder) / "trials_3y.parquet"
    df.to_parquet(path)
    return path


def digest(path):
    """SHA-256 of a file, or None if it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def test_run_end_to_end():
    """Models, score reference and metrics are written; valid/test, references and per-year rows are there."""
    real_grids, train.PARAM_GRIDS = train.PARAM_GRIDS, TINY_GRIDS
    app_model = ROOT / "models" / "gbm.joblib"
    before = digest(app_model)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "horizon_3y"
            report = train_horizon.run(horizon_file(tmp), out, app_model_path=app_model)
            written = {p.name for p in out.iterdir()}
    finally:
        train.PARAM_GRIDS = real_grids
    assert {"gbm.joblib", "gbm_uncalibrated.joblib", "baseline_lr.joblib", "score_reference.npy",
            "metrics.json"} <= written
    assert digest(app_model) == before  # the app model is read, never written
    for split in ("valid", "test"):
        assert 0 <= report[split]["gbm"]["roc_auc"] <= 1 and "status_gbm_2008_2018" in report[split]
        assert (app_model.exists()) == ("app_model" in report[split])
    assert [y["start_year"] for y in report["valid"]["gbm"]["per_start_year"]] == [2019]
    assert {y["start_year"] for y in report["test"]["gbm"]["per_start_year"]} == {2020, 2021, 2022, 2023}
    assert report["split"]["test"]["start_to"] == "2023-05-05"
    assert report["drift_train_vs_test"][0]["feature"]


def test_check_dataset_refuses_bad_files():
    """A training file with an outcome column, a wrong split or no cutoff is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.read_parquet(horizon_file(tmp))
    assert train_horizon.check_dataset(df) == pd.Timestamp("2023-05-05")
    for broken in (df.assign(why_stopped="x"), df.assign(split="train"), df.assign(label=1)):
        try:
            train_horizon.check_dataset(broken)
        except ValueError:
            continue
        raise AssertionError("check_dataset accepted a broken file")
    no_cutoff = df.copy()
    no_cutoff.attrs = {}
    try:
        train_horizon.check_dataset(no_cutoff)
    except ValueError:
        return
    raise AssertionError("check_dataset accepted a file without a follow-up cutoff")


if __name__ == "__main__":
    test_check_dataset_refuses_bad_files()
    test_run_end_to_end()
    print("all ok")

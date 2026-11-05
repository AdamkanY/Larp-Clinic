"""Offline checks for src/predict.py: model inputs, explain() and the shared feature names.

Run with ``python tests/test_predict.py`` (or pytest). No network. The main explain() tests fit a small model on an
invented table where the answer is known; one extra check runs on the saved models and is skipped when
``models/gbm_uncalibrated.joblib`` or ``models/explain_reference.json`` is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, MAX_AGE_FILL,  # noqa: E402
                      build_features)
from labels import FEATURE_NAMES, NO_MAX_AGE, feature_name  # noqa: E402
from predict import explain, model_features, score, score_frame, to_model_frame  # noqa: E402
from test_features import STUDY  # noqa: E402
from train import build_gbm_pipeline  # noqa: E402

_FITTED = {}


def typical():
    """An invented 'typical trial': the reference explain() compares with."""
    ref = {c: 0.0 for c in FEATURE_COLUMNS}
    ref.update({c: None for c in CATEGORICAL_FEATURES})
    ref.update(phase="PHASE2", allocation="RANDOMIZED", masking="NONE", n_arms=2.0, min_age_years=18.0,
               max_age_years=80.0, has_max_age=1.0, eligibility_text_len=1000.0)
    return ref


def toy_model():
    """GBM pipeline fitted on invented trials whose risk depends only on has_dmc (up) and double blinding (down)."""
    if "model" not in _FITTED:
        rng = np.random.default_rng(0)
        n = 4000
        rows = pd.DataFrame([typical()] * n)
        rows["has_dmc"] = rng.integers(0, 2, n).astype(float)
        rows["masking"] = rng.choice(["NONE", "DOUBLE"], n)
        rows["eligibility_text_len"] = rng.integers(200, 4000, n).astype(float)  # noise, no effect
        logit = -2 + 2.0 * rows["has_dmc"] - 1.5 * (rows["masking"] == "DOUBLE")
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        _FITTED["model"] = build_gbm_pipeline(features=FEATURE_COLUMNS).fit(to_model_frame(rows), y)
    return _FITTED["model"]


def test_explain_finds_known_drivers():
    """The two factors that drive the invented outcome come first, with the right direction and plain names."""
    trial = {**typical(), "has_dmc": 1.0, "masking": "DOUBLE"}
    drivers = explain(trial, model=toy_model(), reference=typical())
    assert len(drivers) <= 5 and all(len(d) == 4 for d in drivers)
    names = [d[0] for d in drivers]
    assert set(names[:2]) == {"Has a data monitoring committee", "Blinding"}, drivers
    by_name = {d[0]: d for d in drivers}
    assert by_name["Has a data monitoring committee"][1:3] == ("Yes", "up")
    assert by_name["Blinding"][1:3] == ("Double blind", "down")
    sizes = [d[3] for d in drivers]
    assert all(s > 0 for s in sizes) and sizes == sorted(sizes, reverse=True)


def test_explain_typical_trial_has_no_drivers():
    """A trial equal to the reference pushes nothing; top_n caps the list."""
    model = toy_model()
    assert explain(typical(), model=model, reference=typical()) == []
    trial = {**typical(), "has_dmc": 1.0, "masking": "DOUBLE"}
    assert len(explain(trial, top_n=1, model=model, reference=typical())) == 1


def test_explain_moves_linked_age_columns_together():
    """No maximum age (120 + flag 0) is one factor, 'Maximum age: No upper limit', never 'Has a maximum age'."""
    trial = {**typical(), "has_dmc": 1.0, "max_age_years": MAX_AGE_FILL, "has_max_age": 0.0}
    drivers = explain(trial, top_n=40, model=toy_model(), reference=typical())
    names = [d[0] for d in drivers]
    assert "Has a maximum age" not in names
    assert all(d[1] == "No upper limit" for d in drivers if d[0] == "Maximum age")


def test_model_inputs():
    """Each model gets its own columns: an older model with fewer inputs keeps working."""
    frame = to_model_frame([build_features(STUDY)], ["phase", "n_arms"])
    assert list(frame.columns) == ["phase", "n_arms"] and frame["n_arms"].dtype == "float64"
    assert model_features(object()) == FEATURE_COLUMNS
    assert model_features(toy_model()) == FEATURE_COLUMNS
    try:
        to_model_frame([{"phase": "PHASE2"}], ["phase", "n_arms"])
    except ValueError as exc:
        assert "n_arms" in str(exc)
    else:
        raise AssertionError("a missing feature key must raise")


def test_every_feature_has_a_plain_name():
    """labels.py names every model input (no code-like fallback names in charts)."""
    assert set(FEATURE_COLUMNS) <= set(FEATURE_NAMES)
    assert all("_" not in feature_name(c) for c in FEATURE_COLUMNS)
    assert NO_MAX_AGE == MAX_AGE_FILL


def test_explain_saved_model():
    """On the saved model and reference: a real record gives up to five plain-named factors."""
    if not ((ROOT / "models" / "gbm_uncalibrated.joblib").exists()
            and (ROOT / "models" / "explain_reference.json").exists()):
        print("skipped test_explain_saved_model: model files missing")
        return
    drivers = explain(build_features(STUDY))
    assert 0 < len(drivers) <= 5
    assert all(name in FEATURE_NAMES.values() and direction in ("up", "down") and size > 0
               for name, _, direction, size in drivers)


def test_score_frame_matches_score_and_ignores_extra_columns():
    """Batch scoring equals score() row by row and feeds a model only the columns it was fitted on."""
    rows = pd.DataFrame([typical()] * 3)
    rows["has_dmc"] = [0.0, 1.0, 0.0]
    rows["masking"] = ["NONE", "NONE", "DOUBLE"]
    model = toy_model()
    batch = score_frame(rows, model)
    single = [score(r, model) for r in rows.to_dict("records")]
    assert np.allclose(batch, single)
    small = build_gbm_pipeline(features=FEATURE_COLUMNS[:5]).fit(
        to_model_frame(pd.DataFrame([typical()] * 40), FEATURE_COLUMNS[:5]), [0, 1] * 20)
    assert model_features(small) == FEATURE_COLUMNS[:5]
    assert len(score_frame(rows.assign(not_a_feature="x"), small)) == 3  # extra and unused columns are dropped


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

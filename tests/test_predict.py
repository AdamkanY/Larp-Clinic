"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Offline checks for src/predict.py: model inputs, explain() and the shared feature names.

Run with ``python tests/test_predict.py`` (or pytest). No network. The main explain() tests fit a small model on an
invented table where the answer is known; one extra check runs on the saved models and is skipped when
``models/gbm_uncalibrated.joblib`` or ``models/explain_reference.json`` is missing.
"""

import sys  # edit the module search path
from pathlib import Path  # file paths that work on every system

import numpy as np  # arrays and random numbers
import pandas as pd  # tables (DataFrame)

# Make src/ and tests/ importable, then load the code under test.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
# Feature lists and the feature function (the one shared by the app and the training code).
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, MAX_AGE_FILL, build_features
# Plain-English names of the features.
from labels import FEATURE_NAMES, NO_MAX_AGE, feature_name
# Scoring and explanation code under test.
from predict import explain, model_features, score, score_frame, to_model_frame
# The invented record from the feature tests.
from test_features import STUDY
# Builds the model pipeline that the toy model uses.
from train import build_gbm_pipeline

# The small model is fitted once and then reused by all tests (fitting takes a few seconds).
FITTED = {}


def typical():
    """Build an invented 'typical trial', the reference that explain() compares with.

    Input: none.

    Output: a dict with a value for every model feature (categories None, numbers 0.0, a few typical values set).
    """
    # Start with None for category features and 0.0 for the others.
    ref = {}
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            ref[column] = None
        else:
            ref[column] = 0.0
    # Then set the values that make the trial look typical.
    ref.update({"phase": "PHASE2", "allocation": "RANDOMIZED", "masking": "NONE", "n_arms": 2.0,
                "min_age_years": 18.0, "max_age_years": 80.0, "has_max_age": 1.0, "eligibility_text_len": 1000.0})
    return ref


def typical_with(changes):
    """Return the typical trial with a few values changed.

    Input: changes is a dict {feature name: new value}.

    Output: a dict like typical() with the changes applied.
    """
    # Copy the typical trial and overwrite the given values.
    trial = typical()
    trial.update(changes)
    return trial


def toy_model():
    """Return a GBM pipeline fitted on invented trials (risk depends only on has_dmc and double blinding).

    Input: none.

    Output: the fitted model pipeline; it is fitted on the first call only and reused afterwards.
    """
    # Fit only once; fitting takes a few seconds.
    if "model" not in FITTED:
        # 4000 invented trials, all typical except the three columns changed below.
        rng = np.random.default_rng(0)
        n = 4000
        rows = pd.DataFrame([typical()] * n)
        rows["has_dmc"] = rng.integers(0, 2, n).astype(float)
        rows["masking"] = rng.choice(["NONE", "DOUBLE"], n)
        rows["eligibility_text_len"] = rng.integers(200, 4000, n).astype(float)  # noise, no effect
        # The outcome depends on has_dmc and masking only; eligibility length is noise.
        logit = -2 + 2.0 * rows["has_dmc"] - 1.5 * (rows["masking"] == "DOUBLE")
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        # Fit the pipeline and keep it for the other tests.
        model = build_gbm_pipeline(features=FEATURE_COLUMNS)
        model.fit(to_model_frame(rows), y)
        FITTED["model"] = model
    return FITTED["model"]


def test_explain_finds_known_drivers():
    """Check that the two factors that drive the invented outcome come first, with the right direction and plain names.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # A trial with both known drivers set, explained against the typical trial.
    trial = typical_with({"has_dmc": 1.0, "masking": "DOUBLE"})
    drivers = explain(trial, model=toy_model(), reference=typical())
    assert len(drivers) <= 5 and all(len(d) == 4 for d in drivers)
    names = [d[0] for d in drivers]
    assert set(names[:2]) == {"Has a data monitoring committee", "Blinding"}, drivers
    # Direction: a monitoring committee pushes the risk up, double blinding pushes it down.
    by_name = {d[0]: d for d in drivers}
    assert by_name["Has a data monitoring committee"][1:3] == ("Yes", "up")
    assert by_name["Blinding"][1:3] == ("Double blind", "down")
    # Sizes are positive and sorted from biggest to smallest.
    sizes = [d[3] for d in drivers]
    assert all(s > 0 for s in sizes) and sizes == sorted(sizes, reverse=True)


def test_explain_typical_trial_has_no_drivers():
    """Check that a trial equal to the reference has no drivers and that top_n caps the list.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # The typical trial against itself: nothing pushes the risk.
    model = toy_model()
    assert explain(typical(), model=model, reference=typical()) == []
    # top_n=1 keeps only the biggest driver.
    trial = typical_with({"has_dmc": 1.0, "masking": "DOUBLE"})
    assert len(explain(trial, top_n=1, model=model, reference=typical())) == 1


def test_explain_moves_linked_age_columns_together():
    """Check that the linked age columns move together in explain().

    No maximum age (120 + flag 0) is one factor, 'Maximum age: No upper limit', never 'Has a maximum age'.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # A trial without a maximum age (120 and the flag set to 0), with many drivers allowed.
    trial = typical_with({"has_dmc": 1.0, "max_age_years": MAX_AGE_FILL, "has_max_age": 0.0})
    drivers = explain(trial, top_n=40, model=toy_model(), reference=typical())
    # The flag never shows up alone; the age shows up as 'No upper limit'.
    names = [d[0] for d in drivers]
    assert "Has a maximum age" not in names
    assert all(d[1] == "No upper limit" for d in drivers if d[0] == "Maximum age")


def test_model_inputs():
    """Check that each model gets its own columns, so an older model with fewer inputs keeps working.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Only the requested columns come out, with numbers as floats.
    frame = to_model_frame([build_features(STUDY)], ["phase", "n_arms"])
    assert list(frame.columns) == ["phase", "n_arms"] and frame["n_arms"].dtype == "float64"
    # A model without a feature list gets all features; a fitted pipeline reports its own.
    assert model_features(object()) == FEATURE_COLUMNS
    assert model_features(toy_model()) == FEATURE_COLUMNS
    # A missing feature key must raise a ValueError that names the key.
    try:
        to_model_frame([{"phase": "PHASE2"}], ["phase", "n_arms"])
    except ValueError as exc:
        assert "n_arms" in str(exc)
    else:
        raise AssertionError("a missing feature key must raise")


def test_every_feature_has_a_plain_name():
    """Check that labels.py gives every model input a plain name (no code-like fallback names in charts).

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    assert set(FEATURE_COLUMNS) <= set(FEATURE_NAMES)
    assert all("_" not in feature_name(c) for c in FEATURE_COLUMNS)
    assert NO_MAX_AGE == MAX_AGE_FILL


def test_explain_saved_model():
    """Check explain() on the saved model and reference: a real record gives up to five plain-named factors.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Skip when the saved model files are not in the repository.
    if not ((ROOT / "models" / "gbm_uncalibrated.joblib").exists()
            and (ROOT / "models" / "explain_reference.json").exists()):
        print("skipped test_explain_saved_model: model files missing")
        return
    # Explain the invented record with the saved model; every driver has a plain name and a direction.
    drivers = explain(build_features(STUDY))
    assert 0 < len(drivers) <= 5
    assert all(name in FEATURE_NAMES.values() and direction in ("up", "down") and size > 0
               for name, _, direction, size in drivers)


def test_score_frame_matches_score_and_ignores_extra_columns():
    """Check that batch scoring equals score() row by row and gives a model only the columns it was fitted on.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Three invented trials that differ in has_dmc and masking.
    rows = pd.DataFrame([typical()] * 3)
    rows["has_dmc"] = [0.0, 1.0, 0.0]
    rows["masking"] = ["NONE", "NONE", "DOUBLE"]
    model = toy_model()
    # Batch scores and one-by-one scores must match.
    batch = score_frame(rows, model)
    single = [score(r, model) for r in rows.to_dict("records")]
    assert np.allclose(batch, single)
    # A small model fitted on only five features must ignore the other and extra columns.
    small = build_gbm_pipeline(features=FEATURE_COLUMNS[:5]).fit(
        to_model_frame(pd.DataFrame([typical()] * 40), FEATURE_COLUMNS[:5]), [0, 1] * 20)
    assert model_features(small) == FEATURE_COLUMNS[:5]
    assert len(score_frame(rows.assign(not_a_feature="x"), small)) == 3  # extra and unused columns are dropped


# ---- Run all tests when the file is started directly ----
if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

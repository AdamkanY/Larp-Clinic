"""Offline checks that the saved models score build_features output (no network).

Run with ``python tests/test_predict.py`` (or pytest). Uses the committed
``models/gbm.joblib`` (app model, v1 features) and ``models/gbm_v2.joblib``
(challenger, all features) on the synthetic record of ``test_features.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (CONDITION_FLAGS, FEATURE_COLUMNS,  # noqa: E402
                      V1_FEATURE_COLUMNS, build_features)
from predict import load_model, model_columns, percentile, score  # noqa: E402
from test_features import STUDY  # noqa: E402


def test_app_model_scores_full_rows():
    """The v1 app model takes a full build_features row and returns a probability."""
    model = load_model()
    assert model_columns(model) == V1_FEATURE_COLUMNS
    risk = score(build_features(STUDY))
    assert 0.0 <= risk <= 1.0
    assert 0.0 <= percentile(risk) <= 100.0


def test_condition_flags_reach_only_v2():
    """Condition areas cannot move the v1 score; the v2 model does use them."""
    v2 = load_model("models/gbm_v2.joblib")
    assert model_columns(v2) == FEATURE_COLUMNS
    row = build_features(STUDY)
    v1_base, v2_base = score(row), score(row, model=v2)
    v2_moved = False
    for flag in CONDITION_FLAGS:
        changed = {**row, flag: 1, "n_condition_categories": 1}
        assert score(changed) == v1_base
        v2_moved |= score(changed, model=v2) != v2_base
    assert v2_moved


if __name__ == "__main__":
    test_app_model_scores_full_rows()
    test_condition_flags_reach_only_v2()
    print("all ok")

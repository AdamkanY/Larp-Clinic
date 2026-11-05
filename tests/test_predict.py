"""Offline check for src/predict.py: scoring several trials in one call gives the same numbers as one at a time.

Run with ``python tests/test_predict.py`` (or pytest). No network needed. The trials are invented (the record from
test_features.py and variants of it), but they are scored by the real `models/gbm.joblib`, so the check is skipped
when that file is missing.
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from features import build_features  # noqa: E402
from predict import score, score_many  # noqa: E402
from test_features import STUDY  # noqa: E402


def invented_rows():
    """Feature dicts of four invented trials: the test record, an empty record and two variants of the first."""
    other = copy.deepcopy(STUDY)
    proto = other["protocolSection"]
    proto["designModule"]["phases"] = ["NA"]
    proto["designModule"]["designInfo"] = {"allocation": "NA", "interventionModel": "SINGLE_GROUP",
                                           "primaryPurpose": "PREVENTION", "maskingInfo": {"masking": "NONE"}}
    proto["sponsorCollaboratorsModule"] = {"leadSponsor": {"class": "OTHER"}}
    proto["armsInterventionsModule"]["interventions"] = [{"type": "BEHAVIORAL"}]
    proto["eligibilityModule"].update({"healthyVolunteers": True, "sex": "FEMALE", "maximumAge": None})
    del proto["oversightModule"]  # data monitoring committee not stated
    full = build_features(STUDY)
    # The empty record mixes missing values into columns the others fill, the case where batching could differ.
    return [full, build_features({}), build_features(other), {**full, "eligibility_text_len": 6000, "n_arms": 4}]


def test_score_many_matches_score():
    """One batched call gives each trial its own score() in input order, and no trials give no scores."""
    if not (ROOT / "models" / "gbm.joblib").exists():
        print("skipped: models/gbm.joblib is missing")
        return
    rows = invented_rows()
    one_by_one = [score(row) for row in rows]
    batched = score_many(rows)
    assert isinstance(batched, list) and len(batched) == len(rows) and all(isinstance(p, float) for p in batched)
    assert all(abs(a - b) <= 1e-12 for a, b in zip(batched, one_by_one)), (batched, one_by_one)
    # Different scores, so a mix-up of the order would show; if a retrained model ties two rows, change a row.
    assert len(set(one_by_one)) == len(rows), one_by_one
    assert score_many([]) == []


if __name__ == "__main__":
    test_score_many_matches_score()
    print("all ok")

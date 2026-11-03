"""Offline checks for build_features using a hand-written (synthetic) record.

Run with ``python tests/test_features.py`` (or pytest). No network needed.
The record below is invented test data, not real trial data.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURE_COLUMNS, META_COLUMNS, build_features  # noqa: E402

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001"},
        "statusModule": {
            "overallStatus": "TERMINATED",
            "whyStopped": "slow accrual",
            "startDateStruct": {"date": "2012-03", "type": "ACTUAL"},
            "completionDateStruct": {"date": "2013-05"},
            "lastUpdatePostDateStruct": {"date": "2014-01-01"},
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "X", "class": "INDUSTRY"},
            "collaborators": [{"name": "Y", "class": "NIH"}],
        },
        "oversightModule": {"oversightHasDmc": True, "isFdaRegulatedDrug": False},
        "conditionsModule": {"conditions": ["a", "b"]},
        "designModule": {
            "phases": ["PHASE3", "PHASE2"],
            "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL",
                           "primaryPurpose": "TREATMENT",
                           "maskingInfo": {"masking": "DOUBLE"}},
            "enrollmentInfo": {"count": 12, "type": "ACTUAL"},
        },
        "armsInterventionsModule": {
            "armGroups": [{"label": "a"}, {"label": "b"}],
            "interventions": [{"type": "DRUG"}, {"type": "DEVICE"}, {"type": "DRUG"}],
        },
        "outcomesModule": {"primaryOutcomes": [{}], "secondaryOutcomes": [{}, {}, {}]},
        "eligibilityModule": {"eligibilityCriteria": "x" * 100, "healthyVolunteers": False,
                              "sex": "ALL", "minimumAge": "6 Months",
                              "maximumAge": "65 Years"},
        "contactsLocationsModule": {"locations": [{}] * 40},
    },
    "resultsSection": {"participantFlowModule": {}},
    "hasResults": True,
}


def test_values():
    """Spot-check parsed values and the column set."""
    row = build_features(STUDY)
    assert set(row) == set(META_COLUMNS + FEATURE_COLUMNS)
    assert row["label"] == 1 and row["start_date"] == "2012-03-01"
    assert "start_year" not in row and row["phase"] == "PHASE2/PHASE3"
    assert row["n_arms"] == 2 and row["n_secondary_outcomes"] == 3
    assert row["intervention_drug"] == 1 and row["intervention_device"] == 1
    assert row["intervention_radiation"] == 0
    assert row["min_age_years"] == 0.5 and row["max_age_years"] == 65 and row["has_max_age"] == 1
    assert row["has_collaborators"] == 1 and row["healthy_volunteers"] == 0
    assert row["eligibility_text_len"] == 100 and row["masking"] == "DOUBLE"


def test_leaky_fields_ignored():
    """Changing enrollment, locations, results and non-start dates changes nothing."""
    changed = copy.deepcopy(STUDY)
    p = changed["protocolSection"]
    p["designModule"]["enrollmentInfo"]["count"] = 99999
    p["contactsLocationsModule"]["locations"] = []
    p["statusModule"]["whyStopped"] = "other"
    p["statusModule"]["completionDateStruct"]["date"] = "2030-01"
    changed["resultsSection"] = {}
    changed["hasResults"] = False
    assert build_features(changed) == build_features(STUDY)


def test_missing_and_ongoing():
    """Empty records give None values; ongoing statuses get no label."""
    row = build_features({})
    assert row["label"] is None and row["start_date"] is None and row["n_arms"] == 0
    assert row["n_conditions"] == 0 and row["n_primary_outcomes"] == 0
    assert row["has_max_age"] == 0 and row["max_age_years"] == 120
    ongoing = copy.deepcopy(STUDY)
    ongoing["protocolSection"]["statusModule"]["overallStatus"] = "RECRUITING"
    assert build_features(ongoing)["label"] is None
    bad = copy.deepcopy(STUDY)
    bad["protocolSection"]["statusModule"]["startDateStruct"]["date"] = "2012"
    assert build_features(bad)["start_date"] is None


if __name__ == "__main__":
    test_values()
    test_leaky_fields_ignored()
    test_missing_and_ongoing()
    print("all ok")

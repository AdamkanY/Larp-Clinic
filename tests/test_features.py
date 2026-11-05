"""Offline checks for build_features (and features_from_design) on synthetic records.

Run with ``python tests/test_features.py`` (or pytest). No network needed.
The records below are invented test data, not real trial data.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (DISPLAY_COLS, FEATURE_COLUMNS, INTERVENTION_TYPES,  # noqa: E402
                      META_COLUMNS, build_features, features_from_design)

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "A test trial"},
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

# The example trial the "Check your own trial" form opens with, as the form passes it on ...
EXAMPLE_DESIGN = {
    "phase": "PHASE2", "allocation": "RANDOMIZED", "intervention_model": "PARALLEL",
    "primary_purpose": "TREATMENT", "masking": "DOUBLE", "sponsor_class": "INDUSTRY",
    "sex": "ALL", "intervention_types": ["DRUG"], "n_arms": 2, "has_collaborators": False,
    "has_dmc": False, "healthy_volunteers": False, "min_age_years": 18.0, "max_age_years": None,
    "eligibility_text_len": 1500, "n_primary_outcomes": 1, "n_secondary_outcomes": 3,
    "n_conditions": 1,
}
# ... and that design as the API publishes a trial (no upper age limit, so no maximumAge).
EXAMPLE_RECORD = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000003", "briefTitle": "An example trial"},
        "statusModule": {"overallStatus": "NOT_YET_RECRUITING",
                         "startDateStruct": {"date": "2027-01"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Z", "class": "INDUSTRY"}},
        "oversightModule": {"oversightHasDmc": False},
        "conditionsModule": {"conditions": ["c"]},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE2"],
            "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL",
                           "primaryPurpose": "TREATMENT",
                           "maskingInfo": {"masking": "DOUBLE"}},
        },
        "armsInterventionsModule": {"armGroups": [{"label": "a"}, {"label": "b"}],
                                    "interventions": [{"type": "DRUG", "name": "d"}]},
        "outcomesModule": {"primaryOutcomes": [{}], "secondaryOutcomes": [{}, {}, {}]},
        "eligibilityModule": {"eligibilityCriteria": "x" * 1500, "healthyVolunteers": False,
                              "sex": "ALL", "minimumAge": "18 Years"},
    },
}


def model_inputs(row):
    """Only the model's input columns of a feature dict."""
    return {col: row[col] for col in FEATURE_COLUMNS}


def test_values():
    """Spot-check parsed values and the column set."""
    row = build_features(STUDY)
    assert set(row) == set(META_COLUMNS + DISPLAY_COLS + FEATURE_COLUMNS)
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


def test_display_cols():
    """Display columns are filled and never overlap the model features."""
    row = build_features(STUDY)
    assert row["brief_title"] == "A test trial"
    assert row["conditions"] == "a; b"
    assert row["lead_sponsor_name"] == "X"
    assert row["phase_label"] == "Phase 2/Phase 3"
    assert all(c in row for c in DISPLAY_COLS)
    assert not set(DISPLAY_COLS) & set(FEATURE_COLUMNS)


def test_features_from_design():
    """A typed-in design gets exactly the features of the same design as an API record."""
    typed, registered = features_from_design(**EXAMPLE_DESIGN), build_features(EXAMPLE_RECORD)
    assert model_inputs(typed) == model_inputs(registered)
    # Two phases, categories not stated, an upper age limit, an age in months, several
    # intervention types, flags set, zero counts.
    design = {**EXAMPLE_DESIGN, "phase": "PHASE1/PHASE2", "masking": None,
              "sponsor_class": None, "sex": None, "intervention_types": ["DEVICE", "BEHAVIORAL"],
              "has_collaborators": True, "has_dmc": True, "healthy_volunteers": True,
              "min_age_years": 0.5, "max_age_years": 70.0, "n_arms": 0,
              "n_secondary_outcomes": 0, "eligibility_text_len": 0}
    record = {"protocolSection": {
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Z"},
                                       "collaborators": [{"name": "Y"}]},
        "oversightModule": {"oversightHasDmc": True},
        "conditionsModule": {"conditions": ["c"]},
        "designModule": {"phases": ["PHASE2", "PHASE1"],
                         "designInfo": {"allocation": "RANDOMIZED",
                                        "interventionModel": "PARALLEL",
                                        "primaryPurpose": "TREATMENT"}},
        "armsInterventionsModule": {"interventions": [{"type": "BEHAVIORAL"},
                                                      {"type": "DEVICE"}]},
        "outcomesModule": {"primaryOutcomes": [{}]},
        "eligibilityModule": {"eligibilityCriteria": "", "healthyVolunteers": True,
                              "minimumAge": "6 Months", "maximumAge": "70 Years"},
    }}
    assert model_inputs(features_from_design(**design)) == model_inputs(build_features(record))


def test_design_sets_every_feature():
    """The form can set every model feature: a blank and a full design differ in each one.

    Fails when build_features gains a feature that features_from_design cannot fill, which
    the form would then score without showing it.
    """
    blank = {**{key: None for key in EXAMPLE_DESIGN}, "intervention_types": [], "n_arms": 0,
             "has_collaborators": False, "has_dmc": False, "healthy_volunteers": False,
             "n_primary_outcomes": 0, "n_secondary_outcomes": 0, "n_conditions": 0}
    full = {**EXAMPLE_DESIGN, "intervention_types": INTERVENTION_TYPES, "max_age_years": 65.0,
            "has_collaborators": True, "has_dmc": True, "healthy_volunteers": True}
    empty, filled = features_from_design(**blank), features_from_design(**full)
    assert [col for col in FEATURE_COLUMNS if empty[col] == filled[col]] == []


def test_unknown_codes_rejected():
    """An unknown code raises instead of reaching the model as a silent "missing"."""
    bad_codes = [{"phase": "PHASE5"}, {"phase": "PHASE2/PHASE1"}, {"sex": "BOTH"},
                 {"intervention_types": ["PILL"]}]
    for bad in bad_codes:
        try:
            features_from_design(**{**EXAMPLE_DESIGN, **bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    test_values()
    test_leaky_fields_ignored()
    test_missing_and_ongoing()
    test_display_cols()
    test_features_from_design()
    test_design_sets_every_feature()
    test_unknown_codes_rejected()
    print("all ok")

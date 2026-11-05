"""Offline checks for build_features using a hand-written (synthetic) record.

Run with ``python tests/test_features.py`` (or pytest). No network needed.
The record below is invented test data, not real trial data.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (CONDITION_FEATURES, DISPLAY_COLS, FEATURE_COLUMNS,  # noqa: E402
                      MAX_AGE_FILL, META_COLUMNS, build_features, condition_features,
                      features_from_design)

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


def test_condition_features():
    """Keywords set the right category flags, several at once, and only at word starts."""
    flags = condition_features(["Metastatic Breast Cancer", "Type 2 Diabetes Mellitus"])
    assert flags["cond_oncology"] == 1 and flags["cond_metabolic"] == 1
    assert flags["n_condition_categories"] == 2 and flags["cond_cardio"] == 0
    assert set(flags) == set(CONDITION_FEATURES)
    examples = {
        "Acute Myocardial Infarction": "cardio", "Alzheimer Disease": "neuro", "HIV Infections": "infectious",
        "H1N1 Flu": "infectious", "Schizophrenia": "psych", "Smoking Cessation": "psych", "Asthma": "respiratory",
        "COPD": "respiratory", "Healthy Volunteers": "healthy", "Neuroblastoma": "oncology",
    }
    for text, category in examples.items():
        assert condition_features([text])[f"cond_{category}"] == 1, text
    # Word starts only: "Glasses" is not ALS and "Unhealthy" is not "healthy".
    for text in ["Glasses", "Unhealthy Diet", "Pain"]:
        assert condition_features([text])["n_condition_categories"] == 0, text
    # A trial in two categories through one condition string.
    assert condition_features(["Lung Cancer"])["n_condition_categories"] == 2


def test_condition_features_missing():
    """No conditions (absent, empty or junk) gives all zeros, like the other list counts."""
    for value in (None, [], [None, 3]):
        assert set(condition_features(value).values()) == {0}
    assert build_features({})["n_condition_categories"] == 0
    row = build_features(STUDY)  # conditions "a", "b" match nothing
    assert all(row[c] == 0 for c in CONDITION_FEATURES)


def test_condition_backfill_matches_build_features():
    """Backfilling from the stored "; "-joined text gives what build_features gives on the record."""
    import pandas as pd
    from build_dataset import backfill_conditions
    study = copy.deepcopy(STUDY)
    study["protocolSection"]["conditionsModule"]["conditions"] = ["Lung Cancer", "Depression"]
    row = build_features(study)
    stale = pd.DataFrame([{**row, "split": "train", "start_date": pd.Timestamp(row["start_date"]),
                           **{c: 0 for c in CONDITION_FEATURES}}])
    filled = backfill_conditions(stale).iloc[0]
    assert all(filled[c] == row[c] for c in CONDITION_FEATURES)
    assert filled["n_condition_categories"] == 3


def design_of_study():
    """The design of STUDY as the Check page would enter it."""
    return {
        "phase": "PHASE2/PHASE3", "allocation": "RANDOMIZED", "intervention_model": "PARALLEL",
        "primary_purpose": "TREATMENT", "masking": "DOUBLE", "sponsor_class": "INDUSTRY", "sex": "ALL",
        "n_arms": 2, "has_collaborators": True, "has_dmc": True, "healthy_volunteers": False,
        "min_age_years": 0.5, "max_age_years": 65, "eligibility_text_len": 100,
        "n_primary_outcomes": 1, "n_secondary_outcomes": 3, "intervention_types": ["DRUG", "DEVICE"],
        "conditions": ["a", "b"],
    }


def test_features_from_design_matches_build_features():
    """A design typed into the form gets exactly the features of the equivalent API record."""
    typed, real = features_from_design(design_of_study()), build_features(STUDY)
    assert {c: typed[c] for c in FEATURE_COLUMNS} == {c: real[c] for c in FEATURE_COLUMNS}
    assert typed["label"] is None  # a typed-in design is never a training row


def test_features_from_design_rules():
    """The form gets the same rules as real records: no maximum age = 120 plus flag, unset = missing."""
    row = features_from_design({**design_of_study(), "max_age_years": None, "masking": None,
                                "conditions": ["Breast cancer"], "has_dmc": False})
    assert row["max_age_years"] == MAX_AGE_FILL and row["has_max_age"] == 0
    assert row["masking"] is None and row["has_dmc"] == 0
    assert row["cond_oncology"] == 1 and row["n_conditions"] == 1
    empty = features_from_design({})
    assert all(c in empty for c in FEATURE_COLUMNS) and empty["n_arms"] == 0


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
    print("all ok")

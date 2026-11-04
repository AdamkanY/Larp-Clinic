"""Offline checks for build_features using a hand-written (synthetic) record.

Run with ``python tests/test_features.py`` (or pytest). No network needed.
The record below is invented test data, not real trial data.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (CONDITION_FLAGS, DISPLAY_COLS, FEATURE_COLUMNS,  # noqa: E402
                      META_COLUMNS, V1_FEATURE_COLUMNS, V2_FEATURES, build_features,
                      condition_features)

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
    # DRUG, DEVICE, DRUG: two distinct types.
    assert row["n_intervention_types"] == 2
    # Conditions "a" and "b" match no category.
    assert row["n_condition_categories"] == 0 and row["cond_oncology"] == 0


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
    assert row["n_intervention_types"] == 0 and row["n_condition_categories"] == 0
    assert all(row[flag] == 0 for flag in CONDITION_FLAGS)
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


def categories(conditions):
    """Names of the condition categories condition_features finds."""
    flags = condition_features(conditions)
    return {flag.removeprefix("cond_") for flag in CONDITION_FLAGS if flags[flag]}


def test_condition_categories():
    """Keyword rules: typical conditions land in the expected categories."""
    assert categories(["Breast Cancer"]) == {"oncology"}
    assert categories(["Type 2 Diabetes", "Hypertension", "Major Depressive Disorder"]) == {
        "metabolic", "cardio", "psych"}
    assert categories(["Healthy Volunteers"]) == {"healthy"}
    assert categories(["HIV Infections"]) == {"infectious"}
    assert categories(["Pneumonia"]) == {"infectious", "respiratory"}
    assert categories(["COPD"]) == {"respiratory"}
    assert categories(["Stroke"]) == {"neuro"}
    assert categories(["Alzheimer's Disease"]) == {"neuro"}
    assert categories(["Rheumatoid Arthritis", "Chronic Kidney Disease"]) == set()
    flags = condition_features(["Obesity", "Schizophrenia", "Obesity"])
    assert flags["n_condition_categories"] == 2


def test_tumour_clauses_drop_organ_words():
    """A tumour of an organ is oncology only; disease words still count."""
    assert categories(["Non-small Cell Lung Cancer"]) == {"oncology"}
    assert categories(["Carcinoma, Non-Small-Cell Lung"]) == {"oncology"}
    assert categories(["Brain Metastases"]) == {"oncology"}
    assert categories(["Traumatic Brain Injury"]) == {"neuro"}
    assert categories(["Lung Transplant"]) == {"respiratory"}
    assert categories(["Depression in Cancer Patients"]) == {"oncology", "psych"}
    # Separate conditions are judged separately.
    assert categories(["Lung Cancer", "COPD"]) == {"oncology", "respiratory"}


def test_condition_look_alikes():
    """Words that look like a category but are not."""
    for text in ["Ocular Hypertension", "Heartburn", "Gastroparesis", "Neuromuscular Blockade",
                 "Tympanic Membrane Perforation", "Nerve Block", "Pain, Postoperative"]:
        assert categories([text]) == set(), text
    assert categories(["Respiratory Depression"]) == {"respiratory"}
    assert categories(["Non-alcoholic Fatty Liver Disease"]) == {"metabolic"}
    assert categories(["Bariatric Surgery Candidate"]) == {"metabolic"}
    assert categories(["Autistic Disorder"]) == {"psych"}
    assert categories(["Mycosis Fungoides"]) == {"oncology"}
    assert categories(["Non-infectious Uveitis"]) == set()


def test_condition_inputs():
    """None, empty lists, non-strings and a bare string are handled."""
    empty = {**{flag: 0 for flag in CONDITION_FLAGS}, "n_condition_categories": 0}
    assert condition_features(None) == empty
    assert condition_features([]) == empty
    assert condition_features([None, 3]) == empty
    assert condition_features("Breast Cancer") == condition_features(["Breast Cancer"])
    assert set(condition_features(["x"])) == set(CONDITION_FLAGS) | {"n_condition_categories"}


def test_backfill_parity():
    """Splitting the stored "; "-joined text gives the flags build_features computed.

    trials.parquet was backfilled from its ``conditions`` column this way, so
    the training data and live app scores must agree, also for condition names
    that themselves contain "; ".
    """
    conditions = ["Acute Myeloid Leukemia With t(8;21)(q22;q22)", "Blood Sugar; High",
                  "Healthy", "Lung Cancer"]
    study = copy.deepcopy(STUDY)
    study["protocolSection"]["conditionsModule"]["conditions"] = conditions
    row = build_features(study)
    from_text = condition_features(row["conditions"].split("; "))
    assert from_text == {k: row[k] for k in from_text}
    assert categories(conditions) == {"oncology", "metabolic", "healthy"}


def test_feature_versions():
    """v1 columns keep their old order; v2 only appends the new columns."""
    assert V1_FEATURE_COLUMNS == [c for c in FEATURE_COLUMNS if c not in V2_FEATURES]
    assert len(V1_FEATURE_COLUMNS) == 29 and len(FEATURE_COLUMNS) == 39
    assert set(CONDITION_FLAGS) <= set(V2_FEATURES)


if __name__ == "__main__":
    test_values()
    test_leaky_fields_ignored()
    test_missing_and_ongoing()
    test_display_cols()
    test_condition_categories()
    test_tumour_clauses_drop_organ_words()
    test_condition_look_alikes()
    test_condition_inputs()
    test_backfill_parity()
    test_feature_versions()
    print("all ok")

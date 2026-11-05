"""Offline checks for the app's helper code: labels, chart data, API client, model loaders and risk colours.

Run with ``python tests/test_app.py`` (or pytest). No network needed: the API is replaced by a stub and the chart
data is a small invented table, not real trial data. The risk-colour check uses `models/score_reference.npy` and
is skipped when that file is missing.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from src import api_client, viz  # noqa: E402
from src.labels import label  # noqa: E402


def test_labels():
    """Codes become plain words, whatever spelling they come in; gaps become 'Not stated'."""
    assert label("masking", "DOUBLE") == "Double blind"
    assert label("phase", "PHASE1/PHASE2") == "Phase 1/2" and label("phase", "Phase 1/Phase 2") == "Phase 1/2"
    assert label("phase", "NA") == label("phase", "N/A") == label("phase", "Not Applicable") == "No phase"
    assert label("status", "TERMINATED") == "Stopped early"
    assert label("sponsor_class", None) == label("allocation", float("nan")) == label("masking", "") == "Not stated"
    assert label("anything", "SOME_NEW_CODE") == "Some new code"


def test_choices():
    """The check form and the search filters offer codes from one list in src/, and each has a plain name."""
    from src.features import CATEGORICAL_FEATURES, CATEGORY_CODES
    from src.labels import NAMES, PHASE_ORDER
    assert list(CATEGORY_CODES) == CATEGORICAL_FEATURES
    for kind, codes in CATEGORY_CODES.items():
        assert [code for code in codes if code not in NAMES[kind]] == [], kind  # no raw code in a dropdown
    assert PHASE_ORDER == [label("phase", code) for code in CATEGORY_CODES["phase"]]
    assert api_client.PHASES == ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
    assert api_client.SPONSOR_CLASSES == CATEGORY_CODES["sponsor_class"]


def make_trials(path):
    """Write a tiny invented trials.parquet with the columns the dashboard reads."""
    rows = []
    for i in range(120):
        rows.append({
            "nct_id": f"NCT{i:08d}", "overall_status": "TERMINATED" if i % 4 == 0 else "COMPLETED",
            "label": int(i % 4 == 0), "start_date": pd.Timestamp(2010 + i % 5, 6, 1),
            "phase_label": "Phase 2" if i < 60 else ("Not Applicable" if i < 115 else None),
            "sponsor_class": "INDUSTRY" if i < 90 else "OTHER", "masking": "DOUBLE" if i < 100 else None,
            "allocation": "RANDOMIZED" if i < 80 else "NA", "conditions": "Asthma; Pain" if i < 40 else "Pain",
            "eligibility_text_len": [500, 501, 1000, 2000, 4000, 4001][i % 6], "brief_title": f"Trial {i}",
            "lead_sponsor_name": "Acme",
        })
    pd.DataFrame(rows).to_parquet(path)


def test_viz():
    """Loading maps codes to words and groups eligibility lengths; filters and chart builders behave."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trials.parquet"
        make_trials(path)
        df = viz.load_trials(path)
    assert set(df["phase_label"]) == {"Phase 2", "No phase", "Not stated"}
    assert set(df["masking"]) == {"Double blind", "Not stated"}
    assert set(df["allocation"]) == {"Randomized", "Single group"}
    assert set(df["overall_status"]) == {"Stopped early", "Completed"}
    # The edges belong to the lower group: 500 is "Up to 500", 501 the next one, 4001 "Over 4,000".
    groups = df.groupby("eligibility_text_len")["eligibility_group"].first().astype(str).to_dict()
    assert groups == {500: "Up to 500", 501: "501-1,000", 1000: "501-1,000", 2000: "1,001-2,000",
                      4000: "2,001-4,000", 4001: "Over 4,000"}

    assert len(viz.filter_trials(df, phases=["Phase 2"])) == 60
    assert len(viz.filter_trials(df, condition_kw="asthma")) == 40  # case-insensitive substring
    assert len(viz.filter_trials(df, year_range=(2010, 2010))) == 24
    assert len(viz.filter_trials(df, condition_kw="C++(")) == 0  # regex characters are plain text

    stats = viz.rate_by(df, "phase_label", viz.PHASE_ORDER)
    assert list(stats["phase_label"]) == ["Phase 2", "No phase", "Not stated"]  # clinical order, not by size
    # "Not stated" (5 trials) is too small to count; two real groups remain.
    assert viz.extremes(stats, "phase_label").startswith("Highest: ")
    assert viz.extremes(stats.iloc[:1], "phase_label") == ""
    # A gap in the data ("Not stated") never counts as highest or lowest, even with many trials and the top rate.
    gaps = pd.DataFrame({"phase_label": ["Phase 2", "No phase", "Not stated"], "n": [100, 100, 100],
                         "rate": [0.2, 0.1, 0.5]})
    assert viz.extremes(gaps, "phase_label") == "Highest: Phase 2 at 20%. Lowest: No phase at 10%."

    fig = viz.rate_bar(stats, "phase_label", "Phase", average=df["label"].mean())
    assert list(fig.data[0].x) == ["Phase 2", "No phase", "Not stated"]
    assert fig.data[0].text[2] == ""  # the 5-trial group is faded and unlabelled
    cond = viz.conditions_bar(df)
    assert cond.data[0].y[-1] in {"Asthma", "Pain"} and len(cond.data[0].y) == 2
    assert len(viz.eligibility_stats(df)) == 5
    assert list(viz.trials_table(df.head(1)).columns)[0] == "link"


class Response:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        """Pretend the request succeeded."""

    def json(self):
        """The invented payload."""
        return self.payload


def test_api_client():
    """Search asks for interventional trials with every ongoing status, and failures are never cached."""
    sent = []
    outcomes = [requests.ConnectionError("offline"), Response({"studies": [{"protocolSection": {}}], "totalCount": 1})]

    def fake_get(url, params=None, timeout=None):
        sent.append(params)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    real_get, api_client.requests.get = api_client.requests.get, fake_get
    try:
        api_client._get_json.clear()
        failed = api_client.search_trials(query="asthma", phase="PHASE2", sponsor_class="NIH")
        assert failed["error"].startswith("Could not reach ClinicalTrials.gov") and failed["studies"] == []
        ok = api_client.search_trials(query="asthma", phase="PHASE2", sponsor_class="NIH")  # retried, not served from cache
        assert ok["error"] is None and ok["total"] == 1 and len(sent) == 2
        again = api_client.search_trials(query="asthma", phase="PHASE2", sponsor_class="NIH")
        assert again == ok and len(sent) == 2  # the success is cached
    finally:
        api_client.requests.get = real_get
        api_client._get_json.clear()
    params = sent[0]
    assert "ENROLLING_BY_INVITATION" in params["filter.overallStatus"] and "RECRUITING" in params["filter.overallStatus"]
    assert params["filter.advanced"] == "AREA[StudyType]INTERVENTIONAL AND AREA[Phase]PHASE2 AND AREA[LeadSponsorClass]NIH"
    assert params["query.term"] == "asthma"

    row = api_client.summarize({"protocolSection": {"designModule": {"phases": ["PHASE1", "PHASE2"]}}})
    assert row["Phase"] == "PHASE1, PHASE2"
    assert api_client.summarize({})["Phase"] == "NA"  # a missing phase reads like the API's own "not applicable"


def test_loaders():
    """The app's loaders in src/predict.py: None instead of an error for a missing file, one shared read otherwise."""
    from src import predict
    assert predict.load_metrics("models/missing_metrics.json") is None
    assert predict.get_model("models/missing_model.joblib") is None
    metrics_file = ROOT / "models" / "gbm_metrics.json"
    if not metrics_file.exists():
        print("skipped: models/gbm_metrics.json is missing")
        return
    metrics = predict.load_metrics()
    assert metrics == json.loads(metrics_file.read_text(encoding="utf-8"))
    assert predict.load_metrics() is metrics  # read once, then shared by every page
    if (ROOT / "models" / "gbm.joblib").exists():
        assert predict.get_model() is predict.load_model()


def test_risk_colours():
    """Colour follows the rank among past trials: green below the median, red among the riskiest 10%."""
    reference = ROOT / "models" / "score_reference.npy"
    if not reference.exists():
        print("skipped: models/score_reference.npy is missing")
        return
    import risk_gauge
    from src.predict import percentile
    scores = np.sort(np.load(reference))
    assert risk_gauge.risk_tone(float(scores[len(scores) // 10]))[0] == "green"
    assert risk_gauge.risk_tone(float(scores[int(len(scores) * 0.7)]))[0] == "yellow"
    assert risk_gauge.risk_tone(float(scores[-1]) + 0.01)[0] == "red"
    assert "% risk" in risk_gauge.risk_chip(0.1) and "compact" in risk_gauge.risk_card(0.1, compact=True)
    assert "compact" not in risk_gauge.risk_card(0.1)
    # The rank on screen is src.predict.percentile's, and the caption counts the same reference trials.
    assert risk_gauge.risk_tone(0.2)[2] == percentile(0.2)
    assert f"among {len(scores):,} past trials" in risk_gauge.bar_caption()


def test_risk_without_reference():
    """Without models/score_reference.npy the risk is still shown, just without a rank: grey label, no bar."""
    import risk_gauge
    from src.predict import percentile
    try:  # the error src.predict raises for a missing file is one the display expects
        percentile(0.2, reference_path="models/missing_reference.npy")
    except risk_gauge.NO_REFERENCE:
        pass

    def missing(*args, **kwargs):
        raise FileNotFoundError("models/score_reference.npy")

    real = risk_gauge.percentile, risk_gauge.reference_size
    risk_gauge.percentile = risk_gauge.reference_size = missing
    try:
        assert risk_gauge.risk_tone(0.2) == ("gray", "Estimated risk of stopping early", None)
        assert risk_gauge.risk_bar(0.2) is None
        assert "among the past trials" in risk_gauge.bar_caption()
        card = risk_gauge.risk_card(0.2, compact=True)
        assert "20.0%" in card and "meter" not in card
    finally:
        risk_gauge.percentile, risk_gauge.reference_size = real


if __name__ == "__main__":
    test_labels()
    test_choices()
    test_viz()
    test_api_client()
    test_loaders()
    test_risk_colours()
    test_risk_without_reference()
    print("all ok")

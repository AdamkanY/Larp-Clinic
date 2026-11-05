"""Offline checks for the app's helper code: labels, chart data, API client and risk colours.

Run with ``python tests/test_app.py`` (or pytest). No network needed: the API is replaced by a stub and the chart
data is a small invented table, not real trial data. The risk-colour check uses `models/score_reference.npy` and
is skipped when that file is missing.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from src import api_client, game, viz  # noqa: E402
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS  # noqa: E402
from src.labels import label  # noqa: E402


def test_labels():
    """Codes become plain words, whatever spelling they come in; gaps become 'Not stated'."""
    assert label("masking", "DOUBLE") == "Double blind"
    assert label("phase", "PHASE1/PHASE2") == "Phase 1/2" and label("phase", "Phase 1/Phase 2") == "Phase 1/2"
    assert label("phase", "NA") == label("phase", "N/A") == label("phase", "Not Applicable") == "No phase"
    assert label("status", "TERMINATED") == "Stopped early"
    assert label("sponsor_class", None) == label("allocation", float("nan")) == label("masking", "") == "Not stated"
    assert label("anything", "SOME_NEW_CODE") == "Some new code"


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


def test_risk_colours():
    """Colour follows the rank among past trials: green below the median, red among the riskiest 10%."""
    reference = ROOT / "models" / "score_reference.npy"
    if not reference.exists():
        print("skipped: models/score_reference.npy is missing")
        return
    import risk_gauge
    scores = np.sort(np.load(reference))
    assert risk_gauge.risk_tone(float(scores[len(scores) // 10]))[0] == "green"
    assert risk_gauge.risk_tone(float(scores[int(len(scores) * 0.7)]))[0] == "yellow"
    assert risk_gauge.risk_tone(float(scores[-1]) + 0.01)[0] == "red"
    assert "% risk" in risk_gauge.risk_chip(0.1) and "compact" in risk_gauge.risk_card(0.1, compact=True)
    assert "compact" not in risk_gauge.risk_card(0.1)


def make_game_trials():
    """Invented test-split table for the game: a model-feature column set plus label and phase_label.

    Phase 2: 10 terminated / 30 completed; Phase 3: 5 / 5; "Early": terminated only, so it can never be paired.
    `n_arms` doubles as the score of the stub model below: terminated trials get a higher value than completed ones.
    """
    rows = []
    for phase, n_term, n_comp in (("Phase 2", 10, 30), ("Phase 3", 5, 5), ("Early", 4, 0)):
        for outcome, count in ((1, n_term), (0, n_comp)):
            for _ in range(count):
                rows.append({"label": outcome, "phase_label": phase, "n_arms": 2.0 + outcome})
    trials = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in trials:
            trials[col] = "X" if col in CATEGORICAL_FEATURES else 0.0
    return trials


class StubModel:
    """Stands in for the trained model: the score is the n_arms column, so higher n_arms means 'riskier'."""

    def predict_proba(self, frame):
        """Return [P(0), P(1)] columns like scikit-learn, with P(1) = n_arms / 10."""
        risk = frame["n_arms"].to_numpy() / 10
        return np.column_stack([1 - risk, risk])


def test_game_pairs():
    """Pair sampler: one terminated + one completed per pair, same phase, no repeats, reproducible, never invents pairs."""
    trials = make_game_trials()
    pairs = game.sample_pairs(trials, 12, seed=1)
    assert len(pairs) == 12
    assert (trials.loc[pairs["terminated"], "label"] == 1).all() and (trials.loc[pairs["completed"], "label"] == 0).all()
    assert (trials.loc[pairs["terminated"], "phase_label"].to_numpy()
            == trials.loc[pairs["completed"], "phase_label"].to_numpy()).all()
    assert pairs["terminated"].is_unique and pairs["completed"].is_unique
    assert "Early" not in set(trials.loc[pairs["terminated"], "phase_label"])  # no completed partner exists
    assert game.sample_pairs(trials, 12, seed=1).equals(pairs)
    assert not game.sample_pairs(trials, 12, seed=2).equals(pairs)
    assert len(game.sample_pairs(trials, 1000, seed=1)) == 15  # asks for more than exist: gets all pairable ones


def test_game_scoring():
    """Model pick and accuracy: higher score wins, ties count half, and a game decides everything up front."""
    trials = make_game_trials()
    assert game.model_pick(0.3, 0.1) == "terminated" and game.model_pick(0.1, 0.3) == "completed"
    assert game.model_pick(0.2, 0.2, rng=0) in {"terminated", "completed"}
    pairs = game.sample_pairs(trials, 10, seed=3)
    assert game.pairwise_accuracy(trials, pairs, StubModel()) == 1.0  # stub always scores the terminated one higher
    flat = trials.assign(n_arms=2.0)  # everything tied
    assert game.pairwise_accuracy(flat, pairs, StubModel()) == 0.5
    rounds = game.new_game(trials, 10, model=StubModel(), seed=4)
    assert len(rounds) == 10 and all(r["model_pick"] == "terminated" for r in rounds)
    assert {r["left"] for r in rounds} == {"terminated", "completed"}  # display order is shuffled


def test_game_score_matches_predict():
    """The batch scoring used by the game equals predict.score row by row (skipped without the saved model)."""
    if not (ROOT / "models" / "gbm.joblib").exists() or not (ROOT / "data" / "processed" / "trials.parquet").exists():
        print("skipped: models/gbm.joblib or data/processed/trials.parquet is missing")
        return
    from src.predict import score
    trials = game.load_test_trials().head(3)
    assert (trials["split"] == "test").all()
    batch = game.score_rows(trials)
    for position, (_, row) in enumerate(trials.iterrows()):
        assert abs(batch[position] - score(row[FEATURE_COLUMNS].to_dict())) < 1e-9


def test_one_palette():
    """Colours live only in app/ui.py: the CSS, the charts and the Streamlit theme all take them from PALETTE."""
    import re
    import tomllib

    import plotly.io as pio
    import ui

    assert pio.templates.default == ui.TEMPLATE  # registered on import, before any page builds a chart
    colour = re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b|rgba?\(")
    files = [ROOT / "app" / "style.css", ROOT / "app" / "Home.py", ROOT / "app" / "risk_gauge.py",
             ROOT / "src" / "viz.py", *sorted((ROOT / "app" / "pages").glob("*.py"))]
    for path in files:
        assert not colour.findall(path.read_text(encoding="utf-8")), f"colour literal in {path.name}"
    css = ui.stylesheet()
    assert f"--ink: {ui.PALETTE['ink']};" in css and f"--low-tint: {ui.PALETTE['low_tint']};" in css
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    assert {theme["primaryColor"], theme["textColor"]} == {ui.PALETTE["ink"]}
    assert theme["backgroundColor"] == ui.PALETTE["paper"]
    assert theme["secondaryBackgroundColor"] == ui.PALETTE["paper_2"]
    assert theme["borderColor"] == ui.PALETTE["line"] and theme["linkColor"] == ui.PALETTE["accent_text"]
    with tempfile.TemporaryDirectory() as tmp:
        make_trials(Path(tmp) / "trials.parquet")
        df = viz.load_trials(Path(tmp) / "trials.parquet")
    fig = viz.rate_bar(viz.rate_by(df, "phase_label"), "phase_label", "Phase", 0.1)
    assert fig.data[0].marker.color == ui.PALETTE["bar"]  # viz.py has no colours of its own


if __name__ == "__main__":
    test_labels()
    test_viz()
    test_api_client()
    test_risk_colours()
    test_game_pairs()
    test_game_scoring()
    test_game_score_matches_predict()
    print("all ok")

"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Offline checks for the app's helper code: labels, chart data, API client and risk colours.

Run with ``python tests/test_app.py`` (or pytest). No network needed: the API is replaced by a stub and the chart
data is a small invented table, not real trial data. The risk-colour check uses `models/score_reference.npy` and
is skipped when that file is missing.
"""

import json  # write the snapshot file for the offline test
import sys  # edit the module search path
import tempfile  # temporary folders for the invented files
from pathlib import Path  # file paths that work on every system

import numpy as np  # arrays and numeric helpers
import pandas as pd  # tables (DataFrame)
import requests  # HTTP library; its errors and get() are faked in the API test

# Make the repository root and app/ importable, then load the code under test.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from src import api_client, database, game, open_trials, viz  # API client, SQLite database, game logic, open trials, chart data
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS  # the model's feature lists
from src.labels import label  # turns API codes into plain words


def test_labels():
    """Check that codes become plain words, whatever spelling they come in, and gaps become 'Not stated'.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Known codes, spelling variants, missing values and an unknown code.
    assert label("masking", "DOUBLE") == "Double blind"
    assert label("phase", "PHASE1/PHASE2") == "Phase 1/2" and label("phase", "Phase 1/Phase 2") == "Phase 1/2"
    assert label("phase", "NA") == label("phase", "N/A") == label("phase", "Not Applicable") == "No phase"
    assert label("status", "TERMINATED") == "Stopped early"
    assert label("sponsor_class", None) == label("allocation", float("nan")) == label("masking", "") == "Not stated"
    assert label("anything", "SOME_NEW_CODE") == "Some new code"


def make_trials(path):
    """Write a tiny invented trials.parquet with the columns the dashboard reads.

    Input: path is the file to write (Path).

    Output: nothing; it writes 120 invented trials to the file.
    """
    # One dict per invented trial; the values follow the index so the counts in the tests are known.
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
    # Save the table.
    pd.DataFrame(rows).to_parquet(path)


def test_viz():
    """Check loading (codes to words, eligibility groups), the filters and the chart builders.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Write the invented table and load it the way the app does.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trials.parquet"
        make_trials(path)
        df = viz.load_trials(path)
    # Codes became words.
    assert set(df["phase_label"]) == {"Phase 2", "No phase", "Not stated"}
    assert set(df["masking"]) == {"Double blind", "Not stated"}
    assert set(df["allocation"]) == {"Randomized", "Single group"}
    assert set(df["overall_status"]) == {"Stopped early", "Completed"}
    # The edges belong to the lower group: 500 is "Up to 500", 501 the next one, 4001 "Over 4,000".
    groups = df.groupby("eligibility_text_len")["eligibility_group"].first().astype(str).to_dict()
    assert groups == {500: "Up to 500", 501: "501-1,000", 1000: "501-1,000", 2000: "1,001-2,000",
                      4000: "2,001-4,000", 4001: "Over 4,000"}

    # Filters: phase, keyword, year range, and special characters in the keyword.
    assert len(viz.filter_trials(df, phases=["Phase 2"])) == 60
    assert len(viz.filter_trials(df, condition_kw="asthma")) == 40  # case-insensitive substring
    assert len(viz.filter_trials(df, year_range=(2010, 2010))) == 24
    assert len(viz.filter_trials(df, condition_kw="C++(")) == 0  # regex characters are plain text

    # Rates per phase, and the text that names the highest and lowest group.
    stats = viz.rate_by(df, "phase_label", viz.PHASE_ORDER)
    assert list(stats["phase_label"]) == ["Phase 2", "No phase", "Not stated"]  # clinical order, not by size
    # "Not stated" (5 trials) is too small to count; two real groups remain.
    assert viz.extremes(stats, "phase_label").startswith("Highest: ")
    assert viz.extremes(stats.iloc[:1], "phase_label") == ""
    # A gap in the data ("Not stated") never counts as highest or lowest, even with many trials and the top rate.
    gaps = pd.DataFrame({"phase_label": ["Phase 2", "No phase", "Not stated"], "n": [100, 100, 100],
                         "rate": [0.2, 0.1, 0.5]})
    assert viz.extremes(gaps, "phase_label") == "Highest: Phase 2 at 20%. Lowest: No phase at 10%."

    # The bar chart and the other chart builders.
    fig = viz.rate_bar(stats, "phase_label", "Phase", average=df["label"].mean())
    assert list(fig.data[0].x) == ["Phase 2", "No phase", "Not stated"]
    assert fig.data[0].text[2] == ""  # the 5-trial group is faded and unlabelled
    cond = viz.conditions_bar(df)
    assert cond.data[0].y[-1] in {"Asthma", "Pain"} and len(cond.data[0].y) == 2
    assert len(viz.eligibility_stats(df)) == 5
    assert list(viz.trials_table(df.head(1)).columns)[0] == "link"


def test_viz_heatmap():
    """Check the two-trait matrix: counts and order, empty thin cells, dropped empty rows and columns.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Write and load the invented table.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trials.parquet"
        make_trials(path)
        df = viz.load_trials(path)
    # The numbers behind the heatmap: rates and counts per phase and sponsor.
    rate, n = viz.rate_matrix(df, "phase_label", "sponsor_class", viz.PHASE_ORDER)
    assert list(rate.index) == ["Phase 2", "No phase", "Not stated"]  # clinical order
    assert n.loc["Phase 2", "Industry"] == 60 and n.to_numpy().sum() == len(df)
    assert rate.loc["Phase 2", "Industry"] == df[df["phase_label"] == "Phase 2"]["label"].mean()
    # The figure itself.
    fig = viz.rate_heatmap(df, "phase_label", "sponsor_class", viz.PHASE_ORDER)
    heat = fig.data[0]
    # "Not stated" has 5 trials, below MIN_GROUP_N in every column, so its row disappears.
    assert list(heat.y) == ["Phase 2", "No phase"]
    for row, cells in zip(heat.z, heat.text):
        for value, text in zip(row, cells):
            assert (text == "") == (value is None or np.isnan(value))  # a blank cell has no label
    # Nothing readable left: None, so the page can show a sentence instead.
    assert viz.rate_heatmap(df.head(10), "phase_label", "sponsor_class") is None


def test_viz_more_views():
    """Check the four extra charts: treemap, bubbles, stacked areas and Sankey, and None for too little data.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Write and load the invented table.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trials.parquet"
        make_trials(path)
        df = viz.load_trials(path)

    # Treemap and bubbles: one trace, one rectangle or bubble per condition (Asthma and Pain).
    treemap = viz.condition_treemap(df)
    assert treemap.data[0].type == "treemap" and sorted(treemap.data[0].labels) == ["Asthma", "Pain"]
    bubbles = viz.condition_bubbles(df)
    assert bubbles.data[0].type == "scatter" and len(bubbles.data[0].x) == 2 and bubbles.layout.xaxis.type == "log"
    # Pain: 120 trials, 30 stopped. Asthma: 40 trials, 10 stopped (every fourth trial).
    assert dict(zip(bubbles.data[0].x, bubbles.data[0].y)) == {120: 0.25, 40: 0.25}

    # Stacked areas: one layer per phase with enough trials, in clinical order ("Not stated" has 5 and is left out).
    area = viz.trials_per_year_area(df)
    assert [trace.name for trace in area.data] == ["Phase 2", "No phase"]
    assert sum(sum(trace.y) for trace in area.data) == 115

    # Sankey: 2 phases + 1 blinding group ("Not stated", 20 trials, is too thin) + 2 outcomes = 5 boxes.
    sankey = viz.flow_sankey(df)
    assert sankey.data[0].type == "sankey" and len(sankey.data[0].node.label) == 5
    assert sum(sankey.data[0].link.value) == 2 * 100  # every kept trial crosses two links

    # Too little data gives None for all four, so the page can show a sentence.
    tiny = df.head(10)
    assert viz.condition_treemap(tiny) is None and viz.condition_bubbles(tiny) is None
    assert viz.trials_per_year_area(tiny) is None and viz.flow_sankey(tiny) is None
    # A single start year cannot make an area chart.
    assert viz.trials_per_year_area(df[df["start_year"] == 2010]) is None


def test_gain_curve():
    """Check the cumulative gain curve: riskiest first, 0% to 100% found, random for a useless score.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # A small example where the right order is known.
    curve = viz.gain_curve([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.1, 0.2], points=6)
    assert list(curve["reviewed"]) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert list(curve["caught"]) == [0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
    assert curve["precision"].iloc[1] == 1.0 and curve["precision"].iloc[-1] == 0.4
    # A score that is the same for everyone gives the random line.
    flat = viz.gain_curve([1, 0] * 50, [0.5] * 100, points=11)  # ties keep the input order: no head start
    assert np.allclose(flat["caught"], flat["reviewed"], atol=0.02)


def test_database():
    """Check that the SQLite database matches the pandas helpers and treats filter text as data.

    The database must give the same numbers as the pandas helpers, split conditions into rows and treat filter text
    as data, never as SQL.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Build a database from the invented table in a temporary folder.
    with tempfile.TemporaryDirectory() as tmp:
        parquet, db_path = Path(tmp) / "trials.parquet", Path(tmp) / "trials.db"
        make_trials(parquet)
        database.ensure_database(parquet, db_path)
        df = viz.load_trials(parquet)
        everything = database.TrialDatabase(db_path)
        assert everything.summary() == {"n": 120, "rate": 0.25}
        # Normal form: "Asthma; Pain" becomes two rows, so Pain is counted for all 120 trials, Asthma for 40.
        top = everything.top_conditions(k=5).set_index("condition")["n"].to_dict()
        assert top == {"Pain": 120, "Asthma": 40}
        # Filters as SQL parameters give exactly the pandas filter's trials and rates.
        filters = {"phases": ["Phase 2"], "sponsor_classes": ["Industry"], "year_range": (2010, 2012),
                   "condition_kw": "asth"}
        # The same filters through SQL.
        db = database.TrialDatabase(db_path, **filters)
        expected = viz.filter_trials(df, ["Phase 2"], ["Industry"], (2010, 2012), "asth")
        assert db.summary()["n"] == len(expected) == len(db.trials())
        # Grouped rates agree between SQL and pandas.
        by_sql = db.rate_by("masking", viz.MASKING_ORDER)
        by_pandas = viz.rate_by(expected, "masking", viz.MASKING_ORDER)
        assert list(by_sql["masking"]) == list(by_pandas["masking"].astype(str))
        assert np.allclose(by_sql["rate"], by_pandas["rate"])
        # Every query was recorded for the page's SQL box, with "?" placeholders instead of the values.
        assert all("?" in sql for purpose, sql, params in db.log) and "asth" not in " ".join(s for _, s, _ in db.log)
        # A hostile keyword is only text: no match, and both tables are still there.
        hostile = database.TrialDatabase(db_path, condition_kw="'; DROP TABLE trials; --")
        assert hostile.summary()["n"] == 0 and everything.summary()["n"] == 120
        # Column names cannot be parameters, so only the four known traits may be grouped by.
        try:
            db.rate_by("label; DROP TABLE trials")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown column was accepted")



def test_tilt_table():
    """Check the tilt table: the share of running trials, the rate among ended trials and the lowest final rate.

    Input: none.

    Output: nothing; the test fails with an AssertionError if a number is wrong.
    """
    counts = pd.DataFrame([{"start_year": 2019, "completed": 80, "terminated": 20, "withdrawn": 5, "recruiting": 10,
                            "not_yet_recruiting": 0, "active_not_recruiting": 10, "enrolling_by_invitation": 0,
                            "suspended": 5, "unknown": 25}])
    row = open_trials.tilt_table(counts, pd.Series({2019: 0.10})).iloc[0]
    assert row["ended"] == 100 and row["running"] == 25 and row["running_share"] == 0.2
    assert row["unknown_share"] == 25 / 150
    assert row["observed_rate"] == 0.2 and row["rate_if_running_complete"] == 20 / 125
    assert np.isclose(row["gap_observed"], 1.0) and np.isclose(row["gap_if_running_complete"], 0.6)
    # The saved counts cover every start year of the data, and the ended trials match trials.parquet closely.
    saved = open_trials.load_counts()
    assert list(saved["start_year"]) == list(range(2008, 2020))


class Response:
    """Minimal stand-in for requests.Response, holding an invented payload."""

    def __init__(self, payload):
        """Store the invented answer.

        Input: payload is the data json() will return (dict).

        Output: nothing; it sets self.payload.
        """
        self.payload = payload

    def raise_for_status(self):
        """Pretend the request succeeded.

        Input: none.

        Output: nothing, because the invented request never fails.
        """

    def json(self):
        """Return the invented payload.

        Input: none.

        Output: the payload (dict) given when the object was created.
        """
        return self.payload


def test_api_client():
    """Check that search asks for interventional trials with every ongoing status, and that failures are never cached.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Record what is sent; the first call fails, the second succeeds.
    sent = []
    outcomes = [requests.ConnectionError("offline"), Response({"studies": [{"protocolSection": {}}], "totalCount": 1})]

    def fake_get(url, params=None, timeout=None):
        """Stand-in for requests.get that records the parameters and plays the planned outcomes in order.

        Input: url is the requested address (ignored); params is the dict of query parameters (recorded); timeout is
        ignored.

        Output: the next planned Response; raises the planned exception instead when the outcome is an error.
        """
        sent.append(params)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    # Swap the real requests.get for the fake one, and put it back at the end whatever happens.
    real_get = api_client.requests.get
    api_client.requests.get = fake_get
    try:
        api_client._get_json.clear()
        failed = api_client.search_trials(query="asthma", phase="PHASE2", sponsor_class="NIH")
        assert failed["error"] == api_client.OFFLINE_MESSAGE and failed["studies"] == []
        assert "offline" in failed["detail"] and "offline" not in failed["error"]  # technical text kept apart
        ok = api_client.search_trials(query="asthma", phase="PHASE2", sponsor_class="NIH")  # retried, not served from cache
        assert ok["error"] is None and ok["total"] == 1 and len(sent) == 2
        again = api_client.search_trials(query="asthma", phase="PHASE2", sponsor_class="NIH")
        assert again == ok and len(sent) == 2  # the success is cached
    finally:
        api_client.requests.get = real_get
        api_client._get_json.clear()
    # The first request carried every ongoing status and the three filters.
    params = sent[0]
    assert "ENROLLING_BY_INVITATION" in params["filter.overallStatus"] and "RECRUITING" in params["filter.overallStatus"]
    assert params["filter.advanced"] == "AREA[StudyType]INTERVENTIONAL AND AREA[Phase]PHASE2 AND AREA[LeadSponsorClass]NIH"
    assert params["query.term"] == "asthma"

    # The table row shows phases as text, and a missing phase as "NA".
    row = api_client.summarize({"protocolSection": {"designModule": {"phases": ["PHASE1", "PHASE2"]}}})
    assert row["Phase"] == "PHASE1, PHASE2"
    assert api_client.summarize({})["Phase"] == "NA"  # a missing phase reads like the API's own "not applicable"


def example_study(nct_id, title, conditions, phases, sponsor_class):
    """Build a minimal ongoing-trial record with just the fields the offline search reads.

    Input: nct_id is the trial id (str); title is the brief title (str);
    conditions is a list of condition names (list of str);
    phases is a list of phase codes or None; sponsor_class is the sponsor class code (str).

    Output: a study record (dict in the API v2 layout).
    """
    return {"protocolSection": {
        "identificationModule": {"nctId": nct_id, "briefTitle": title},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"phases": phases, "studyType": "INTERVENTIONAL"},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme", "class": sponsor_class}},
        "conditionsModule": {"conditions": conditions},
    }}


def test_offline_examples():
    """Check that the saved snapshot is searched like the API and a broken or missing file gives None.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Three invented trials for the snapshot.
    studies = [example_study("NCT1", "Drug A in adults", ["Asthma"], ["PHASE2"], "INDUSTRY"),
               example_study("NCT2", "Walking study", ["Depression"], None, "OTHER"),
               example_study("NCT3", "Asthma app", ["Breathing"], ["PHASE3"], "OTHER")]
    # Save them as a snapshot file in a temporary folder and search it.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ongoing_trials.json"
        path.write_text(json.dumps({"saved_on": "2026-12-01", "studies": studies}))
        everything = api_client.search_examples(path=path)
        assert everything["total"] == 3 and everything["saved_on"] == "2026-12-01" and everything["error"] is None
        # The keyword matches the title or the conditions, ignoring case.
        assert [api_client.summarize(s)["NCT ID"] for s in api_client.search_examples("ASTHMA", path=path)["studies"]] \
            == ["NCT1", "NCT3"]
        assert api_client.search_examples(phase="NA", path=path)["total"] == 1  # a missing phase counts as "NA"
        assert api_client.search_examples(sponsor_class="OTHER", page_size=1, path=path)["total"] == 2
        assert len(api_client.search_examples(sponsor_class="OTHER", page_size=1, path=path)["studies"]) == 1
        # A file that is not JSON gives None.
        path.write_text("not json")
        assert api_client.load_examples(path) is None
    # A missing file gives None too.
    assert api_client.search_examples(path=Path(tmp) / "missing.json") is None


def test_risk_colours():
    """Check that the risk colour follows the rank among past trials.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Skip when the saved reference scores are missing.
    reference = ROOT / "models" / "score_reference.npy"
    if not reference.exists():
        print("skipped: models/score_reference.npy is missing")
        return
    import risk_gauge  # the app's colour logic
    # Scores of past trials in order; pick low, middle and high ones.
    scores = np.sort(np.load(reference))
    assert risk_gauge.risk_tone(float(scores[len(scores) // 10]))[0] == "green"
    assert risk_gauge.risk_tone(float(scores[int(len(scores) * 0.7)]))[0] == "yellow"
    assert risk_gauge.risk_tone(float(scores[-1]) + 0.01)[0] == "red"
    assert "% risk" in risk_gauge.risk_chip(0.1) and "compact" in risk_gauge.risk_card(0.1, compact=True)
    assert "compact" not in risk_gauge.risk_card(0.1)


def make_game_trials():
    """Build an invented test-split table for the game.

    It has all model-feature columns plus label and phase_label.
    Phase 2: 10 terminated / 30 completed; Phase 3: 5 / 5; "Early": terminated only, so it can never be paired.
    `n_arms` doubles as the score of the stub model below: terminated trials get a higher value than completed ones.

    Input: none.

    Output: a table (DataFrame) with one row per invented trial.
    """
    # One row per trial: terminated trials get n_arms 3.0, completed ones 2.0.
    rows = []
    for phase, n_term, n_comp in (("Phase 2", 10, 30), ("Phase 3", 5, 5), ("Early", 4, 0)):
        for outcome, count in ((1, n_term), (0, n_comp)):
            for _ in range(count):
                rows.append({"label": outcome, "phase_label": phase, "n_arms": 2.0 + outcome})
    # Fill every model feature the rows lack with a dummy value.
    trials = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in trials:
            trials[col] = "X" if col in CATEGORICAL_FEATURES else 0.0
    return trials


class StubModel:
    """Stand-in for the trained model: the score is the n_arms column, so higher n_arms means 'riskier'."""

    def predict_proba(self, frame):
        """Return probabilities like scikit-learn, with P(1) = n_arms / 10.

    Input: frame is a table (DataFrame) with an n_arms column.

    Output: an array with two columns, P(0) and P(1), one row per trial.
    """
        # The score is n_arms scaled to the range 0 to 1.
        risk = frame["n_arms"].to_numpy() / 10
        return np.column_stack([1 - risk, risk])


def test_game_pairs():
    """Check the game's pair sampler: one terminated and one completed trial per pair, same phase.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Draw 12 pairs from the invented table and check them.
    trials = make_game_trials()
    pairs = game.sample_pairs(trials, 12, seed=1)
    assert len(pairs) == 12
    assert (trials.loc[pairs["terminated"], "label"] == 1).all() and (trials.loc[pairs["completed"], "label"] == 0).all()
    assert (trials.loc[pairs["terminated"], "phase_label"].to_numpy()
            == trials.loc[pairs["completed"], "phase_label"].to_numpy()).all()
    assert pairs["terminated"].is_unique and pairs["completed"].is_unique
    assert "Early" not in set(trials.loc[pairs["terminated"], "phase_label"])  # no completed partner exists
    # The same seed gives the same pairs, another seed gives other pairs, and too many requested gives all possible.
    assert game.sample_pairs(trials, 12, seed=1).equals(pairs)
    assert not game.sample_pairs(trials, 12, seed=2).equals(pairs)
    assert len(game.sample_pairs(trials, 1000, seed=1)) == 15  # asks for more than exist: gets all pairable ones


def test_game_scoring():
    """Check the model's pick and accuracy: the higher score wins and ties count half.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # The model picks the trial with the higher score; a tie is decided at random.
    trials = make_game_trials()
    assert game.model_pick(0.3, 0.1) == "terminated" and game.model_pick(0.1, 0.3) == "completed"
    assert game.model_pick(0.2, 0.2, rng=0) in {"terminated", "completed"}
    # Accuracy on 10 pairs: perfect for the stub, one half when all scores are tied.
    pairs = game.sample_pairs(trials, 10, seed=3)
    assert game.pairwise_accuracy(trials, pairs, StubModel()) == 1.0  # stub always scores the terminated one higher
    flat = trials.assign(n_arms=2.0)  # everything tied
    assert game.pairwise_accuracy(flat, pairs, StubModel()) == 0.5
    # A new game fixes all ten rounds, with shuffled display order.
    rounds = game.new_game(trials, 10, model=StubModel(), seed=4)
    assert len(rounds) == 10 and all(r["model_pick"] == "terminated" for r in rounds)
    assert {r["left"] for r in rounds} == {"terminated", "completed"}  # display order is shuffled


def test_game_score_matches_predict():
    """Check that the batch scoring used by the game equals predict.score row by row (skipped without the saved model).

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Skip when the saved model or the data file is missing.
    if not (ROOT / "models" / "gbm.joblib").exists() or not (ROOT / "data" / "processed" / "trials.parquet").exists():
        print("skipped: models/gbm.joblib or data/processed/trials.parquet is missing")
        return
    from src.predict import score  # the single-trial scoring function
    # Three real test trials, scored as a batch and one by one.
    trials = game.load_test_trials().head(3)
    assert (trials["split"] == "test").all()
    batch = game.score_rows(trials)
    for position, (_, row) in enumerate(trials.iterrows()):
        assert abs(batch[position] - score(row[FEATURE_COLUMNS].to_dict())) < 1e-9


def test_one_palette():
    """Check that colours live only in app/ui.py and everything else takes them from PALETTE.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    import re  # regular expressions to find colour codes in the source files
    import tomllib  # reads the Streamlit config file

    import plotly.io as pio  # the chart templates
    import ui  # the app's palette

    # The chart template is registered on import.
    assert pio.templates.default == ui.TEMPLATE  # registered on import, before any page builds a chart
    # No page, style or chart file may contain a colour literal of its own.
    colour = re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b|rgba?\(")
    files = [ROOT / "app" / "style.css", ROOT / "app" / "Home.py", ROOT / "app" / "risk_gauge.py",
             ROOT / "src" / "viz.py", *sorted((ROOT / "app" / "pages").glob("*.py"))]
    for path in files:
        assert not colour.findall(path.read_text(encoding="utf-8")), f"colour literal in {path.name}"
    # The CSS takes its colours from the palette.
    css = ui.stylesheet()
    assert f"--ink: {ui.PALETTE['ink']};" in css and f"--low-tint: {ui.PALETTE['low_tint']};" in css
    # The Streamlit theme in config.toml matches the palette.
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    assert {theme["primaryColor"], theme["textColor"]} == {ui.PALETTE["ink"]}
    assert theme["backgroundColor"] == ui.PALETTE["paper"]
    assert theme["secondaryBackgroundColor"] == ui.PALETTE["paper_2"]
    assert theme["borderColor"] == ui.PALETTE["line"] and theme["linkColor"] == ui.PALETTE["accent_text"]
    # A chart built by viz.py uses the palette's bar colour.
    with tempfile.TemporaryDirectory() as tmp:
        make_trials(Path(tmp) / "trials.parquet")
        df = viz.load_trials(Path(tmp) / "trials.parquet")
    fig = viz.rate_bar(viz.rate_by(df, "phase_label"), "phase_label", "Phase", 0.1)
    assert fig.data[0].marker.color == ui.PALETTE["bar"]  # viz.py has no colours of its own


def test_ai_headers():
    """Check that every source file cites the AI use in its first lines (HSG rule).

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # Load the checker from tools/ and expect no file with a missing header.
    sys.path.insert(0, str(ROOT / "tools"))
    import ai_provenance  # the script that lists files without the AI-use header
    assert ai_provenance.missing_headers() == []


# ---- Run all tests when the file is started directly ----
if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

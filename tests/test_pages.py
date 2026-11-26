"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Smoke tests that open every page of the app in Streamlit's test runner and click through the main paths.

Run with ``python tests/test_pages.py``. No network: ClinicalTrials.gov is replaced by an invented
record. The pages read the real `data/processed/trials.parquet` and `models/` files, so this needs the repository
as downloaded; the tests print "skipped" if those files are missing.
"""

import copy  # deep copies of the invented record
import json  # write the offline snapshot file
import re  # regular expressions to find text in the pages
import sys  # edit the module search path
import tempfile  # a temporary folder for the snapshot file
from pathlib import Path  # file paths that work on every system
from unittest import mock  # replace functions with fakes during a test

import requests  # HTTP library; its get() is replaced so no test reaches the network
import streamlit as st  # the app framework (used here to clear its cache)
from streamlit.testing.v1 import AppTest  # runs a Streamlit page without a browser

# Make the repository root and tests/ importable, then load the shared test record and the API client.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for src, when the file is run directly instead of through pytest
sys.path.insert(0, str(ROOT / "tests"))
from test_features import STUDY  # the invented record
import src.api_client  # the offline tests point its snapshot path somewhere else

# Paths and titles the tests use: the entry page, the five pages, and the files the pages need.
HOME = str(ROOT / "app" / "Home.py")
# The five pages in navigation order, with the title they carry in the navigation, the Home card and the header.
PAGES = [("pages/1_Find_a_Trial.py", "Find a trial"), ("pages/2_Check_Your_Trial.py", "Check your own trial"),
         ("pages/3_Explore_Data.py", "Explore the data"), ("pages/4_How_the_Model_Works.py", "How the model works"),
         ("pages/5_Game.py", "Play the game")]
NEEDS = [ROOT / "data" / "processed" / "trials.parquet", ROOT / "models" / "gbm.joblib",
         ROOT / "models" / "gbm_metrics.json"]


def open_page(page):
    """Start the app at Home.py and switch to one page, the way a visitor uses the sidebar.

    Input: page is the path of the page file such as "pages/5_Game.py" (str), or None to stay on the Home page.

    Output: the AppTest object showing that page after it ran.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Clear cached data, start the app and check that Home runs without an exception.
    st.cache_data.clear()
    at = AppTest.from_file(HOME, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    # Go to the requested page, or stay on Home.
    return at.switch_page(page).run() if page else at


def ongoing_study(study_type="INTERVENTIONAL", nct_id="NCT00000001"):
    """Return the invented record from the feature tests, turned into an ongoing trial of the given type and id.

    Input: study_type is the study type (str, "INTERVENTIONAL" or "OBSERVATIONAL"); nct_id is the trial id (str).

    Output: a new study record (dict in the API v2 layout) with status RECRUITING.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Copy the record so STUDY stays unchanged, then set the id, the status and the type.
    study = copy.deepcopy(STUDY)
    study["protocolSection"]["identificationModule"]["nctId"] = nct_id
    study["protocolSection"]["statusModule"]["overallStatus"] = "RECRUITING"
    study["protocolSection"]["designModule"]["studyType"] = study_type
    return study


def stub_api(studies):
    """Make requests.get answer with the given studies, or fail like an offline computer.

    Input: studies is a list of study records (dict), or None to simulate no connection.

    Output: the undo function; calling it puts the real requests.get back.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Remember the real function so it can be restored.
    real = requests.get

    def fake_get(url, params=None, timeout=None):
        """Stand-in for requests.get that answers from the invented studies or raises ConnectionError.

        Input: url is the requested address (str; a trial id in it means a single-trial request); params and timeout are
        ignored.

        Output: a Resp object holding the invented answer; raises ConnectionError when studies is None.
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # No studies means no connection.
        if studies is None:
            raise requests.ConnectionError("offline")

        class Resp:
            """Minimal stand-in for requests.Response."""

            def raise_for_status(self):
                """Pretend the request succeeded.

                Input: none.

                Output: nothing.
                """
                # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
                pass

            def json(self):
                """Return the invented answer.

                Input: none.

                Output: a dict with the studies and their count, or the first study for a single-trial request.
                """
                # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
                return {"studies": studies, "totalCount": len(studies)} if "NCT" not in url else studies[0]

        # The same fake answer for every request.
        return Resp()

    def undo():
        """Put the real requests.get back.

        Input: none.

        Output: nothing; it restores requests.get.
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        requests.get = real

    # Swap in the fake and hand back the way to undo it.
    requests.get = fake_get
    return undo


def markdown(at):
    """Join all markdown text of the page into one string.

    Input: at is the AppTest object of the page.

    Output: one string with the value of every markdown element, separated by spaces.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return " ".join(m.value for m in at.markdown)


def charts(at):
    """List the Plotly figure specs on the page.

    Input: at is the AppTest object of the page.

    Output: a list of strings, one JSON text per Plotly chart.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return [el.proto.spec for el in at.get("plotly_chart")]


def has_drivers_chart(at):
    """Check whether the page shows the 'what pushes the risk' chart with at least one bar.

    Input: at is the AppTest object of the page.

    Output: True if such a horizontal bar chart is on the page, otherwise False.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return any("What pushes this trial" in spec and '"orientation":"h"' in spec.replace(" ", "")
               for spec in charts(at))


def test_page_files():
    """Check that the pages folder holds exactly the five pages, numbered in navigation order.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    assert sorted(p.name for p in (ROOT / "app" / "pages").glob("*.py")) == [Path(p).name for p, _ in PAGES]


def test_home():
    """Check the landing page: one card per page in navigation order, "ongoing" trials, and the colour legend.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Open Home and read its text.
    at = open_page(None)
    text = markdown(at)
    # The card titles must match the page titles, in order.
    titles = [re.search(r'class="band-title">([^<]+)<', m.value) for m in at.markdown]
    assert [t.group(1) for t in titles if t] == [title for _, title in PAGES]
    # The wording and the colour legend are there.
    assert "Search ongoing trials" in text
    assert "chip green" in text and "chip red" in text


def test_explore():
    """Check the Explore page: key numbers, a filter that narrows them, a message when nothing matches, and the SQL box.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Open the page: three key numbers and no exception.
    at = open_page("pages/3_Explore_Data.py")
    assert not at.exception and len(at.metric) == 3
    total = int(at.metric[0].value.replace(",", ""))
    # A phase filter lowers the trial count.
    at.multiselect(key="explore_phases").set_value(["Phase 2"]).run()
    assert not at.exception and int(at.metric[0].value.replace(",", "")) < total
    # A keyword that matches nothing gives a warning, not an error.
    at.text_input(key="explore_condition").set_value("zzzzqqqq").run()
    assert not at.exception and any("No trials match" in w.value for w in at.warning)
    # Clearing the filters brings the full count back.
    clear = [b for b in at.button if b.label == "Clear filters"][0]
    clear.click().run()
    assert int(at.metric[0].value.replace(",", "")) == total
    # The SQL box lists the queries that made the numbers, with GROUP BY and the JOIN for the conditions.
    assert any("Under the hood" in e.label for e in at.expander)
    queries = " ".join(c.value for c in at.code)
    assert "GROUP BY phase_label" in queries and "JOIN trials" in queries


def test_explore_heatmap():
    """Check the Explore page: the two-trait heatmap is shown and the pair control switches the traits.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The first heatmap compares phase and lead sponsor.
    at = open_page("pages/3_Explore_Data.py")
    assert not at.exception
    assert any('"type":"heatmap"' in spec.replace(" ", "") and "Lead sponsor" in spec for spec in charts(at))
    # Choosing another pair redraws the heatmap with blinding.
    at.get("button_group")[0].set_value("Phase × blinding").run()
    assert not at.exception
    assert any('"type":"heatmap"' in spec.replace(" ", "") and "Blinding" in spec for spec in charts(at))


def test_explore_more_views():
    """Check the Explore page: the section "More ways to look at it" with its four tabs and four chart types.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The section and its four tabs are there.
    at = open_page("pages/3_Explore_Data.py")
    assert not at.exception
    assert any(s.value == "More ways to look at it" for s in at.subheader)
    assert [t.label for t in at.tabs[:4]] == ["Conditions map", "Conditions: size vs risk", "Trials per year",
                                              "How trials flow"]
    # The four chart types are drawn.
    specs = " ".join(spec.replace(" ", "") for spec in charts(at))
    for kind in ['"type":"treemap"', '"type":"sankey"', '"stackgroup"']:
        assert kind in specs
    # A keyword that matches nothing stops the page with a warning, without an error in the new section.
    at.text_input(key="explore_condition").set_value("zzzzqqqq").run()
    assert not at.exception


def test_check_your_trial():
    """Check the Check page: no raw API codes, the result strip is there, and Reset restores the example.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Open the page: the risk card is shown.
    at = open_page("pages/2_Check_Your_Trial.py")
    assert not at.exception, [e.value for e in at.exception]
    assert "risk-card" in markdown(at) and "compact" in markdown(at)
    # The table of values has plain words, no codes like PHASE2.
    table = " ".join(c for df in at.table for c in df.value.index.astype(str))
    assert table and not re.search(r"[A-Z]{4,}|_", table), table  # no PHASE2, NON_RANDOMIZED, ...
    assert has_drivers_chart(at)
    # Changing the phase works, and Reset brings the example back.
    assert at.selectbox(key="chk_phase").value == "PHASE2"
    at.selectbox(key="chk_phase").set_value("PHASE3").run()
    assert at.selectbox(key="chk_phase").value == "PHASE3"
    [b for b in at.button if b.label == "Reset to the example trial"][0].click().run()
    assert at.selectbox(key="chk_phase").value == "PHASE2" and not at.exception


def test_check_levers_and_conditions():
    """Check the Check page: the table only suggests design levers, and typed conditions are recognised.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    at = open_page("pages/2_Check_Your_Trial.py")
    # An open-label, single-arm design leaves room for blinding / arm changes.
    at.selectbox(key="chk_masking").set_value("NONE").run()
    # Every suggested row is a design lever (never phase, sponsor, purpose or model).
    rows = [str(r) for df in at.table for r in df.value.index]
    allowed = ("Blinding", "Allocation", "Number of arms", "Length of eligibility", "Minimum age", "Maximum age",
               "Has a data monitoring committee")
    assert rows and all(r.startswith(allowed) for r in rows), rows
    assert not any(r.startswith(("Phase", "Sponsor", "Primary purpose", "Intervention model")) for r in rows)
    # Typed conditions are matched to condition areas.
    at.text_input(key="chk_conditions").set_value("Breast cancer, depression").run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "Recognised condition areas: cancer, mental health and addiction" in captions, captions


def test_check_model_mismatch_message():
    """Check that the page explains a model/form mismatch instead of crashing.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    import src.predict  # the module whose function is faked
    # Pretend the model needs one more input than the form has; restore it at the end.
    real = src.predict.model_features
    src.predict.model_features = lambda model: real(model) + ["a_future_feature"]
    try:
        at = open_page("pages/2_Check_Your_Trial.py")
        assert not at.exception, [e.value for e in at.exception]
        assert any("cannot be scored right now" in e.value for e in at.error)
    finally:
        src.predict.model_features = real


def test_search_with_results_and_details():
    """Check the search page: result cards show the risk and Details opens the full view.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The fake API returns one interventional and one observational trial.
    undo = stub_api([ongoing_study(), ongoing_study("OBSERVATIONAL", "NCT00000002")])
    try:
        at = open_page("pages/1_Find_a_Trial.py")
        assert not at.exception, [e.value for e in at.exception]
        # Two cards; only the first one shows a risk.
        cards = [m.value for m in at.markdown if 'class="card-title"' in m.value]
        assert len(cards) == 2 and "% risk" in cards[0] and "% risk" not in cards[1]
        # The details view shows the risk card and the drivers chart; Back returns to the two cards.
        [b for b in at.button if b.label == "See details"][0].click().run()
        assert not at.exception and "risk-card" in markdown(at)
        assert has_drivers_chart(at)
        [b for b in at.button if "Back to results" in b.label][0].click().run()
        assert len([m for m in at.markdown if 'class="card-title"' in m.value]) == 2
    finally:
        undo()


def test_search_offline():
    """Check the search page: no connection and no saved examples gives a plain message and a retry button, not a crash.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # No connection, and the snapshot path points to a missing file.
    undo = stub_api(None)
    saved = src.api_client.EXAMPLES_PATH
    src.api_client.EXAMPLES_PATH = ROOT / "no_such_folder" / "ongoing_trials.json"  # as if no snapshot was saved
    try:
        # The page shows a plain error, no raw exception text, a retry button and folded technical details.
        at = open_page("pages/1_Find_a_Trial.py")
        assert not at.exception
        assert any("Could not reach ClinicalTrials.gov" in e.value for e in at.error)
        assert not any("ConnectionError" in e.value for e in at.error)  # no raw exception text in the message
        assert any(b.label == "Try again" for b in at.button)
        assert [x.label for x in at.expander] == ["Technical details"]  # the reason is still there, folded away
    finally:
        src.api_client.EXAMPLES_PATH = saved
        undo()


def test_search_offline_examples():
    """Check the search page when offline with a saved snapshot: labelled, scored examples.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # No connection, but a snapshot with two trials in a temporary folder.
    undo = stub_api(None)
    saved = src.api_client.EXAMPLES_PATH
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "ongoing_trials.json"
        snapshot.write_text(json.dumps({"saved_on": "2026-12-01",
                                        "studies": [ongoing_study(), ongoing_study(nct_id="NCT00000003")]}))
        src.api_client.EXAMPLES_PATH = snapshot
        try:
            # The page warns that these are saved examples and still scores both.
            at = open_page("pages/1_Find_a_Trial.py")
            assert not at.exception, [e.value for e in at.exception]
            assert any("saved example trials" in w.value and "2026-12-01" in w.value for w in at.warning)
            cards = [m.value for m in at.markdown if 'class="card-title"' in m.value]
            assert len(cards) == 2 and all("% risk" in c for c in cards)
            [b for b in at.button if b.label == "See details"][0].click().run()
            assert not at.exception and "risk-card" in markdown(at)  # get_trial found it in the snapshot
        finally:
            src.api_client.EXAMPLES_PATH = saved
            undo()


def test_how_the_model_works():
    """Check the model page: title, boxes, six tabs, calibration plot, sliders, confusion matrix, horse race and tables.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Title, boxes, tabs, metrics and charts.
    at = open_page("pages/4_How_the_Model_Works.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown(at)
    assert "<h1" in text and ">How the model works</h1>" in text
    assert "In short" in text and "How to read a risk score" in text and "Red is not a verdict" in text
    assert [t.label for t in at.tabs] == ["What is predicted", "How good is it?", "Which model won?",
                                          "What matters most", "Can you trust it?", "Versions and checks"]
    assert [m.label for m in at.metric] == ["ROC-AUC", "PR-AUC", "Brier score"]
    assert any(c.proto.id and "calibration_chart" in c.proto.id for c in at.get("plotly_chart"))
    # The tilt check: the counted running trials, their chart and the sentence that names the share for 2019.
    assert any(c.proto.id and "tilt_chart" in c.proto.id for c in at.get("plotly_chart"))
    assert any("We counted them" in m.value for m in at.markdown)
    assert any(c.proto.id and "score_violin" in c.proto.id for c in at.get("plotly_chart"))
    # The review-budget slider moves the marked point and the sentence above the gain chart.
    found = [m.value for m in at.markdown if "Reviewing the riskiest" in m.value]
    assert found and "**20%**" in found[0]
    at.slider(key="gain_share").set_value(40).run()
    assert not at.exception
    assert any("**40%**" in m.value for m in at.markdown if "Reviewing the riskiest" in m.value)
    # The cost slider moves the yes/no line (1 / (1 + ratio)) and the confusion matrix below it.
    assert any(c.proto.id and "confusion_chart" in c.proto.id for c in at.get("plotly_chart"))
    assert any("1 / (1 + 5) = 17%" in m.value for m in at.markdown)
    at.slider(key="cost_ratio").set_value(9).run()
    assert not at.exception
    assert any("1 / (1 + 9) = 10%" in m.value for m in at.markdown)
    # Tables: collect them, then check the horse race, the versions and the ablations.
    tables = [df.value for df in at.dataframe]
    # The horse race: six horses, best first, the baseline among them, and the race chart (needs horse_race.json).
    if (ROOT / "models" / "horse_race.json").exists():
        race = [t for t in tables if "PR-AUC, tuned" in t.columns][0]
        assert len(race) == 6 and "Baseline: always the average rate" in list(race["Model"])
        assert any(c.proto.id and "race_chart" in c.proto.id for c in at.get("plotly_chart"))
    versions = [t for t in tables if "Status" in t.columns][0]
    assert list(versions["Status"]) == ["Used in the app", "Tested, not used"]
    assert list(versions["Inputs"]) == [29, 38]
    ablations = [t.value for t in at.table if "Change in test ROC-AUC" in t.value.columns][0]
    assert len(ablations) == 4 and ablations["Variant"].iloc[0] == "All inputs"


def test_every_page_with_mocked_api_client():
    """Check that every page opens with src.api_client mocked and nothing reaches the network.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    from src import api_client  # the module whose functions are mocked
    # A fake answer, and a function that fails the test if anything calls the real network.
    found = {"studies": [ongoing_study()], "total": 1, "error": None}
    no_network = mock.Mock(side_effect=AssertionError("a page tried to reach the network"))
    with mock.patch.object(api_client, "search_trials", return_value=found) as search, \
            mock.patch.object(api_client, "get_trial", return_value=ongoing_study()) as get_trial, \
            mock.patch.object(requests, "get", no_network):
        # Open each page: no exception, the right title, and the search is called only by Find a trial.
        for page, title in PAGES:
            calls = search.call_count
            at = open_page(page)
            assert not at.exception, (page, [e.value for e in at.exception])
            assert f">{title}</h1>" in markdown(at), page
            assert (search.call_count > calls) == page.endswith("1_Find_a_Trial.py"), page
        # Opening the details of a trial uses the mocked get_trial.
        at = open_page("pages/1_Find_a_Trial.py")
        [b for b in at.button if b.label == "See details"][0].click().run()
        assert not at.exception and get_trial.called and "risk-card" in markdown(at)
    assert not no_network.called


def play_round(at, user_side="A"):
    """Click one of the two pick buttons of the game and run the page.

    Input: at is the AppTest object of the game page; user_side is "A" or "B", the card the player picks (str).

    Output: the same AppTest object after the answer is shown.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    [b for b in at.button if b.label == f"Trial {user_side} was stopped early"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_game():
    """Check the game: cards without the outcome, a reveal, ten rounds, a final scoreboard, play again.

    Input: none.

    Output: nothing; the assertions raise an AssertionError when a check fails.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Round 1: two cards, no spoilers, two pick buttons.
    at = open_page("pages/5_Game.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown(at)
    assert "Round 1 of 10" in text and "of the time" in text
    assert text.count('class="card-title"') == 2 and "Stopped early" not in text and "Completed" not in text  # no spoilers
    assert {b.label for b in at.button} >= {"Trial A was stopped early", "Trial B was stopped early"}

    # After a pick the answer, the colours, a Next button and two links are shown.
    play_round(at, "A")
    text = markdown(at)
    assert "was stopped early.**" in text and "chip red" in text and "chip green" in text
    assert [b.label for b in at.button] == ["Next round"]
    links = [e for e in at.get("link_button")]
    assert len(links) == 2 and all(l.proto.url.startswith("https://clinicaltrials.gov/study/NCT") for l in links)

    # Play rounds 2 to 10.
    for round_number in range(2, 11):
        [b for b in at.button if b.label in ("Next round", "See the final score")][0].click().run()
        assert f"Round {round_number} of 10" in markdown(at)
        play_round(at, "B")
    # The final scoreboard has three numbers and ten rows; Play again starts a new game.
    assert [b.label for b in at.button] == ["See the final score"]
    at.button[0].click().run()
    assert not at.exception and [m.label for m in at.metric] == ["You", "Model", "Model on 2,000 pairs"]
    assert len(at.table[0].value) == 10
    [b for b in at.button if b.label == "Play again"][0].click().run()
    assert not at.exception and "Round 1 of 10" in markdown(at)


# ---- Run all tests when the file is started directly ----
if __name__ == "__main__":
    # Skip when the data or model files are missing.
    if not all(p.exists() for p in NEEDS):
        print("skipped: data/ or models/ files are missing")
        sys.exit(0)
    # Run every test function and print its name.
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

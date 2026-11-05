"""Smoke tests that open every page of the app in Streamlit's test runner and click through the main paths.

Run with ``python tests/test_pages.py``. No network: ClinicalTrials.gov is replaced by an invented
record. The pages read the real `data/processed/trials.parquet` and `models/` files, so this needs the repository
as downloaded; the tests print "skipped" if those files are missing.
"""

import copy
import re
import sys
from pathlib import Path

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_features import STUDY  # noqa: E402

HOME = str(ROOT / "app" / "Home.py")
NEEDS = [ROOT / "data" / "processed" / "trials.parquet", ROOT / "models" / "gbm.joblib",
         ROOT / "models" / "gbm_metrics.json"]


def open_page(page):
    """Start the app at Home.py and switch to one page, the way a visitor uses the sidebar."""
    st.cache_data.clear()
    at = AppTest.from_file(HOME, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at.switch_page(page).run() if page else at


def ongoing_study(study_type="INTERVENTIONAL", nct_id="NCT00000001"):
    """The invented record from the feature tests, turned into an ongoing trial of the given type and id."""
    study = copy.deepcopy(STUDY)
    study["protocolSection"]["identificationModule"]["nctId"] = nct_id
    study["protocolSection"]["statusModule"]["overallStatus"] = "RECRUITING"
    study["protocolSection"]["designModule"]["studyType"] = study_type
    return study


def stub_api(studies):
    """Make requests.get answer with the given studies (or fail offline when studies is None); returns an undo function."""
    real = requests.get

    def fake_get(url, params=None, timeout=None):
        if studies is None:
            raise requests.ConnectionError("offline")

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"studies": studies, "totalCount": len(studies)} if "NCT" not in url else studies[0]

        return Resp()

    requests.get = fake_get
    return lambda: setattr(requests, "get", real)


def markdown(at):
    """All markdown text of the page in one string."""
    return " ".join(m.value for m in at.markdown)


def charts(at):
    """The Plotly figure specs (JSON text) on the page."""
    return [el.proto.spec for el in at.get("plotly_chart")]


def has_drivers_chart(at):
    """True if the page shows the 'what pushes the risk' chart with at least one bar."""
    return any("What pushes this trial" in spec and '"orientation":"h"' in spec.replace(" ", "")
               for spec in charts(at))


def test_home():
    """Landing page: four cards and the colour legend."""
    at = open_page(None)
    text = markdown(at)
    for title in ("Find a trial", "Check your own trial", "Explore the data", "How the model works"):
        assert title in text
    assert "chip green" in text and "chip red" in text


def test_explore():
    """Explore: key numbers, a filter that narrows them, and a message instead of an error when nothing matches."""
    at = open_page("pages/2_Explore.py")
    assert not at.exception and len(at.metric) == 3
    total = int(at.metric[0].value.replace(",", ""))
    at.multiselect(key="explore_phases").set_value(["Phase 2"]).run()
    assert not at.exception and int(at.metric[0].value.replace(",", "")) < total
    at.text_input(key="explore_condition").set_value("zzzzqqqq").run()
    assert not at.exception and any("No trials match" in w.value for w in at.warning)
    clear = [b for b in at.button if b.label == "Clear filters"][0]
    clear.click().run()
    assert int(at.metric[0].value.replace(",", "")) == total


def test_check_your_trial():
    """Check page: raw API codes stay out of the text, the result strip is there, and Reset restores the example."""
    at = open_page("pages/5_Check_Your_Trial.py")
    assert not at.exception, [e.value for e in at.exception]
    assert "risk-card" in markdown(at) and "compact" in markdown(at)
    table = " ".join(c for df in at.table for c in df.value.index.astype(str))
    assert table and not re.search(r"[A-Z]{4,}|_", table), table  # no PHASE2, NON_RANDOMIZED, ...
    assert has_drivers_chart(at)
    assert at.selectbox(key="chk_phase").value == "PHASE2"
    at.selectbox(key="chk_phase").set_value("PHASE3").run()
    assert at.selectbox(key="chk_phase").value == "PHASE3"
    [b for b in at.button if b.label == "Reset to the example trial"][0].click().run()
    assert at.selectbox(key="chk_phase").value == "PHASE2" and not at.exception


def test_check_levers_and_conditions():
    """Check page: the table only suggests design levers, and typed conditions are recognised."""
    at = open_page("pages/5_Check_Your_Trial.py")
    # An open-label, single-arm design leaves room for blinding / arm changes.
    at.selectbox(key="chk_masking").set_value("NONE").run()
    rows = [str(r) for df in at.table for r in df.value.index]
    allowed = ("Blinding", "Allocation", "Number of arms", "Length of eligibility", "Minimum age", "Maximum age",
               "Has a data monitoring committee")
    assert rows and all(r.startswith(allowed) for r in rows), rows
    assert not any(r.startswith(("Phase", "Sponsor", "Primary purpose", "Intervention model")) for r in rows)
    at.text_input(key="chk_conditions").set_value("Breast cancer, depression").run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "Recognised condition areas: cancer, mental health and addiction" in captions, captions


def test_check_model_mismatch_message():
    """If the saved model expects an input the form lacks, the page explains it instead of crashing."""
    import src.predict
    real = src.predict.model_features
    src.predict.model_features = lambda model: real(model) + ["a_future_feature"]
    try:
        at = open_page("pages/5_Check_Your_Trial.py")
        assert not at.exception, [e.value for e in at.exception]
        assert any("cannot be scored right now" in e.value for e in at.error)
    finally:
        src.predict.model_features = real


def test_search_with_results_and_details():
    """Search: a result card shows its risk, interventional trials only get one, and Details opens the full view."""
    undo = stub_api([ongoing_study(), ongoing_study("OBSERVATIONAL", "NCT00000002")])
    try:
        at = open_page("pages/1_Search.py")
        assert not at.exception, [e.value for e in at.exception]
        cards = [m.value for m in at.markdown if 'class="card-title"' in m.value]
        assert len(cards) == 2 and "% risk" in cards[0] and "% risk" not in cards[1]
        [b for b in at.button if b.label == "See details"][0].click().run()
        assert not at.exception and "risk-card" in markdown(at)
        assert has_drivers_chart(at)
        [b for b in at.button if "Back to results" in b.label][0].click().run()
        assert len([m for m in at.markdown if 'class="card-title"' in m.value]) == 2
    finally:
        undo()


def test_search_offline():
    """Search: no connection gives a plain message and a retry button, not a crash."""
    undo = stub_api(None)
    try:
        at = open_page("pages/1_Search.py")
        assert not at.exception
        assert any("Could not reach ClinicalTrials.gov" in e.value for e in at.error)
        assert any(b.label == "Try again" for b in at.button)
    finally:
        undo()


def test_explain():
    """Explain: summary box and four tabs, with the numbers filled in."""
    at = open_page("pages/2_Explain.py")
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4 and "In short" in markdown(at)
    assert [m.label for m in at.metric] == ["ROC-AUC", "PR-AUC", "Brier score"]


def play_round(at, user_side="A"):
    """Click one of the two pick buttons and return the page after the answer is shown."""
    [b for b in at.button if b.label == f"Trial {user_side} was stopped early"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_game():
    """Game: two cards without the outcome, a reveal with link and scores, ten rounds, a final scoreboard, play again."""
    at = open_page("pages/5_Game.py")
    assert not at.exception, [e.value for e in at.exception]
    text = markdown(at)
    assert "Round 1 of 10" in text and "of the time" in text
    assert text.count('class="card-title"') == 2 and "Stopped early" not in text and "Completed" not in text  # no spoilers
    assert {b.label for b in at.button} >= {"Trial A was stopped early", "Trial B was stopped early"}

    play_round(at, "A")
    text = markdown(at)
    assert "was stopped early.**" in text and "chip red" in text and "chip green" in text
    assert [b.label for b in at.button] == ["Next round"]
    links = [e for e in at.get("link_button")]
    assert len(links) == 2 and all(l.proto.url.startswith("https://clinicaltrials.gov/study/NCT") for l in links)

    for round_number in range(2, 11):
        [b for b in at.button if b.label in ("Next round", "See the final score")][0].click().run()
        assert f"Round {round_number} of 10" in markdown(at)
        play_round(at, "B")
    assert [b.label for b in at.button] == ["See the final score"]
    at.button[0].click().run()
    assert not at.exception and [m.label for m in at.metric] == ["You", "Model", "Model on 2,000 pairs"]
    assert len(at.table[0].value) == 10
    [b for b in at.button if b.label == "Play again"][0].click().run()
    assert not at.exception and "Round 1 of 10" in markdown(at)


if __name__ == "__main__":
    if not all(p.exists() for p in NEEDS):
        print("skipped: data/ or models/ files are missing")
        sys.exit(0)
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

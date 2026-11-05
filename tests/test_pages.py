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
    """Start the app at Home.py and switch to one page, the way a visitor uses the top navigation."""
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


def test_home():
    """Landing page: four cards and the colour legend."""
    at = open_page(None)
    text = markdown(at)
    for title in ("Find a trial", "Check your own trial", "Explore the data", "How the model works"):
        assert title in text
    assert "chip green" in text and "chip red" in text


def test_no_pages_folder():
    """Page scripts live in app/views/: a folder named app/pages/ would switch on Streamlit's old multipage mode.

    In that mode a fresh server whose first request is a page URL runs that page without Home.py (no top
    navigation), and keeps doing so for every visitor until someone opens Home.
    """
    assert not (ROOT / "app" / "pages").exists(), "keep page scripts in app/views/ and delete app/pages/"


def test_explore():
    """Explore: key numbers, a filter that narrows them, and a message instead of an error when nothing matches."""
    at = open_page("views/4_Explore.py")
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
    at = open_page("views/2_Check_Your_Trial.py")
    assert not at.exception, [e.value for e in at.exception]
    assert "risk-card" in markdown(at) and "compact" in markdown(at)
    table = " ".join(c for df in at.table for c in df.value.index.astype(str))
    assert table and not re.search(r"[A-Z]{4,}|_", table), table  # no PHASE2, NON_RANDOMIZED, ...
    assert at.selectbox(key="chk_phase").value == "PHASE2"
    at.selectbox(key="chk_phase").set_value("PHASE3").run()
    assert at.selectbox(key="chk_phase").value == "PHASE3"
    [b for b in at.button if b.label == "Reset to the example trial"][0].click().run()
    assert at.selectbox(key="chk_phase").value == "PHASE2" and not at.exception


def test_search_with_results_and_details():
    """Search: a result card shows its risk, interventional trials only get one, and Details opens the full view."""
    undo = stub_api([ongoing_study(), ongoing_study("OBSERVATIONAL", "NCT00000002")])
    try:
        at = open_page("views/1_Search.py")
        assert not at.exception, [e.value for e in at.exception]
        cards = [m.value for m in at.markdown if 'class="card-title"' in m.value]
        assert len(cards) == 2 and "% risk" in cards[0] and "% risk" not in cards[1]
        [b for b in at.button if b.label == "See details"][0].click().run()
        assert not at.exception and "risk-card" in markdown(at)
        [b for b in at.button if "Back to results" in b.label][0].click().run()
        assert len([m for m in at.markdown if 'class="card-title"' in m.value]) == 2
    finally:
        undo()


def test_search_offline():
    """Search: no connection gives a plain message and a retry button, not a crash."""
    undo = stub_api(None)
    try:
        at = open_page("views/1_Search.py")
        assert not at.exception
        assert any("Could not reach ClinicalTrials.gov" in e.value for e in at.error)
        assert any(b.label == "Try again" for b in at.button)
    finally:
        undo()


def test_explain():
    """Explain: summary box and four tabs, with the numbers filled in."""
    at = open_page("views/5_Explain.py")
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4 and "In short" in markdown(at)
    assert [m.label for m in at.metric] == ["ROC-AUC", "PR-AUC", "Brier score"]


if __name__ == "__main__":
    if not all(p.exists() for p in NEEDS):
        print("skipped: data/ or models/ files are missing")
        sys.exit(0)
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("all ok")

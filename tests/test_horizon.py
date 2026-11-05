"""Offline checks for the 3-year horizon label (src/horizon.py) and the two horizon tables of build_dataset.py.

Run with ``python tests/test_horizon.py`` (or pytest). No network: every record below is invented test data,
not real trial data. Also checks that the old ``trials.parquet`` build is unchanged by the extra statuses.
"""

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import build_dataset  # noqa: E402
import fetch_data  # noqa: E402
import horizon  # noqa: E402
from features import FEATURE_COLUMNS, build_features  # noqa: E402
from test_features import STUDY  # noqa: E402

SNAPSHOT = "2026-11-05"  # -> follow-up cutoff 2023-05-05 with a 3-year horizon and a 6-month lag


def study(nct, status, start, stop=None, stop_type="ACTUAL", why=None, phases=("PHASE2",)):
    """The invented record of test_features.py with a new id, status, dates, stop reason and phase."""
    record = copy.deepcopy(STUDY)
    proto = record["protocolSection"]
    proto["identificationModule"]["nctId"] = nct
    status_module = {"overallStatus": status, "startDateStruct": {"date": start, "type": "ACTUAL"}}
    if stop:
        status_module["completionDateStruct"] = {"date": stop, "type": stop_type}
    if why:
        status_module["whyStopped"] = why
    proto["statusModule"] = status_module
    if phases is None:
        proto["designModule"].pop("phases")
    else:
        proto["designModule"]["phases"] = list(phases)
    return record


# nct id -> (record, expected horizon label: 1, 0 or None for "no label")
CASES = {
    "NCT01": (study("NCT01", "TERMINATED", "2012-03", "2013-05", why="Slow accrual"), 1),
    "NCT02": (study("NCT02", "TERMINATED", "2012-03", "2016-01", why="Funding ended"), 0),
    "NCT03": (study("NCT03", "TERMINATED", "2012-03", "2014-01", stop_type="ESTIMATED"), None),
    "NCT04": (study("NCT04", "TERMINATED", "2012-03"), None),  # no stop date at all
    "NCT05": (study("NCT05", "COMPLETED", "2015-01", "2019-06"), 0),
    "NCT06": (study("NCT06", "RECRUITING", "2018-06", "2027-01", stop_type="ESTIMATED"), 0),
    "NCT07": (study("NCT07", "RECRUITING", "2024-01"), None),  # too recent
    "NCT08": (study("NCT08", "NOT_YET_RECRUITING", "2019-01"), None),  # never started
    "NCT09": (study("NCT09", "COMPLETED", "2007-05", "2009-01"), 0),  # labelled, but before the window
    "NCT10": (study("NCT10", "TERMINATED", "2020-05", "2020-09",
                    why="Halted because of the COVID-19 pandemic; slow accrual"), 1),
    "NCT11": (study("NCT11", "TERMINATED", "2015-01", "2018-01"), 1),  # exactly 3 years: still inside
    "NCT12": (study("NCT12", "ACTIVE_NOT_RECRUITING", "2023-05"), 0),  # 2023-05-01 <= cutoff 2023-05-05
    "NCT13": (study("NCT13", "ENROLLING_BY_INVITATION", "2023-06"), None),  # after the cutoff
    "NCT14": (study("NCT14", "COMPLETED", "2016-02", phases=None), 0),  # labelled, but no phase
    "NCT15": (study("NCT15", "TERMINATED", "2016-02", "2015-01"), None),  # stops before it starts
    "NCT16": (study("NCT16", "WITHDRAWN", "2016-02"), None),
    "NCT17": (study("NCT17", "COMPLETED", "2019-07", "2020-01"), 0),
}


def records():
    """Feature rows and outcome rows of every case, the way build_dataset.load_rows produces them."""
    studies = [record for record, _ in CASES.values()]
    return [build_features(s) for s in studies], [horizon.outcome_fields(s) for s in studies]


def merged():
    """All cases merged with their outcome fields."""
    rows, outcomes = records()
    return build_dataset.merge_outcomes(rows, outcomes)


def test_labels():
    """Every case gets the label the rules say."""
    df = merged().set_index("nct_id")
    labels = horizon.horizon_labels(df, SNAPSHOT)
    for nct, (_, expected) in CASES.items():
        got = labels[nct]
        assert (pd.isna(got) if expected is None else got == expected), (nct, got, expected)


def test_cutoff_and_years_to_stop():
    """The cutoff is the snapshot minus horizon and lag; years to stop only exist for terminated trials."""
    assert horizon.followup_cutoff(SNAPSHOT) == pd.Timestamp("2023-05-05")
    assert horizon.followup_cutoff("2026-11-05", years=5, lag_months=0) == pd.Timestamp("2021-11-05")
    df = merged().set_index("nct_id")
    years = horizon.years_to_stop(df)
    assert years["NCT01"] == 14 / 12 and years["NCT11"] == 3.0  # whole months, like the label
    assert pd.isna(years["NCT03"]) and pd.isna(years["NCT05"]) and pd.isna(years["NCT15"])


def test_horizon_dataset():
    """Rows, splits and labels of trials_3y.parquet; no outcome column ever reaches it."""
    df = build_dataset.build_horizon_dataset(merged(), SNAPSHOT)
    by_id = df.set_index("nct_id")
    assert set(by_id.index) == {"NCT01", "NCT02", "NCT05", "NCT06", "NCT10", "NCT11", "NCT12", "NCT17"}
    assert by_id.loc["NCT01", "split"] == "train" and by_id.loc["NCT06", "split"] == "train"
    assert by_id.loc["NCT17", "split"] == "valid"
    assert by_id.loc["NCT10", "split"] == "test" and by_id.loc["NCT12", "split"] == "test"
    assert by_id["label"].to_dict() == {nct: CASES[nct][1] for nct in by_id.index}
    assert not set(df.columns) & set(horizon.OUTCOME_COLUMNS)
    assert df.attrs["followup_cutoff"] == "2023-05-05" and df.attrs["horizon_years"] == 3
    assert str(df["label"].dtype) == "int8" and str(df["has_dmc"].dtype) == "Int8"


def test_outcomes_table():
    """The EDA table keeps every status since 2008, both labels, the stop reason and the COVID flag."""
    out = build_dataset.build_outcomes(merged(), SNAPSHOT).set_index("nct_id")
    assert "NCT09" not in out.index and {"NCT07", "NCT08", "NCT16"} <= set(out.index)
    assert out.loc["NCT01", "label_status"] == 1 and out.loc["NCT01", "label_3y"] == 1
    assert out.loc["NCT02", "label_status"] == 1 and out.loc["NCT02", "label_3y"] == 0
    assert pd.isna(out.loc["NCT06", "label_status"]) and out.loc["NCT06", "label_3y"] == 0
    assert out.loc["NCT01", "stop_reason"] == "Recruitment" and out.loc["NCT10", "stop_reason"] == "COVID-19"
    assert bool(out.loc["NCT10", "mentions_covid"]) and not bool(out.loc["NCT01", "mentions_covid"])
    assert pd.isna(out.loc["NCT05", "stop_reason"])  # only terminated trials get a reason group


def test_status_table_unchanged():
    """trials.parquet comes out the same whether or not the download holds still-running trials."""
    rows, _ = records()
    ended = [r for r in rows if r["overall_status"] in ("COMPLETED", "TERMINATED")]
    pd.testing.assert_frame_equal(build_dataset.build_dataset(rows), build_dataset.build_dataset(ended))


def test_no_leakage():
    """Outcome fields never become features, and build_features ignores them."""
    assert not set(FEATURE_COLUMNS) & set(horizon.OUTCOME_COLUMNS)
    assert not set(FEATURE_COLUMNS) & set(horizon.outcome_fields(STUDY))
    changed = copy.deepcopy(STUDY)
    changed["protocolSection"]["statusModule"]["whyStopped"] = "COVID-19"
    changed["protocolSection"]["statusModule"]["completionDateStruct"] = {"date": "2031-01", "type": "ESTIMATED"}
    assert build_features(changed) == build_features(STUDY)


def test_stop_reasons():
    """Rough reason groups, COVID first."""
    assert horizon.stop_reason("Slow accrual") == "Recruitment"
    assert horizon.stop_reason("Low enrollment due to COVID-19") == "COVID-19"
    assert horizon.stop_reason("Sponsor decision") == "Sponsor or business decision"
    assert horizon.stop_reason("Lack of funding") == "Funding"
    assert horizon.stop_reason("Futility at interim analysis") == "Efficacy or futility"
    assert horizon.stop_reason(None) == horizon.stop_reason("   ") == horizon.OTHER_REASON
    assert horizon.mentions_covid("SARS-CoV-2 outbreak") and not horizon.mentions_covid(None)


def test_download_checks():
    """The horizon files need still-running trials and the completion fields; the snapshot date is read."""
    new = {"params": fetch_data.base_params(full=False), "fetched_on": "2026-11-05"}
    assert build_dataset.horizon_problem(new) is None
    assert build_dataset.horizon_problem({"params": fetch_data.base_params(full=True)}) is None
    old = copy.deepcopy(new)
    old["params"]["filter.overallStatus"] = "COMPLETED,TERMINATED"
    assert "RECRUITING" in build_dataset.horizon_problem(old)
    old_fields = copy.deepcopy(new)
    old_fields["params"]["fields"] = "NCTId,OverallStatus,StartDate"
    assert "CompletionDate" in build_dataset.horizon_problem(old_fields)
    assert build_dataset.horizon_problem({}) is not None
    assert build_dataset.snapshot_date(new) == pd.Timestamp("2026-11-05")
    assert build_dataset.snapshot_date({"started": "Tue Nov  3 15:09:39 2026"}) == pd.Timestamp("2026-11-03")
    assert set(horizon.STILL_RUNNING) <= set(fetch_data.STATUSES)


if __name__ == "__main__":
    test_labels()
    test_cutoff_and_years_to_stop()
    test_horizon_dataset()
    test_outcomes_table()
    test_status_table_unchanged()
    test_no_leakage()
    test_stop_reasons()
    test_download_checks()
    print("all ok")

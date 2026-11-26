"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

How many trials of each start year have not ended yet: the check behind the "test-year tilt".

The model predicts too few stopped trials for 2017-2019. Our explanation: only trials that have ended are in the
data. A stopped trial ends early by definition, so it is already in; a long trial that will complete may still be
running, so it is missing, and the later the start year, the more of them. This module turns that explanation into
numbers. It counts, per start year, the interventional trials in every status on ClinicalTrials.gov, and
``tilt_table`` shows how much of the gap the still-running trials can explain.

Statuses: *ended* = COMPLETED + TERMINATED (what the model learned from); *still running* = RECRUITING,
NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION and SUSPENDED; *unknown* = UNKNOWN (the sponsor
has not confirmed the status for over two years, so nobody knows how these trials ended); WITHDRAWN trials never
enrolled anyone and are left out, as in training.

The counts are saved in ``data/processed/open_trials_by_year.csv`` (made on 2026-11-23 with the same queries from a
browser; the counts of ended trials match ``trials.parquet`` within 0.3%).

Usage (needs internet access to clinicaltrials.gov):
    python src/open_trials.py      # refresh data/processed/open_trials_by_year.csv
"""

import time  # pause between two API calls, so we stay well below the rate limit
from pathlib import Path  # file paths that work on every operating system

import pandas as pd  # the table of counts (DataFrame)

ROOT = Path(__file__).resolve().parents[1]
COUNTS_PATH = ROOT / "data" / "processed" / "open_trials_by_year.csv"
API = "https://clinicaltrials.gov/api/v2/studies"
YEARS = range(2008, 2020)
ENDED = ["completed", "terminated"]
RUNNING = ["recruiting", "not_yet_recruiting", "active_not_recruiting", "enrolling_by_invitation", "suspended"]
STATUSES = ENDED + ["withdrawn"] + RUNNING + ["unknown"]


def count(year: int, status: str) -> int:
    """Count the interventional trials of one start year that have one status today (one API call).

    Input: year is the start year (integer); status is the status in lower case, for example "completed" (text).

    Output: the number of trials as an integer.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    import requests  # only needed when refreshing the counts

    params = {"filter.overallStatus": status.upper(),
              "filter.advanced": f"AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[{year}-01-01,{year}-12-31]",
              "countTotal": "true", "pageSize": 1, "fields": "NCTId"}
    response = requests.get(API, params=params, timeout=30,
                            headers={"User-Agent": "trial-radar-student-project/0.1"})
    response.raise_for_status()
    return int(response.json().get("totalCount", 0))


def fetch_counts() -> pd.DataFrame:
    """Count every status for every start year (about 100 API calls, paced well below the limit).

    Input: none.

    Output: a table with one row per start year and one column per status.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    rows = []
    for year in YEARS:
        row = {"start_year": year}
        for status in STATUSES:
            row[status] = count(year, status)
            time.sleep(0.7)
        rows.append(row)
    return pd.DataFrame(rows)


def load_counts(path=COUNTS_PATH) -> pd.DataFrame:
    """Read the saved counts per start year.

    Input: path is the CSV file with the counts (default: data/processed/open_trials_by_year.csv).

    Output: the table of counts, one row per start year.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return pd.read_csv(path)


def tilt_table(counts: pd.DataFrame, predicted: pd.Series) -> pd.DataFrame:
    """Per start year: how many trials are still running, and the real share stopped with and without them.

    Input: counts is the table from load_counts(); predicted is the model's average prediction per start year
    (a Series with the year as index), for example on the test trials.

    Output: one row per year in predicted, with ``ended``, ``running``, ``running_share`` (of ended + running),
    ``unknown_share`` (of ended + running + unknown), ``observed_rate`` (stopped among the ended trials, what the
    data shows), ``rate_if_running_complete`` (the lowest the final rate can be: every running trial completes),
    ``predicted``, and the gaps ``gap_observed`` / ``gap_if_running_complete`` (real rate above the prediction).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    c = counts.set_index("start_year")
    ended = c[ENDED].sum(axis=1)
    running = c[RUNNING].sum(axis=1)
    out = pd.DataFrame({
        "ended": ended,
        "running": running,
        "running_share": running / (ended + running),
        "unknown_share": c["unknown"] / (ended + running + c["unknown"]),
        "observed_rate": c["terminated"] / ended,
        "rate_if_running_complete": c["terminated"] / (ended + running),
    }).loc[predicted.index]
    out["predicted"] = predicted
    out["gap_observed"] = out["observed_rate"] / out["predicted"] - 1
    out["gap_if_running_complete"] = out["rate_if_running_complete"] / out["predicted"] - 1
    return out.rename_axis("start_year").reset_index()


if __name__ == "__main__":
    # Refresh the counts from the API (needs internet) and save them.
    table = fetch_counts()
    table.to_csv(COUNTS_PATH, index=False)
    print(table.to_string(index=False))
    print(f"Saved {COUNTS_PATH.relative_to(ROOT)}")

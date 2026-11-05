"""The 3-year horizon label and the outcome fields it needs. Label and EDA only, never model features.

Fixed-horizon question: "was the trial terminated within ``HORIZON_YEARS`` years of its start?"

* 1 = status TERMINATED and the trial stopped (its completion date) at most ``HORIZON_YEARS`` x 12 months
  after the start month (months, because most registry dates are month precision: ``YYYY-MM``);
* 0 = COMPLETED (whenever), TERMINATED later than that, or still running (RECRUITING,
  ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION) when the data was downloaded;
* no label (the row is dropped) = every other status (NOT_YET_RECRUITING, WITHDRAWN, SUSPENDED,
  UNKNOWN, ...), a TERMINATED trial without a usable stop date, and every trial that started too
  recently to have been followed for the whole horizon plus a reporting lag.

Unlike the status label in ``features.LABELS`` (TERMINATED vs COMPLETED among trials that have ended),
every labelled trial here had the same time to be stopped, so recent start years are not biased towards
early terminations and still-running trials stop being dropped.

Leakage: the completion date and ``whyStopped`` are read here and nowhere else. They build the label and
the EDA table (``data/processed/outcomes.parquet``); ``build_features`` never reads them, they are not in
``FEATURE_COLUMNS`` and they are never written to a file a model trains on.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import parse_start_date  # noqa: E402  (same 'YYYY-MM[-DD]' parsing as the start date)

HORIZON_YEARS = 3
# Sponsors update a record weeks or months after a trial stops. Trials must have started this much more than
# HORIZON_YEARS before the download, so a stop near the end of the horizon has had time to reach the registry.
REPORTING_LAG_MONTHS = 6

STILL_RUNNING = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]
# Fetched for the app and the charts but never labelled: such a trial has not started (a past start date
# on a not-yet-recruiting record is a stale estimate).
NOT_STARTED = ["NOT_YET_RECRUITING"]
LABELLED_STATUSES = ["COMPLETED", "TERMINATED"] + STILL_RUNNING
# A stop date of these types is the planned end, not the day the trial stopped.
NOT_ACTUAL_DATE_TYPES = {"ESTIMATED", "ANTICIPATED"}

# Start-date windows of the horizon experiment (inclusive). The test window ends at followup_cutoff().
WINDOW_START = pd.Timestamp("2008-01-01")
SPLITS = {
    "train": (pd.Timestamp("2008-01-01"), pd.Timestamp("2018-12-31")),
    "valid": (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")),
    "test": (pd.Timestamp("2020-01-01"), None),
}

# Columns that hold outcome information. They may appear in outcomes.parquet (EDA) but never in a
# training file; src/train_horizon.py refuses a dataset that has any of them.
OUTCOME_COLUMNS = ["completion_date", "completion_date_type", "why_stopped", "stop_reason",
                   "mentions_covid", "years_to_stop"]

# Rough stop-reason groups for the dashboard, first match wins (COVID first, so "slow accrual due to
# COVID-19" counts as COVID). Free text is messy: these are approximate and never a model input.
STOP_REASONS = [
    ("COVID-19", r"covid|corona ?virus|sars-?cov|pandemic|lockdown"),
    ("Recruitment", r"accru|recruit|enrol|enroll|low participation|lack of (?:participants|patients|subjects)"
                    r"|insufficient (?:participants|patients|subjects)|no participants|slow"),
    ("Funding", r"fund|financ|budget|grant|money|cost"),
    ("Sponsor or business decision", r"business|sponsor|strateg|commercial|company|portfolio|priorit|"
                                     r"administrative|corporate|management decision"),
    ("Safety", r"safety|adverse|toxicit|side effect"),
    ("Efficacy or futility", r"futil|efficacy|lack of (?:effect|benefit)|ineffective|interim analysis"),
    ("Investigator or site", r"investigator|\bpi\b|staff|site closed|site closure|relocat"),
]
OTHER_REASON = "Other or not stated"
_REASON_PATTERNS = [(name, re.compile(pattern, re.I)) for name, pattern in STOP_REASONS]
_COVID = _REASON_PATTERNS[0][1]


def _get(obj, *path):
    """Walk nested dicts and return None as soon as a key is missing."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def outcome_fields(study_json: dict) -> dict:
    """What happened to one trial after it started: stop date and stop reason. Label and EDA only.

    Args:
        study_json: one element of the API's ``studies`` list.

    Returns:
        ``nct_id``, ``completion_date`` (ISO date or None), ``completion_date_type`` (``ACTUAL`` /
        ``ESTIMATED`` or None) and ``why_stopped`` (free text or None).
    """
    status = _get(study_json, "protocolSection", "statusModule") or {}
    completion = parse_start_date(_get(status, "completionDateStruct", "date"))
    why = status.get("whyStopped")
    return {
        "nct_id": _get(study_json, "protocolSection", "identificationModule", "nctId"),
        "completion_date": completion.isoformat() if completion else None,
        "completion_date_type": _get(status, "completionDateStruct", "type"),
        "why_stopped": (why.strip() or None) if isinstance(why, str) else None,
    }


def followup_cutoff(snapshot, years: int = HORIZON_YEARS, lag_months: int = REPORTING_LAG_MONTHS) -> pd.Timestamp:
    """Latest start date with full follow-up: the download date minus the horizon and the reporting lag."""
    return pd.Timestamp(snapshot) - pd.DateOffset(years=years, months=lag_months)


def stop_date(df: pd.DataFrame) -> pd.Series:
    """The date each trial stopped, where it is usable: an actual completion date on or after the start.

    Missing for planned (estimated) dates, missing dates and impossible ones (before the start).
    """
    start = pd.to_datetime(df["start_date"])
    stop = pd.to_datetime(df["completion_date"], errors="coerce")
    kind = df["completion_date_type"].astype("string").str.upper()
    planned = kind.isin(NOT_ACTUAL_DATE_TYPES).fillna(False).astype(bool)
    return stop.where(~planned & (stop >= start))


def months_to_stop(df: pd.DataFrame) -> pd.Series:
    """Whole calendar months from the start month to the stop month (usable stop dates only, else missing)."""
    start = pd.to_datetime(df["start_date"])
    stopped = stop_date(df)
    return (stopped.dt.year - start.dt.year) * 12 + (stopped.dt.month - start.dt.month)


def horizon_labels(df: pd.DataFrame, snapshot, years: int = HORIZON_YEARS,
                   lag_months: int = REPORTING_LAG_MONTHS) -> pd.Series:
    """The horizon label (1 / 0 / missing) for every row, following the rules in the module docstring.

    Args:
        df: needs ``overall_status``, ``start_date``, ``completion_date``, ``completion_date_type``.
        snapshot: date the raw data was downloaded (``fetched_on`` in ``data/raw/manifest.json``).
        years, lag_months: the horizon and the reporting lag.

    Returns:
        Nullable ``Int8`` series aligned with ``df``; missing means "no label, drop for training".
    """
    start = pd.to_datetime(df["start_date"])
    status = df["overall_status"]
    months = months_to_stop(df)

    label = pd.Series(pd.NA, index=df.index, dtype="Int8")
    label[status.eq("COMPLETED") | status.isin(STILL_RUNNING)] = 0
    terminated = status.eq("TERMINATED") & months.notna()
    label[terminated & (months <= 12 * years)] = 1
    label[terminated & (months > 12 * years)] = 0
    followed = start.notna() & (start <= followup_cutoff(snapshot, years, lag_months))
    return label.where(followed, pd.NA)


def years_to_stop(df: pd.DataFrame) -> pd.Series:
    """Years (whole months / 12) from start to stop for terminated trials with a usable stop date."""
    return (months_to_stop(df) / 12).where(df["overall_status"].eq("TERMINATED"))


def stop_reason(text) -> str:
    """Rough group of a free-text stop reason, e.g. 'Slow accrual' -> 'Recruitment'. EDA only."""
    if not isinstance(text, str) or not text.strip():
        return OTHER_REASON
    for name, pattern in _REASON_PATTERNS:
        if pattern.search(text):
            return name
    return OTHER_REASON


def mentions_covid(text) -> bool:
    """True if a stop reason mentions COVID-19 or the pandemic."""
    return isinstance(text, str) and bool(_COVID.search(text))


def split_for(start_dates: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    """``train`` / ``valid`` / ``test`` by start date (see SPLITS); missing outside every window."""
    out = pd.Series(pd.NA, index=start_dates.index, dtype="string")
    for name, (first, last) in SPLITS.items():
        out[start_dates.between(first, last if last is not None else cutoff)] = name
    return out

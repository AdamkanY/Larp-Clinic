"""Turn raw API pages into the processed tables in ``data/processed/``.

* ``trials.parquet`` (the app and ``src/train.py``): COMPLETED/TERMINATED trials that started
  2008-01-01 .. 2019-12-31, status label (TERMINATED = 1), ``split`` train = start 2008-2016,
  test = start 2017-2019. The horizon experiment does not change it.
* ``trials_3y.parquet`` (only ``src/train_horizon.py``; git-ignored): the 3-year horizon label of
  ``src/horizon.py`` for completed, terminated and still-running trials that started 2008-01-01 up
  to the follow-up cutoff and have a phase; ``split`` train = 2008-2018, valid = 2019, test = 2020
  onwards. Same feature columns, no outcome columns (stop date, stop reason).
* ``outcomes.parquet`` (EDA only: the Explore page): every trial that started 2008 or later, whatever
  its status, with both labels, the stop date, ``why_stopped`` and a rough reason group. Never a
  training file.

The two horizon files need a download that includes still-running trials and the completion date
(``python src/fetch_data.py --restart`` once); with an older download only ``trials.parquet`` is built.

Usage:
    python src/build_dataset.py                       # all three files
    python src/build_dataset.py --horizon-only        # leave trials.parquet untouched
    python src/build_dataset.py --snapshot 2026-11-05 # override the download date
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import horizon  # noqa: E402
from features import (BINARY_FEATURES, EXTRA_DISPLAY_COLS, FEATURE_COLUMNS,  # noqa: E402
                      META_COLUMNS, build_features)

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "data" / "raw" / "pages"
MANIFEST = ROOT / "data" / "raw" / "manifest.json"
OUT_PATH = ROOT / "data" / "processed" / "trials.parquet"
HORIZON_PATH = ROOT / "data" / "processed" / "trials_3y.parquet"
OUTCOMES_PATH = ROOT / "data" / "processed" / "outcomes.parquet"

WINDOW_START = pd.Timestamp("2008-01-01")
WINDOW_END = pd.Timestamp("2019-12-31")
TEST_FROM_YEAR = 2017  # train: 2008-2016, test: 2017-2019

COUNT_COLUMNS = ["n_arms", "n_primary_outcomes", "n_secondary_outcomes", "n_conditions",
                 "eligibility_text_len"]
OUTCOME_FIELDS = ["nct_id", "completion_date", "completion_date_type", "why_stopped"]
# Fields the horizon files need in the raw download (absent from downloads made before Nov 2026).
HORIZON_FIELDS = {"CompletionDate", "CompletionDateType", "WhyStopped"}
OUTCOMES_SIZE_WARNING_MB = 5  # outcomes.parquet is committed for the app; keep it small


def load_rows() -> tuple:
    """Read every raw page once: ``build_features`` rows and the outcome fields of the same studies.

    The outcome fields (stop date, stop reason) come from ``horizon.outcome_fields``, never from
    ``build_features``, so they cannot reach a feature column.
    """
    pages = sorted(PAGES_DIR.glob("page_*.json"))
    if not pages:
        raise SystemExit(f"No pages in {PAGES_DIR}; run src/fetch_data.py first.")
    rows, outcomes = [], []
    for page in pages:
        data = json.loads(page.read_text(encoding="utf-8"))
        for study in data.get("studies", []):
            rows.append(build_features(study))
            outcomes.append(horizon.outcome_fields(study))
    print(f"Read {len(pages)} pages, {len(rows)} study records.")
    return rows, outcomes


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable ints keep real missing values instead of turning flags and counts into floats."""
    for col in BINARY_FEATURES:
        df[col] = df[col].astype("Int8")
    for col in COUNT_COLUMNS:
        df[col] = df[col].astype("Int32")
    return df


def build_dataset(rows: list) -> pd.DataFrame:
    """Dedupe, apply label/date filters and add the train/test split column."""
    df = pd.DataFrame(rows, columns=META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS)
    n_dupes = df.duplicated("nct_id").sum()
    df = df.drop_duplicates("nct_id")
    print(f"Dropped {n_dupes} duplicate NCT ids.")
    print("\nRows per status (before filtering):")
    print(df["overall_status"].value_counts(dropna=False).to_string())

    df["start_date"] = pd.to_datetime(df["start_date"])
    no_label = df["label"].isna()
    invalid = df["start_date"].isna()
    too_old = df["start_date"] < WINDOW_START
    too_new = df["start_date"] > WINDOW_END
    print(f"\nDropped: no label {no_label.sum()}, invalid start date "
          f"{(invalid & ~no_label).sum()}, start < 2008 {too_old.sum()}, "
          f"start >= 2020 {too_new.sum()} (counts can overlap)")
    df = df[~no_label & ~invalid & ~too_old & ~too_new].copy()

    df["split"] = (df["start_date"].dt.year >= TEST_FROM_YEAR).map(
        {False: "train", True: "test"})
    df["label"] = df["label"].astype("int8")
    return set_dtypes(df).reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    """Print row counts, terminated rate per split and missingness per feature."""
    print("\nRows per status (in window):")
    print(df["overall_status"].value_counts().to_string())
    print("\nPer split:")
    summary = df.groupby("split")["label"].agg(rows="size", terminated="sum",
                                                terminated_rate="mean")
    print(summary.round(4).to_string())
    print("\n% missing per feature:")
    print((df[FEATURE_COLUMNS].isna().mean() * 100).round(2).to_string())


def read_manifest(path: Path = MANIFEST) -> dict:
    """The query and download date of the raw pages (written by fetch_data.py); {} if missing."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def horizon_problem(manifest: dict):
    """Why the raw download cannot build the horizon files, or None if it can."""
    params = manifest.get("params")
    if not params:
        return "data/raw/manifest.json is missing, so the query is unknown"
    statuses = set(params.get("filter.overallStatus", "").split(","))
    missing = sorted(set(horizon.STILL_RUNNING) - statuses)
    if missing:
        return f"the download has no {', '.join(missing)} trials"
    fields = params.get("fields")  # absent for --full downloads, which hold every field
    if fields is not None:
        absent = sorted(HORIZON_FIELDS - set(fields.split(",")))
        if absent:
            return f"the download lacks the fields {', '.join(absent)}"
    return None


def snapshot_date(manifest: dict) -> pd.Timestamp:
    """The day the raw pages were downloaded, from ``fetched_on`` (or the older ``started`` stamp)."""
    if manifest.get("fetched_on"):
        return pd.Timestamp(manifest["fetched_on"])
    if manifest.get("started"):
        return pd.Timestamp(datetime.strptime(manifest["started"], "%a %b %d %H:%M:%S %Y").date())
    raise SystemExit("data/raw/manifest.json has no download date; pass --snapshot YYYY-MM-DD.")


def merge_outcomes(rows: list, outcomes: list) -> pd.DataFrame:
    """All study records with their outcome fields, one row per NCT id, dates parsed."""
    df = pd.DataFrame(rows, columns=META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS)
    out = pd.DataFrame(outcomes, columns=OUTCOME_FIELDS)
    # Records without an id cannot be matched (and never occur in practice).
    df = df[df["nct_id"].notna()].drop_duplicates("nct_id")
    out = out[out["nct_id"].notna()].drop_duplicates("nct_id")
    merged = df.merge(out, on="nct_id", how="left", validate="one_to_one")
    merged["start_date"] = pd.to_datetime(merged["start_date"])
    merged["completion_date"] = pd.to_datetime(merged["completion_date"])
    return merged


def build_horizon_dataset(merged: pd.DataFrame, snapshot) -> pd.DataFrame:
    """The training table of the horizon experiment: horizon label, train/valid/test split, no outcome columns."""
    df = merged.copy()
    cutoff = horizon.followup_cutoff(snapshot)
    df["label"] = horizon.horizon_labels(df, snapshot)
    start = df["start_date"]
    drops = {
        "status never labelled (not yet recruiting, other)": ~df["overall_status"].isin(horizon.LABELLED_STATUSES),
        "terminated without a usable stop date": df["overall_status"].eq("TERMINATED") & horizon.stop_date(df).isna(),
        "invalid start date": start.isna(),
        "start before 2008": start < WINDOW_START,
        f"start after {cutoff:%Y-%m-%d} (less than {horizon.HORIZON_YEARS} years "
        f"+ {horizon.REPORTING_LAG_MONTHS} months of follow-up)": start > cutoff,
        "phase missing": df["phase"].isna(),
    }
    print(f"\nHorizon label (snapshot {pd.Timestamp(snapshot):%Y-%m-%d}): dropped "
          + ", ".join(f"{name} {int(mask.sum())}" for name, mask in drops.items()) + " (counts can overlap)")
    keep = df["label"].notna() & (start >= WINDOW_START) & df["phase"].notna()
    df = df[keep].copy()
    df["split"] = horizon.split_for(df["start_date"], cutoff).astype(object)
    if df["split"].isna().any():
        raise ValueError("Some horizon rows fall outside every split window")
    df["label"] = df["label"].astype("int8")
    df = set_dtypes(df)[META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS + ["split"]]
    df.attrs = {
        "label": f"TERMINATED within {horizon.HORIZON_YEARS} years of the start = 1; completed, stopped later "
                 "or still running = 0 (src/horizon.py)",
        "horizon_years": horizon.HORIZON_YEARS,
        "reporting_lag_months": horizon.REPORTING_LAG_MONTHS,
        "snapshot": f"{pd.Timestamp(snapshot):%Y-%m-%d}",
        "followup_cutoff": f"{cutoff:%Y-%m-%d}",
    }
    return df.reset_index(drop=True)


def report_horizon(df: pd.DataFrame) -> None:
    """Print rows, label rate per split and per start year, and where the label differs from the status."""
    print("\nHorizon dataset, per split:")
    summary = df.groupby("split", sort=False)["label"].agg(rows="size", terminated_3y="sum", rate="mean")
    print(summary.round(4).to_string())
    print("\nPer start year (rate = share terminated within the horizon):")
    years = df.groupby(df["start_date"].dt.year)["label"].agg(rows="size", rate="mean")
    print(years.round(4).T.to_string())
    print("\nStatus x horizon label (terminated with label 0 stopped after the horizon):")
    print(pd.crosstab(df["overall_status"], df["label"]).to_string())


def build_outcomes(merged: pd.DataFrame, snapshot) -> pd.DataFrame:
    """The EDA table: every trial that started 2008 or later, its status, both labels and how it stopped."""
    df = merged[merged["start_date"] >= WINDOW_START].copy()
    terminated = df["overall_status"].eq("TERMINATED")
    out = pd.DataFrame({
        "nct_id": df["nct_id"],
        "overall_status": df["overall_status"],
        "start_date": df["start_date"],
        "phase_label": df["phase_label"],
        "sponsor_class": df["sponsor_class"],
        "label_status": df["label"].astype("Int8"),
        "label_3y": horizon.horizon_labels(df, snapshot),
        "completion_date": df["completion_date"],
        "completion_date_type": df["completion_date_type"],
        "years_to_stop": horizon.years_to_stop(df),
        "why_stopped": df["why_stopped"],
        "stop_reason": df["why_stopped"].map(horizon.stop_reason).where(terminated),
        "mentions_covid": df["why_stopped"].map(horizon.mentions_covid) & terminated,
    })
    out.attrs = {"snapshot": f"{pd.Timestamp(snapshot):%Y-%m-%d}",
                 "followup_cutoff": f"{horizon.followup_cutoff(snapshot):%Y-%m-%d}",
                 "horizon_years": horizon.HORIZON_YEARS}
    return out.reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> None:
    """Write one parquet file and say how big it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    """Build, report and save the processed tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--horizon-only", action="store_true",
                        help="build only trials_3y.parquet and outcomes.parquet")
    parser.add_argument("--snapshot", help="download date YYYY-MM-DD (default: fetched_on in the manifest)")
    args = parser.parse_args()

    rows, outcomes = load_rows()
    if not args.horizon_only:
        df = build_dataset(rows)
        report(df)
        print()
        save(df, OUT_PATH)

    manifest = read_manifest()
    problem = horizon_problem(manifest)
    if problem:
        message = (f"\nSkipped {HORIZON_PATH.name} and {OUTCOMES_PATH.name}: {problem}. "
                   "Download again with `python src/fetch_data.py --restart`.")
        if args.horizon_only:
            raise SystemExit(message)
        print(message)
        return
    snapshot = pd.Timestamp(args.snapshot) if args.snapshot else snapshot_date(manifest)
    merged = merge_outcomes(rows, outcomes)

    horizon_df = build_horizon_dataset(merged, snapshot)
    report_horizon(horizon_df)
    save(horizon_df, HORIZON_PATH)

    outcomes_df = build_outcomes(merged, snapshot)
    save(outcomes_df, OUTCOMES_PATH)
    if OUTCOMES_PATH.stat().st_size > OUTCOMES_SIZE_WARNING_MB * 1e6:
        print(f"Warning: {OUTCOMES_PATH.name} is over {OUTCOMES_SIZE_WARNING_MB} MB; it is meant to be committed.")


if __name__ == "__main__":
    main()

"""Turn raw API pages into the model-ready table ``data/processed/trials.parquet``.

Runs ``build_features`` over every study in ``data/raw/pages/``, keeps
COMPLETED/TERMINATED trials that started 2008-01-01 .. 2019-12-31, and adds a
``split`` column (train = start 2008-2016, test = start 2017-2019).

Usage:
    python src/build_dataset.py              # full rebuild from data/raw/pages/
    python src/build_dataset.py --backfill   # add new condition features to the
                                             # existing parquet (no raw pages needed)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, CONDITION_FLAG_FEATURES,  # noqa: E402
                      EXTRA_DISPLAY_COLS, FEATURE_COLUMNS, META_COLUMNS,
                      build_features, condition_features)

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "data" / "raw" / "pages"
OUT_PATH = ROOT / "data" / "processed" / "trials.parquet"

WINDOW_START = pd.Timestamp("2008-01-01")
WINDOW_END = pd.Timestamp("2019-12-31")
TEST_FROM_YEAR = 2017  # train: 2008-2016, test: 2017-2019


def load_rows() -> list:
    """Run build_features on every study in every raw page."""
    pages = sorted(PAGES_DIR.glob("page_*.json"))
    if not pages:
        raise SystemExit(f"No pages in {PAGES_DIR}; run src/fetch_data.py first.")
    rows = []
    for page in pages:
        data = json.loads(page.read_text(encoding="utf-8"))
        rows.extend(build_features(study) for study in data.get("studies", []))
    print(f"Read {len(pages)} pages, {len(rows)} study records.")
    return rows


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


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable ints keep real missing values instead of turning flags into floats."""
    for col in BINARY_FEATURES:
        df[col] = df[col].astype("Int8")
    for col in ["n_arms", "n_primary_outcomes", "n_secondary_outcomes",
                "n_conditions", "eligibility_text_len"]:
        df[col] = df[col].astype("Int32")
    df["n_condition_categories"] = df["n_condition_categories"].astype("Int8")
    return df


def backfill_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the condition-category features from the stored ``conditions`` column.

    ``conditions`` holds the registered conditions joined with "; " (see
    ``build_features``), so splitting it back gives the same list
    ``condition_features`` sees on a raw record. This updates an existing
    parquet when the condition features change, without re-downloading the
    raw pages (which are not in git).
    """
    lists = df["conditions"].map(lambda text: text.split("; ") if isinstance(text, str) else None)
    flags = pd.DataFrame(list(lists.map(condition_features)), index=df.index)
    for col in flags.columns:
        df[col] = flags[col]
    # Keep the column order build_dataset writes: meta, display, features, split.
    order = META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS + ["split"]
    return set_dtypes(df[order + [c for c in df.columns if c not in order]].copy())


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


def report_conditions(df: pd.DataFrame) -> None:
    """Print trials and terminated rate per condition category (train and test)."""
    rows = []
    for col in CONDITION_FLAG_FEATURES + ["no category"]:
        mask = df["n_condition_categories"] == 0 if col == "no category" else df[col] == 1
        part = df[mask]
        rows.append({"category": col.removeprefix("cond_"), "trials": len(part),
                     "share": len(part) / len(df), "terminated_rate": part["label"].mean()})
    print("\nCondition categories (multi-hot; a trial can be in several):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    """Build (or backfill), report and save the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backfill", action="store_true",
                        help="recompute condition features in the existing parquet")
    if parser.parse_args().backfill:
        df = backfill_conditions(pd.read_parquet(OUT_PATH))
        report_conditions(df)
        df.to_parquet(OUT_PATH, index=False)
        print(f"\nBackfilled condition features into {OUT_PATH.relative_to(ROOT)} ({len(df)} rows)")
        return
    df = build_dataset(load_rows())
    report(df)
    report_conditions(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

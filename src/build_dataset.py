"""Turn raw API pages into the model-ready table ``data/processed/trials.parquet``.

Runs ``build_features`` over every study in ``data/raw/pages/``, keeps
COMPLETED/TERMINATED trials that started 2008-01-01 .. 2019-12-31, and adds a
``split`` column (train = start 2008-2016, test = start 2017-2019).

Usage:
    python src/build_dataset.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, FEATURE_COLUMNS, META_COLUMNS,  # noqa: E402
                      build_features)

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
    df = pd.DataFrame(rows, columns=META_COLUMNS + FEATURE_COLUMNS)
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
    # Nullable ints keep real missing values instead of turning flags into floats.
    for col in BINARY_FEATURES:
        df[col] = df[col].astype("Int8")
    for col in ["n_arms", "n_primary_outcomes", "n_secondary_outcomes",
                "n_conditions", "eligibility_text_len"]:
        df[col] = df[col].astype("Int32")
    return df.reset_index(drop=True)


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


def main() -> None:
    """Build, report and save the dataset."""
    df = build_dataset(load_rows())
    report(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

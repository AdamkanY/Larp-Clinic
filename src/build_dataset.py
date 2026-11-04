"""Turn raw API pages into the model-ready table ``data/processed/trials.parquet``.

Runs ``build_features`` over every study in ``data/raw/pages/``, keeps
COMPLETED/TERMINATED trials that started 2008-01-01 .. 2019-12-31, and adds a
``split`` column (train = start 2008-2016, test = start 2017-2019).

``--backfill`` instead updates an existing parquet without the raw pages: it
recomputes the features that derive from stored columns (the condition-category
flags from the ``conditions`` text, ``n_intervention_types`` from the
intervention flags) and keeps every row and every other value as it is.

Usage:
    python src/build_dataset.py              # full build from data/raw/pages/
    python src/build_dataset.py --backfill   # add derived columns, no refetch
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, CONDITION_FLAGS, EXTRA_DISPLAY_COLS,  # noqa: E402
                      FEATURE_COLUMNS, INTERVENTION_FLAGS, META_COLUMNS,
                      build_features, condition_features)

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "data" / "raw" / "pages"
OUT_PATH = ROOT / "data" / "processed" / "trials.parquet"

WINDOW_START = pd.Timestamp("2008-01-01")
WINDOW_END = pd.Timestamp("2019-12-31")
TEST_FROM_YEAR = 2017  # train: 2008-2016, test: 2017-2019

COUNT_COLUMNS = ["n_arms", "n_primary_outcomes", "n_secondary_outcomes", "n_conditions",
                 "eligibility_text_len", "n_condition_categories", "n_intervention_types"]
# Column order of the parquet; a backfilled file gets the same layout as a fresh build.
COLUMN_ORDER = META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS + ["split"]


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


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable ints keep real missing values instead of turning flags into floats."""
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
    return cast_dtypes(df).reset_index(drop=True)


def backfill(df: pd.DataFrame) -> pd.DataFrame:
    """Add or recompute the features that can be derived from stored columns.

    ``conditions`` holds the raw condition list joined with "; ". Splitting it
    on "; " and passing the pieces to ``condition_features`` reproduces
    exactly what ``build_features`` computes from the raw record, because
    ``condition_features`` joins its input with "; " again before matching.
    ``n_intervention_types`` is the sum of the intervention flags (missing
    flags count as absent, as in ``build_features``). Rows, row order and all
    other columns stay untouched, so the split and the CV folds do not move.
    """
    conditions = df["conditions"].map(lambda text: text.split("; ") if isinstance(text, str) else [])
    derived = pd.DataFrame(list(conditions.map(condition_features)), index=df.index)
    for col in CONDITION_FLAGS + ["n_condition_categories"]:
        df[col] = derived[col]
    df["n_intervention_types"] = df[INTERVENTION_FLAGS].sum(axis=1, skipna=True)
    return cast_dtypes(df)[COLUMN_ORDER]


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
    print("\nCondition categories (share of trials, terminated rate):")
    rates = pd.DataFrame({flag: {"share": df[flag].mean(),
                                 "terminated_rate": df.loc[df[flag] == 1, "label"].mean()}
                          for flag in CONDITION_FLAGS}).T
    print(rates.round(4).to_string())


def main() -> None:
    """Build (or backfill), report and save the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backfill", action="store_true",
                        help="update the existing parquet from its own columns (no raw pages)")
    args = parser.parse_args()
    if args.backfill:
        if not OUT_PATH.exists():
            raise SystemExit(f"{OUT_PATH.relative_to(ROOT)} not found; run a full build first.")
        df = backfill(pd.read_parquet(OUT_PATH))
    else:
        df = build_dataset(load_rows())
    report(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

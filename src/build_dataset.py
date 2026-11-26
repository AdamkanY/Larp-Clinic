"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Turn the downloaded API pages into the table the model learns from: data/processed/trials.parquet.

Every study in data/raw/pages/ goes through build_features(). We keep the COMPLETED and TERMINATED trials that
started between 2008-01-01 and 2019-12-31 and add the column "split": train = start 2008-2016, test = start
2017-2019.

How to run it:
    python src/build_dataset.py              # build the whole table from data/raw/pages/
    python src/build_dataset.py --backfill   # only recompute the condition features in the existing table
                                             # (works without the raw pages)
"""

# ---- 1. Imports ----
import argparse  # read the option --backfill
import json  # read the saved pages
import sys  # change the module search path
from pathlib import Path  # file paths

import pandas as pd  # tables (DataFrame) and the parquet file

# The other files of src/ are imported by their plain name, so this works from the terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import BINARY_FEATURES, CONDITION_FLAG_FEATURES, EXTRA_DISPLAY_COLS, FEATURE_COLUMNS, META_COLUMNS
from features import build_features, condition_features

# ---- 2. Settings ----
ROOT = Path(__file__).resolve().parents[1]  # the project folder
PAGES_DIR = ROOT / "data" / "raw" / "pages"
OUT_PATH = ROOT / "data" / "processed" / "trials.parquet"

# The years we study and the year where the test data starts (rule 3 in the main CLAUDE.md).
# train.py checks the finished table against the same dates.
WINDOW_START = pd.Timestamp("2008-01-01")
WINDOW_END = pd.Timestamp("2019-12-31")
TEST_FROM_YEAR = 2017  # train: 2008-2016, test: 2017-2019


def load_rows():
    """Read every downloaded page and run build_features() on every study.

    Input: none (it reads the pages in data/raw/pages/).

    Output: a list of feature rows, one dictionary per study.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Find the saved pages; stop with a message if there are none.
    pages = sorted(PAGES_DIR.glob("page_*.json"))
    if not pages:
        raise SystemExit(f"No pages in {PAGES_DIR}; run src/fetch_data.py first.")

    # One page holds up to 1,000 studies. Every study goes through the one feature function.
    rows = []
    for page in pages:
        data = json.loads(page.read_text(encoding="utf-8"))
        for study in data.get("studies", []):
            rows.append(build_features(study))
    print(f"Read {len(pages)} pages, {len(rows)} study records.")
    return rows


def build_dataset(rows):
    """Make the final table from the feature rows: no duplicates, only labelled trials in our years, with split.

    Input: rows is a list of feature dictionaries, as load_rows() returns it.

    Output: a pandas DataFrame with one row per usable trial and the extra column "split" ("train" or "test").
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # One row per study. A study can be on two pages if the registry changed while we were downloading.
    df = pd.DataFrame(rows, columns=META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS)
    n_dupes = df.duplicated("nct_id").sum()
    df = df.drop_duplicates("nct_id")
    print(f"Dropped {n_dupes} duplicate NCT ids.")
    print("\nRows per status (before filtering):")
    print(df["overall_status"].value_counts(dropna=False).to_string())

    # Drop the rows we cannot use: no label (the status is neither COMPLETED nor TERMINATED), a start date we
    # could not read, or a start outside 2008-2019.
    df["start_date"] = pd.to_datetime(df["start_date"])
    no_label = df["label"].isna()
    invalid = df["start_date"].isna()
    too_old = df["start_date"] < WINDOW_START
    too_new = df["start_date"] > WINDOW_END
    print(f"\nDropped: no label {no_label.sum()}, invalid start date "
          f"{(invalid & ~no_label).sum()}, start < 2008 {too_old.sum()}, "
          f"start >= 2020 {too_new.sum()} (counts can overlap)")
    df = df[~no_label & ~invalid & ~too_old & ~too_new].copy()

    # Split by start year and never at random: every test trial started after all the training trials, the
    # same way the app later scores trials that are newer than everything the model has seen.
    is_test = df["start_date"].dt.year >= TEST_FROM_YEAR
    df["split"] = is_test.map({False: "train", True: "test"})
    df["label"] = df["label"].astype("int8")
    df = set_dtypes(df)
    return df.reset_index(drop=True)


def set_dtypes(df):
    """Store flags and counts as whole numbers that can still be missing.

    With normal integers pandas would turn a column with a missing value into decimal numbers. The types "Int8"
    and "Int32" (with a capital I) keep whole numbers and mark a missing value as <NA>.

    Input: df is the trials table (DataFrame).

    Output: the same DataFrame with the new column types.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Flags are 0/1 values.
    for col in BINARY_FEATURES:
        df[col] = df[col].astype("Int8")

    # Counts and lengths are larger whole numbers.
    for col in ["n_arms", "n_primary_outcomes", "n_secondary_outcomes", "n_conditions", "eligibility_text_len"]:
        df[col] = df[col].astype("Int32")
    df["n_condition_categories"] = df["n_condition_categories"].astype("Int8")
    return df


def backfill_conditions(df):
    """Recompute the condition features of an existing table from its "conditions" column.

    We need this when the keywords in features.py change: the raw pages are not in git, so we cannot always
    rebuild everything. The "conditions" column holds the registered conditions joined with "; " (see
    build_features). Splitting it gives back the list that condition_features() gets from a raw record.

    Input: df is the existing trials table with a "conditions" column (DataFrame).

    Output: a DataFrame with the new condition columns and the usual column order.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # One row of flags per trial. A trial without conditions gets None, which gives all zeros.
    flag_rows = []
    for text in df["conditions"]:
        if isinstance(text, str):
            conditions = text.split("; ")
        else:
            conditions = None
        flag_rows.append(condition_features(conditions))
    flags = pd.DataFrame(flag_rows, index=df.index)

    # Overwrite the old columns with the new values.
    for col in flags.columns:
        df[col] = flags[col]

    # Keep the column order that build_dataset() writes: meta, display, features, split. Other columns go last.
    order = META_COLUMNS + EXTRA_DISPLAY_COLS + FEATURE_COLUMNS + ["split"]
    for col in df.columns:
        if col not in order:
            order.append(col)
    return set_dtypes(df[order].copy())


def report(df):
    """Print how many trials we have, the share of terminated trials per split and the missing values.

    Input: df is the finished trials table (DataFrame).

    Output: nothing. It prints three small tables.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Number of trials per status.
    print("\nRows per status (in window):")
    print(df["overall_status"].value_counts().to_string())

    # Size and share of terminated trials for train and test.
    print("\nPer split:")
    summary = df.groupby("split")["label"].agg(rows="size", terminated="sum", terminated_rate="mean")
    print(summary.round(4).to_string())

    # How many percent of the values are missing in each feature.
    print("\n% missing per feature:")
    print((df[FEATURE_COLUMNS].isna().mean() * 100).round(2).to_string())


def report_conditions(df):
    """Print the number of trials and the share of terminated trials for every condition category.

    Input: df is the trials table with the condition columns (DataFrame).

    Output: nothing. It prints one table.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # One line per category, and a last line for the trials that are in no category.
    rows = []
    for col in CONDITION_FLAG_FEATURES + ["no category"]:
        if col == "no category":
            part = df[df["n_condition_categories"] == 0]
        else:
            part = df[df[col] == 1]
        rows.append({
            "category": col.removeprefix("cond_"),
            "trials": len(part),
            "share": len(part) / len(df),
            "terminated_rate": part["label"].mean(),
        })
    print("\nCondition categories (multi-hot; a trial can be in several):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    """Build the table (or only backfill the condition features), print the reports and save the file.

    Input: none (the option --backfill comes from the command line).

    Output: nothing. It writes data/processed/trials.parquet and prints the progress.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Read the command-line option.
    parser = argparse.ArgumentParser(
        description="Turn raw API pages into the model-ready table ``data/processed/trials.parquet``.")
    parser.add_argument("--backfill", action="store_true",
                        help="recompute condition features in the existing parquet")
    args = parser.parse_args()

    # --backfill: update the condition features in the existing table and stop.
    if args.backfill:
        df = backfill_conditions(pd.read_parquet(OUT_PATH))
        report_conditions(df)
        df.to_parquet(OUT_PATH, index=False)
        print(f"\nBackfilled condition features into {OUT_PATH.relative_to(ROOT)} ({len(df)} rows)")
        return

    # Normal run: raw pages -> feature rows -> filtered table -> parquet file.
    rows = load_rows()
    df = build_dataset(rows)
    report(df)
    report_conditions(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH.relative_to(ROOT)}")


# ---- 3. Run the script ----
if __name__ == "__main__":
    main()

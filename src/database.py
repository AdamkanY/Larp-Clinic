# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""SQLite database behind the Explore page: two tables and the SQL queries that read them.

Why a database: the Explore page asks the same few questions over and over (how many trials match the filters,
what share was stopped early, per phase / sponsor / blinding / allocation, which conditions are most common).
Those are exactly the questions SQL is made for (CS course, chapter 3: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT,
JOIN and ``pd.read_sql``).

Design (CS course 3.2 and 3.3, keys and normal forms):

* ``trials``: one row per trial, primary key ``nct_id``, with the columns the page shows, already in plain words
  (built from ``viz.load_trials``, so the database and the page use the same names).
* ``trial_conditions``: one row per trial and condition. The ``conditions`` text of a record lists several
  conditions in one cell ("Asthma; COPD"), which breaks the first normal form (one value per cell). Splitting it
  into its own table, with ``nct_id`` as a foreign key to ``trials``, makes "trials per condition" a plain
  ``JOIN`` + ``GROUP BY``.

The database file ``data/processed/trials.db`` is not in git: ``ensure_database`` builds it from
``trials.parquet`` the first time it is needed (a few seconds) and again whenever the parquet is newer.
User input never becomes part of the SQL text: every filter value is passed as a ``?`` parameter, so a keyword
like ``'; DROP TABLE trials; --`` is just text to search for (no SQL injection).

Usage:
    python src/database.py      # (re)build data/processed/trials.db and print a few example queries
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz  # noqa: E402

DB_PATH = viz.DATA_PATH.with_name("trials.db")

# Columns of the trials table: what the Explore page filters, groups, charts and lists. Model inputs that the page
# never shows stay in the parquet.
TRIAL_COLUMNS = ["nct_id", "brief_title", "lead_sponsor_name", "overall_status", "label", "start_year",
                 "phase_label", "sponsor_class", "masking", "allocation", "eligibility_text_len"]

SCHEMA = """
CREATE TABLE trials (
    nct_id               TEXT PRIMARY KEY,
    brief_title          TEXT,
    lead_sponsor_name    TEXT,
    overall_status       TEXT NOT NULL,
    label                INTEGER NOT NULL CHECK (label IN (0, 1)),  -- 1 = stopped early (TERMINATED)
    start_year           INTEGER NOT NULL,
    phase_label          TEXT NOT NULL,
    sponsor_class        TEXT NOT NULL,
    masking              TEXT NOT NULL,
    allocation           TEXT NOT NULL,
    eligibility_text_len INTEGER
);
CREATE TABLE trial_conditions (
    nct_id    TEXT NOT NULL REFERENCES trials (nct_id),
    condition TEXT NOT NULL,
    PRIMARY KEY (nct_id, condition)
);
CREATE INDEX idx_trials_filters ON trials (phase_label, sponsor_class, start_year);
CREATE INDEX idx_conditions_condition ON trial_conditions (condition);
"""

# The four traits the page breaks the share down by. Only these names may go into a GROUP BY: column names cannot
# be "?" parameters in SQL, so they come from this fixed list, never from user input.
GROUP_COLUMNS = ["phase_label", "sponsor_class", "masking", "allocation"]


def build_database(parquet_path=viz.DATA_PATH, db_path=DB_PATH) -> Path:
    """Create the SQLite file from the trials parquet: the trials table and the split-out conditions table.

    Written to a temporary file first and renamed at the end, so a page never opens a half-built database.
    """
    trials = viz.load_trials(parquet_path)
    # One row per (trial, condition): split "A; B" into rows, trim, drop empties and repeats within a trial.
    conditions = (trials[["nct_id", "conditions"]]
                  .assign(condition=trials["conditions"].str.split(";"))
                  .explode("condition"))
    conditions["condition"] = conditions["condition"].str.strip()
    conditions = conditions[conditions["condition"] != ""].drop_duplicates(["nct_id", "condition"])

    db_path = Path(db_path)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as conn:
        # Tables with keys first, then the rows (to_sql appends into the tables the schema defined).
        conn.executescript(SCHEMA)
        trials[TRIAL_COLUMNS].to_sql("trials", conn, if_exists="append", index=False)
        conditions[["nct_id", "condition"]].to_sql("trial_conditions", conn, if_exists="append", index=False)
    tmp.replace(db_path)
    return db_path


def ensure_database(parquet_path=viz.DATA_PATH, db_path=DB_PATH) -> Path:
    """The database file, built first if it is missing or older than the parquet it comes from."""
    db_path, parquet_path = Path(db_path), Path(parquet_path)
    if not db_path.exists() or db_path.stat().st_mtime < parquet_path.stat().st_mtime:
        build_database(parquet_path, db_path)
    return db_path


def where_clause(phases=None, sponsor_classes=None, year_range=None, condition_kw="", table="") -> tuple:
    """The WHERE part for the page's filters, with its parameters; an empty filter adds no condition.

    Args:
        phases, sponsor_classes, year_range, condition_kw: the page's filters (plain names, as on screen).
        table: alias of the trials table in a JOIN (e.g. "t"), so the column names are unambiguous.

    Returns:
        (sql, params): e.g. ("WHERE phase_label IN (?, ?) AND start_year BETWEEN ? AND ?", ["Phase 1", ...]).
    """
    col = f"{table}." if table else ""
    parts, params = [], []
    if phases:
        parts.append(f"{col}phase_label IN ({', '.join('?' for _ in phases)})")
        params += list(phases)
    if sponsor_classes:
        parts.append(f"{col}sponsor_class IN ({', '.join('?' for _ in sponsor_classes)})")
        params += list(sponsor_classes)
    if year_range:
        parts.append(f"{col}start_year BETWEEN ? AND ?")
        params += [int(year_range[0]), int(year_range[1])]
    kw = (condition_kw or "").strip()
    if kw:
        # Case-insensitive substring match on any of the trial's conditions. instr() instead of LIKE, so "%" and
        # "_" in a keyword are plain characters, like the old pandas filter (regex=False).
        parts.append(f"{col}nct_id IN (SELECT nct_id FROM trial_conditions "
                     "WHERE instr(lower(condition), lower(?)) > 0)")
        params.append(kw)
    return ("WHERE " + " AND ".join(parts)) if parts else "", params


class TrialDatabase:
    """Read-only access to trials.db for one page run; records every query it runs, so the page can show them.

    Args:
        db_path: the SQLite file (see ``ensure_database``).
        filters: the page's filters as keyword arguments of ``where_clause`` (phases, sponsor_classes,
            year_range, condition_kw); every query below applies them.
    """

    def __init__(self, db_path=DB_PATH, **filters):
        self.db_path = Path(db_path)
        self.filters = filters
        self.log = []  # (purpose, SQL text, parameters) of every query, in order

    def query(self, purpose: str, sql: str, params=()) -> pd.DataFrame:
        """Run one SELECT with ``?`` parameters and return the result as a DataFrame (``pd.read_sql``)."""
        self.log.append((purpose, " ".join(sql.split()), list(params)))
        # Opened read-only ("mode=ro"): the page can look, never change anything.
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            return pd.read_sql(sql, conn, params=list(params))

    def options(self) -> dict:
        """What the filters can offer: phases present, sponsor types by size, first and last start year."""
        phases = self.query("Filter choices: phases", "SELECT DISTINCT phase_label FROM trials")["phase_label"]
        sponsors = self.query("Filter choices: sponsors",
                              "SELECT sponsor_class, COUNT(*) AS n FROM trials GROUP BY sponsor_class "
                              "ORDER BY n DESC")["sponsor_class"]
        years = self.query("Filter choices: years",
                           "SELECT MIN(start_year) AS first, MAX(start_year) AS last FROM trials").iloc[0]
        return {"phases": set(phases), "sponsors": list(sponsors),
                "years": (int(years["first"]), int(years["last"]))}

    def summary(self, filtered: bool = True) -> dict:
        """How many trials match (all trials with ``filtered=False``), and the share stopped early among them."""
        sql, params = where_clause(**self.filters) if filtered else ("", [])
        row = self.query("Key numbers",
                         f"SELECT COUNT(*) AS n, AVG(label) AS rate FROM trials {sql}", params).iloc[0]
        return {"n": int(row["n"]), "rate": None if pd.isna(row["rate"]) else float(row["rate"])}

    def trials(self) -> pd.DataFrame:
        """The matching trials, one row each, for the charts the page still draws in pandas and the trial table."""
        sql, params = where_clause(**self.filters)
        frame = self.query("Matching trials", f"SELECT * FROM trials {sql} ORDER BY start_year, nct_id", params)
        # The same eligibility groups as viz.load_trials (a pandas category), so viz.eligibility_stats works as before.
        frame["eligibility_group"] = pd.cut(frame["eligibility_text_len"].astype(float), viz.ELIGIBILITY_EDGES,
                                            labels=viz.ELIGIBILITY_LABELS)
        return frame

    def rate_by(self, column: str, order=None) -> pd.DataFrame:
        """Trials, stopped trials and share stopped early per value of one trait (GROUP BY), like ``viz.rate_by``."""
        if column not in GROUP_COLUMNS:
            raise ValueError(f"cannot group by {column!r}")
        sql, params = where_clause(**self.filters)
        out = self.query(f"Share stopped early by {column}",
                         f"""SELECT {column}, COUNT(*) AS n, SUM(label) AS terminated, AVG(label) AS rate
                             FROM trials {sql} GROUP BY {column} ORDER BY n DESC""", params)
        # Clinical or natural order where the page wants one (SQL sorts by size).
        if order:
            rank = {v: i for i, v in enumerate(order)}
            out = out.sort_values(column, key=lambda s: s.map(rank).fillna(len(order)))
        return out.reset_index(drop=True)

    def top_conditions(self, k: int = 15) -> pd.DataFrame:
        """The k most common conditions among the matching trials (JOIN + GROUP BY + ORDER BY + LIMIT)."""
        # In the JOIN both tables have nct_id, so the filters name the trials table by its alias t.
        sql, params = where_clause(**self.filters, table="t")
        return self.query(
            "The most common conditions",
            f"""SELECT c.condition AS condition, COUNT(*) AS n, SUM(t.label) AS terminated, AVG(t.label) AS rate
                FROM trial_conditions AS c JOIN trials AS t ON t.nct_id = c.nct_id
                {sql}
                GROUP BY c.condition ORDER BY n DESC, c.condition LIMIT ?""", list(params) + [int(k)])


if __name__ == "__main__":
    path = build_database()
    db = TrialDatabase(path)
    print(f"Built {path.relative_to(viz.DATA_PATH.parents[2])}: {db.summary()['n']:,} trials")
    print(db.rate_by("phase_label", order=viz.PHASE_ORDER).to_string(index=False))
    print(db.top_conditions(k=5).to_string(index=False))

"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

The SQLite database behind the page "Explore the data": two tables and the SQL queries that read them.

Why a database? The page asks the same few questions again and again: how many trials match the filters, which
share of them was stopped early, the same per phase / sponsor / blinding / allocation, and which conditions are
the most common. SQL is made for exactly these questions (CS course, chapter 3: SELECT, WHERE, GROUP BY,
ORDER BY, LIMIT, JOIN and pd.read_sql).

The two tables (CS course 3.2 and 3.3, keys and normal forms):

* trials: one row per trial with the primary key nct_id. It has the columns the page shows, already in plain
  words. It is built from viz.load_trials(), so the database and the page use the same names.
* trial_conditions: one row per trial and condition. In the trials file one cell can hold several conditions
  ("Asthma; COPD"). That breaks the first normal form (only one value per cell). So the conditions get their own
  table, with nct_id as a foreign key to trials. "Trials per condition" is then a simple JOIN + GROUP BY.

The file data/processed/trials.db is not in git. ensure_database() builds it from trials.parquet the first time
it is needed (this takes a few seconds), and again whenever the parquet file is newer.

What the user types never becomes part of the SQL text. Every filter value is passed as a "?" parameter, so a
keyword like "'; DROP TABLE trials; --" is only a text to search for (no SQL injection).

How to run it:
    python src/database.py      # build data/processed/trials.db again and print a few example queries
"""

# ---- 1. Imports ----
import sqlite3  # the SQLite database (part of Python)
import sys  # change the module search path
from pathlib import Path  # file paths

import pandas as pd  # tables (DataFrame) and pd.read_sql

# The other files of src/ are imported by their plain name, so this works from the app and from the terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz  # loads the trials table and has the names and groups the page uses

# ---- 2. Settings and table layout ----
DB_PATH = viz.DATA_PATH.with_name("trials.db")

# The columns of the table trials: everything the Explore page filters, groups, draws or lists. Model inputs
# that the page never shows stay in the parquet file.
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

# The four columns the page can group by. A column name cannot be a "?" parameter in SQL, it has to be written
# into the query text. So we only allow the names in this list and never a name that comes from the user.
GROUP_COLUMNS = ["phase_label", "sponsor_class", "masking", "allocation"]


def build_database(parquet_path=viz.DATA_PATH, db_path=DB_PATH):
    """Create the SQLite file from the trials file: the table trials and the table trial_conditions.

    We write to a temporary file first and rename it at the end, so the page never opens a half-built database.

    Input: parquet_path is the trials file (Path or string); db_path is where the SQLite file goes (Path or string).

    Output: the Path of the finished database file.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Load the trials with the plain names the page uses.
    trials = viz.load_trials(parquet_path)

    # One row per trial and condition: split "A; B" into a list, explode() then makes one row per list entry.
    # After that we remove spaces around the names, empty names and conditions that a trial lists twice.
    conditions = trials[["nct_id", "conditions"]].copy()
    conditions["condition"] = trials["conditions"].str.split(";")
    conditions = conditions.explode("condition")
    conditions["condition"] = conditions["condition"].str.strip()
    conditions = conditions[conditions["condition"] != ""]
    conditions = conditions.drop_duplicates(["nct_id", "condition"])

    # Build in a temporary file, so an old good database stays until the new one is ready.
    db_path = Path(db_path)
    temporary_path = db_path.with_suffix(".tmp")
    temporary_path.unlink(missing_ok=True)

    # First create the empty tables with their keys, then fill in the rows. if_exists="append" adds the rows
    # to the tables from SCHEMA. Without it pandas would make its own tables without keys.
    conn = sqlite3.connect(temporary_path)
    conn.executescript(SCHEMA)
    trials[TRIAL_COLUMNS].to_sql("trials", conn, if_exists="append", index=False)
    conditions[["nct_id", "condition"]].to_sql("trial_conditions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    # Rename the finished file to the real name.
    temporary_path.replace(db_path)
    return db_path


def ensure_database(parquet_path=viz.DATA_PATH, db_path=DB_PATH):
    """Return the path of the database file. Build it first if it is missing or older than the trials file.

    Input: parquet_path is the trials file (Path or string); db_path is the SQLite file (Path or string).

    Output: the Path of an up-to-date database file.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Rebuild only when there is no file or the trials file is newer.
    db_path = Path(db_path)
    parquet_path = Path(parquet_path)
    if not db_path.exists() or db_path.stat().st_mtime < parquet_path.stat().st_mtime:
        build_database(parquet_path, db_path)
    return db_path


def where_clause(phases=None, sponsor_classes=None, year_range=None, condition_kw="", table=""):
    """Build the WHERE part of a query from the filters of the page. Returns (SQL text, list of parameters).

    Example: ("WHERE phase_label IN (?, ?) AND start_year BETWEEN ? AND ?", ["Phase 1", "Phase 2", 2010, 2015]).
    A filter that is empty adds no condition. Without any filter the text is "". table is the short name of the
    trials table in a JOIN (for example "t"), so SQL knows which table a column belongs to.

    Input: phases is a list of phase names or None; sponsor_classes is a list of sponsor type names or None;
    year_range is a pair (first year, last year) or None; condition_kw is a keyword to find in the conditions
    (string); table is the short name of the trials table in a JOIN (string, "" for none).

    Output: a tuple (SQL text, list of parameters). The text is "" if no filter is set.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # The column names get the table name in front when a JOIN needs it.
    prefix = ""
    if table:
        prefix = f"{table}."

    # One condition and its parameters per filter that is set.
    parts = []
    params = []
    if phases:
        # One "?" per chosen phase, for example "phase_label IN (?, ?, ?)".
        question_marks = ", ".join(["?"] * len(phases))
        parts.append(f"{prefix}phase_label IN ({question_marks})")
        params += list(phases)
    if sponsor_classes:
        question_marks = ", ".join(["?"] * len(sponsor_classes))
        parts.append(f"{prefix}sponsor_class IN ({question_marks})")
        params += list(sponsor_classes)
    if year_range:
        parts.append(f"{prefix}start_year BETWEEN ? AND ?")
        params += [int(year_range[0]), int(year_range[1])]
    keyword = (condition_kw or "").strip()
    if keyword:
        # The keyword must appear in one of the trial's conditions, upper and lower case do not matter.
        # We use instr() and not LIKE, because in LIKE the characters "%" and "_" have a special meaning.
        parts.append(f"{prefix}nct_id IN (SELECT nct_id FROM trial_conditions "
                     "WHERE instr(lower(condition), lower(?)) > 0)")
        params.append(keyword)

    # Join the conditions with AND.
    if not parts:
        return "", params
    return "WHERE " + " AND ".join(parts), params


class TrialDatabase:
    """Reads trials.db for one run of the page and remembers every query, so the page can show them.

    db_path is the SQLite file (see ensure_database). phases, sponsor_classes, year_range and condition_kw are
    the filters of the page. Every method below uses them, except options().
    """

    def __init__(self, db_path=DB_PATH, phases=None, sponsor_classes=None, year_range=None, condition_kw=""):
        """Remember the file and the filters. Nothing is read yet.

        Input: db_path is the SQLite file (Path or string); phases, sponsor_classes, year_range and condition_kw
        are the filters of the page, as in where_clause().

        Output: nothing (the object is created).
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # Store the file and the filters.
        self.db_path = Path(db_path)
        self.phases = phases
        self.sponsor_classes = sponsor_classes
        self.year_range = year_range
        self.condition_kw = condition_kw
        self.log = []  # every query that was run, in order: (purpose, SQL text, parameters)

    def where(self, table=""):
        """The WHERE part and its parameters for the filters of this object (see where_clause).

        Input: table is the short name of the trials table in a JOIN (string, "" for none).

        Output: a tuple (SQL text, list of parameters).
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # Hand the stored filters to the function.
        return where_clause(self.phases, self.sponsor_classes, self.year_range, self.condition_kw, table)

    def query(self, purpose, sql, params=()):
        """Run one SELECT with "?" parameters and return the result as a DataFrame (pd.read_sql).

        Input: purpose is a short description of the query for the log (string); sql is the SELECT text with "?"
        in place of values (string); params are the values for the "?" (list or tuple).

        Output: the result as a pandas DataFrame.
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # Write the query into the log. split() and join() remove the line breaks and extra spaces.
        self.log.append((purpose, " ".join(sql.split()), list(params)))
        # "mode=ro" opens the file read-only: the page can look at the data but can never change it.
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        result = pd.read_sql(sql, conn, params=list(params))
        conn.close()
        return result

    def options(self):
        """What the filters of the page can offer: the phases, the sponsor types (biggest first), the years.

        Input: none.

        Output: a dictionary with "phases" (set of names), "sponsors" (list, biggest group first) and "years"
        (tuple of the first and the last start year).
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # Three small queries on the whole table (the filters do not apply here).
        phases = self.query("Filter choices: phases", "SELECT DISTINCT phase_label FROM trials")
        sponsors = self.query(
            "Filter choices: sponsors",
            "SELECT sponsor_class, COUNT(*) AS n FROM trials GROUP BY sponsor_class ORDER BY n DESC")
        years = self.query(
            "Filter choices: years",
            "SELECT MIN(start_year) AS first, MAX(start_year) AS last FROM trials")
        # Turn the results into plain Python values.
        first_year = int(years["first"].iloc[0])
        last_year = int(years["last"].iloc[0])
        return {
            "phases": set(phases["phase_label"]),
            "sponsors": list(sponsors["sponsor_class"]),
            "years": (first_year, last_year),
        }

    def summary(self, filtered=True):
        """How many trials match the filters and which share of them was stopped early.

        With filtered=False the numbers are for all trials. If no trial matches, the share is None.

        Input: filtered is True to use the filters of the page, False to count all trials (boolean).

        Output: a dictionary with "n" (number of trials, integer) and "rate" (share stopped early between 0 and 1,
        or None if no trial matches).
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # With or without the filters.
        if filtered:
            where, params = self.where()
        else:
            where, params = "", []
        # COUNT gives the number of trials, AVG of the 0/1 label gives the share stopped early.
        result = self.query("Key numbers", f"SELECT COUNT(*) AS n, AVG(label) AS rate FROM trials {where}", params)
        n = result["n"].iloc[0]
        rate = result["rate"].iloc[0]
        # AVG of zero rows is NULL in SQL, which pandas shows as a missing value.
        if pd.isna(rate):
            return {"n": int(n), "rate": None}
        return {"n": int(n), "rate": float(rate)}

    def trials(self):
        """The trials that match the filters, one row each (for the trial table and the charts drawn in pandas).

        Input: none.

        Output: a DataFrame with the columns of the table trials plus "eligibility_group".
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # SELECT all matching rows in a fixed order.
        where, params = self.where()
        frame = self.query("Matching trials", f"SELECT * FROM trials {where} ORDER BY start_year, nct_id", params)
        # The same length groups as in viz.load_trials(), so viz.eligibility_stats() works on this table too.
        frame["eligibility_group"] = pd.cut(frame["eligibility_text_len"].astype(float), viz.ELIGIBILITY_EDGES,
                                            labels=viz.ELIGIBILITY_LABELS)
        return frame

    def rate_by(self, column, order=None):
        """Trials, stopped trials and share stopped early for every value of one column (GROUP BY).

        The result looks like the one of viz.rate_by(). SQL sorts the groups by size. With order (a list of
        values) the rows come in that order and values that are not in the list go last.

        Input: column is the column to group by, one of GROUP_COLUMNS (string); order is a list of values in the
        order the rows should have, or None to sort by group size.

        Output: a DataFrame with the columns of column, "n", "terminated" and "rate" (share between 0 and 1).
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # The column name goes into the SQL text, so it must be one of our four known names (see GROUP_COLUMNS).
        if column not in GROUP_COLUMNS:
            raise ValueError(f"cannot group by {column!r}")
        where, params = self.where()
        out = self.query(
            f"Share stopped early by {column}",
            f"""SELECT {column}, COUNT(*) AS n, SUM(label) AS terminated, AVG(label) AS rate
                FROM trials {where} GROUP BY {column} ORDER BY n DESC""", params)

        # Put the rows in the wanted order, if there is one.
        if order:
            # Give every value its place in the list and sort by this place.
            place_in_order = {}
            for i in range(len(order)):
                place_in_order[order[i]] = i
            out["place"] = out[column].map(place_in_order).fillna(len(order))
            out = out.sort_values("place").drop(columns="place")
        return out.reset_index(drop=True)

    def top_conditions(self, k=15):
        """The k most common conditions among the matching trials (JOIN + GROUP BY + ORDER BY + LIMIT).

        Input: k is how many conditions to return (integer).

        Output: a DataFrame with the columns "condition", "n", "terminated" and "rate" (share between 0 and 1).
        """
        # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
        # Both tables of the JOIN have a column nct_id. So the filters must name the trials table by its short
        # name t.
        where, params = self.where(table="t")
        return self.query(
            "The most common conditions",
            f"""SELECT c.condition AS condition, COUNT(*) AS n, SUM(t.label) AS terminated, AVG(t.label) AS rate
                FROM trial_conditions AS c JOIN trials AS t ON t.nct_id = c.nct_id
                {where}
                GROUP BY c.condition ORDER BY n DESC, c.condition LIMIT ?""", list(params) + [int(k)])


# ---- 3. Run the script ----
if __name__ == "__main__":
    # Build the database again and print a few example queries as a quick check.
    path = build_database()
    db = TrialDatabase(path)
    print(f"Built {path.relative_to(viz.DATA_PATH.parents[2])}: {db.summary()['n']:,} trials")
    print(db.rate_by("phase_label", order=viz.PHASE_ORDER).to_string(index=False))
    print(db.top_conditions(k=5).to_string(index=False))

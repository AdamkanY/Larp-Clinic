"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Measure, for every source file, how much of it comes from commits made together with an AI, and check the
AI headers.

The HSG rules on AI use say that AI-generated code must be cited in the source code. Claude Code adds a line
"Co-Authored-By: Claude ..." to every commit it makes, so git can tell which lines came from an AI-assisted
commit. This script turns that into the table in docs/AI_USE.md:

    python tools/ai_provenance.py            # print the table (Markdown) for docs/AI_USE.md
    python tools/ai_provenance.py --check    # stop with an error if a source file lacks the "AI use:" header

How to read the share: "git blame" gives every line to the last commit that changed it. So the share means
"lines last changed in an AI-assisted commit". A line the team rewrote by hand in a later commit counts as the
team's. A line the AI wrote and nobody touched since counts as the AI's. Commits made on the GitHub website
(uploads, merges) have no such line, so their lines count as the team's even if the text came from a chat with
an AI. The table in docs/AI_USE.md says so.
"""
import re  # regular expressions: recognise the commit ids in the output of "git blame"
import subprocess  # run git commands from Python
import sys  # read the command line (--check) and stop with an error message
from datetime import date  # today's date for the heading of the table
from pathlib import Path  # file paths that work on every operating system

ROOT = Path(__file__).resolve().parents[1]
# The source files that must carry the header: code, styles and the app settings (not data, models or docs).
SOURCE_GLOBS = ["app/*.py", "app/pages/*.py", "app/style.css", "src/*.py", "tests/*.py", "tools/*.py",
                ".streamlit/config.toml", "notebooks/*.ipynb"]
HEADER = "AI use:"
# A Python file carries its header inside the module docstring, so the file has to start with this. Python only
# reads a docstring that is the very first thing in the file.
PYTHON_HEADER_START = '"""AI use:'


def git(*args):
    """Run a git command in the project folder and return what it prints.

    Input: args are the words of the command after "git", for example "log" and "--oneline".

    Output: the text that git printed. An error in git stops the script.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def ai_commits():
    """Find all commits that were written by Claude or have a Claude co-author line.

    Input: none.

    Output: a set with the ids of these commits.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # One record per commit: id, author name and the co-author lines. \x1f separates the three parts and \x1e
    # the commits, because these characters never occur in names.
    log = git("log", "--format=%H%x1f%an%x1f%(trailers:key=Co-Authored-By,valueonly,separator=;)%x1e")
    hashes = set()
    for record in log.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        parts = record.split("\x1f") + ["", ""]
        commit = parts[0]
        author = parts[1]
        co_authors = parts[2]
        if author == "Claude" or "anthropic.com" in co_authors.lower():
            hashes.add(commit)
    return hashes


def source_files():
    """List the source files in a fixed order.

    Inside a git checkout only the files git knows are returned. In the unpacked submission zip there is no
    git, so there every file on disk is returned (the header check must work there too).

    Input: none.

    Output: a sorted list of paths, relative to the project folder.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Collect every file that matches one of the patterns.
    found = set()
    for pattern in SOURCE_GLOBS:
        for path in ROOT.glob(pattern):
            found.add(str(path.relative_to(ROOT)))
    found = sorted(found)

    # Keep only the files that git tracks (without git: all of them).
    try:
        tracked = set(git("ls-files").split())
    except (OSError, subprocess.CalledProcessError):
        return found  # no git here: check every file on disk

    files = []
    for path in found:
        if path in tracked:
            files.append(path)
    return files


def share(path, ai):
    """Count the lines of one file and how many of them come from AI-assisted commits.

    Input: path is the file (relative to the project folder); ai is the set of AI-assisted commit ids from
    ai_commits().

    Output: a pair (number of lines, number of lines last changed in an AI-assisted commit).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    blame = git("blame", "--line-porcelain", "--", path)
    total = 0
    from_ai = 0
    for line in blame.splitlines():
        # A line that starts with a 40-character commit id belongs to one line of the file.
        if re.match(r"^[0-9a-f]{40} ", line):
            total += 1
            if line.split()[0] in ai:
                from_ai += 1
    return total, from_ai


def missing_headers():
    """Find the source files that do not have the AI header.

    A Python file needs it on its first line (inside the module docstring); the other files (styles, settings,
    notebooks) need it somewhere in their first lines.

    Input: none.

    Output: a list with the paths of the files without the header (empty if all have it).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    missing = []
    for path in source_files():
        head = (ROOT / path).read_text(encoding="utf-8")[:2000]
        if path.endswith(".py"):
            has_header = head.startswith(PYTHON_HEADER_START)
        else:
            has_header = HEADER in head
        if not has_header:
            missing.append(path)
    return missing


def table():
    """Build the Markdown table: file, number of lines, share from AI-assisted commits.

    Input: none.

    Output: the table as one Markdown text, with a first line that says when and at which commit it was measured.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    ai = ai_commits()
    commit = git("rev-parse", "--short", "HEAD").strip()
    n_commits = len(git("rev-list", "HEAD").split())
    rows = [f"Measured on {date.today().isoformat()} at commit {commit} "
            f"({len(ai)} of {n_commits} commits are AI-assisted).", "",
            "| File | Lines | Lines from AI-assisted commits |", "|---|---|---|"]
    for path in source_files():
        total, from_ai = share(path, ai)
        if total:
            rows.append(f"| `{path}` | {total} | {from_ai / total:.0%} |")
        else:
            rows.append(f"| `{path}` | 0 | - |")
    return "\n".join(rows)


def main():
    """Print the table. With --check only test the headers and stop with an error if one is missing.

    Input: none (the option --check is read from the command line).

    Output: nothing returned; the table or the result of the check is printed.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    if "--check" in sys.argv:
        missing = missing_headers()
        if missing:
            sys.exit("Missing the AI header: " + ", ".join(missing))
        print("Every source file has the AI header.")
        return
    print(table())


if __name__ == "__main__":
    main()

# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Measure, per source file, how much of it comes from commits co-authored by an AI, and check the AI headers.

The HSG rules on AI use require AI-generated code to be cited in the source code. Claude Code adds a
"Co-Authored-By: Claude ..." trailer to every commit it makes, so git can tell which lines came from an
AI-assisted commit. This script turns that into the table in docs/AI_USE.md:

    python tools/ai_provenance.py            # print the table (Markdown) for docs/AI_USE.md
    python tools/ai_provenance.py --check    # exit 1 if a source file lacks the "AI use:" header

How to read the share: `git blame` attributes every line to the last commit that changed it, so the share is
"lines last changed in an AI-assisted commit". A line the team rewrote by hand in a later commit without the
trailer counts as the team's; a line the AI wrote and nobody touched since counts as the AI's. Commits made in the
GitHub web interface (uploads, merges) have no trailer, so their lines count as the team's even if the text came
from a chat with an AI; the table in docs/AI_USE.md says so.
"""
import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Source files that must carry the header: code, styles and the app settings (not data, models or docs).
SOURCE_GLOBS = ["app/*.py", "app/pages/*.py", "app/style.css", "src/*.py", "tests/*.py", "tools/*.py",
                ".streamlit/config.toml", "notebooks/*.ipynb"]
HEADER = "AI use:"
# Who counts as an AI author: the commit author "Claude" or a co-author trailer with an Anthropic address.
AI_TRAILER = re.compile(r"anthropic\.com", re.I)


def git(*args) -> str:
    """Run a git command in the repository and return its output (fails loudly if git fails)."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def ai_commits() -> set:
    """Hashes of every commit authored by Claude or carrying a Claude co-author trailer."""
    hashes = set()
    log = git("log", "--format=%H%x1f%an%x1f%(trailers:key=Co-Authored-By,valueonly,separator=;)%x1e")
    for record in log.split("\x1e"):
        if not record.strip():
            continue
        commit, author, trailers = (record.strip().split("\x1f") + ["", ""])[:3]
        if author == "Claude" or AI_TRAILER.search(trailers):
            hashes.add(commit)
    return hashes


def source_files() -> list:
    """The source files, in a stable order: only tracked ones inside a git checkout, all of them in an unpacked zip.

    The header check must also work in the submitted zip, which has no .git folder.
    """
    found = sorted({str(p.relative_to(ROOT)) for pattern in SOURCE_GLOBS for p in ROOT.glob(pattern)})
    try:
        tracked = set(git("ls-files").split())
    except (OSError, subprocess.CalledProcessError):
        return found  # not a git checkout (or git is missing): check every file on disk
    return [f for f in found if f in tracked]


def share(path: str, ai: set) -> tuple:
    """(lines, lines last changed in an AI-assisted commit) for one file, from git blame."""
    blame = git("blame", "--line-porcelain", "--", path)
    owners = [line.split()[0] for line in blame.splitlines() if re.match(r"^[0-9a-f]{40} ", line)]
    return len(owners), sum(owner in ai for owner in owners)


def missing_headers() -> list:
    """Source files whose first lines do not contain the AI header."""
    missing = []
    for path in source_files():
        head = (ROOT / path).read_text(encoding="utf-8")[:2000]
        if HEADER not in head:
            missing.append(path)
    return missing


def table() -> str:
    """Markdown table: file, number of lines, share from AI-assisted commits."""
    ai = ai_commits()
    rows = [f"Measured on {date.today().isoformat()} at commit {git('rev-parse', '--short', 'HEAD').strip()} "
            f"({len(ai)} of {len(git('rev-list', 'HEAD').split())} commits are AI-assisted).", "",
            "| File | Lines | Lines from AI-assisted commits |", "|---|---|---|"]
    for path in source_files():
        total, from_ai = share(path, ai)
        rows.append(f"| `{path}` | {total} | {from_ai / total:.0%} |" if total else f"| `{path}` | 0 | - |")
    return "\n".join(rows)


def main() -> None:
    """Print the table, or with --check fail if a header is missing."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only check that every source file has the header")
    if parser.parse_args().check:
        missing = missing_headers()
        if missing:
            sys.exit("Missing the AI header: " + ", ".join(missing))
        print("Every source file has the AI header.")
        return
    print(table())


if __name__ == "__main__":
    main()

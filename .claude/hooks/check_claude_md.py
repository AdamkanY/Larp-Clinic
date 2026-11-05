#!/usr/bin/env python3
"""Keep the CLAUDE.md guides in step with the repository. A Claude Code hook; standard library only.

``.claude/settings.json`` runs it at two moments:

* SessionStart: ``check_claude_md.py --session-start`` remembers the commit the session started from
  and which files were already changed, so only this session's work is checked.
* Stop (Claude is about to end its turn): ``check_claude_md.py`` checks that
    1. every file is listed in the file table of its folder's CLAUDE.md (the nearest guide up the tree),
       and every top-level folder appears in the directory map of the root CLAUDE.md;
    2. no table lists a file that is gone. Rows marked "(expected)" and git-ignored generated files are
       fine; an "(expected)" file that now exists should lose the marker;
    3. files changed since the last check have a guide that was updated with them, and the root
       CLAUDE.md got a "Recent changes" line. This is a reminder, given once per version of a file.
  If anything is found, the stop is blocked once and Claude gets the list, so it updates the guides
  (or says why no update is needed) before it finishes. The stop right after that is never blocked,
  so the hook cannot loop.

``find_problems(root)`` (checks 1 and 2) is also run by ``tests/test_claude_md.py``. Any internal error
makes the hook pass silently: it must never break a session.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

GUIDE = "CLAUDE.md"
IGNORED_NAMES = {GUIDE, ".gitkeep", ".DS_Store"}
FILE_TABLE_HEADER = "File"  # first header cell of a folder guide's file table
MAP_TABLE_HEADER = "Path"  # first header cell of the root guide's directory map
EXPECTED = "(expected)"
PATTERN_MARKS = ("<", "NNNNN", "*")  # `<model>_metrics.json`, `page_NNNNN.json`, `*.py`
TOKEN = re.compile(r"`([^`]+)`")
SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
STATE_DIR = "claude-md-check"  # inside .git, so it is never committed
STATE_MAX_AGE_DAYS = 14
MAX_LISTED = 12


def git(root: Path, *args, stdin: str = None) -> str:
    """Run git in the repository and return its output; raises if git fails."""
    return subprocess.run(["git", *args], cwd=root, input=stdin, capture_output=True, text=True,
                          check=True, timeout=20).stdout


def repo_files(root: Path) -> list:
    """Tracked and untracked-but-not-ignored files that exist, as POSIX paths relative to the root."""
    out = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted({path for path in out.split("\0") if path and (root / path).is_file()})


def is_ignored(root: Path, path: str) -> bool:
    """True if git ignores this path: a generated file that is never committed."""
    result = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=root,
                            capture_output=True, timeout=20)
    return result.returncode == 0


def _cells(line: str) -> list:
    """The cells of one markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_rows(text: str, first_header: str) -> list:
    """Rows (lists of cells) of every markdown table whose first header cell is ``first_header``."""
    rows, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and SEPARATOR.match(lines[i + 1].strip()):
            header = _cells(lines[i])
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                if header[0] == first_header:
                    rows.append(_cells(lines[i]))
                i += 1
            continue
        i += 1
    return rows


def token_regex(token: str):
    """Regex for a table token: a file, a folder (``static/fonts/``) or a pattern (``<model>_metrics.json``)."""
    if token.endswith("/"):
        return re.compile(re.escape(token) + ".+")
    pattern = re.sub(r"<[^>]*>", "[^/]+", re.escape(token))
    return re.compile(pattern.replace("NNNNN", r"\d+").replace(r"\*", "[^/]*"))


def load_guides(root: Path, files: list) -> dict:
    """{folder ('' for the root): table rows} for every CLAUDE.md in the repository."""
    guides = {}
    for path in files:
        if PurePosixPath(path).name == GUIDE:
            folder = _folder(path)
            text = (root / path).read_text(encoding="utf-8")
            guides[folder] = table_rows(text, MAP_TABLE_HEADER if folder == "" else FILE_TABLE_HEADER)
    return guides


def _folder(path: str) -> str:
    """Parent folder of a path, '' for the root."""
    parent = str(PurePosixPath(path).parent)
    return "" if parent == "." else parent


def governing(path: str, guides: dict) -> str:
    """The folder of the nearest CLAUDE.md above ``path`` ('' = the root guide)."""
    parts = PurePosixPath(path).parent.parts
    for depth in range(len(parts), 0, -1):
        folder = "/".join(parts[:depth])
        if folder in guides:
            return folder
    return ""


def _tokens(rows: list) -> list:
    """Every backticked token in the first cell of the rows."""
    return [token for row in rows if row for token in TOKEN.findall(row[0])]


def find_problems(root) -> list:
    """Out-of-date file tables (checks 1 and 2 of the module docstring), as readable lines."""
    root = Path(root)
    files = repo_files(root)
    guides = load_guides(root, files)
    if "" not in guides:
        return [f"{GUIDE} is missing at the repository root"]
    problems = []
    map_tokens = _tokens(guides[""])

    def in_map(top):
        return any(token == top or token.startswith(top) for token in map_tokens)

    missing_tops = set()
    for folder in guides:
        if folder and not in_map(folder.split("/")[0] + "/"):
            missing_tops.add(folder.split("/")[0] + "/")
    for path in files:
        if PurePosixPath(path).name in IGNORED_NAMES:
            continue
        folder = governing(path, guides)
        if folder == "":
            if "/" in path and not in_map(path.split("/")[0] + "/"):
                missing_tops.add(path.split("/")[0] + "/")
            continue  # root-level files (README, requirements) need no row
        relative = path[len(folder) + 1:]
        if not any(token_regex(token).fullmatch(relative) for token in _tokens(guides[folder])):
            problems.append(f"{folder}/{GUIDE}: no row for `{relative}`")
    problems += [f"{GUIDE}: the directory map has no row for `{top}`" for top in sorted(missing_tops)]

    for folder, rows in guides.items():
        for row in rows:
            found = TOKEN.findall(row[0]) if row else []
            if not found or is_pattern(found[0]):
                continue
            token = found[0]
            if folder == "" and not token.endswith("/"):
                continue  # the directory map only lists folders
            target = f"{folder}/{token}" if folder else token
            exists = (root / target).is_dir() if token.endswith("/") else (root / target).is_file()
            expected = EXPECTED in row[0]
            where = f"{folder}/{GUIDE}" if folder else GUIDE
            if exists and expected:
                problems.append(f"{where}: `{token}` exists now; drop \"(expected)\" from its row")
            elif not exists and not expected and not is_ignored(root, target.rstrip("/")):
                problems.append(f"{where}: lists `{token}`, which does not exist "
                                "(remove the row, or mark a planned file \"(expected)\")")
    return problems


def is_pattern(token: str) -> bool:
    """True for tokens that describe many files, such as ``<model>_metrics.json``."""
    return any(mark in token for mark in PATTERN_MARKS)


# ---- per-session reminder (check 3) ----

def _head(root: Path) -> str:
    """The current commit."""
    return git(root, "rev-parse", "HEAD").strip()


def _commit_exists(root: Path, sha: str) -> bool:
    """True if ``sha`` names a commit in this repository."""
    try:
        git(root, "cat-file", "-e", f"{sha}^{{commit}}")
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def changed_since(root: Path, base: str) -> list:
    """Files that differ from commit ``base`` (committed or not) plus untracked files."""
    tracked = git(root, "diff", "--name-only", "-z", base, "--").split("\0")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    return sorted({path for path in tracked + untracked if path})


def fingerprints(root: Path, paths: list) -> dict:
    """{path: git blob hash of its current content, or 'deleted'}."""
    existing = [path for path in paths if (root / path).is_file()]
    hashes = git(root, "hash-object", "--stdin-paths", stdin="\n".join(existing) + "\n").split() if existing else []
    prints = dict(zip(existing, hashes))
    return {path: prints.get(path, "deleted") for path in paths}


def state_file(root: Path, session_id) -> Path:
    """Where this session's start commit and already-checked file versions are kept."""
    git_dir = Path(git(root, "rev-parse", "--absolute-git-dir").strip())
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id or "default"))
    return git_dir / STATE_DIR / f"{safe}.json"


def _read_state(path: Path):
    """The saved state, or None."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_state(path: Path, state: dict) -> None:
    """Save the state and drop the files of sessions older than two weeks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    cutoff = time.time() - STATE_MAX_AGE_DAYS * 86400
    for old in path.parent.glob("*.json"):
        if old != path and old.stat().st_mtime < cutoff:
            old.unlink(missing_ok=True)


def session_start(root: Path, payload: dict) -> None:
    """Remember where the session started; a resumed or compacted session keeps its first start."""
    path = state_file(root, payload.get("session_id"))
    if _read_state(path):
        return
    base = _head(root)
    # Changes that were already there belong to someone else's work, not to this session.
    _write_state(path, {"base": base, "seen": fingerprints(root, changed_since(root, base))})


def reminders(root: Path, payload: dict) -> tuple:
    """Check 3: ({guide folder: changed files whose guide was not updated with them}, root note needed?)."""
    path = state_file(root, payload.get("session_id"))
    state = _read_state(path) or {"base": _head(root), "seen": {}}
    if not _commit_exists(root, state.get("base", "")):
        state["base"] = _head(root)
    prints = fingerprints(root, changed_since(root, state["base"]))
    seen = state.setdefault("seen", {})
    fresh = [p for p, digest in prints.items() if seen.get(p) != digest]
    seen.update({p: prints[p] for p in fresh})
    _write_state(path, state)

    updated_guides = {_folder(p) for p in fresh if PurePosixPath(p).name == GUIDE}
    guides = load_guides(root, repo_files(root))
    pending = {}
    for p in fresh:
        name = PurePosixPath(p).name
        if name in IGNORED_NAMES or "/" not in p:
            continue  # guides themselves and root-level files (README, requirements)
        folder = governing(p, guides)
        if folder not in updated_guides:
            pending.setdefault(folder, []).append(p)
    needs_root_note = any("/" in p and PurePosixPath(p).name not in IGNORED_NAMES for p in fresh) \
        and "" not in updated_guides
    return pending, needs_root_note


def _listed(items: list) -> str:
    """Comma-separated, cut after MAX_LISTED items."""
    more = f" and {len(items) - MAX_LISTED} more" if len(items) > MAX_LISTED else ""
    return ", ".join(items[:MAX_LISTED]) + more


def message(problems: list, pending: dict, needs_root_note: bool) -> str:
    """The text Claude gets when the stop is blocked."""
    lines = ["CLAUDE.md check (.claude/hooks/check_claude_md.py) before you finish:"]
    if problems:
        lines.append("Out of date, please fix:")
        lines += [f"- {problem}" for problem in problems[:MAX_LISTED]]
        if len(problems) > MAX_LISTED:
            lines.append(f"- ... and {len(problems) - MAX_LISTED} more")
    if pending or needs_root_note:
        lines.append("Changed since the last check, guide not updated with it (update the row if what a "
                     "file does or holds changed; if nothing a reader needs changed, just say so):")
        for folder, paths in sorted(pending.items()):
            lines.append(f"- {folder + '/' if folder else ''}{GUIDE}: {_listed(paths)}")
        if needs_root_note:
            lines.append(f"- {GUIDE} (root): add a dated line under \"Recent changes\" for anything notable, "
                         "and update Status / Roadmap / Open decisions if they moved")
    return "\n".join(lines)


def run(payload: dict, root: Path, session_start_mode: bool = False):
    """One hook call. Returns the JSON to print (dict) or None to let Claude stop without comment."""
    if session_start_mode:
        session_start(root, payload)
        return None
    problems = find_problems(root)
    pending, needs_root_note = reminders(root, payload)
    if payload.get("stop_hook_active"):
        # Claude already had one chance in this stop; never block twice, but tell the person what is left.
        if problems:
            return {"systemMessage": "CLAUDE.md guides still out of date: " + "; ".join(problems[:MAX_LISTED])}
        return None
    if problems or pending or needs_root_note:
        return {"decision": "block", "reason": message(problems, pending, needs_root_note)}
    return None


def main() -> int:
    """Hook entry point: read the event from stdin, print a decision, always exit 0."""
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        payload = json.loads(raw) if raw.strip() else {}
        start = Path(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
        root = Path(git(start, "rev-parse", "--show-toplevel").strip())
        output = run(payload, root, session_start_mode="--session-start" in sys.argv[1:])
        if output:
            print(json.dumps(output))
    except Exception:  # noqa: BLE001 - a broken check must never break the session
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

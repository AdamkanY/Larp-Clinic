"""The CLAUDE.md guides match the files in the repository: the same check the Claude Code Stop hook runs.

Run with ``python tests/test_claude_md.py`` (or pytest). Needs git, no network. The first test checks this
repository; the others build a tiny throwaway git repository to show what the hook catches and when it
reminds Claude to update a guide.
"""

import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".claude" / "hooks" / "check_claude_md.py"

spec = importlib.util.spec_from_file_location("check_claude_md", CHECKER)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

HAS_GIT = shutil.which("git") is not None and (ROOT / ".git").exists()

ROOT_GUIDE = "# Toy\n\n| Path | Holds |\n|---|---|\n| `pkg/` | code |\n"
PKG_GUIDE = ("# pkg\n\n| File | Contains |\n|---|---|\n| `a.py` | a |\n| `data_<n>.csv` | data files |\n"
             "| `sub/` | helpers |\n| `later.py` (expected) | planned |\n| `out.bin` | generated, git-ignored |\n")


def sh(root, *args):
    """Run git quietly in the toy repository."""
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", *args], cwd=root, check=True, capture_output=True)


def write(root, path, text):
    """Create or overwrite one file of the toy repository."""
    target = Path(root) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def toy_repo(tmp):
    """A committed toy repository whose guides are complete."""
    root = Path(tmp)
    sh(root, "init", "-q")
    write(root, "CLAUDE.md", ROOT_GUIDE)
    write(root, "pkg/CLAUDE.md", PKG_GUIDE)
    write(root, "pkg/a.py", "A = 1\n")
    write(root, "pkg/data_1.csv", "x\n")
    write(root, "pkg/sub/x.txt", "x\n")
    write(root, ".gitignore", "pkg/out.bin\n")
    write(root, "README.md", "toy\n")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "init")
    return root


def test_repository_guides_are_current():
    """Every file of Trial-Radar is in its folder's table, and no table lists a missing file."""
    if not HAS_GIT:
        print("skipped (no git)")
        return
    problems = check.find_problems(ROOT)
    assert not problems, "\n".join(problems)


def test_detects_drift():
    """New files, missing files, unknown top-level folders and stale '(expected)' markers are reported."""
    if not HAS_GIT:
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = toy_repo(tmp)
        assert check.find_problems(root) == []  # patterns, folders, ignored and expected rows are fine
        write(root, "pkg/new.py", "")
        write(root, "extra/z.txt", "")
        write(root, "pkg/CLAUDE.md", PKG_GUIDE + "| `gone.py` | removed long ago |\n")
        write(root, "pkg/later.py", "")
        problems = "\n".join(check.find_problems(root))
        assert "pkg/CLAUDE.md: no row for `new.py`" in problems
        assert "directory map has no row for `extra/`" in problems
        assert "lists `gone.py`, which does not exist" in problems
        assert "`later.py` exists now" in problems
        assert "out.bin" not in problems and "data_1.csv" not in problems


def test_stop_hook_reminds_once_per_change():
    """A changed file without a guide update blocks one stop; the next stop and unchanged files pass."""
    if not HAS_GIT:
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = toy_repo(tmp)
        session = {"session_id": "test-session"}
        assert check.run(session, root, session_start_mode=True) is None
        assert check.run(session, root) is None  # nothing changed yet

        write(root, "pkg/a.py", "A = 2\n")
        blocked = check.run(session, root)
        assert blocked["decision"] == "block"
        assert "pkg/CLAUDE.md: pkg/a.py" in blocked["reason"] and "Recent changes" in blocked["reason"]
        assert check.run({**session, "stop_hook_active": True}, root) is None  # never twice in a row
        assert check.run(session, root) is None  # same version of the file: already reminded

        write(root, "pkg/a.py", "A = 3\n")
        write(root, "pkg/CLAUDE.md", PKG_GUIDE.replace("| a |", "| a, now 3 |"))
        write(root, "CLAUDE.md", ROOT_GUIDE + "\n## Recent changes\n\n- a.py holds 3\n")
        assert check.run(session, root) is None  # guides updated together with the change

        sh(root, "add", "-A")
        sh(root, "commit", "-q", "-m", "work")
        write(root, "README.md", "toy, longer\n")
        assert check.run(session, root) is None  # committed work stays seen; root-level files need no guide


if __name__ == "__main__":
    test_repository_guides_are_current()
    test_detects_drift()
    test_stop_hook_reminds_once_per_change()
    print("all ok")

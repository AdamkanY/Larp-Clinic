"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Check that every Python file follows the docstring format of the course.

The rules (one place, so the whole team checks the same thing):

* The file starts on line 1 with the module docstring, and its first line starts with "AI use:".
* Every function, method and nested function has a docstring: a one-line summary, an empty line, a paragraph that
  starts with "Input:", an empty line, a paragraph that starts with "Output:".
* Every class has a docstring.

Usage (from the project folder):

    python tools/check_docstrings.py            # check all Python files
    python tools/check_docstrings.py src/viz.py # check only some files

The script prints one line per file and stops with an error if a file breaks a rule.
"""
import ast  # read a Python file as a tree, so docstrings can be found without running the code
import glob  # find the files that match a pattern
import sys  # read the file names from the command line and stop with an error code
from pathlib import Path  # file paths that work on every operating system

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ["app", "app/pages", "src", "tests", "tools"]
HEADER_START = '"""AI use:'


def python_files():
    """Find the Python files that must follow the format.

    Input: none.

    Output: a sorted list of paths, relative to the project folder.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    files = []
    for folder in FOLDERS:
        for path in sorted(glob.glob(str(ROOT / folder / "*.py"))):
            files.append(str(Path(path).relative_to(ROOT)))
    return files


def function_problems(node):
    """Find the format problems of one function.

    Input: node is the function as a tree node (from the ast module).

    Output: a list of short problem texts (empty if the docstring is fine).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    where = f"line {node.lineno}: {node.name}"
    doc = ast.get_docstring(node)
    if not doc:
        return [where + " has no docstring"]

    # Find the lines where the Input and the Output paragraphs start.
    lines = doc.splitlines()
    starts = {}
    for i in range(len(lines)):
        for key in ("Input:", "Output:"):
            if lines[i].startswith(key) and key not in starts:
                starts[key] = i
    if "Input:" not in starts or "Output:" not in starts:
        return [where + " needs 'Input:' and 'Output:' paragraphs"]

    # Summary line, then an empty line, and an empty line before each of the two paragraphs.
    problems = []
    if starts["Output:"] < starts["Input:"]:
        problems.append(where + " has Output before Input")
    if not lines[0].strip() or len(lines) < 2 or lines[1].strip():
        problems.append(where + " needs a one-line summary followed by an empty line")
    for key in ("Input:", "Output:"):
        if starts[key] == 0 or lines[starts[key] - 1].strip():
            problems.append(where + f" needs an empty line before '{key}'")
    return problems


def file_problems(path):
    """Find all format problems of one file.

    Input: path is the file, relative to the project folder.

    Output: a list of short problem texts (empty if the file is fine).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # The header must be the first thing in the file, so Python reads it as the module docstring.
    problems = []
    if not source.startswith(HEADER_START):
        problems.append('line 1 must start with """AI use:')
    if not ast.get_docstring(tree):
        problems.append("no module docstring")

    # Every function and class, also those inside other functions.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            problems += function_problems(node)
        elif isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
            problems.append(f"line {node.lineno}: class {node.name} has no docstring")
    return problems


def main():
    """Check the files named on the command line (or all Python files) and print the result.

    Input: none (the file names are read from the command line).

    Output: nothing returned; one line per file is printed, and the script stops with an error if a file has
    problems.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    files = sys.argv[1:]
    if not files:
        files = python_files()

    # One line per file, with the problems below a file that fails.
    bad = 0
    for path in files:
        problems = file_problems(path)
        if problems:
            bad += 1
            print("FAIL", path)
            for problem in problems:
                print("   ", problem)
        else:
            print("ok  ", path)
    if bad:
        sys.exit(f"{bad} file(s) break the docstring format.")
    print("Every file follows the docstring format.")


if __name__ == "__main__":
    main()

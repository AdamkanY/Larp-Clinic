# tools/

Developer scripts that are neither part of the app nor of the data/model pipeline. Not imported by `app/` or `src/`.

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `ai_provenance.py` | Reads the git history: which commits are AI-assisted (author "Claude" or a `Co-Authored-By` trailer with an Anthropic address) and, per source file, the share of lines last changed in such a commit (`git blame`). `python tools/ai_provenance.py` prints the table for `docs/AI_USE.md`; `--check` fails if a Python file does not start with `"""AI use:` on line 1, or if another source file lacks the "AI use:" header in its first lines (also checked by `tests/test_app.py`; works in the unpacked submission zip too, without git) | Claude | team, before the submission |
| `check_docstrings.py` | Checks the lecturer's docstring format in every Python file: module docstring on line 1 starting with `"""AI use:`, and for every function (also nested ones) a one-line summary, an empty line, an `Input:` paragraph, an empty line and an `Output:` paragraph; classes need a docstring. `python tools/check_docstrings.py [files]`; stops with an error if a file breaks a rule | Claude | team, before every commit that touches Python files |

Every new Python file (`app/`, `src/`, `tests/`, `tools/`) starts on line 1 with the module docstring
`"""AI use: ...` (copy the first line from any existing file), and every function has a docstring with a summary, an
`Input:` and an `Output:` paragraph. `app/style.css`, `.streamlit/config.toml` and the notebooks keep the two-line
"AI use:" comment at the top.

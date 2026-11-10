# tools/

Developer scripts that are neither part of the app nor of the data/model pipeline. Not imported by `app/` or `src/`.

| File | Contains / does | Produced by | Used by |
|---|---|---|---|
| `ai_provenance.py` | Reads the git history: which commits are AI-assisted (author "Claude" or a `Co-Authored-By` trailer with an Anthropic address) and, per source file, the share of lines last changed in such a commit (`git blame`). `python tools/ai_provenance.py` prints the table for `docs/AI_USE.md`; `--check` fails if a source file lacks the "AI use:" header in its first lines (also checked by `tests/test_app.py`; works in the unpacked submission zip too, without git) | Claude | team, before the submission |

Every new source file (`app/`, `src/`, `tests/`, `tools/`, `notebooks/`, `app/style.css`, `.streamlit/config.toml`)
starts with the two-line "AI use:" header; copy it from any existing file.

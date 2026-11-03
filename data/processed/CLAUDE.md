# data/processed/

Cleaned, model-ready files produced by code in `src/`. Safe to delete and regenerate. Not for: raw exports,
hand-edited files, or notebooks' scratch output.

Currently empty (`.gitkeep` only).

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `trials_clean.parquet` (expected) | Raw CSV after dropping WITHDRAWN, parsing dates and design fields, removing LEAKAGE columns | `src/data.py` | `src/features.py`, `src/train.py`, app visuals |
| `features.parquet` (expected) | Feature matrix plus label and start date | `src/features.py` | `src/train.py` |

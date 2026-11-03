# tests/

Offline checks that need no network or real data (synthetic records only). Run `python tests/test_features.py`.
Not for: exploration or tests that call the live API.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `test_features.py` | Checks `build_features` values, missing handling, and that leaky fields (enrollment, locations, results, other dates) are ignored | Simon | everyone, before committing feature changes |

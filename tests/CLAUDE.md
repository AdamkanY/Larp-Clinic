# tests/

Offline checks that need no network or real data (synthetic records only). Run `python tests/test_features.py`.
Not for: exploration or tests that call the live API.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `test_features.py` | Checks `build_features` values, missing handling, and that leaky fields (enrollment, locations, results, other dates) are ignored; `condition_features` categories, tumour-of-an-organ rule, look-alikes that must not match (ocular hypertension, heartburn, nerve block, ...), odd inputs, parity between the raw list and the stored `"; "`-joined text (the backfill path), and the v1/v2 column lists | Simon / Adam | everyone, before committing feature changes |

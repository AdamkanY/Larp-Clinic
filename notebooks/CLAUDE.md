# notebooks/

Exploration only. **Nothing in `src/` or `app/` may import from here.** Code worth keeping moves to `src/`.
Not for: final pipeline code, app code, or large outputs. `01_eda.ipynb` is committed *with* its outputs (small PNG charts, about 200 KB) so it can be read on GitHub without running it; keep other notebooks that small or clear their outputs.

Naming: `NN_topic_initials.ipynb`. Run a notebook from `notebooks/` or the repo root; it finds `data/` itself.

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `01_eda.ipynb` | EDA of `data/processed/trials.parquet`: status counts, terminated rate by phase, sponsor class, masking, allocation and start year, missingness table, eligibility text length (terminated vs completed), each chart with a short takeaway. Needs `matplotlib` and a Jupyter install (`pip install jupyter`, not in `requirements-dev.txt`) | Team | Nobody (reference) |

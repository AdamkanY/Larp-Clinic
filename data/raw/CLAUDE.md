# data/raw/

Original ClinicalTrials.gov exports. **Never edited, never overwritten.** Cleaned versions go to `data/processed/`.
Not for: intermediate files, samples, or anything produced by code.

## Files

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `ctg-studies.csv` | 95,574 interventional trials (21 columns, ~140 MB, Git LFS). Statuses: COMPLETED 76,027 / TERMINATED 13,949 / WITHDRAWN 5,598. Start dates 2008 to 2026 (a few placeholder future dates). | Manual export from clinicaltrials.gov | `src/data.py` (expected) |

Expected later: an export (or API pull) that includes ongoing trials and an enrollment-type field.

## Data dictionary (`ctg-studies.csv`)

Tags: **SAFE** feature, **LEAKAGE** (never use), **ID** metadata, **TARGET**.

| Column | Meaning | Type | % missing | Tag |
|---|---|---|---|---|
| NCT Number | Trial registry ID (unique) | str | 0 | ID |
| Study Title | Free-text title | str | 0 | ID (text features optional, design-time) |
| Study URL | Link to the registry page | str | 0 | ID |
| Study Status | Overall status | str | 0 | TARGET (COMPLETED=0, TERMINATED=1, drop WITHDRAWN) |
| Conditions | Conditions studied, `\|`-separated | str | 0 | SAFE |
| Interventions | `TYPE: name` entries, `\|`-separated | str | 0 | SAFE |
| Sponsor | Lead sponsor | str | 0 | SAFE |
| Collaborators | Other sponsors, `\|`-separated | str | 69.4 | SAFE (missing = none) |
| Sex | ALL / MALE / FEMALE | str | 0.03 | SAFE |
| Age | Age groups, e.g. `ADULT, OLDER_ADULT` | str | 0 | SAFE |
| Phases | PHASE1, PHASE2, PHASE3, `PHASE1\|PHASE2`, `PHASE2\|PHASE3` | str | 0 | SAFE |
| Enrollment | Participant count; **actual** for finished trials, no type column | float | 0.06 | LEAKAGE in this CSV (see root CLAUDE.md) |
| Funder Type | INDUSTRY, OTHER, NIH, OTHER_GOV, NETWORK, FED, INDIV, UNKNOWN | str | 0 | SAFE |
| Study Type | Always INTERVENTIONAL in this export | str | 0 | SAFE but constant (drop) |
| Study Design | `Allocation \| Intervention Model \| Masking \| Primary Purpose` string | str | 0.03 | SAFE (parse into 4 fields) |
| Start Date | `YYYY-MM` or `YYYY-MM-DD` | str | 0 | SAFE (design-time) and the split key |
| Primary Completion Date | End of primary data collection | str | 0.26 | LEAKAGE |
| Completion Date | End of the trial | str | 1.55 | LEAKAGE |
| First Posted | Date first registered publicly | str | 0 | SAFE (known at registration) |
| Last Update Posted | Last registry edit | str | 0 | LEAKAGE |
| Locations | Sites, full address strings | str | 5.97 | SAFE (count/countries only; sites can be added after start, so use with care) |

Not in the export (do not expect them): why-stopped, results-posted, enrollment type.

## Quirks to handle in `src/`

- All three remaining statuses are finished; there are **no ongoing trials** in this file.
- `Enrollment` is 0 for 99% of WITHDRAWN trials (dropped anyway).
- Start dates in 2027 to 2099 (about 22 rows) are data-entry errors or placeholders.
- Terminated share by start year rises from 13% (2013) to 21% (2021), then falls for recent years because
  recent trials have not had time to terminate. Mind this in the date split.

# data/raw/

Raw ClinicalTrials.gov API v2 pages, written by `src/fetch_data.py`. **Not committed** (git-ignored, except this
file): regenerate with `python src/fetch_data.py`. Never edit by hand; cleaned data goes to `data/processed/`.
Not for: intermediate files, samples, or anything produced by `build_dataset.py`.

## Files (all generated, git-ignored)

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `pages/page_NNNNN.json` | One API response per file (up to 1000 studies, plus `nextPageToken`, first page has `totalCount`) | `src/fetch_data.py` | `src/build_dataset.py` |
| `manifest.json` | Query parameters of the download and `fetched_on` (the download date = the data snapshot for the horizon label); a rerun with different parameters is refused (use `--restart`) | `src/fetch_data.py` | `src/fetch_data.py` (resume check), `src/build_dataset.py` (snapshot date, horizon check) |

## Query

`GET https://clinicaltrials.gov/api/v2/studies` with
`filter.overallStatus=COMPLETED,TERMINATED,RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION,NOT_YET_RECRUITING`,
`filter.advanced=AREA[StudyType]INTERVENTIONAL`, `pageSize=1000`, paginated by `nextPageToken`. By default a
`fields` projection limits the download to the design-time fields below plus a few display and outcome fields, so
enrollment, results, locations and every date other than the start and completion date are never stored;
`--full` downloads whole records. Resumable: rerun to continue after an interruption. Retries 429/5xx with
exponential backoff; any other 4xx (for example a field name the API does not know) stops with the API's message.

Downloads made before 2026-11-05 hold only COMPLETED/TERMINATED trials and no completion date or `whyStopped`.
`build_dataset.py` then still builds `trials.parquet` but skips the two horizon files; download again with
`python src/fetch_data.py --restart`.

## Decisions

- **Enrollment is permanently excluded** as a feature (finished trials report actual counts, which leak).
- **Locations / number of sites excluded from v1** (sites are added after start).
- **Date window:** start date 2008-01-01 to 2019-12-31. Train = start 2008-2016, test = start 2017-2019.
  Start dates >= 2020 and unparseable ones are dropped.
- **Label:** TERMINATED = 1, COMPLETED = 0; every other status is dropped for training.
- **Horizon experiment** (root `CLAUDE.md`, "Experiment exception"): still-running trials and the completion date
  are downloaded for the 3-year label of `src/horizon.py`; `whyStopped` only for the stop-reason charts.

## API fields used (JSON path under `protocolSection`) and tags

Tags: **SAFE** feature, **LEAKAGE** (never use), **META** non-feature column, **LABEL** builds a label only,
**EDA** charts only. LABEL and EDA fields are read by `src/horizon.py` alone, never by `build_features`.

| Path | Used as | Tag |
|---|---|---|
| `identificationModule.nctId` | `nct_id` | META |
| `identificationModule.briefTitle` / `sponsorCollaboratorsModule.leadSponsor.name` / `conditionsModule.conditions` (joined) | `brief_title` / `lead_sponsor_name` / `conditions` | DISPLAY (never a feature) |
| `statusModule.overallStatus` | `overall_status`, `label` | META / TARGET |
| `statusModule.startDateStruct.date` | `start_date` (only date allowed; `split` is derived from it, not a feature) | META + SAFE |
| `statusModule.completionDateStruct.{date,type}` | stop date of a terminated trial for the 3-year horizon label (`ESTIMATED` dates are not used) | LABEL (LEAKAGE as a feature) |
| `statusModule.whyStopped` | `why_stopped`, `stop_reason`, `mentions_covid` in `outcomes.parquet` | EDA (LEAKAGE as a feature) |
| `designModule.phases` | `phase` | SAFE |
| `designModule.designInfo.{allocation,interventionModel,primaryPurpose,maskingInfo.masking}` | same-named columns | SAFE |
| `armsInterventionsModule.armGroups` | `n_arms` (absent = 0) | SAFE |
| `armsInterventionsModule.interventions[].type` | `intervention_<type>` multi-hot | SAFE |
| `sponsorCollaboratorsModule.leadSponsor.class` / `.collaborators` | `sponsor_class` / `has_collaborators` | SAFE |
| `oversightModule.{oversightHasDmc}` | `has_dmc` (`isFdaRegulated*` dropped from features) | SAFE |
| `eligibilityModule.{healthyVolunteers,sex,minimumAge,maximumAge,eligibilityCriteria}` | `healthy_volunteers`, `sex`, `min_age_years`, `max_age_years` (missing = 120), `has_max_age`, `eligibility_text_len` | SAFE |
| `outcomesModule.{primaryOutcomes,secondaryOutcomes}` / `conditionsModule.conditions` | counts (absent = 0) | SAFE |
| `designModule.enrollmentInfo` | - | LEAKAGE (excluded permanently) |
| `contactsLocationsModule` | - | excluded from v1 |
| `primaryCompletionDateStruct`, `lastUpdate*`, `resultsFirst*` | - (not downloaded) | LEAKAGE |
| `resultsSection`, `hasResults` | - | LEAKAGE |

## Quirks

- Missing source fields become missing values, not 0 (older registrations lack arms, DMC and FDA flags).
  `minimumAge` / `maximumAge` of `N/A` also become missing.
- Start and completion dates are mostly month precision (`YYYY-MM`); stored as the 1st of the month. The
  horizon label therefore counts whole months (36 months = 3 years).
- Follow-up bias: the terminated share among *ended* trials rises for recent start years, because trials still
  running are left out and a stopped trial ends sooner than one that runs its course. The 2017-2019 test window
  of `trials.parquet` is affected; the 3-year horizon label is built to avoid it.
- Sponsors report a stop weeks or months late, so the horizon label only uses trials that started at least
  3 years and 6 months before `fetched_on` (`REPORTING_LAG_MONTHS` in `src/horizon.py`).

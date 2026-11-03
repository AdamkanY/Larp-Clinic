# data/raw/

Raw ClinicalTrials.gov API v2 pages, written by `src/fetch_data.py`. **Not committed** (git-ignored, except this
file): regenerate with `python src/fetch_data.py`. Never edit by hand; cleaned data goes to `data/processed/`.
Not for: intermediate files, samples, or anything produced by `build_dataset.py`.

## Files (all generated, git-ignored)

| File | Contains | Produced by | Used by |
|---|---|---|---|
| `pages/page_NNNNN.json` | One API response per file (up to 1000 studies, plus `nextPageToken`, first page has `totalCount`) | `src/fetch_data.py` | `src/build_dataset.py` |
| `manifest.json` | Query parameters of the download; a rerun with different parameters is refused (use `--restart`) | `src/fetch_data.py` | `src/fetch_data.py` (resume check) |

## Query

`GET https://clinicaltrials.gov/api/v2/studies` with `filter.overallStatus=COMPLETED,TERMINATED`,
`filter.advanced=AREA[StudyType]INTERVENTIONAL`, `pageSize=1000`, paginated by `nextPageToken`. By default a
`fields` projection limits the download to the design-time fields below (so enrollment, results and
completion dates are never stored); `--full` downloads whole records. Resumable: rerun to continue after an
interruption. Retries 429/5xx with exponential backoff.

## Decisions

- **Enrollment is permanently excluded** as a feature (finished trials report actual counts, which leak).
- **Locations / number of sites excluded from v1** (sites are added after start).
- **Date window:** start date 2008-01-01 to 2019-12-31. Train = start 2008-2016, test = start 2017-2019.
  Start dates >= 2020 and unparseable ones are dropped.
- **Label:** TERMINATED = 1, COMPLETED = 0; every other status is dropped for training.

## API fields used (JSON path under `protocolSection`) and tags

Tags: **SAFE** feature, **LEAKAGE** (never use), **META** non-feature column.

| Path | Used as | Tag |
|---|---|---|
| `identificationModule.nctId` | `nct_id` | META |
| `statusModule.overallStatus` | `overall_status`, `label` | META / TARGET |
| `statusModule.startDateStruct.date` | `start_date`, `start_year` (only date allowed) | META + SAFE |
| `designModule.phases` | `phase` | SAFE |
| `designModule.designInfo.{allocation,interventionModel,primaryPurpose,maskingInfo.masking}` | same-named columns | SAFE |
| `armsInterventionsModule.armGroups` | `n_arms` | SAFE |
| `armsInterventionsModule.interventions[].type` | `intervention_<type>` multi-hot | SAFE |
| `sponsorCollaboratorsModule.leadSponsor.class` / `.collaborators` | `sponsor_class` / `has_collaborators` | SAFE |
| `oversightModule.{oversightHasDmc,isFdaRegulatedDrug,isFdaRegulatedDevice}` | `has_dmc`, `fda_regulated_*` | SAFE |
| `eligibilityModule.{healthyVolunteers,sex,minimumAge,maximumAge,eligibilityCriteria}` | `healthy_volunteers`, `sex`, `min/max_age_years`, `eligibility_text_len` | SAFE |
| `outcomesModule.{primaryOutcomes,secondaryOutcomes}` / `conditionsModule.conditions` | counts | SAFE |
| `designModule.enrollmentInfo` | - | LEAKAGE (excluded permanently) |
| `contactsLocationsModule` | - | excluded from v1 |
| `statusModule.whyStopped`, `completionDateStruct`, `primaryCompletionDateStruct`, `lastUpdate*`, `resultsFirst*` | - | LEAKAGE |
| `resultsSection`, `hasResults` | - | LEAKAGE |

## Quirks

- Missing source fields become missing values, not 0 (older registrations lack arms, DMC and FDA flags).
  `minimumAge` / `maximumAge` of `N/A` also become missing.
- Start dates are month precision (`YYYY-MM`); stored as the 1st of the month.
- Terminated share rises for recent start years partly because recent trials have had less time to terminate;
  the 2017-2019 test window may therefore differ in base rate from train.

# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Download all COMPLETED/TERMINATED interventional studies from ClinicalTrials.gov.

Raw API pages are written unchanged to ``data/raw/pages/page_NNNNN.json``.
The script is resumable: each page stores the ``nextPageToken`` that leads to
the following one, so a rerun continues after the last complete page.

Usage:
    python src/fetch_data.py            # fetch or resume
    python src/fetch_data.py --restart  # delete existing pages, start over
    python src/fetch_data.py --full     # no field projection (whole records)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PAGES_DIR = RAW_DIR / "pages"
MANIFEST = RAW_DIR / "manifest.json"
PAGE_SIZE = 1000
MAX_RETRIES = 6
REQUEST_TIMEOUT = 60
PAUSE_BETWEEN_PAGES = 0.5  # seconds; be polite to the public API

# Only the design-time fields build_features reads. Projecting keeps the raw
# download small and means leaky fields (enrollment, results...) never touch disk.
# Use --full to download whole records instead.
FIELDS = [
    "NCTId", "OverallStatus", "StartDate", "Phase", "DesignAllocation",
    "DesignInterventionModel", "DesignPrimaryPurpose", "DesignMasking",
    "ArmGroupLabel", "InterventionType", "LeadSponsorClass", "CollaboratorName",
    "OversightHasDMC", "IsFDARegulatedDrug", "IsFDARegulatedDevice",
    "HealthyVolunteers", "Sex", "MinimumAge", "MaximumAge",
    "EligibilityCriteria", "PrimaryOutcomeMeasure", "SecondaryOutcomeMeasure",
    "Condition",
    # Display-only (never features): shown in the app's trial cards and tables.
    "BriefTitle", "LeadSponsorName",
]


def base_params(full: bool) -> dict:
    """Query parameters shared by every page of the download."""
    # Every interventional trial that is completed or terminated, 1,000 per page.
    params = {
        "format": "json",
        "pageSize": PAGE_SIZE,
        "filter.overallStatus": "COMPLETED,TERMINATED",
        "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
        "countTotal": "true",
    }
    if not full:
        params["fields"] = ",".join(FIELDS)
    return params


def get_with_retry(session: requests.Session, params: dict) -> dict:
    """GET one page, retrying 429/5xx/network errors with exponential backoff."""
    # Exponential back-off: 2, 4, 8, ... seconds between attempts.
    for attempt in range(MAX_RETRIES + 1):
        wait = 2 ** (attempt + 1)
        try:
            resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                # Respect the server's own hint when it gives one.
                wait = max(wait, int(resp.headers.get("Retry-After", 0) or 0))
                reason = f"HTTP {resp.status_code}"
            else:
                # 4xx other than 429 will not fix itself: fail loudly with the body.
                raise SystemExit(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            reason = type(exc).__name__
        if attempt == MAX_RETRIES:
            raise SystemExit(f"Giving up after {MAX_RETRIES} retries ({reason}).")
        print(f"  {reason}; retry {attempt + 1}/{MAX_RETRIES} in {wait}s", flush=True)
        time.sleep(wait)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write via a temp file so an interrupted run never leaves a half page."""
    # Write to a temporary file first and rename it, so an interrupted run never leaves half a page behind.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def resume_state(params: dict, restart: bool):
    """Return (next page number, next page token) from what is already on disk."""
    # Resume logic: count the pages already on disk and continue from the token stored in the last one.
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(PAGES_DIR.glob("page_*.json"))
    if restart:
        for page in pages:
            page.unlink()
        MANIFEST.unlink(missing_ok=True)
        pages = []
    # A resume must use the same query, otherwise the pages would mix two different downloads.
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        if old.get("params") != params:
            raise SystemExit("Existing pages used different query parameters; "
                             "rerun with --restart.")
    if not pages:
        write_json_atomic(MANIFEST, {"params": params, "started": time.ctime()})
        return 1, None
    last = json.loads(pages[-1].read_text(encoding="utf-8"))
    number = int(pages[-1].stem.split("_")[1])
    return number + 1, last.get("nextPageToken")


def main() -> None:
    """Download pages until the API stops returning a nextPageToken."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--restart", action="store_true", help="delete pages and start over")
    parser.add_argument("--full", action="store_true", help="download whole records")
    args = parser.parse_args()

    # Where to start: page 1 with no token, or the next page after the last one on disk.
    params = base_params(args.full)
    number, token = resume_state(params, args.restart)
    if number > 1 and token is None:
        print(f"Already complete ({number - 1} pages in {PAGES_DIR}).")
        return
    if number > 1:
        print(f"Resuming at page {number}.")

    session = requests.Session()
    # Generic client name; no personal contact details.
    session.headers["User-Agent"] = "trial-radar-student-project/0.1"
    fetched, total, start = 0, None, time.time()

    # Page loop: fetch, save, print progress with an estimate of the time left, follow the next token.
    while True:
        query = dict(params)
        if token:
            query["pageToken"] = token
        data = get_with_retry(session, query)
        write_json_atomic(PAGES_DIR / f"page_{number:05d}.json", data)

        fetched += len(data.get("studies", []))
        total = data.get("totalCount", total)
        elapsed = time.time() - start
        eta = f", ETA {elapsed / fetched * (total - fetched) / 60:.0f} min" \
            if total and fetched and total > fetched else ""
        print(f"page {number}: {fetched} studies this run"
              f"{f' of ~{total} total' if total else ''}{eta}", flush=True)

        token = data.get("nextPageToken")
        if not token:
            print("Done.")
            return
        number += 1
        time.sleep(PAUSE_BETWEEN_PAGES)


if __name__ == "__main__":
    sys.exit(main())

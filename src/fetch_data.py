"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Download all COMPLETED/TERMINATED interventional studies from ClinicalTrials.gov.

The API gives the studies in pages of 1,000. Every page is saved unchanged as data/raw/pages/page_NNNNN.json.
Each page also holds a "nextPageToken", the key to the page after it. So when the download is interrupted, we can
run the script again and it goes on after the last saved page.

How to run it:
    python src/fetch_data.py            # download, or go on with an interrupted download
    python src/fetch_data.py --restart  # delete the saved pages and start again
    python src/fetch_data.py --full     # download whole records, not only the fields in FIELDS
"""

# ---- 1. Imports ----
import argparse  # read the options --restart and --full
import json  # read and write the saved pages
import time  # pauses, waiting times and the time estimate
from pathlib import Path  # file paths

import requests  # download the pages from the API

# ---- 2. Settings ----
API_URL = "https://clinicaltrials.gov/api/v2/studies"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PAGES_DIR = RAW_DIR / "pages"
MANIFEST = RAW_DIR / "manifest.json"  # remembers with which query the saved pages were downloaded
PAGE_SIZE = 1000
MAX_RETRIES = 6
REQUEST_TIMEOUT = 60  # seconds
PAUSE_BETWEEN_PAGES = 0.5  # seconds, so we do not flood the public API

# We only ask for the fields that build_features() reads. This keeps the download small, and fields that would
# leak the outcome (enrollment, results, ...) are never even saved. With --full we get whole records instead.
FIELDS = [
    "NCTId", "OverallStatus", "StartDate", "Phase", "DesignAllocation",
    "DesignInterventionModel", "DesignPrimaryPurpose", "DesignMasking",
    "ArmGroupLabel", "InterventionType", "LeadSponsorClass", "CollaboratorName",
    "OversightHasDMC", "IsFDARegulatedDrug", "IsFDARegulatedDevice",
    "HealthyVolunteers", "Sex", "MinimumAge", "MaximumAge",
    "EligibilityCriteria", "PrimaryOutcomeMeasure", "SecondaryOutcomeMeasure",
    "Condition",
    # Only shown in the app (trial cards and tables), never used as features.
    "BriefTitle", "LeadSponsorName",
]


def base_params(full):
    """The query parameters that are the same for every page of the download.

    Input: full is True to download whole records, False to ask only for the fields in FIELDS (boolean).

    Output: a dictionary with the query parameters of the API.
    """
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


def get_with_retry(session, params):
    """Download one page and return it as a dictionary. If it fails, wait and try again (at most 6 more times).

    Worth another try: the server is busy (code 429), has an error of its own (500 and above), or the connection
    broke. Any other error code would not go away by waiting, so the script stops with the message of the server.

    Input: session is the requests.Session used for the download; params is the dictionary of query parameters
    for this page (including the page token, if there is one).

    Output: the page as a dictionary (the parsed JSON). The script stops with an error message if all tries fail.
    """
    # Try the download again and again until it works or the tries are used up.
    for attempt in range(MAX_RETRIES + 1):
        # We wait longer after every failed try: 2, 4, 8, ... seconds.
        wait = 2 ** (attempt + 1)
        try:
            response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                # Sometimes the server says how long we should wait. Then we wait at least that long.
                server_wait = int(response.headers.get("Retry-After", 0) or 0)
                if server_wait > wait:
                    wait = server_wait
                reason = f"HTTP {response.status_code}"
            else:
                raise SystemExit(f"HTTP {response.status_code}: {response.text[:500]}")
        except (requests.ConnectionError, requests.Timeout, ValueError) as error:
            # ValueError: the answer was not valid JSON (for example a page that was cut off).
            reason = type(error).__name__

        if attempt == MAX_RETRIES:
            raise SystemExit(f"Giving up after {MAX_RETRIES} retries ({reason}).")
        print(f"  {reason}; retry {attempt + 1}/{MAX_RETRIES} in {wait}s", flush=True)
        time.sleep(wait)


def write_json(path, content):
    """Save content as a JSON file.

    We write to a temporary file first and rename it at the end. So if the script is stopped in the middle of
    writing, no half page is left behind under the real name.

    Input: path is the file to write (Path); content is the data to save (dictionary or list that JSON can store).

    Output: nothing. It writes the file.
    """
    # Write under a temporary name, then rename.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content), encoding="utf-8")
    temporary.replace(path)


def resume_state(params, restart):
    """Look at the pages that are already saved and return where to go on: (next page number, its token).

    A new download starts at (1, None). With restart=True the saved pages are deleted first.

    Input: params is the query dictionary of this download (dictionary); restart is True to delete the saved
    pages first (boolean).

    Output: a tuple (page number, token) for the next page to download. The token is None for page 1 and also
    after the last page, when there is nothing more to download.
    """
    # Find the saved pages; with --restart delete them and the manifest.
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(PAGES_DIR.glob("page_*.json"))
    if restart:
        for page in pages:
            page.unlink()
        MANIFEST.unlink(missing_ok=True)
        pages = []

    # We can only go on with the same query. Otherwise the pages of two different downloads would be mixed.
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        if old.get("params") != params:
            raise SystemExit("Existing pages used different query parameters; rerun with --restart.")

    # Nothing saved yet: note the query and start with page 1.
    if not pages:
        write_json(MANIFEST, {"params": params, "started": time.ctime()})
        return 1, None

    # Go on after the last saved page. Its number is in the file name ("page_00012" -> 12), and the page itself
    # holds the token of the next one. The last page of all has no token.
    last_page = json.loads(pages[-1].read_text(encoding="utf-8"))
    number = int(pages[-1].stem.split("_")[1])
    return number + 1, last_page.get("nextPageToken")


def main():
    """Download page after page until the API gives no token for a next page.

    Input: none (the options come from the command line).

    Output: nothing. It saves the pages in data/raw/pages/ and prints the progress.
    """
    # Read the command-line options.
    parser = argparse.ArgumentParser(
        description="Download all COMPLETED/TERMINATED interventional studies from ClinicalTrials.gov.")
    parser.add_argument("--restart", action="store_true", help="delete pages and start over")
    parser.add_argument("--full", action="store_true", help="download whole records")
    args = parser.parse_args()

    # Where do we start? At page 1, or after the last page that is already saved.
    params = base_params(args.full)
    number, token = resume_state(params, args.restart)
    if number > 1 and token is None:
        print(f"Already complete ({number - 1} pages in {PAGES_DIR}).")
        return
    if number > 1:
        print(f"Resuming at page {number}.")

    # One session for all pages, so the connection is reused.
    session = requests.Session()
    # The API wants to know who is asking. A general name is enough, no personal contact details.
    session.headers["User-Agent"] = "trial-radar-student-project/0.1"

    # Counters for the progress message.
    fetched = 0  # studies downloaded in this run
    total = None  # studies the API has in all (it tells us with every page)
    start = time.time()

    # The download loop.
    while True:
        # Download one page and save it. Every page after the first one needs the token of the page before.
        query = dict(params)
        if token:
            query["pageToken"] = token
        data = get_with_retry(session, query)
        write_json(PAGES_DIR / f"page_{number:05d}.json", data)

        # Print the progress. If we know the total, we also estimate the minutes left from the speed so far.
        fetched += len(data.get("studies", []))
        total = data.get("totalCount", total)
        elapsed = time.time() - start
        message = f"page {number}: {fetched} studies this run"
        if total:
            message += f" of ~{total} total"
        if total and fetched and total > fetched:
            minutes_left = elapsed / fetched * (total - fetched) / 60
            message += f", ETA {minutes_left:.0f} min"
        print(message, flush=True)

        # Go to the next page, or stop if this was the last one.
        token = data.get("nextPageToken")
        if not token:
            print("Done.")
            return
        number += 1
        time.sleep(PAUSE_BETWEEN_PAGES)


# ---- 3. Run the script ----
if __name__ == "__main__":
    main()

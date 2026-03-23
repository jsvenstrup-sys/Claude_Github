#!/usr/bin/env python3
"""
Scrape the ComEd smaller-generator interconnection queue page and save results to CSV.

Source page:
  https://www.comed.com/smart-energy/my-green-power-connection/
      developers-contractors/smaller-generators/interconnection-queue

The page either:
  (a) links to a downloadable Excel/CSV queue file, or
  (b) embeds an HTML table with queue entries.

This script handles both cases:
  1. Downloads a linked spreadsheet if one is found.
  2. Falls back to parsing any HTML tables on the page.

Usage:
    python scrape_interconnection_queue.py [--output OUTPUT.csv]

Examples:
    python scrape_interconnection_queue.py
    python scrape_interconnection_queue.py --output comed_queue.csv
"""

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUEUE_URL = (
    "https://www.comed.com/smart-energy/my-green-power-connection/"
    "developers-contractors/smaller-generators/interconnection-queue"
)

SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv", ".xlsm"}

# Browser-like headers to avoid 403 responses
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.comed.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch_page(url: str, retries: int = 4, timeout: int = 30) -> requests.Response:
    """GET a URL with exponential-backoff retry."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries - 1}] {exc} – waiting {wait}s …")
            time.sleep(wait)


def find_spreadsheet_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Return absolute URLs of any spreadsheet links found on the page."""
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        abs_url = urljoin(base_url, href)
        ext = Path(urlparse(abs_url).path).suffix.lower()
        if ext in SPREADSHEET_EXTENSIONS:
            links.append(abs_url)
    return links


def download_spreadsheet(url: str) -> pd.DataFrame:
    """Download a spreadsheet URL and return its contents as a DataFrame."""
    print(f"  Downloading spreadsheet: {url}")
    resp = fetch_page(url, timeout=60)
    ext = Path(urlparse(url).path).suffix.lower()

    import io
    buf = io.BytesIO(resp.content)

    if ext == ".csv":
        return pd.read_csv(buf)
    else:
        # Try all sheets; if multiple, concatenate with a 'sheet' column
        xl = pd.ExcelFile(buf)
        if len(xl.sheet_names) == 1:
            return xl.parse(xl.sheet_names[0])
        frames = []
        for name in xl.sheet_names:
            df = xl.parse(name)
            if not df.empty:
                df.insert(0, "_sheet", name)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_html_tables(soup: BeautifulSoup) -> list[pd.DataFrame]:
    """Parse all <table> elements from the page into DataFrames."""
    tables = soup.find_all("table")
    frames = []
    for i, tbl in enumerate(tables, start=1):
        try:
            # pd.read_html expects a string or file-like object
            dfs = pd.read_html(str(tbl))
            for df in dfs:
                if not df.empty:
                    frames.append(df)
                    print(f"  Parsed HTML table {i}: {len(df)} rows × {len(df.columns)} cols")
        except ValueError:
            pass  # no parseable table
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("comed_interconnection_queue.csv"),
        help="Output CSV path (default: comed_interconnection_queue.csv)",
    )
    p.add_argument(
        "--url",
        default=QUEUE_URL,
        help="Override the queue page URL",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Fetching queue page: {args.url}")
    resp = fetch_page(args.url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # ── 1. Look for a linked spreadsheet ────────────────────────────────────
    spreadsheet_links = find_spreadsheet_links(soup, args.url)
    df: pd.DataFrame | None = None

    if spreadsheet_links:
        print(f"Found {len(spreadsheet_links)} spreadsheet link(s):")
        for link in spreadsheet_links:
            print(f"  {link}")
        # Use the first link (most relevant)
        df = download_spreadsheet(spreadsheet_links[0])
        print(f"  Loaded {len(df):,} rows from spreadsheet.")

    # ── 2. Fall back to HTML tables ──────────────────────────────────────────
    if df is None or df.empty:
        print("No spreadsheet found (or empty). Parsing HTML tables …")
        tables = parse_html_tables(soup)
        if tables:
            df = pd.concat(tables, ignore_index=True) if len(tables) > 1 else tables[0]
            print(f"  Combined {len(tables)} table(s) → {len(df):,} rows.")
        else:
            print("ERROR: No queue data found on the page.", file=sys.stderr)
            print(
                "The page may require JavaScript rendering. "
                "Try opening the URL in a browser and downloading the queue file manually.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── 3. Save to CSV ───────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} rows → {args.output.resolve()}")

    # ── 4. Preview ───────────────────────────────────────────────────────────
    print("\nColumn names:")
    for col in df.columns:
        print(f"  {col}")
    print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")


if __name__ == "__main__":
    main()

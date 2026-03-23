#!/usr/bin/env python3
"""
Scrape the ComEd smaller-generator interconnection queue page and save results to CSV.

Source page:
  https://www.comed.com/smart-energy/my-green-power-connection/
      developers-contractors/smaller-generators/interconnection-queue

The page renders content via JavaScript, so this script uses Playwright to
load the page in a real browser, then:
  1. Downloads any linked spreadsheet (xlsx/xls/csv).
  2. Falls back to extracting HTML table data from the rendered DOM.

Dependencies:
    pip install playwright pandas openpyxl lxml beautifulsoup4
    playwright install chromium

Usage:
    python scrape_interconnection_queue.py [--output OUTPUT.csv] [--headless]

Examples:
    python scrape_interconnection_queue.py
    python scrape_interconnection_queue.py --output comed_queue.csv
    python scrape_interconnection_queue.py --no-headless   # show browser window
"""

import argparse
import io
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUEUE_URL = (
    "https://www.comed.com/smart-energy/my-green-power-connection/"
    "developers-contractors/smaller-generators/interconnection-queue"
)

SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv", ".xlsm"}

# How long to wait for the page to finish rendering (seconds)
PAGE_LOAD_TIMEOUT = 30_000   # ms (Playwright uses ms)
CONTENT_WAIT_TIMEOUT = 15_000


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------

def _load_page_with_playwright(url: str, headless: bool, download_dir: Path):
    """
    Navigate to *url* in a Playwright Chromium browser.

    Returns (html_content, downloaded_files) where:
      - html_content: fully-rendered HTML string of the page
      - downloaded_files: list of Paths for any files downloaded via link clicks
    """
    from playwright.sync_api import sync_playwright

    downloaded: list[Path] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print(f"  Opening browser → {url}")
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until="networkidle")

        # Give any lazy-loaded content a moment to settle
        try:
            page.wait_for_load_state("networkidle", timeout=CONTENT_WAIT_TIMEOUT)
        except Exception:
            pass  # proceed even if networkidle times out

        html = page.content()

        # ── Try to click any spreadsheet download links ──────────────────────
        soup_quick = BeautifulSoup(html, "html.parser")
        sheet_hrefs = [
            tag["href"].strip()
            for tag in soup_quick.find_all("a", href=True)
            if Path(urlparse(urljoin(url, tag["href"].strip())).path).suffix.lower()
            in SPREADSHEET_EXTENSIONS
        ]

        for href in sheet_hrefs:
            abs_href = urljoin(url, href)
            print(f"  Found spreadsheet link: {abs_href}")
            try:
                with page.expect_download(timeout=30_000) as dl_info:
                    page.evaluate(f"window.location.href = '{abs_href}'")
                download = dl_info.value
                dest = download_dir / download.suggested_filename
                download.save_as(str(dest))
                downloaded.append(dest)
                print(f"  Downloaded → {dest.name}")
            except Exception as exc:
                print(f"  Could not download {abs_href}: {exc}")

        context.close()
        browser.close()

    return html, downloaded


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_spreadsheet(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    xl = pd.ExcelFile(path)
    if len(xl.sheet_names) == 1:
        return xl.parse(xl.sheet_names[0])
    frames = []
    for name in xl.sheet_names:
        df = xl.parse(name)
        if not df.empty:
            df.insert(0, "_sheet", name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_spreadsheet_bytes(data: bytes, url: str) -> pd.DataFrame:
    ext = Path(urlparse(url).path).suffix.lower()
    buf = io.BytesIO(data)
    if ext == ".csv":
        return pd.read_csv(buf)
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


def _parse_html_tables(html: str) -> list[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    frames = []
    for i, tbl in enumerate(soup.find_all("table"), start=1):
        try:
            dfs = pd.read_html(io.StringIO(str(tbl)))
            for df in dfs:
                if not df.empty:
                    frames.append(df)
                    print(f"  HTML table {i}: {len(df)} rows × {len(df.columns)} cols")
        except ValueError:
            pass
    return frames


# ---------------------------------------------------------------------------
# Fallback: plain requests (no JS)
# ---------------------------------------------------------------------------

def _fetch_with_requests(url: str) -> tuple[str, list[str]]:
    """Return (html, spreadsheet_urls). Used when Playwright is unavailable."""
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.comed.com/",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    sheet_urls = [
        urljoin(url, tag["href"].strip())
        for tag in soup.find_all("a", href=True)
        if Path(urlparse(urljoin(url, tag["href"].strip())).path).suffix.lower()
        in SPREADSHEET_EXTENSIONS
    ]
    return resp.text, sheet_urls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("comed_interconnection_queue.csv"),
        help="Output CSV path (default: comed_interconnection_queue.csv)",
    )
    p.add_argument(
        "--url",
        default=QUEUE_URL,
        help="Override the queue page URL",
    )
    p.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        default=True,
        help="Show the browser window while scraping",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    df: pd.DataFrame | None = None

    # ── Try Playwright (handles JS-rendered pages) ───────────────────────────
    try:
        import playwright  # noqa: F401 – just checking availability
        playwright_available = True
    except ImportError:
        playwright_available = False

    with tempfile.TemporaryDirectory() as tmp:
        dl_dir = Path(tmp)

        if playwright_available:
            print(f"Fetching queue page with Playwright: {args.url}")
            html, downloaded = _load_page_with_playwright(
                args.url, headless=args.headless, download_dir=dl_dir
            )

            # Prefer downloaded spreadsheet files
            for path in downloaded:
                try:
                    df = _parse_spreadsheet(path)
                    if not df.empty:
                        print(f"  Loaded {len(df):,} rows from {path.name}")
                        break
                except Exception as exc:
                    print(f"  Could not parse {path.name}: {exc}")

            # Fall back to HTML tables from rendered DOM
            if df is None or df.empty:
                print("No spreadsheet downloaded. Parsing rendered HTML tables …")
                tables = _parse_html_tables(html)
                if tables:
                    df = pd.concat(tables, ignore_index=True) if len(tables) > 1 else tables[0]
                    print(f"  Extracted {len(df):,} rows from HTML tables.")

        else:
            # ── Fallback: plain requests ─────────────────────────────────────
            print("Playwright not installed. Falling back to plain HTTP fetch.")
            print(f"Fetching: {args.url}")
            import requests as req_lib

            try:
                html, sheet_urls = _fetch_with_requests(args.url)
            except req_lib.HTTPError as exc:
                print(f"ERROR: HTTP {exc.response.status_code} fetching page.", file=sys.stderr)
                print(
                    "Install Playwright for JavaScript-rendered page support:\n"
                    "  pip install playwright && playwright install chromium",
                    file=sys.stderr,
                )
                sys.exit(1)

            for sheet_url in sheet_urls:
                print(f"  Downloading spreadsheet: {sheet_url}")
                try:
                    resp = req_lib.get(sheet_url, timeout=60)
                    resp.raise_for_status()
                    df = _parse_spreadsheet_bytes(resp.content, sheet_url)
                    if not df.empty:
                        print(f"  Loaded {len(df):,} rows.")
                        break
                except Exception as exc:
                    print(f"  Could not download/parse {sheet_url}: {exc}")

            if df is None or df.empty:
                print("No spreadsheet found. Parsing HTML tables …")
                tables = _parse_html_tables(html)
                if tables:
                    df = pd.concat(tables, ignore_index=True) if len(tables) > 1 else tables[0]

    # ── Validate ─────────────────────────────────────────────────────────────
    if df is None or df.empty:
        print("ERROR: No queue data could be extracted.", file=sys.stderr)
        print(
            "The page may block automated access even with a real browser.\n"
            "Try:\n"
            "  1. Run with --no-headless to watch the browser for CAPTCHA prompts.\n"
            "  2. Download the queue spreadsheet manually from the page and load it directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Save ─────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} rows → {args.output.resolve()}")

    # ── Preview ──────────────────────────────────────────────────────────────
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")
    print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")


if __name__ == "__main__":
    main()

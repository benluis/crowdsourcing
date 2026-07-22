"""
Download Kicktraq daily chart images and store metadata in SQLite.
Created: 2026-07-22

Usage:
    python src/scrapers/scrape_kicktraq_charts.py --db PATH [--force] [INPUT_CSV]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import (  # noqa: E402
    connect_db,
    default_db_path,
    extract_slug,
    init_schema,
    kicktraq_charts_complete,
    log_scrape_event,
    upsert_kicktraq_chart,
    upsert_kicktraq_metadata,
    upsert_project,
    utc_now_iso,
)
from scrapers.kicktraq_parser import (  # noqa: E402
    CHART_FILENAMES,
    chart_image_url,
    kickstarter_to_kicktraq_url,
    parse_project_info_from_html,
)
from scrapers.ks_session import CloudflareBlockedError, create_kickstarter_session, fetch_page  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_INPUT_CSV = "data/my_file.csv"
DEFAULT_CHARTS_DIR = "data/kicktraq/charts"
DELAY_SECONDS = 2.5
MIN_PNG_BYTES = 1024


def resolve_url_column(df: pd.DataFrame) -> str | None:
    for col in ("project_url", "url", "combined.url"):
        if col in df.columns:
            return col
    return None


class KicktraqChartScraper:
    def __init__(self):
        self.session, self._http_backend = create_kickstarter_session()

    def fetch_text(self, url: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            try:
                response = fetch_page(self.session, url, max_attempts=1)
                if response is None:
                    continue
                if response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logging.warning("Rate limit on %s; sleeping %ss", url, wait)
                    time.sleep(wait)
                    continue
                if response.status_code == 200:
                    return response.text
                logging.error("HTTP %s for %s", response.status_code, url)
                return None
            except CloudflareBlockedError as exc:
                logging.error("Blocked fetch for %s: %s", url, exc)
                time.sleep(30 * (attempt + 1))
            except Exception as exc:
                logging.error("Request failed for %s: %s", url, exc)
                time.sleep(10 * (attempt + 1))
        return None

    def fetch_binary(self, url: str, max_retries: int = 3) -> tuple[bytes, str] | None:
        for attempt in range(max_retries):
            try:
                response = fetch_page(self.session, url, max_attempts=1)
                if response is None:
                    continue
                if response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logging.warning("Rate limit on %s; sleeping %ss", url, wait)
                    time.sleep(wait)
                    continue
                if response.status_code != 200:
                    logging.error("HTTP %s for %s", response.status_code, url)
                    return None
                content_type = response.headers.get("content-type", "")
                if "image" not in content_type and not url.endswith(".png"):
                    logging.error("Unexpected content-type %s for %s", content_type, url)
                    return None
                data = response.content
                if len(data) < MIN_PNG_BYTES:
                    logging.error("Image too small (%d bytes) for %s", len(data), url)
                    return None
                return data, content_type
            except CloudflareBlockedError as exc:
                logging.error("Blocked image fetch for %s: %s", url, exc)
                time.sleep(30 * (attempt + 1))
            except Exception as exc:
                logging.error("Image request failed for %s: %s", url, exc)
                time.sleep(10 * (attempt + 1))
        return None


def chart_output_dir(base_dir: Path, project_slug: str) -> Path:
    safe_slug = project_slug.replace("/", "__")
    return base_dir / safe_slug


def should_skip_project(conn, project_id: str, force: bool) -> bool:
    if force:
        return False
    return kicktraq_charts_complete(conn, project_id)


def scrape_project(
    conn,
    scraper: KicktraqChartScraper,
    project_id: str,
    kickstarter_url: str,
    charts_dir: Path,
    *,
    force: bool = False,
) -> bool:
    if should_skip_project(conn, project_id, force):
        logging.info("Skipping project %s (charts already downloaded)", project_id)
        return True

    kicktraq_url = kickstarter_to_kicktraq_url(kickstarter_url)
    if not kicktraq_url:
        logging.warning("Could not derive Kicktraq URL for project %s", project_id)
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_charts",
            "fetch",
            status="error",
            error_message="missing kicktraq url",
        )
        return False

    slug = extract_slug(kickstarter_url)
    upsert_project(conn, project_id, kickstarter_url, date_added=utc_now_iso())

    html_text = scraper.fetch_text(kicktraq_url)
    if html_text is None:
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_charts",
            "fetch",
            status="error",
            error_message="failed to fetch kicktraq page",
        )
        return False

    try:
        info = parse_project_info_from_html(html_text, kicktraq_url)
    except ValueError as exc:
        logging.error("Failed to parse Kicktraq page for %s: %s", project_id, exc)
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_charts",
            "parse",
            status="error",
            error_message=str(exc),
        )
        return False

    upsert_kicktraq_metadata(conn, project_id, asdict(info))

    out_dir = chart_output_dir(charts_dir, slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for chart_type, filename in CHART_FILENAMES.items():
        image_url = chart_image_url(kicktraq_url, chart_type)
        result = scraper.fetch_binary(image_url)
        if result is None:
            logging.warning("Failed to download %s for project %s", chart_type, project_id)
            continue

        data, content_type = result
        file_path = out_dir / filename
        file_path.write_bytes(data)
        upsert_kicktraq_chart(
            conn,
            project_id,
            chart_type,
            source_url=image_url,
            file_path=str(file_path.as_posix()),
            file_size=len(data),
            content_type=content_type,
        )
        downloaded += 1
        logging.info("Saved %s (%d bytes) for project %s", file_path, len(data), project_id)

    success = downloaded == len(CHART_FILENAMES)
    log_scrape_event(
        conn,
        project_id,
        "kicktraq_charts",
        "download",
        rows_fetched=downloaded,
        expected_count=len(CHART_FILENAMES),
        status="complete" if success else "partial",
        error_message="" if success else f"downloaded {downloaded}/{len(CHART_FILENAMES)} charts",
    )
    return success


def process_projects(
    conn,
    df: pd.DataFrame,
    url_col: str,
    scraper: KicktraqChartScraper,
    charts_dir: Path,
    *,
    force: bool = False,
    delay: float = DELAY_SECONDS,
) -> tuple[int, int, int]:
    processed = 0
    skipped = 0
    failed = 0
    for _, row in df.iterrows():
        project_id = str(row.get("id", "")).strip()
        project_url = str(row.get(url_col, "")).strip()
        if not project_id or not project_url:
            continue
        if should_skip_project(conn, project_id, force):
            skipped += 1
            continue

        ok = scrape_project(
            conn,
            scraper,
            project_id,
            project_url,
            charts_dir,
            force=force,
        )
        conn.commit()
        if ok:
            processed += 1
        else:
            failed += 1
        time.sleep(delay + random.uniform(0, 1))
    return processed, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kicktraq daily chart images into SQLite")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--db", default=None)
    parser.add_argument("--charts-dir", default=DEFAULT_CHARTS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download charts for all projects")
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logging.error("Input CSV not found: %s", args.input_csv)
        sys.exit(1)

    db_path = Path(args.db) if args.db else default_db_path()
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)
    init_schema(conn)

    df = pd.read_csv(args.input_csv)
    url_col = resolve_url_column(df)
    if url_col is None:
        logging.error("No URL column found in %s", args.input_csv)
        sys.exit(1)
    df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
    logging.info("Downloading Kicktraq charts for %d Kickstarter projects", len(df))

    scraper = KicktraqChartScraper()
    processed, skipped, failed = process_projects(
        conn,
        df,
        url_col,
        scraper,
        charts_dir,
        force=args.force,
        delay=args.delay,
    )
    logging.info(
        "Done: processed=%d skipped=%d failed=%d charts_dir=%s db=%s",
        processed,
        skipped,
        failed,
        charts_dir,
        db_path,
    )
    conn.close()


if __name__ == "__main__":
    main()

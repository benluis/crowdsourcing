"""
Fetch Kickstarter nav comment/update totals and store in SQLite.
Created: 2025-06-15

Usage:
    python src/scrapers/fetch_project_counts.py --db PATH [--force] [INPUT_CSV]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import cloudscraper
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import (  # noqa: E402
    connect_db,
    default_db_path,
    init_schema,
    refresh_completeness_status,
    update_project_ks_counts,
    upsert_project,
    utc_now_iso,
)
from scrapers.nav_counts import parse_nav_counts_from_html  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_INPUT_CSV = "data/my_file.csv"
DELAY_SECONDS = 2.5


def resolve_url_column(df: pd.DataFrame) -> str | None:
    for col in ("project_url", "url", "combined.url"):
        if col in df.columns:
            return col
    return None


class ProjectCountFetcher:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.scraper.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def fetch_page_html(self, url: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            try:
                response = self.scraper.get(url)
                if response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logging.warning("Rate limit on %s; sleeping %ss", url, wait)
                    time.sleep(wait)
                    continue
                if response.status_code == 200:
                    return response.text
                logging.error("HTTP %s for %s", response.status_code, url)
                return None
            except Exception as exc:
                logging.error("Request failed for %s: %s", url, exc)
                time.sleep(10 * (attempt + 1))
        return None


def should_skip_project(conn, project_id: str, force: bool) -> bool:
    if force:
        return False
    row = conn.execute(
        "SELECT ks_counts_fetched_at FROM projects WHERE project_id = ?",
        (str(project_id),),
    ).fetchone()
    return row is not None and row["ks_counts_fetched_at"] is not None


def process_projects(
    conn,
    df: pd.DataFrame,
    url_col: str,
    fetcher: ProjectCountFetcher,
    *,
    force: bool = False,
    delay: float = DELAY_SECONDS,
) -> tuple[int, int]:
    processed = 0
    skipped = 0
    for _, row in df.iterrows():
        project_id = str(row.get("id", "")).strip()
        project_url = str(row.get(url_col, "")).strip()
        if not project_id or not project_url:
            continue
        if should_skip_project(conn, project_id, force):
            skipped += 1
            continue

        upsert_project(conn, project_id, project_url, date_added=utc_now_iso())
        html_text = fetcher.fetch_page_html(project_url)
        if html_text is None:
            logging.warning("Failed to fetch counts for %s", project_id)
            conn.commit()
            time.sleep(delay + random.uniform(0, 1))
            continue

        counts = parse_nav_counts_from_html(html_text)
        update_project_ks_counts(
            conn,
            project_id,
            ks_comments_nav=counts.ks_comments_nav,
            ks_comments_emoji=counts.ks_comments_emoji,
            ks_updates_nav=counts.ks_updates_nav,
        )
        conn.commit()
        processed += 1
        logging.info(
            "Project %s: comments_nav=%s updates_nav=%s",
            project_id,
            counts.ks_comments_nav,
            counts.ks_updates_nav,
        )
        time.sleep(delay + random.uniform(0, 1))
    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Kickstarter nav counts into SQLite")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--db", default=None)
    parser.add_argument("--force", action="store_true", help="Re-fetch counts for all projects")
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logging.error("Input CSV not found: %s", args.input_csv)
        sys.exit(1)

    db_path = Path(args.db) if args.db else default_db_path()
    conn = connect_db(db_path)
    init_schema(conn)

    df = pd.read_csv(args.input_csv)
    url_col = resolve_url_column(df)
    if url_col is None:
        logging.error("No URL column found in %s", args.input_csv)
        sys.exit(1)
    df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
    logging.info("Fetching nav counts for %d Kickstarter projects", len(df))

    fetcher = ProjectCountFetcher()
    processed, skipped = process_projects(
        conn, df, url_col, fetcher, force=args.force, delay=args.delay
    )
    report = refresh_completeness_status(conn)
    logging.info(
        "Done: processed=%d skipped=%d completeness_rows=%d db=%s",
        processed,
        skipped,
        len(report),
        db_path,
    )
    conn.close()


if __name__ == "__main__":
    main()

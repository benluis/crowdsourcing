"""
SQLite-backed Kickstarter updates scraper with completeness-aware queue.
Created: 2025-06-15

Usage:
    python src/scrapers/scrape_updates_sqlite.py --db PATH [--status partial,missing] [INPUT_CSV]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import (  # noqa: E402
    connect_db,
    default_db_path,
    get_projects_needing_scrape,
    init_schema,
    log_scrape_event,
    refresh_completeness_status,
    refresh_scraped_counts,
    replace_updates_for_project,
    upsert_project,
    utc_now_iso,
)
from scrapers.scrape_updates import KickstarterUpdatesScraper  # noqa: E402
from scrapers.ks_session import CloudflareBlockedError, CsrfTokenError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_INPUT_CSV = "data/my_file.csv"
MAX_RUNTIME_SECONDS = 9.8 * 24 * 3600
MAX_CONSECUTIVE_BLOCKS = 3


def resolve_url_column(df: pd.DataFrame) -> str | None:
    for col in ("project_url", "url", "combined.url"):
        if col in df.columns:
            return col
    return None


def seed_projects_from_csv(conn, csv_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    url_col = resolve_url_column(df)
    if url_col is None:
        return
    df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
    now = utc_now_iso()
    for _, row in df.iterrows():
        pid = str(row.get("id", "")).strip()
        if pid:
            upsert_project(conn, pid, str(row.get(url_col, "")), date_added=now)
    conn.commit()


def scrape_project(conn, scraper: KickstarterUpdatesScraper, project_id: str, project_url: str) -> bool:
    """Scrape updates for one project. Returns False if blocked (existing rows kept)."""
    rows = []
    try:
        for update in scraper.fetch_updates_with_body(project_url):
            update["project_id"] = project_id
            rows.append(update)
        replace_updates_for_project(conn, project_id, rows)
        refresh_scraped_counts(conn, project_id=project_id)
        refresh_completeness_status(conn, project_id=project_id)
        conn.execute(
            "UPDATE projects SET last_scraped_at = ? WHERE project_id = ?",
            (utc_now_iso(), project_id),
        )
        log_scrape_event(
            conn,
            project_id,
            "updates",
            "rescrape",
            rows_fetched=len(rows),
            status="Success",
        )
        conn.commit()
        logging.info("Project %s: stored %d updates", project_id, len(rows))
        return True
    except (CloudflareBlockedError, CsrfTokenError) as exc:
        conn.rollback()
        log_scrape_event(
            conn,
            project_id,
            "updates",
            "rescrape",
            rows_fetched=0,
            status="Blocked",
            error_message=str(exc),
        )
        conn.commit()
        logging.error(
            "Blocked for project %s (%s); keeping existing updates",
            project_id,
            exc,
        )
        return False
    except Exception as exc:
        conn.rollback()
        log_scrape_event(
            conn,
            project_id,
            "updates",
            "rescrape",
            rows_fetched=0,
            status="Failed",
            error_message=str(exc),
        )
        conn.commit()
        logging.error("Failed project %s: %s", project_id, exc)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape updates into SQLite")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--status",
        default="partial,missing",
        help="Comma-separated updates_status values to rescrape",
    )
    parser.add_argument("--all", action="store_true", help="Scrape all projects from input CSV")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else default_db_path()
    conn = connect_db(db_path)
    init_schema(conn)
    seed_projects_from_csv(conn, args.input_csv)

    if args.all:
        df = pd.read_csv(args.input_csv)
        url_col = resolve_url_column(df)
        if url_col is None:
            logging.error("No URL column in %s", args.input_csv)
            sys.exit(1)
        queue = [
            (str(r["id"]), str(r[url_col]))
            for _, r in df.iterrows()
            if str(r.get("id", "")).strip() and str(r.get(url_col, "")).strip()
        ]
    else:
        statuses = [s.strip() for s in args.status.split(",") if s.strip()]
        rows = get_projects_needing_scrape(conn, "updates", statuses=statuses)
        queue = [(r["project_id"], r["project_url"]) for r in rows]

    logging.info("Updates scrape queue: %d projects", len(queue))
    scraper = KickstarterUpdatesScraper()
    start = time.time()
    consecutive_blocks = 0

    for i, (project_id, project_url) in enumerate(queue, 1):
        if time.time() - start > MAX_RUNTIME_SECONDS:
            logging.warning("Max runtime reached; exiting gracefully")
            break
        if not project_url:
            continue
        logging.info("Scraping updates %d/%d: %s", i, len(queue), project_id)
        ok = scrape_project(conn, scraper, project_id, project_url)
        if not ok:
            consecutive_blocks += 1
            if consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                logging.error(
                    "Stopping: Cloudflare blocked %d projects in a row. "
                    "Install curl_cffi on the compute node or retry later.",
                    consecutive_blocks,
                )
                break
        else:
            consecutive_blocks = 0

    conn.close()
    logging.info("Updates SQLite scrape complete")


if __name__ == "__main__":
    main()

"""
SQLite-backed Kickstarter comments scraper with completeness-aware queue.
Created: 2025-06-15

Usage:
    python src/scrapers/scrape_comments_sqlite.py --db PATH [--status partial,missing] [INPUT_CSV]
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
    replace_comments_for_project,
    upsert_project,
    utc_now_iso,
)
from scrapers.scrape_comments import KickstarterCommentsScraper  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_INPUT_CSV = "data/my_file.csv"
MAX_RUNTIME_SECONDS = 9.8 * 24 * 3600


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


def scrape_project(conn, scraper: KickstarterCommentsScraper, project_id: str, project_url: str) -> None:
    rows = []
    try:
        for comment in scraper.fetch_comments(project_url):
            comment["project_id"] = project_id
            rows.append(comment)
        replace_comments_for_project(conn, project_id, rows)
        refresh_scraped_counts(conn, project_id=project_id)
        refresh_completeness_status(conn, project_id=project_id)
        conn.execute(
            "UPDATE projects SET last_scraped_at = ? WHERE project_id = ?",
            (utc_now_iso(), project_id),
        )
        log_scrape_event(
            conn,
            project_id,
            "comments",
            "rescrape",
            rows_fetched=len(rows),
            status="Success",
        )
        conn.commit()
        logging.info("Project %s: stored %d comments", project_id, len(rows))
    except Exception as exc:
        conn.rollback()
        log_scrape_event(
            conn,
            project_id,
            "comments",
            "rescrape",
            rows_fetched=0,
            status="Failed",
            error_message=str(exc),
        )
        conn.commit()
        logging.error("Failed project %s: %s", project_id, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape comments into SQLite")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--status",
        default="partial,missing",
        help="Comma-separated comments_status values to rescrape",
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
        rows = get_projects_needing_scrape(conn, "comments", statuses=statuses)
        queue = [(r["project_id"], r["project_url"]) for r in rows]

    logging.info("Comments scrape queue: %d projects", len(queue))
    scraper = KickstarterCommentsScraper()
    start = time.time()

    for i, (project_id, project_url) in enumerate(queue, 1):
        if time.time() - start > MAX_RUNTIME_SECONDS:
            logging.warning("Max runtime reached; exiting gracefully")
            break
        if not project_url:
            continue
        logging.info("Scraping comments %d/%d: %s", i, len(queue), project_id)
        scrape_project(conn, scraper, project_id, project_url)

    conn.close()
    logging.info("Comments SQLite scrape complete")


if __name__ == "__main__":
    main()

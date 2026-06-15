"""
One-time migration: ingest existing batch CSVs into SQLite.
Created: 2025-06-15

Usage:
    python src/processing/consolidate_to_sqlite.py [--db PATH] [INPUT_CSV]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import (  # noqa: E402
    connect_db,
    default_db_path,
    init_schema,
    insert_comment,
    insert_update,
    log_scrape_event,
    parse_batch_timestamp,
    refresh_completeness_status,
    refresh_scraped_counts,
    upsert_project,
    utc_now_iso,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SCRAPED_COMMENTS_DIR = "data/scraped"
SCRAPED_UPDATES_DIR = "data/scraped_updates_only"
DEFAULT_INPUT_CSV = "data/my_file.csv"


def resolve_url_column(df: pd.DataFrame) -> str | None:
    for col in ("project_url", "url", "combined.url"):
        if col in df.columns:
            return col
    return None


def load_projects_from_csv(csv_path: str, conn) -> int:
    if not os.path.exists(csv_path):
        logging.warning("Input CSV not found: %s", csv_path)
        return 0
    df = pd.read_csv(csv_path)
    url_col = resolve_url_column(df)
    if url_col is None:
        logging.warning("No URL column in %s", csv_path)
        return 0
    df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
    now = utc_now_iso()
    count = 0
    for _, row in df.iterrows():
        pid = str(row.get("id", "")).strip()
        if not pid:
            continue
        upsert_project(conn, pid, str(row.get(url_col, "")), date_added=now)
        count += 1
    conn.commit()
    logging.info("Seeded %d projects from %s", count, csv_path)
    return count


def ingest_comment_batches(conn, comments_dir: str) -> tuple[int, int]:
    pattern = os.path.join(comments_dir, "kickstarter_comments_batch_*.csv")
    files = sorted(glob.glob(pattern))
    inserted = 0
    skipped = 0
    for fpath in files:
        batch_ts = parse_batch_timestamp(os.path.basename(fpath)) or utc_now_iso()
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            logging.warning("Could not read %s: %s", fpath, exc)
            continue
        if "project_id" not in df.columns or "id" not in df.columns:
            logging.warning("Skipping %s: missing id/project_id columns", fpath)
            continue
        for _, row in df.iterrows():
            upsert_project(conn, str(row["project_id"]), date_added=batch_ts)
            if insert_comment(conn, row.to_dict(), date_added=batch_ts):
                inserted += 1
            else:
                skipped += 1
        conn.commit()
    logging.info("Comments: inserted=%d duplicates_skipped=%d from %d files", inserted, skipped, len(files))
    return inserted, skipped


def ingest_update_batches(conn, updates_dir: str) -> tuple[int, int]:
    pattern = os.path.join(updates_dir, "kickstarter_updates_full_batch_*.csv")
    files = sorted(glob.glob(pattern))
    inserted = 0
    skipped = 0
    for fpath in files:
        batch_ts = parse_batch_timestamp(os.path.basename(fpath)) or utc_now_iso()
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            logging.warning("Could not read %s: %s", fpath, exc)
            continue
        if "project_id" not in df.columns or "id" not in df.columns:
            logging.warning("Skipping %s: missing id/project_id columns", fpath)
            continue
        for _, row in df.iterrows():
            upsert_project(conn, str(row["project_id"]), date_added=batch_ts)
            if insert_update(conn, row.to_dict(), date_added=batch_ts):
                inserted += 1
            else:
                skipped += 1
        conn.commit()
    logging.info("Updates: inserted=%d duplicates_skipped=%d from %d files", inserted, skipped, len(files))
    return inserted, skipped


def ingest_summary_batches(conn, summary_dir: str, summary_glob: str, scrape_type: str, count_col: str) -> None:
    pattern = os.path.join(summary_dir, summary_glob)
    for fpath in glob.glob(pattern):
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            logging.warning("Could not read summary %s: %s", fpath, exc)
            continue
        for _, row in df.iterrows():
            pid = str(row.get("id", row.get("project_id", ""))).strip()
            if not pid:
                continue
            url = str(row.get("project_url", ""))
            upsert_project(conn, pid, url)
            status = str(row.get("status", ""))
            count = row.get(count_col)
            try:
                count_int = int(count) if pd.notna(count) else None
            except (TypeError, ValueError):
                count_int = None
            log_scrape_event(
                conn,
                pid,
                scrape_type,
                "migrate_summary",
                rows_fetched=count_int,
                status=status,
            )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate batch CSVs into SQLite")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--db", default=None, help="SQLite DB path (default: data/kickstarter/kickstarter_YYYYMMDD.db)")
    parser.add_argument("--comments-dir", default=SCRAPED_COMMENTS_DIR)
    parser.add_argument("--updates-dir", default=SCRAPED_UPDATES_DIR)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else default_db_path()
    source_globs = f"{args.comments_dir}/*;{args.updates_dir}/*"
    conn = connect_db(db_path)
    init_schema(conn, source_globs=source_globs)

    logging.info("Consolidating into %s", db_path)
    load_projects_from_csv(args.input_csv, conn)
    ingest_summary_batches(conn, args.comments_dir, "kickstarter_summary_batch_*.csv", "comments", "comments_count")
    ingest_summary_batches(
        conn, args.updates_dir, "kickstarter_updates_summary_batch_*.csv", "updates", "updates_count"
    )
    ingest_comment_batches(conn, args.comments_dir)
    ingest_update_batches(conn, args.updates_dir)

    refresh_scraped_counts(conn)
    report = refresh_completeness_status(conn)
    logging.info("Completeness computed for %d projects", len(report))
    conn.execute(
        "UPDATE meta SET last_migration_at = ? WHERE id = 1",
        (utc_now_iso(),),
    )
    conn.commit()
    conn.close()
    logging.info("Migration complete: %s", db_path)


if __name__ == "__main__":
    main()

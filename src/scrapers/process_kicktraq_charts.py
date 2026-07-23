"""
Extract per-day Kicktraq metrics from downloaded chart images via Gemini vision.
Created: 2026-07-22

Usage:
    python src/scrapers/process_kicktraq_charts.py --db PATH [--force] [PROJECT_ID ...]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.env_config import GeminiSettings, get_gemini_settings  # noqa: E402
from processing.kicktraq_vision import (  # noqa: E402
    daily_rows_to_records,
    extract_project_daily_rows,
    get_vision_model_name,
)
from processing.sqlite_schema import (  # noqa: E402
    checkpoint_db,
    connect_db,
    default_db_path,
    get_kicktraq_metadata,
    get_projects_with_kicktraq_charts,
    init_schema,
    kicktraq_daily_complete,
    log_scrape_event,
    replace_kicktraq_daily_for_project,
    utc_now_iso,
)
from scrapers.kicktraq_parser import CHART_FILENAMES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_CHARTS_DIR = "data/kicktraq/charts"
DEFAULT_DELAY_SECONDS = 0.5
CHECKPOINT_EVERY = 100


def resolve_chart_dir(charts_dir: Path, project_slug: str) -> Path:
    safe_slug = project_slug.replace("/", "__")
    return charts_dir / safe_slug


def chart_dir_from_db(conn, project_id: str, charts_dir: Path) -> Path | None:
    row = conn.execute(
        "SELECT file_path FROM kicktraq_charts WHERE project_id = ? LIMIT 1",
        (str(project_id),),
    ).fetchone()
    if row and row["file_path"]:
        return Path(row["file_path"]).parent
    meta = get_kicktraq_metadata(conn, project_id)
    if meta and meta["kicktraq_url"]:
        slug = meta["kicktraq_url"].split("/projects/")[1].strip("/").rstrip("/")
        candidate = resolve_chart_dir(charts_dir, slug)
        if candidate.is_dir():
            return candidate
    return None


def process_project(
    conn,
    project_id: str,
    charts_dir: Path,
    settings: GeminiSettings,
    *,
    force: bool = False,
) -> bool:
    if not force and kicktraq_daily_complete(conn, project_id):
        return True

    meta = get_kicktraq_metadata(conn, project_id)
    if meta is None:
        logging.warning("No kicktraq_metadata for project %s", project_id)
        return False
    if not meta["start_date"] or not meta["end_date"]:
        logging.warning(
            "Project %s missing campaign dates in kicktraq_metadata", project_id
        )
        return False

    chart_dir = chart_dir_from_db(conn, project_id, charts_dir)
    if chart_dir is None:
        logging.warning("Could not locate chart directory for project %s", project_id)
        return False

    for chart_type, filename in CHART_FILENAMES.items():
        if not (chart_dir / filename).is_file():
            logging.warning(
                "Missing %s for project %s at %s", filename, project_id, chart_dir
            )
            return False

    try:
        rows = extract_project_daily_rows(
            chart_dir,
            start_date=meta["start_date"],
            end_date=meta["end_date"],
            campaign_days=meta["campaign_days"],
            settings=settings,
        )
    except Exception as exc:
        logging.error("Vision extraction failed for %s: %s", project_id, exc)
        log_scrape_event(
            conn,
            project_id,
            "kicktraq_daily",
            "extract",
            status="error",
            error_message=str(exc),
        )
        return False

    records = daily_rows_to_records(
        project_id,
        rows,
        model=get_vision_model_name(settings),
    )
    inserted = replace_kicktraq_daily_for_project(conn, project_id, records)
    log_scrape_event(
        conn,
        project_id,
        "kicktraq_daily",
        "extract",
        rows_fetched=inserted,
        status="complete",
    )
    logging.info("Project %s: stored %d daily rows", project_id, inserted)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-day Kicktraq metrics from chart images using Gemini vision"
    )
    parser.add_argument(
        "project_ids", nargs="*", help="Optional project IDs to process"
    )
    parser.add_argument("--db", default=None)
    parser.add_argument("--charts-dir", default=DEFAULT_CHARTS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Re-extract even if daily rows exist"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Seconds to wait between projects (rate limiting)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N projects (0 = no limit)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Run WAL checkpoint every N committed projects",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else default_db_path()
    charts_dir = Path(args.charts_dir)
    settings = get_gemini_settings(require_key=True)

    conn = connect_db(db_path)
    init_schema(conn)

    if args.project_ids:
        targets = args.project_ids
    else:
        rows = get_projects_with_kicktraq_charts(conn, require_daily=False)
        targets = [row["project_id"] for row in rows]

    if args.limit > 0:
        targets = targets[: args.limit]

    if not targets:
        logging.info("No projects with downloaded Kicktraq charts need extraction")
        conn.close()
        return

    logging.info(
        "Processing %d projects with model=%s db=%s",
        len(targets),
        settings.model,
        db_path,
    )

    processed = 0
    skipped = 0
    failed = 0
    for index, project_id in enumerate(
        tqdm(targets, desc="Kicktraq extract", unit="project"),
        start=1,
    ):
        if not args.force and kicktraq_daily_complete(conn, project_id):
            skipped += 1
            continue

        ok = process_project(
            conn,
            project_id,
            charts_dir,
            settings,
            force=args.force,
        )
        conn.commit()
        if ok:
            processed += 1
        else:
            failed += 1

        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            checkpoint_db(conn)

        if args.delay > 0 and index < len(targets):
            time.sleep(args.delay + random.uniform(0, args.delay * 0.25))

    checkpoint_db(conn)
    logging.info(
        "Done: processed=%d skipped=%d failed=%d total=%d db=%s extracted_at=%s",
        processed,
        skipped,
        failed,
        len(targets),
        db_path,
        utc_now_iso(),
    )
    conn.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

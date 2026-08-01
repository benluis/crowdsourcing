"""
Full recompile: batch CSVs + all SQLite sources -> one canonical DB.
Created: 2025-06-15

Usage:
    python src/processing/recompile_kickstarter_db.py --db data/kickstarter/kickstarter_main.db [INPUT_CSV]

Steps:
  1. Fresh schema
  2. Seed projects from input CSV
  3. Ingest all comment/update batch CSVs
  4. Merge every other kickstarter*.db in the same directory
  5. Overlay rescraped comment rows from the richest DB (latest last_scraped_at wins)
  6. Refresh counts + completeness report
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.audit_completeness import REPORT_DIR  # noqa: E402
from processing.consolidate_to_sqlite import (  # noqa: E402
    DEFAULT_INPUT_CSV,
    SCRAPED_COMMENTS_DIR,
    SCRAPED_UPDATES_DIR,
    ingest_comment_batches,
    ingest_summary_batches,
    ingest_update_batches,
    load_projects_from_csv,
)
from processing.sqlite_schema import (  # noqa: E402
    connect_db,
    init_schema,
    refresh_completeness_status,
    refresh_scraped_counts,
    utc_now_iso,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PROJECT_MERGE_COLUMNS = (
    "project_url",
    "project_slug",
    "ks_comments_nav",
    "ks_comments_emoji",
    "ks_comments_api",
    "ks_updates_nav",
    "ks_updates_api",
    "ks_counts_fetched_at",
    "last_scraped_at",
)


def discover_source_dbs(db_path: Path, extra: list[str]) -> list[Path]:
    found: dict[str, Path] = {}
    parent = db_path.parent
    for p in sorted(parent.glob("kickstarter*.db")):
        if p.resolve() == db_path.resolve():
            continue
        found[str(p.resolve())] = p
    for raw in extra:
        p = Path(raw)
        if p.exists() and p.resolve() != db_path.resolve():
            found[str(p.resolve())] = p
    # Prefer larger DBs last so COALESCE fills from smaller first then richer
    return sorted(found.values(), key=lambda p: p.stat().st_size)


def merge_db_rows(conn, source: Path, *, label: str) -> dict[str, int]:
    alias = "srcdb"
    stats = {"comments": 0, "updates": 0, "projects": 0}
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
    try:
        before_c = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        before_u = conn.execute("SELECT COUNT(*) FROM updates").fetchone()[0]
        conn.execute(f"INSERT OR IGNORE INTO main.comments SELECT * FROM {alias}.comments")
        conn.execute(f"INSERT OR IGNORE INTO main.updates SELECT * FROM {alias}.updates")
        conn.execute(f"INSERT OR IGNORE INTO main.projects SELECT * FROM {alias}.projects")
        after_c = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        after_u = conn.execute("SELECT COUNT(*) FROM updates").fetchone()[0]
        stats["comments"] = after_c - before_c
        stats["updates"] = after_u - before_u

        set_clause = ",\n".join(
            f"{col} = COALESCE(main.projects.{col}, "
            f"(SELECT {col} FROM {alias}.projects p WHERE p.project_id = main.projects.project_id))"
            for col in PROJECT_MERGE_COLUMNS
        )
        conn.execute(
            f"""
            UPDATE main.projects SET
            {set_clause},
            date_added = (
                SELECT MIN(x) FROM (
                    SELECT main.projects.date_added AS x
                    UNION ALL
                    SELECT date_added FROM {alias}.projects p
                    WHERE p.project_id = main.projects.project_id
                ) WHERE x IS NOT NULL
            )
            """
        )
        stats["projects"] = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        logging.info(
            "Merged %s: +%d comments, +%d updates, %d project fields filled",
            label,
            stats["comments"],
            stats["updates"],
            stats["projects"],
        )
    finally:
        conn.execute(f"DETACH DATABASE {alias}")
    return stats


def pick_overlay_source(sources: list[Path]) -> Path | None:
    """DB with the most rescraped projects (last_scraped_at)."""
    best: Path | None = None
    best_count = -1
    for path in sources:
        try:
            c = connect_db(path, create=False)
            n = c.execute(
                "SELECT COUNT(*) FROM projects WHERE last_scraped_at IS NOT NULL"
            ).fetchone()[0]
            c.close()
        except Exception:
            continue
        if n > best_count:
            best_count = n
            best = path
    return best


def overlay_rescraped_comments(conn, source: Path) -> int:
    """Replace comments for projects rescraped in source (rescrape wins over batches)."""
    alias = "overlay"
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
    try:
        project_ids = [
            r[0]
            for r in conn.execute(
                f"""
                SELECT project_id FROM {alias}.projects
                WHERE last_scraped_at IS NOT NULL
                """
            ).fetchall()
        ]
        replaced = 0
        for pid in project_ids:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {alias}.comments WHERE project_id = ?", (pid,)
            ).fetchone()[0]
            if count == 0:
                continue
            conn.execute("DELETE FROM main.comments WHERE project_id = ?", (pid,))
            conn.execute(
                f"""
                INSERT INTO main.comments
                SELECT * FROM {alias}.comments WHERE project_id = ?
                """,
                (pid,),
            )
            replaced += 1
        conn.commit()
        logging.info(
            "Overlay rescraped comments from %s: %d projects replaced",
            source.name,
            replaced,
        )
        return replaced
    finally:
        conn.execute(f"DETACH DATABASE {alias}")


def overlay_rescraped_updates(conn, source: Path) -> int:
    alias = "overlay"
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
    try:
        project_ids = [
            r[0]
            for r in conn.execute(
                f"""
                SELECT project_id FROM {alias}.projects
                WHERE last_scraped_at IS NOT NULL
                """
            ).fetchall()
        ]
        replaced = 0
        for pid in project_ids:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {alias}.updates WHERE project_id = ?", (pid,)
            ).fetchone()[0]
            if count == 0:
                continue
            conn.execute("DELETE FROM main.updates WHERE project_id = ?", (pid,))
            conn.execute(
                f"""
                INSERT INTO main.updates
                SELECT * FROM {alias}.updates WHERE project_id = ?
                """,
                (pid,),
            )
            replaced += 1
        conn.commit()
        logging.info(
            "Overlay rescraped updates from %s: %d projects replaced",
            source.name,
            replaced,
        )
        return replaced
    finally:
        conn.execute(f"DETACH DATABASE {alias}")


def print_summary(conn, db_path: Path) -> None:
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM comments),
          (SELECT COUNT(*) FROM updates),
          (SELECT COUNT(*) FROM projects),
          (SELECT COUNT(*) FROM projects WHERE ks_counts_fetched_at IS NOT NULL),
          (SELECT COUNT(*) FROM projects WHERE comments_status='complete'),
          (SELECT COUNT(*) FROM projects WHERE comments_status='partial'),
          (SELECT COUNT(*) FROM projects WHERE updates_status='complete')
        """
    ).fetchone()
    logging.info("=== RECOMPILED %s ===", db_path)
    logging.info(
        "comments=%s updates=%s projects=%s nav_fetched=%s "
        "comments_complete=%s comments_partial=%s updates_complete=%s",
        *row,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompile all Kickstarter data into one DB")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--db",
        default="data/kickstarter/kickstarter_main.db",
        help="Output canonical DB path",
    )
    parser.add_argument("--comments-dir", default=SCRAPED_COMMENTS_DIR)
    parser.add_argument("--updates-dir", default=SCRAPED_UPDATES_DIR)
    parser.add_argument(
        "--merge-db",
        action="append",
        default=[],
        help="Extra SQLite file to merge (repeatable)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip replacing comments/updates for rescraped projects",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If --db exists, move it to .bak before recompiling",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    sources_to_merge: list[Path] = []
    bak_path: Path | None = None

    if db_path.exists():
        if args.backup:
            bak_path = db_path.with_suffix(db_path.suffix + ".bak")
            if bak_path.exists():
                bak_path.unlink()
            try:
                shutil.move(db_path, bak_path)
                logging.info("Backed up existing DB to %s", bak_path)
            except OSError as exc:
                logging.warning(
                    "Could not move %s (%s); writing fresh data in place after delete",
                    db_path,
                    exc,
                )
                try:
                    db_path.unlink()
                except OSError:
                    alt = db_path.with_name(db_path.stem + "_recompiled.db")
                    logging.error(
                        "Database locked. Use --db %s or close programs using the file.",
                        alt,
                    )
                    raise SystemExit(1) from exc
        else:
            try:
                db_path.unlink()
            except OSError as exc:
                alt = db_path.with_name(db_path.stem + "_recompiled.db")
                logging.error(
                    "Cannot replace %s (%s). Try: --db %s",
                    db_path,
                    exc,
                    alt,
                )
                raise SystemExit(1) from exc
            logging.info("Removed existing %s for fresh recompile", db_path)

    sources_to_merge = discover_source_dbs(db_path, args.merge_db)
    if bak_path is not None:
        sources_to_merge.append(bak_path)
    sources_to_merge = sorted(
        {str(p.resolve()): p for p in sources_to_merge}.values(),
        key=lambda p: p.stat().st_size,
    )

    source_globs = f"{args.comments_dir}/*;{args.updates_dir}/*"
    conn = connect_db(db_path)
    init_schema(conn, source_globs=source_globs)

    logging.info("Step 1/6: seed projects from %s", args.input_csv)
    load_projects_from_csv(args.input_csv, conn)

    logging.info("Step 2/6: ingest summary batches")
    ingest_summary_batches(
        conn, args.comments_dir, "kickstarter_summary_batch_*.csv", "comments", "comments_count"
    )
    ingest_summary_batches(
        conn, args.updates_dir, "kickstarter_updates_summary_batch_*.csv", "updates", "updates_count"
    )

    logging.info("Step 3/6: ingest comment/update batch CSVs")
    ingest_comment_batches(conn, args.comments_dir)
    ingest_update_batches(conn, args.updates_dir)

    logging.info("Step 4/6: merge %d SQLite sources", len(sources_to_merge))
    for src in sources_to_merge:
        merge_db_rows(conn, src, label=src.name)

    if not args.no_overlay:
        overlay_src = pick_overlay_source(sources_to_merge)
        if overlay_src:
            logging.info("Step 5/6: overlay rescrapes from %s", overlay_src.name)
            overlay_rescraped_comments(conn, overlay_src)
            overlay_rescraped_updates(conn, overlay_src)
        else:
            logging.info("Step 5/6: no overlay source found")
    else:
        logging.info("Step 5/6: overlay skipped")

    logging.info("Step 6/6: refresh counts and completeness")
    refresh_scraped_counts(conn)
    report = refresh_completeness_status(conn)
    conn.execute("UPDATE meta SET last_migration_at = ? WHERE id = 1", (utc_now_iso(),))
    conn.commit()

    print_summary(conn, db_path)

    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"completeness_report_{datetime.now().strftime('%Y%m%d')}.csv"
    pd.DataFrame(report).to_csv(out_path, index=False)
    logging.info("Completeness report: %s", out_path)

    conn.close()
    logging.info("Recompile complete: %s", db_path)


if __name__ == "__main__":
    main()

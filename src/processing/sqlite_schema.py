"""
SQLite schema and helpers for Kickstarter scrape consolidation.
Created: 2025-06-15
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

COMMENTS_STATUSES = (
    "not_checked",
    "complete",
    "partial",
    "missing",
    "expected_zero",
)
UPDATES_STATUSES = COMMENTS_STATUSES


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_db_path(base_dir: str | Path = "data/kickstarter") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    return base / f"kickstarter_{date_str}.db"


def extract_slug(url: str) -> str:
    if not url:
        return ""
    clean = url.split("?")[0].split("#")[0]
    if "/projects/" not in clean:
        return ""
    return clean.split("/projects/")[1].strip("/")


def parse_batch_timestamp(filename: str) -> Optional[str]:
    """Parse YYYYMMDD_HHMMSS from batch filename into ISO timestamp."""
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if not match:
        return None
    d, t = match.group(1), match.group(2)
    try:
        dt = datetime.strptime(f"{d}_{t}", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def connect_db(db_path: str | Path, create: bool = True) -> sqlite3.Connection:
    path = Path(db_path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection, source_globs: str = "") -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            db_created_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            last_migration_at TEXT,
            source_batch_globs TEXT
        );

        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            project_url TEXT,
            project_slug TEXT,
            ks_comments_nav INTEGER,
            ks_comments_emoji INTEGER,
            ks_comments_api INTEGER,
            ks_updates_nav INTEGER,
            ks_updates_api INTEGER,
            ks_counts_fetched_at TEXT,
            scraped_comments_total INTEGER DEFAULT 0,
            scraped_comments_top_level INTEGER DEFAULT 0,
            scraped_updates_total INTEGER DEFAULT 0,
            scraped_counts_computed_at TEXT,
            comments_status TEXT DEFAULT 'not_checked',
            updates_status TEXT DEFAULT 'not_checked',
            date_added TEXT NOT NULL,
            last_scraped_at TEXT
        );

        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            parent_id TEXT,
            project_slug TEXT,
            author TEXT,
            author_id TEXT,
            body TEXT,
            created_at TEXT,
            date_added TEXT NOT NULL,
            scraped_at TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );

        CREATE TABLE IF NOT EXISTS updates (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            project_slug TEXT,
            title TEXT,
            number INTEGER,
            body TEXT,
            author TEXT,
            author_id TEXT,
            published_at TEXT,
            date_added TEXT NOT NULL,
            scraped_at TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );

        CREATE TABLE IF NOT EXISTS scrape_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            scrape_type TEXT NOT NULL,
            action TEXT NOT NULL,
            rows_fetched INTEGER,
            expected_count INTEGER,
            status TEXT,
            error_message TEXT,
            timestamp TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_comments_project_id ON comments(project_id);
        CREATE INDEX IF NOT EXISTS idx_updates_project_id ON updates(project_id);
        CREATE INDEX IF NOT EXISTS idx_projects_comments_status ON projects(comments_status);
        CREATE INDEX IF NOT EXISTS idx_projects_updates_status ON projects(updates_status);
        """
    )
    now = utc_now_iso()
    row = conn.execute("SELECT id FROM meta WHERE id = 1").fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO meta (id, db_created_at, schema_version, last_migration_at, source_batch_globs)
            VALUES (1, ?, ?, ?, ?)
            """,
            (now, SCHEMA_VERSION, now, source_globs),
        )
    conn.commit()


def checkpoint_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def upsert_project(
    conn: sqlite3.Connection,
    project_id: str,
    project_url: str = "",
    date_added: Optional[str] = None,
) -> None:
    now = date_added or utc_now_iso()
    slug = extract_slug(project_url)
    conn.execute(
        """
        INSERT INTO projects (project_id, project_url, project_slug, date_added)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(project_id) DO UPDATE SET
            project_url = COALESCE(excluded.project_url, projects.project_url),
            project_slug = COALESCE(excluded.project_slug, projects.project_slug),
            date_added = MIN(projects.date_added, excluded.date_added)
        """,
        (str(project_id), project_url or None, slug or None, now),
    )


def _normalize_parent_id(parent_id: Any) -> Optional[str]:
    if parent_id is None:
        return None
    if isinstance(parent_id, float) and str(parent_id) == "nan":
        return None
    s = str(parent_id).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def insert_comment(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    date_added: Optional[str] = None,
) -> bool:
    """Insert comment if new. Returns True if inserted."""
    now = date_added or utc_now_iso()
    scraped_at = row.get("scraped_at") or now
    parent_id = _normalize_parent_id(row.get("parent_id"))

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO comments (
            id, project_id, parent_id, project_slug, author, author_id,
            body, created_at, date_added, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["id"]),
            str(row["project_id"]),
            parent_id,
            row.get("project_slug"),
            row.get("author"),
            str(row.get("author_id")) if row.get("author_id") is not None else None,
            row.get("body"),
            str(row.get("created_at")) if row.get("created_at") is not None else None,
            now,
            str(scraped_at),
        ),
    )
    return cur.rowcount > 0


def insert_update(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    date_added: Optional[str] = None,
) -> bool:
    now = date_added or utc_now_iso()
    scraped_at = row.get("scraped_at") or now
    number = row.get("number")
    if number is not None:
        try:
            number = int(number)
        except (TypeError, ValueError):
            number = None

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO updates (
            id, project_id, project_slug, title, number, body,
            author, author_id, published_at, date_added, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["id"]),
            str(row["project_id"]),
            row.get("project_slug"),
            row.get("title"),
            number,
            row.get("body"),
            row.get("author"),
            str(row.get("author_id")) if row.get("author_id") is not None else None,
            str(row.get("published_at")) if row.get("published_at") is not None else None,
            now,
            str(scraped_at),
        ),
    )
    return cur.rowcount > 0


def replace_comments_for_project(
    conn: sqlite3.Connection,
    project_id: str,
    rows: list[dict[str, Any]],
) -> int:
    now = utc_now_iso()
    conn.execute("DELETE FROM comments WHERE project_id = ?", (str(project_id),))
    inserted = 0
    for row in rows:
        row = {**row, "project_id": str(project_id)}
        conn.execute(
            """
            INSERT INTO comments (
                id, project_id, parent_id, project_slug, author, author_id,
                body, created_at, date_added, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["id"]),
                str(project_id),
                _normalize_parent_id(row.get("parent_id")),
                row.get("project_slug"),
                row.get("author"),
                str(row.get("author_id")) if row.get("author_id") is not None else None,
                row.get("body"),
                str(row.get("created_at")) if row.get("created_at") is not None else None,
                now,
                str(row.get("scraped_at") or now),
            ),
        )
        inserted += 1
    return inserted


def replace_updates_for_project(
    conn: sqlite3.Connection,
    project_id: str,
    rows: list[dict[str, Any]],
) -> int:
    now = utc_now_iso()
    conn.execute("DELETE FROM updates WHERE project_id = ?", (str(project_id),))
    inserted = 0
    for row in rows:
        row = {**row, "project_id": str(project_id)}
        number = row.get("number")
        if number is not None:
            try:
                number = int(number)
            except (TypeError, ValueError):
                number = None
        conn.execute(
            """
            INSERT INTO updates (
                id, project_id, project_slug, title, number, body,
                author, author_id, published_at, date_added, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["id"]),
                str(project_id),
                row.get("project_slug"),
                row.get("title"),
                number,
                row.get("body"),
                row.get("author"),
                str(row.get("author_id")) if row.get("author_id") is not None else None,
                str(row.get("published_at")) if row.get("published_at") is not None else None,
                now,
                str(row.get("scraped_at") or now),
            ),
        )
        inserted += 1
    return inserted


def refresh_scraped_counts(conn: sqlite3.Connection, project_id: Optional[str] = None) -> None:
    now = utc_now_iso()
    if project_id:
        _refresh_one_project_counts(conn, str(project_id), now)
    else:
        ids = [r[0] for r in conn.execute("SELECT project_id FROM projects").fetchall()]
        for pid in ids:
            _refresh_one_project_counts(conn, pid, now)
    conn.commit()


def _refresh_one_project_counts(conn: sqlite3.Connection, project_id: str, now: str) -> None:
    total = conn.execute(
        "SELECT COUNT(*) FROM comments WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    top_level = conn.execute(
        "SELECT COUNT(*) FROM comments WHERE project_id = ? AND parent_id IS NULL",
        (project_id,),
    ).fetchone()[0]
    updates = conn.execute(
        "SELECT COUNT(*) FROM updates WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    conn.execute(
        """
        UPDATE projects SET
            scraped_comments_total = ?,
            scraped_comments_top_level = ?,
            scraped_updates_total = ?,
            scraped_counts_computed_at = ?
        WHERE project_id = ?
        """,
        (total, top_level, updates, now, project_id),
    )


def compute_item_status(expected: Optional[int], scraped: int) -> str:
    if expected is None:
        return "not_checked"
    if expected == 0:
        return "expected_zero" if scraped == 0 else "partial"
    if scraped == 0:
        return "missing"
    if scraped == expected:
        return "complete"
    return "partial"


def update_project_ks_counts(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    ks_comments_nav: Optional[int] = None,
    ks_comments_emoji: Optional[int] = None,
    ks_comments_api: Optional[int] = None,
    ks_updates_nav: Optional[int] = None,
    ks_updates_api: Optional[int] = None,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE projects SET
            ks_comments_nav = COALESCE(?, ks_comments_nav),
            ks_comments_emoji = COALESCE(?, ks_comments_emoji),
            ks_comments_api = COALESCE(?, ks_comments_api),
            ks_updates_nav = COALESCE(?, ks_updates_nav),
            ks_updates_api = COALESCE(?, ks_updates_api),
            ks_counts_fetched_at = ?
        WHERE project_id = ?
        """,
        (
            ks_comments_nav,
            ks_comments_emoji,
            ks_comments_api,
            ks_updates_nav,
            ks_updates_api,
            now,
            str(project_id),
        ),
    )


def _primary_expected_comments(row: sqlite3.Row) -> Optional[int]:
    if row["ks_comments_nav"] is not None:
        return row["ks_comments_nav"]
    if row["ks_comments_emoji"] is not None:
        return row["ks_comments_emoji"]
    return row["ks_comments_api"]


def refresh_completeness_status(
    conn: sqlite3.Connection, project_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """Recompute comments_status and updates_status. Returns report rows."""
    query = "SELECT * FROM projects"
    params: tuple = ()
    if project_id:
        query += " WHERE project_id = ?"
        params = (str(project_id),)

    report = []
    for row in conn.execute(query, params).fetchall():
        pid = row["project_id"]
        expected_comments = _primary_expected_comments(row)
        expected_updates = row["ks_updates_nav"]
        if expected_updates is None and row["ks_updates_api"] is not None:
            expected_updates = row["ks_updates_api"]

        comments_status = compute_item_status(
            expected_comments, row["scraped_comments_top_level"] or 0
        )
        updates_status = compute_item_status(
            expected_updates, row["scraped_updates_total"] or 0
        )
        conn.execute(
            """
            UPDATE projects SET comments_status = ?, updates_status = ?
            WHERE project_id = ?
            """,
            (comments_status, updates_status, pid),
        )
        report.append(
            {
                "project_id": pid,
                "project_url": row["project_url"],
                "ks_comments_nav": row["ks_comments_nav"],
                "ks_comments_emoji": row["ks_comments_emoji"],
                "scraped_comments_top_level": row["scraped_comments_top_level"],
                "scraped_comments_total": row["scraped_comments_total"],
                "comments_status": comments_status,
                "ks_updates_nav": row["ks_updates_nav"],
                "scraped_updates_total": row["scraped_updates_total"],
                "updates_status": updates_status,
            }
        )
    conn.commit()
    return report


def log_scrape_event(
    conn: sqlite3.Connection,
    project_id: str,
    scrape_type: str,
    action: str,
    *,
    rows_fetched: Optional[int] = None,
    expected_count: Optional[int] = None,
    status: str = "",
    error_message: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO scrape_log (
            project_id, scrape_type, action, rows_fetched,
            expected_count, status, error_message, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(project_id),
            scrape_type,
            action,
            rows_fetched,
            expected_count,
            status,
            error_message[:500] if error_message else "",
            utc_now_iso(),
        ),
    )


def get_projects_needing_scrape(
    conn: sqlite3.Connection,
    scrape_type: str,
    statuses: Optional[list[str]] = None,
) -> list[sqlite3.Row]:
    col = "comments_status" if scrape_type == "comments" else "updates_status"
    nav_col = "ks_comments_nav" if scrape_type == "comments" else "ks_updates_nav"
    if statuses is None:
        statuses = ["partial", "missing"]
    placeholders = ",".join("?" * len(statuses))
    return conn.execute(
        f"""
        SELECT project_id, project_url, project_slug
        FROM projects
        WHERE {col} IN ({placeholders})
           OR ({col} = 'not_checked' AND {nav_col} > 0)
        ORDER BY project_id
        """,
        statuses,
    ).fetchall()

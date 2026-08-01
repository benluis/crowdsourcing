"""
Validate Kickstarter SQLite DB readiness for statistical analysis.
Created: 2025-07-25

Usage:
    python src/processing/validate_kickstarter_for_analysis.py --db PATH [--input-csv PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.consolidate_to_sqlite import DEFAULT_INPUT_CSV, resolve_url_column  # noqa: E402
from processing.sqlite_schema import CANONICAL_DB_PATH, connect_db, init_schema  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CANONICAL_DB = CANONICAL_DB_PATH
REPORT_DIR = Path("data/kickstarter/validation")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _null_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().sum() + (series.astype(str).str.strip() == "").sum()) / len(series)


def _parseable_dates(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return float(parsed.notna().sum()) / len(series)


def validate_db(db_path: Path, input_csv: Path) -> tuple[list[CheckResult], dict]:
    conn = connect_db(db_path, create=False)
    init_schema(conn)

    checks: list[CheckResult] = []
    summary: dict = {}

    n_projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    n_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    n_updates = conn.execute("SELECT COUNT(*) FROM updates").fetchone()[0]
    summary["counts"] = {
        "projects": n_projects,
        "comments": n_comments,
        "updates": n_updates,
    }

    checks.append(
        CheckResult(
            "minimum_volume",
            n_comments >= 1_000_000 and n_updates >= 300_000 and n_projects >= 47_000,
            f"projects={n_projects:,} comments={n_comments:,} updates={n_updates:,}",
        )
    )

    input_df = pd.read_csv(input_csv, low_memory=False)
    url_col = resolve_url_column(input_df)
    if url_col:
        input_df = input_df[
            input_df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)
        ]
    id_col = next(
        (c for c in ("id", "project_id", "projectId") if c in input_df.columns),
        None,
    )
    if id_col is None:
        raise ValueError(f"No project id column found in {input_csv}")
    input_ids = set(input_df[id_col].astype(str))
    db_ids = {
        row[0]
        for row in conn.execute("SELECT project_id FROM projects").fetchall()
    }
    missing_from_input = db_ids - input_ids
    extra_in_input = input_ids - db_ids
    summary["project_alignment"] = {
        "input_csv_kickstarter_projects": len(input_ids),
        "db_projects": len(db_ids),
        "db_not_in_input_csv": len(missing_from_input),
        "input_csv_not_in_db": len(extra_in_input),
    }
    checks.append(
        CheckResult(
            "db_projects_traceable_to_input_csv",
            len(missing_from_input) == 0,
            f"db_not_in_csv={len(missing_from_input)} csv_not_scraped={len(extra_in_input)}",
        )
    )

    orphan_comments = conn.execute(
        """
        SELECT COUNT(*) FROM comments c
        LEFT JOIN projects p ON c.project_id = p.project_id
        WHERE p.project_id IS NULL
        """
    ).fetchone()[0]
    orphan_updates = conn.execute(
        """
        SELECT COUNT(*) FROM updates u
        LEFT JOIN projects p ON u.project_id = p.project_id
        WHERE p.project_id IS NULL
        """
    ).fetchone()[0]
    summary["orphans"] = {"comments": orphan_comments, "updates": orphan_updates}
    checks.append(
        CheckResult(
            "no_orphan_rows",
            orphan_comments == 0 and orphan_updates == 0,
            f"comment_orphans={orphan_comments} update_orphans={orphan_updates}",
        )
    )

    dup_comments = conn.execute(
        """
        SELECT COUNT(*) - COUNT(DISTINCT id) FROM comments
        """
    ).fetchone()[0]
    dup_updates = conn.execute(
        """
        SELECT COUNT(*) - COUNT(DISTINCT id) FROM updates
        """
    ).fetchone()[0]
    summary["duplicate_ids"] = {"comments": dup_comments, "updates": dup_updates}
    checks.append(
        CheckResult(
            "unique_row_ids",
            dup_comments == 0 and dup_updates == 0,
            f"comment_dupes={dup_comments} update_dupes={dup_updates}",
        )
    )

    comments_df = pd.read_sql_query(
        "SELECT project_id, body, created_at, parent_id FROM comments",
        conn,
    )
    updates_df = pd.read_sql_query(
        "SELECT project_id, title, body, published_at FROM updates",
        conn,
    )

    comment_body_null = _null_rate(comments_df["body"])
    comment_date_parse = _parseable_dates(comments_df["created_at"])
    update_body_null = _null_rate(updates_df["body"])
    update_title_null = _null_rate(updates_df["title"])
    has_update_title = updates_df["title"].fillna("").astype(str).str.strip() != ""
    has_update_body = updates_df["body"].fillna("").astype(str).str.strip() != ""
    update_text_null = float((~has_update_title & ~has_update_body).sum()) / max(
        len(updates_df), 1
    )
    update_date_parse = _parseable_dates(updates_df["published_at"])
    summary["field_quality"] = {
        "comment_body_null_rate": round(comment_body_null, 4),
        "comment_created_at_parse_rate": round(comment_date_parse, 4),
        "update_body_null_rate": round(update_body_null, 4),
        "update_title_null_rate": round(update_title_null, 4),
        "update_title_or_body_null_rate": round(update_text_null, 4),
        "update_published_at_parse_rate": round(update_date_parse, 4),
        "comment_reply_share": round(
            float(comments_df["parent_id"].notna().sum()) / max(len(comments_df), 1),
            4,
        ),
    }
    checks.append(
        CheckResult(
            "comment_body_present",
            comment_body_null <= 0.01,
            f"null_or_empty_rate={comment_body_null:.2%}",
        )
    )
    checks.append(
        CheckResult(
            "comment_dates_parseable",
            comment_date_parse >= 0.95,
            f"parse_rate={comment_date_parse:.2%}",
        )
    )
    checks.append(
        CheckResult(
            "update_text_present",
            update_text_null <= 0.01,
            f"title_or_body_null_rate={update_text_null:.2%} (body_only_null={update_body_null:.2%})",
        )
    )
    checks.append(
        CheckResult(
            "update_dates_parseable",
            update_date_parse >= 0.95,
            f"parse_rate={update_date_parse:.2%}",
        )
    )

    projects_df = pd.read_sql_query(
        """
        SELECT project_id, comments_status, updates_status,
               scraped_comments_total, scraped_updates_total,
               ks_comments_nav, ks_updates_nav
        FROM projects
        """,
        conn,
    )
    status_counts = (
        projects_df.groupby(["comments_status", "updates_status"])
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    summary["completeness_status"] = status_counts

    projects_with_comments = int((projects_df["scraped_comments_total"] > 0).sum())
    projects_with_updates = int((projects_df["scraped_updates_total"] > 0).sum())
    summary["coverage"] = {
        "projects_with_any_comment": projects_with_comments,
        "projects_with_any_update": projects_with_updates,
        "share_with_comments": round(projects_with_comments / max(n_projects, 1), 4),
        "share_with_updates": round(projects_with_updates / max(n_projects, 1), 4),
    }
    checks.append(
        CheckResult(
            "broad_comment_coverage",
            projects_with_comments / max(n_projects, 1) >= 0.45,
            f"{projects_with_comments:,}/{n_projects:,} projects ({projects_with_comments / max(n_projects, 1):.1%})",
        )
    )
    checks.append(
        CheckResult(
            "broad_update_coverage",
            projects_with_updates / max(n_projects, 1) >= 0.60,
            f"{projects_with_updates:,}/{n_projects:,} projects ({projects_with_updates / max(n_projects, 1):.1%})",
        )
    )


    # Per-project scrape completeness is strict (top-level == nav); partial is OK for text analysis.
    comment_complete = int((projects_df["comments_status"] == "complete").sum())
    update_complete = int((projects_df["updates_status"] == "complete").sum())
    summary["strict_completeness"] = {
        "comments_complete": comment_complete,
        "updates_complete": update_complete,
        "comments_partial": int((projects_df["comments_status"] == "partial").sum()),
    }

    median_comment_len = (
        comments_df["body"].dropna().astype(str).str.len().median()
        if len(comments_df)
        else 0
    )
    summary["text_stats"] = {
        "median_comment_body_chars": float(median_comment_len),
        "median_update_body_chars": float(
            updates_df["body"].dropna().astype(str).str.len().median()
            if len(updates_df)
            else 0
        ),
    }

    conn.close()
    return checks, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Kickstarter DB for analysis")
    parser.add_argument("--db", default=str(CANONICAL_DB))
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    parser.add_argument("--strict", action="store_true", help="Exit 1 unless all checks pass")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        logging.error("Database not found: %s", db_path)
        sys.exit(1)

    checks, summary = validate_db(db_path, Path(args.input_csv))
    passed = sum(1 for c in checks if c.passed)
    failed = [c for c in checks if not c.passed]

    logging.info("Validated %s", db_path)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        logging.info("[%s] %s: %s", status, check.name, check.detail)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    report_path = out_dir / f"validation_report_{date_str}.json"
    payload = {
        "db": str(db_path),
        "input_csv": args.input_csv,
        "checks": [asdict(c) for c in checks],
        "summary": summary,
        "analysis_ready": len(failed) == 0,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Report written to %s", report_path)
    logging.info(
        "Result: %d/%d checks passed; analysis_ready=%s",
        passed,
        len(checks),
        payload["analysis_ready"],
    )

    if args.strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

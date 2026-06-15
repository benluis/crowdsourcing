"""
Export SQLite comments/updates to dated CSVs for analysis pipelines.
Created: 2025-06-15

Usage:
    python src/processing/export_sqlite_for_analysis.py --db PATH
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import connect_db, default_db_path, init_schema  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

EXPORT_DIR = "data/kickstarter/exports"


def export_table(conn, table: str, out_path: Path) -> int:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SQLite tables to CSV")
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", default=EXPORT_DIR)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else default_db_path()
    conn = connect_db(db_path)
    init_schema(conn)

    date_str = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.output_dir)
    comments_path = out_dir / f"comments_{date_str}.csv"
    updates_path = out_dir / f"updates_{date_str}.csv"
    projects_path = out_dir / f"projects_{date_str}.csv"

    n_comments = export_table(conn, "comments", comments_path)
    n_updates = export_table(conn, "updates", updates_path)
    n_projects = export_table(conn, "projects", projects_path)

    logging.info("Exported comments=%d -> %s", n_comments, comments_path)
    logging.info("Exported updates=%d -> %s", n_updates, updates_path)
    logging.info("Exported projects=%d -> %s", n_projects, projects_path)
    conn.close()


if __name__ == "__main__":
    main()

"""
Recompute completeness status and write dated report CSV.
Created: 2025-06-15

Usage:
    python src/processing/audit_completeness.py --db PATH
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.sqlite_schema import (  # noqa: E402
    CANONICAL_DB_PATH,
    connect_db,
    init_schema,
    refresh_completeness_status,
    refresh_scraped_counts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

REPORT_DIR = "data/kickstarter"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scrape completeness")
    parser.add_argument("--db", default=None)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else CANONICAL_DB_PATH
    conn = connect_db(db_path)
    init_schema(conn)

    refresh_scraped_counts(conn)
    report = refresh_completeness_status(conn)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"completeness_report_{date_str}.csv"
    pd.DataFrame(report).to_csv(out_path, index=False)

    status_counts = pd.DataFrame(report).groupby(["comments_status", "updates_status"]).size()
    logging.info("Report written to %s (%d projects)", out_path, len(report))
    logging.info("Status breakdown:\n%s", status_counts.to_string())
    conn.close()


if __name__ == "__main__":
    main()

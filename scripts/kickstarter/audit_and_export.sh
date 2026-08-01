#!/usr/bin/env bash
# Audit, validate, and export Kickstarter SQLite for analysis (HPC/local).
# Usage: ./scripts/kickstarter/audit_and_export.sh [DB_PATH] [INPUT_CSV]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DB_PATH="${1:-data/kickstarter/kickstarter_main.db}"
INPUT_CSV="${2:-data/my_file.csv}"

echo "--- Kickstarter audit + export ---"
echo "DB: $DB_PATH"

uv run python src/processing/audit_completeness.py --db "$DB_PATH"
uv run python src/processing/validate_kickstarter_for_analysis.py \
  --db "$DB_PATH" --input-csv "$INPUT_CSV" --strict
uv run python src/processing/export_sqlite_for_analysis.py --db "$DB_PATH"

echo "--- Done ---"

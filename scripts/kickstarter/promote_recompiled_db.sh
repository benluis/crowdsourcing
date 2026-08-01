#!/usr/bin/env bash
# Promote recompiled DB to canonical kickstarter_main.db (with backup).
# Usage: ./scripts/kickstarter/promote_recompiled_db.sh [RECOMPILED_PATH] [MAIN_PATH]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RECOMPILED="${1:-data/kickstarter/kickstarter_recompiled.db}"
MAIN="${2:-data/kickstarter/kickstarter_main.db}"
ARCHIVE_DIR="data/kickstarter/archive_dbs"

if [ ! -f "$RECOMPILED" ]; then
  echo "Recompiled DB not found: $RECOMPILED" >&2
  exit 1
fi

mkdir -p "$ARCHIVE_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -f "$MAIN" ]; then
  BACKUP="$ARCHIVE_DIR/kickstarter_main_${STAMP}.db"
  echo "Backing up $MAIN -> $BACKUP"
  mv "$MAIN" "$BACKUP"
fi

echo "Promoting $RECOMPILED -> $MAIN"
mv "$RECOMPILED" "$MAIN"
echo "Done. Run scripts/kickstarter/audit_and_export.sh to validate and export."

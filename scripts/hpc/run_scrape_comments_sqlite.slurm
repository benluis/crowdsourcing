#!/bin/bash
#SBATCH --job-name=ks-scrape-comments-sqlite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=logs/scrape_comments_sqlite_%j.out
#SBATCH --error=logs/scrape_comments_sqlite_%j.err
# Created: 2025-06-15

set -e

module purge
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv add cloudscraper curl-cffi pandas beautifulsoup4 requests
uv sync

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/kickstarter

INPUT_CSV=${1:-"data/my_file.csv"}
DB_PATH=${2:-"data/kickstarter/kickstarter_main.db"}
STATUS=${3:-"partial,missing"}
PROJECT_DELAY=${4:-12}
REQUEST_DELAY=${5:-6}
BLOCK_COOLDOWN=${6:-180}

echo "--- Scrape comments into SQLite ---"
echo "Date: $(date)"
echo "DB: $DB_PATH"
echo "Throttle: project_delay=${PROJECT_DELAY}s request_delay=${REQUEST_DELAY}s block_cooldown=${BLOCK_COOLDOWN}s"

CMD=(uv run python src/scrapers/scrape_comments_sqlite.py "$INPUT_CSV" --status "$STATUS"
  --delay "$PROJECT_DELAY" --request-delay "$REQUEST_DELAY" --block-cooldown "$BLOCK_COOLDOWN"
  --db "$DB_PATH")
"${CMD[@]}"

echo "--- Done: $(date) ---"

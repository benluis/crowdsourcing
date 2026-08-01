#!/bin/bash
#SBATCH --job-name=kt-scrape-charts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=logs/scrape_kicktraq_charts_%j.out
#SBATCH --error=logs/scrape_kicktraq_charts_%j.err
# Created: 2026-07-22

set -e

module purge
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv add cloudscraper curl-cffi pandas beautifulsoup4 requests
uv sync

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/kickstarter data/kicktraq/charts

INPUT_CSV=${1:-"data/my_file.csv"}
DB_PATH=${2:-"data/kickstarter/kickstarter_main.db"}
CHARTS_DIR=${3:-"data/kicktraq/charts"}
PROJECT_DELAY=${4:-12}
REQUEST_DELAY=${5:-6}
BLOCK_COOLDOWN=${6:-180}
PASS_COOLDOWN=${7:-300}
# Default matches SBATCH --time minus a small buffer (9.8 days).
MAX_RUNTIME_HOURS=${8:-235.2}

echo "--- Download Kicktraq daily chart images ---"
echo "Date: $(date)"
echo "Input CSV: $INPUT_CSV"
echo "DB: $DB_PATH"
echo "Charts dir: $CHARTS_DIR"
echo "Throttle: project_delay=${PROJECT_DELAY}s request_delay=${REQUEST_DELAY}s block_cooldown=${BLOCK_COOLDOWN}s"
echo "Requeue: pass_cooldown=${PASS_COOLDOWN}s max_runtime_hours=${MAX_RUNTIME_HOURS}"

CMD=(uv run python src/scrapers/scrape_kicktraq_charts.py "$INPUT_CSV"
  --db "$DB_PATH"
  --charts-dir "$CHARTS_DIR"
  --delay "$PROJECT_DELAY"
  --request-delay "$REQUEST_DELAY"
  --block-cooldown "$BLOCK_COOLDOWN"
  --pass-cooldown "$PASS_COOLDOWN"
  --max-runtime-hours "$MAX_RUNTIME_HOURS")
"${CMD[@]}"

echo "--- Done: $(date) ---"

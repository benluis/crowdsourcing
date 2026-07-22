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
DB_PATH=${2:-""}
CHARTS_DIR=${3:-"data/kicktraq/charts"}
EXTRA_ARGS=${4:-""}

echo "--- Download Kicktraq daily chart images ---"
echo "Date: $(date)"
echo "Input CSV: $INPUT_CSV"
echo "Charts dir: $CHARTS_DIR"

CMD=(uv run python src/scrapers/scrape_kicktraq_charts.py "$INPUT_CSV" --charts-dir "$CHARTS_DIR")
if [ -n "$DB_PATH" ]; then
  CMD+=(--db "$DB_PATH")
fi
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi
"${CMD[@]}"

echo "--- Done: $(date) ---"

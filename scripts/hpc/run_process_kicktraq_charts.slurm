#!/bin/bash
#SBATCH --job-name=kt-process-charts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=logs/process_kicktraq_charts_%j.out
#SBATCH --error=logs/process_kicktraq_charts_%j.err
# Created: 2026-07-22

set -e

module purge
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv add cloudscraper curl-cffi pandas beautifulsoup4 requests python-dotenv
uv sync

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/kickstarter data/kicktraq/charts

if [ ! -f .env ]; then
  echo "Missing .env with GEMINI_API_KEY. Copy .env.example to .env first."
  exit 1
fi

DB_PATH=${1:-""}
CHARTS_DIR=${2:-"data/kicktraq/charts"}
EXTRA_ARGS=${3:-""}

echo "--- Extract Kicktraq daily metrics from chart images ---"
echo "Date: $(date)"

CMD=(uv run python src/scrapers/process_kicktraq_charts.py --charts-dir "$CHARTS_DIR")
if [ -n "$DB_PATH" ]; then
  CMD+=(--db "$DB_PATH")
fi
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi
"${CMD[@]}"

echo "--- Done: $(date) ---"

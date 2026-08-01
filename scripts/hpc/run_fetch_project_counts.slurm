#!/bin/bash
#SBATCH --job-name=ks-fetch-counts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=logs/fetch_counts_%j.out
#SBATCH --error=logs/fetch_counts_%j.err
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
EXTRA_ARGS=${3:-""}

echo "--- Fetch Kickstarter nav counts ---"
echo "Date: $(date)"
echo "DB: $DB_PATH"

CMD=(uv run python src/scrapers/fetch_project_counts.py "$INPUT_CSV" --db "$DB_PATH")
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi
"${CMD[@]}"

uv run python src/processing/audit_completeness.py --db "$DB_PATH"

echo "--- Done: $(date) ---"

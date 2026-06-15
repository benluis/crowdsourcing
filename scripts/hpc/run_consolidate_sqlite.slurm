#!/bin/bash
#SBATCH --job-name=ks-consolidate-sqlite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=logs/consolidate_sqlite_%j.out
#SBATCH --error=logs/consolidate_sqlite_%j.err
# Created: 2025-06-15

set -e

module purge
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv add pandas
uv sync

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/kickstarter

INPUT_CSV=${1:-"data/my_file.csv"}
DB_PATH=${2:-""}

echo "--- Consolidate batch CSVs to SQLite ---"
echo "Date: $(date)"
echo "Input: $INPUT_CSV"

if [ -n "$DB_PATH" ]; then
  uv run python src/processing/consolidate_to_sqlite.py "$INPUT_CSV" --db "$DB_PATH"
else
  uv run python src/processing/consolidate_to_sqlite.py "$INPUT_CSV"
fi

echo "--- Done: $(date) ---"

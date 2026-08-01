#!/bin/bash
#SBATCH --job-name=ks-audit-export
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=datamover
#SBATCH --output=logs/audit_export_%j.out
#SBATCH --error=logs/audit_export_%j.err
# Created: 2025-07-25

set -e

module purge
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv sync

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/kickstarter

DB_PATH=${1:-"data/kickstarter/kickstarter_main.db"}
INPUT_CSV=${2:-"data/my_file.csv"}

bash scripts/kickstarter/audit_and_export.sh "$DB_PATH" "$INPUT_CSV"

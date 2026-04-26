#!/bin/bash
#SBATCH --job-name=ks-updates-detective
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/updates_detective_pipeline_%j.out
#SBATCH --error=logs/updates_detective_pipeline_%j.err

# Exit immediately if any command fails
set -e

# Load modules
module purge
module add anaconda/2024.02
# Need Java 17+ for language_tool_python
module add java/17.0.2

# DeTeCtive paths can be overridden when submitting the job, for example:
#   sbatch --export=ALL,DETECTIVE_MODEL_PATH=/path/Deepfake_best.pth,DETECTIVE_DATABASE_PATH=/path/database/deepfake scripts/hpc/run_robust_updates_detective.slurm
: "${DETECTIVE_MODEL_PATH:=models/detective/Deepfake_best.pth}"
: "${DETECTIVE_DATABASE_PATH:=models/detective/database/deepfake}"
: "${DETECTIVE_MODEL_NAME:=princeton-nlp/unsup-simcse-roberta-base}"
: "${DETECTIVE_K:=5}"
export DETECTIVE_MODEL_PATH
export DETECTIVE_DATABASE_PATH
export DETECTIVE_MODEL_NAME
export DETECTIVE_K

# Ensure environment exists and install dependencies
echo "Installing uv for fast package management..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

echo "Adding all required dependencies to project and syncing..."
uv add pandas numpy matplotlib seaborn nltk cloudscraper beautifulsoup4 requests language_tool_python transformers torch textstat scikit-learn accelerate sentencepiece tqdm protobuf pydantic selenium webdriver-manager

echo "Adding FAISS dependency for DeTeCtive..."
if ! uv add faiss-gpu; then
  echo "faiss-gpu was unavailable; falling back to faiss-cpu."
  uv add faiss-cpu
fi
uv sync

# NLTK data for sentence tokenization
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "--- STARTING DETECTIVE KICKSTARTER UPDATES PIPELINE ---"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "DETECTIVE_MODEL_PATH: $DETECTIVE_MODEL_PATH"
echo "DETECTIVE_DATABASE_PATH: $DETECTIVE_DATABASE_PATH"
echo "DETECTIVE_MODEL_NAME: $DETECTIVE_MODEL_NAME"
echo "DETECTIVE_K: $DETECTIVE_K"

# Run from project root (sbatch is typically run from project root)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# Create logs and analysis output directories
mkdir -p logs
mkdir -p data/analysis/detective

echo "----------------------------------------"
echo "Starting DeTeCtive Updates Pipeline..."
echo "----------------------------------------"
uv run python src/pipelines/pipeline_updates_detective.py

echo "----------------------------------------"
echo "DeTeCtive Updates Pipeline Complete."
echo "Date: $(date)"

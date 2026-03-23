#!/bin/bash
#SBATCH --job-name=ks-comments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/comments_pipeline_%j.out
#SBATCH --error=logs/comments_pipeline_%j.err

# Exit immediately if any command fails
set -e

# Load modules
module purge
module add anaconda/2024.02
# Need Java 17+ for language_tool_python
module add java/17.0.2

# Ensure environment exists and install dependencies
echo "Installing uv for fast package management..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

echo "Adding all required dependencies to project and syncing..."
# Scraping, analysis, and ML dependencies compiled from previous scripts
uv add pandas numpy matplotlib seaborn nltk cloudscraper beautifulsoup4 requests language_tool_python transformers torch textstat scikit-learn accelerate sentencepiece tqdm protobuf pydantic selenium webdriver-manager
uv sync

# NLTK data for sentence tokenization
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "--- STARTING ROBUST KICKSTARTER PIPELINE ---"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run from project root (sbatch is typically run from project root)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# Create logs and analysis output directories
mkdir -p logs
mkdir -p data/analysis

# 1. Run Comments Pipeline (Load Existing or Scrape -> Analyze -> Save)
echo "----------------------------------------"
echo "Starting Comments Pipeline..."
echo "----------------------------------------"
uv run python src/pipelines/pipeline_comments.py

echo "----------------------------------------"
echo "Comments Pipeline Complete."
echo "Date: $(date)"

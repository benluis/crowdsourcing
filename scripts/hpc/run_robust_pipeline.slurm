#!/bin/bash
#SBATCH --job-name=ks-robust-pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/robust_pipeline_%j.out
#SBATCH --error=logs/robust_pipeline_%j.err

# Exit immediately if any command fails
set -e

# Load modules
module purge
module add anaconda/2024.02

# Ensure environment exists and install dependencies
if ! conda info --envs | grep -q "ai-detect-sentences"; then
    echo "Creating environment ai-detect-sentences..."
    conda create -n ai-detect-sentences python=3.9 -y
fi
source activate ai-detect-sentences
echo "Installing/updating dependencies (pandas, nltk, language_tool_python, etc.)..."
pip install pandas nltk cloudscraper beautifulsoup4 requests language_tool_python transformers torch textstat

# NLTK data for sentence tokenization
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

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
python src/pipelines/pipeline_comments.py

# 2. Run Updates Pipeline (Load Existing or Scrape -> Analyze -> Save)
echo "----------------------------------------"
echo "Starting Updates Pipeline..."
echo "----------------------------------------"
python src/pipelines/pipeline_updates.py

echo "----------------------------------------"
echo "All Pipelines Complete."
echo "Date: $(date)"

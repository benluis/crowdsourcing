#!/bin/bash
#SBATCH --job-name=ks-pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# Load modules
module purge
module add anaconda/2024.02

# Activate environment (ensure it has pandas, nltk, textstat, language_tool_python)
# Using the environment created in previous steps
source activate ai-detect-sentences

echo "--- STARTING KICKSTARTER PIPELINES ---"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Create logs directory if it doesn't exist
mkdir -p logs

# 1. Run Comments Pipeline
echo "----------------------------------------"
echo "Starting Comments Pipeline..."
echo "----------------------------------------"
python src/pipelines/pipeline_comments.py

# 2. Run Updates Pipeline
echo "----------------------------------------"
echo "Starting Updates Pipeline..."
echo "----------------------------------------"
python src/pipelines/pipeline_updates.py

echo "----------------------------------------"
echo "All Pipelines Complete."
echo "Date: $(date)"

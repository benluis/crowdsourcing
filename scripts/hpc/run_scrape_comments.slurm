#!/bin/bash
#SBATCH --job-name=ks-scrape-comments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=ks_scrape_%j.out

# REMOVED: #SBATCH --qos=datamover  <-- This was causing the error

# Exit immediately if any command fails
set -e

echo "--- STARTING KICKSTARTER COMMENT SCRAPER ---"

# 1. SET UP THE ENVIRONMENT
echo "Loading environment..."
module purge

echo "Installing uv for fast package management..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

echo "Adding scraper dependencies and syncing..."
uv add cloudscraper pandas beautifulsoup4 requests
uv sync

# 2. RUN THE PYTHON SCRIPT
echo "--- Starting Scraper Script ---"
# Assume the job is submitted from the root of the cloned repo
cd $SLURM_SUBMIT_DIR

# Accept CSV file from command line argument, default to data/my_file.csv if not provided
INPUT_CSV=${1:-"data/my_file.csv"}

echo "Processing file: $INPUT_CSV"

# Run the scraper script using uv
uv run python src/scrapers/scrape_comments.py "$INPUT_CSV"

echo "--- Scraper Finished Successfully ---"
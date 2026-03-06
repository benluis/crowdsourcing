#!/bin/bash
#SBATCH --job-name=ks-scrape-updates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --partition=datamover
#SBATCH --output=ks_updates_%j.out

# Exit immediately if any command fails
set -e

echo "--- STARTING KICKSTARTER UPDATES SCRAPER ---"

# 1. SET UP THE ENVIRONMENT
echo "Loading environment..."
module purge
module add anaconda/2024.02

# Check if environment exists, create if not
if ! conda info --envs | grep -q "ai-detect-sentences"; then
    echo "Creating environment ai-detect-sentences..."
    conda create -n ai-detect-sentences python=3.9 -y
    source activate ai-detect-sentences
    # Install scraper dependencies
    pip install cloudscraper pandas beautifulsoup4 requests
else
    echo "Activating existing environment..."
    source activate ai-detect-sentences
    # Ensure dependencies are installed even if env exists
    pip install cloudscraper pandas beautifulsoup4 requests
fi 

# 2. RUN THE PYTHON SCRIPT
echo "--- Starting Scraper Script ---"
# Assume the job is submitted from the root of the cloned repo
cd $SLURM_SUBMIT_DIR

# Accept CSV file from command line argument, default to "my_file.csv" if not provided
INPUT_CSV=${1:-"my_file.csv"}

echo "Processing file: $INPUT_CSV"

# Run the scraper script
python src/scrapers/scrape_updates.py "$INPUT_CSV"

echo "--- Scraper Finished Successfully ---"

#!/bin/bash
#SBATCH --job-name=ks-sentiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06-00:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=ks_sentiment_%j.out

# Exit immediately if any command fails
set -e

# echo "--- STARTING KICKSTARTER SENTIMENT ANALYSIS ---"

# 1. SET UP THE ENVIRONMENT
echo "Loading environment..."
module purge
module add anaconda/2024.02

# Check if environment exists, create if not
if ! conda info --envs | grep -q "ai-detect-sentences"; then
    echo "Creating environment ai-detect-sentences..."
    conda create -n ai-detect-sentences python=3.9 -y
    source activate ai-detect-sentences
    # Install dependencies
    pip install pandas nltk requests torch
else
    echo "Activating existing environment..."
    source activate ai-detect-sentences
    # Ensure dependencies are installed
    pip install pandas nltk requests torch
fi 

# 2. DOWNLOAD NLTK DATA (if not already present)
echo "Checking/Downloading NLTK data..."
python -m nltk.downloader subjectivity vader_lexicon punkt

# 3. RUN THE PYTHON SCRIPT
echo "--- Starting Analysis Script ---"
# Assume the job is submitted from the root of the cloned repo
cd /work/users/b/e/benluis/crowdsourcing

# Accept input path (file or directory) from command line argument
# Defaults to "data/scraped" directory if not provided
INPUT_PATH=${1:-"data/scraped"}

echo "Processing path: $INPUT_PATH"

# Run the analysis script
python analyze_kickstarter_comments.py "$INPUT_PATH"

echo "--- Analysis Finished Successfully ---"

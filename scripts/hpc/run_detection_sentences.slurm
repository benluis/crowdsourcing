#!/bin/bash
#SBATCH --job-name=deberta-sentences
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00                 # 4 hours for sentence-level processing (more data)
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=deberta-sentences_%j.out

# Exit immediately if any command fails
set -e

echo "--- STARTING DEBERTA SENTENCE-LEVEL DETECTION JOB ---"

# 1. SET UP THE ENVIRONMENT
echo "Purging modules and creating a clean Conda environment..."
module purge
module add anaconda/2024.02

# We'll use a new environment name to avoid any conflicts
conda env remove -n ai-detect-sentences -y
conda create -n ai-detect-sentences python=3.9 -y
source activate ai-detect-sentences

# 2. INSTALL ALL DEPENDENCIES
echo "Installing all required packages..."
conda install -c conda-forge protobuf -y
pip install torch transformers pandas scikit-learn accelerate tiktoken sentencepiece numpy tqdm nltk

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "Installation complete."

# 3. RUN THE PYTHON SCRIPT
echo "--- Starting Python Sentence Detection Script ---"
cd /work/users/b/e/benluis/crowdsourcing
python src/modeling/run_deberta_detection_sentences.py
echo "--- Python Script Finished Successfully ---"
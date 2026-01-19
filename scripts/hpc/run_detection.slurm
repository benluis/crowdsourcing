#!/bin/bash
#SBATCH --job-name=deberta-detect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00                 # 2 hours is more than enough for this
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=deberta-detect_%j.out

# Exit immediately if any command fails
set -e

echo "--- STARTING DEBERTA DETECTION JOB ---"

# 1. SET UP THE ENVIRONMENT
echo "Purging modules and creating a clean Conda environment..."
module purge
module add anaconda/2024.02

# We'll use a new environment name to avoid any conflicts with the training one
conda env remove -n ai-detect-run -y
conda create -n ai-detect-run python=3.9 -y
source activate ai-detect-run

# 2. INSTALL ALL DEPENDENCIES
echo "Installing all required packages..."
conda install -c conda-forge protobuf -y
pip install torch transformers pandas scikit-learn accelerate tiktoken sentencepiece

echo "Installation complete."

# 3. RUN THE PYTHON SCRIPT
echo "--- Starting Python Detection Script ---"
python run_deberta_detection.py
echo "--- Python Script Finished Successfully ---"

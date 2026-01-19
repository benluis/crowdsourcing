#!/bin/bash
#SBATCH --job-name=deberta-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=predict_%j.out

# Exit immediately if any command fails
set -e

echo "--- STARTING SELF-CONTAINED PREDICTION JOB ---"

echo "Purging modules and recreating the Conda environment..."
module purge
module add anaconda/2024.02
conda env remove -n ai-detect-predict -y
conda create -n ai-detect-predict python=3.9 -y
source activate ai-detect-predict

echo "Installing all required packages..."
conda install -c conda-forge protobuf -y
pip install transformers datasets torch scikit-learn pandas accelerate tiktoken sentencepiece

echo "--- Starting Python Prediction Script ---"
python predict_scores.py
echo "--- Python Script Finished ---"

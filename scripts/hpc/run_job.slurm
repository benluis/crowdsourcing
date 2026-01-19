#!/bin/bash
#SBATCH --job-name=deberta-finetune-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=finetune-a100_%j.out

set -e

echo "--- STARTING DEFINITIVE FINETUNE JOB ---"

echo "Purging modules and recreating the Conda environment..."
module purge
module add anaconda/2024.02
conda env remove -n ai-detect -y
conda create -n ai-detect python=3.9 -y
source activate ai-detect

# --- ROBUST INSTALLATION METHOD ---
echo "Installing 'protobuf' using conda-forge..."
conda install -c conda-forge protobuf -y

echo "Installing remaining packages with pip..."
pip install transformers datasets torch scikit-learn pandas accelerate tiktoken sentencepiece
# --- END INSTALLATION ---

echo "--- Verifying 'protobuf' installation ---"
conda list | grep protobuf

echo "--- Starting Python Finetune Script ---"
python finetune_deberta.py
echo "--- Python Script Finished ---"

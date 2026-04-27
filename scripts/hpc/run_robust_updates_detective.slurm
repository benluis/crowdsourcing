#!/bin/bash
#SBATCH --job-name=ks-updates-detective
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/updates_detective_pipeline_%j.out
#SBATCH --error=logs/updates_detective_pipeline_%j.err

# Exit immediately if any command fails
set -e

# Load modules
module purge
module add anaconda/2024.02
# Need Java 17+ for language_tool_python
module add java/17.0.2

# Run from project root (sbatch is typically run from project root)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT" || exit 1

# DeTeCtive lives under the crowdsourcing checkout, but its own requirements
# and generated database are inside the upstream DeTeCtive repository.
: "${DETECTIVE_REPO:=$PROJECT_ROOT/models/detective/DeTeCtive}"
: "${DETECTIVE_MODEL_PATH:=$DETECTIVE_REPO/Deepfake_best.pth}"
: "${DETECTIVE_DATABASE_PATH:=$DETECTIVE_REPO/database/deepfake}"
: "${DETECTIVE_MODEL_NAME:=princeton-nlp/unsup-simcse-roberta-base}"
: "${DETECTIVE_K:=5}"
export DETECTIVE_REPO
export DETECTIVE_MODEL_PATH
export DETECTIVE_DATABASE_PATH
export DETECTIVE_MODEL_NAME
export DETECTIVE_K

# Use a fixed conda environment instead of mutating the project pyproject on
# every run. This keeps DeTeCtive's fragile numpy/faiss stack reproducible.
ENV_NAME="${DETECTIVE_ENV_NAME:-detective_env}"
echo "--- SETTING UP CONDA ENVIRONMENT: $ENV_NAME ---"
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 numpy=1.26.4 -y
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

echo "Installing crowdsourcing pipeline helper dependencies..."
# Keep heavyweight model dependencies (torch/transformers/faiss/sklearn) under
# the DeTeCtive requirements file below to avoid version drift.
python -m pip install \
  pandas matplotlib seaborn nltk cloudscraper beautifulsoup4 requests \
  language_tool_python textstat tqdm pydantic selenium webdriver-manager

echo "Installing DeTeCtive authors' requirements last so pinned versions win..."
python -m pip install -r "$DETECTIVE_REPO/requirements.txt"

# NLTK data for sentence tokenization
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "--- STARTING DETECTIVE KICKSTARTER UPDATES PIPELINE ---"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DETECTIVE_REPO: $DETECTIVE_REPO"
echo "DETECTIVE_MODEL_PATH: $DETECTIVE_MODEL_PATH"
echo "DETECTIVE_DATABASE_PATH: $DETECTIVE_DATABASE_PATH"
echo "DETECTIVE_MODEL_NAME: $DETECTIVE_MODEL_NAME"
echo "DETECTIVE_K: $DETECTIVE_K"

if [ ! -f "$DETECTIVE_MODEL_PATH" ]; then
  echo "ERROR: Missing DeTeCtive model checkpoint: $DETECTIVE_MODEL_PATH" >&2
  exit 1
fi

for f in index.faiss index_meta.faiss label_dict.pkl; do
  if [ ! -f "$DETECTIVE_DATABASE_PATH/$f" ]; then
    echo "ERROR: Missing DeTeCtive database file: $DETECTIVE_DATABASE_PATH/$f" >&2
    exit 1
  fi
done

# Create logs and analysis output directories
mkdir -p logs
mkdir -p data/analysis/detective

echo "----------------------------------------"
echo "Starting DeTeCtive Updates Pipeline..."
echo "----------------------------------------"
python src/pipelines/pipeline_updates_detective.py

echo "----------------------------------------"
echo "DeTeCtive Updates Pipeline Complete."
echo "Date: $(date)"

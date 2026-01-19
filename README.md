# Crowdsourcing AI Detection Project

This repository contains the code and analysis for investigating the impact of AI-generated text on crowdfunding success (Kickstarter & Indiegogo). The project includes web scrapers, an AI detection pipeline (DeBERTa), text feature engineering, and statistical mediation analysis.

## 📂 Project Structure

```
crowdsourcing/
├── data/                    # (Ignored by Git)
│   ├── raw/                 # Original CSVs (IG_Data.csv, KS_Data.csv)
│   ├── processed/           # Cleaned data (mediation_analysis_data.csv)
│   └── dictionaries/        # Helpers (Paetzold_2016.xlsx)
├── src/
│   ├── scrapers/            # Data collection (main.py)
│   ├── processing/          # Data cleaning & merging
│   ├── modeling/            # AI Model training (finetune_deberta.py)
│   └── analysis/            # Statistical models (R scripts)
├── notebooks/               # Jupyter notebooks for exploration
├── models/                  # Trained DeBERTa model artifacts
├── scripts/                 # HPC/SLURM scripts
└── results/                 # Output logs and tables
```

## 🚀 Quick Start

### 1. Installation
This project uses Python for data processing/AI and R for statistical analysis.

```bash
# Python
pip install -r requirements.txt

# R (Run in R console)
# install.packages(c("mediation", "dplyr", "tidyverse", "readxl", "furrr"))
```

### 2. Pipeline Overview

The research follows this sequential flow:

**Phase 1: Data Collection**
- Run `src/scrapers/main.py` to scrape project details.
- Raw data lands in `data/raw/`.

**Phase 2: Processing & AI Scoring**
- `src/processing/combine_stories_to_csv.py`: Merges scraped JSON/PKL files into a master CSV.
- `src/modeling/run_deberta_detection.py`: Runs the fine-tuned DeBERTa model to assign an `ai_score` to every project.

**Phase 3: Feature Engineering**
- `src/analysis/concreteness.R`: Calculates linguistic concreteness using the Paetzold dictionary.
- `src/processing/create_text_quality_trend.py`: Generates quality metrics over time.

**Phase 4: Statistical Analysis (The Core Results)**
- `src/analysis/joint_mediation.R`: **Primary Result Script.** Runs the Joint Interaction Model and Causal Mediation Analysis to test the "Broken Signal" hypothesis.
- `notebooks/did_analysis.ipynb`: Difference-in-Differences (DiD) checks.

## 📊 Key Scripts

| File | Purpose |
| :--- | :--- |
| `src/analysis/joint_mediation.R` | **Main Paper Results.** Proves the market shift Post-GPT. |
| `src/analysis/concreteness.R` | Calculates "Concreteness" scores for text analysis. |
| `src/modeling/finetune_deberta.py` | Code used to train the AI detector. |
| `notebooks/ai-detect.ipynb` | Walkthrough of the AI detection methodology. |

## 🧠 Model Info
The AI detector is a fine-tuned `microsoft/deberta-v3-small` model located in `models/deberta_v3/`. It outputs a probability score (0-1) indicating the likelihood that a text was written by AI.

"""
Analysis pipeline for Kickstarter updates using DeTeCtive AI detection.
- Loads from scraper batch files
- Runs sentiment + text quality + DeTeCtive AI detection on updates
- Saves in batches to data/analysis/detective/, merges to all_updates_analyzed_detective.csv at end
- Uses separate checkpoint and batch filenames from the DeBERTa updates pipeline
"""

import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
except Exception:
    pass

try:
    import language_tool_python
    from analysis.analyze_kickstarter_comments import KickstarterSentimentAnalyzer
    from modeling.detective_detector import DeTeCtiveDetector
    from pipelines.pipeline_helpers import (
        append_to_checkpoint,
        load_processed_ids_from_checkpoint,
        load_project_ids_with_data_from_summary,
        merge_batch_files,
        record_failure,
    )
    from processing.text_quality_analysis import grammar_quality
except ImportError as e:
    logging.error(f"Import Error: {e}")
    logging.error("Make sure you are running this from the project root.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline_updates_detective.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

INPUT_CSV = "data/my_file.csv"  # Input row 'id' = project id. Output row 'id' = update post id.
OUTPUT_DIR = "data/analysis/detective"
SCRAPED_UPDATES_DIR = "data/scraped_updates_only"
BATCH_SIZE_PROJECTS = 50
MAX_RUNTIME_HOURS = 139  # ~5.8 days to be safe under a 6-day SLURM limit.

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "updates_detective_processed_ids.txt")
FAILURES_PATH = os.path.join(OUTPUT_DIR, "updates_detective_failures.csv")
BATCH_GLOB = os.path.join(OUTPUT_DIR, "updates_detective_batch_*.csv")
MERGED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "all_updates_analyzed_detective.csv")


def _ai_scores_sentence_level(ai_detector, text: str) -> dict:
    """
    Split text into sentences, run DeTeCtive on each, and return aggregate AI scores.
    """
    empty_result = {
        "ai_scores_sentences": [],
        "ai_sentences": [],
        "ai_score_mean": 0.0,
        "ai_score_median": 0.0,
        "ai_score_max": 0.0,
    }
    if not text or not isinstance(text, str) or not text.strip():
        return empty_result

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        try:
            nltk.download("punkt_tab")
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]

    sentences = [sentence for sentence in sentences if len(sentence.strip()) > 5]
    if not sentences:
        return empty_result

    scores = ai_detector.predict_batch(sentences)
    arr = np.array(scores)
    return {
        "ai_scores_sentences": list(scores),
        "ai_sentences": sentences,
        "ai_score_mean": float(np.mean(arr)) if len(arr) else 0.0,
        "ai_score_median": float(np.median(arr)) if len(arr) else 0.0,
        "ai_score_max": float(np.max(arr)) if len(arr) else 0.0,
    }


def load_updates_for_project(project_id: str, scraped_dir: str) -> list:
    """
    Load updates from kickstarter_updates_full batch files.
    Aggregate across all files, dedupe by update id.
    """
    updates = []
    batch_files = glob.glob(os.path.join(scraped_dir, "kickstarter_updates_full*.csv"))
    seen_ids = set()

    for file_path in batch_files:
        try:
            loaded_df = pd.read_csv(file_path)
            if "project_id" not in loaded_df.columns:
                continue

            subset = loaded_df[loaded_df["project_id"].astype(str) == project_id]
            if len(subset) == 0:
                continue

            for _, row in subset.iterrows():
                rec = row.to_dict()
                update_id = rec.get("id")
                if update_id is not None and update_id in seen_ids:
                    continue
                if update_id is not None:
                    seen_ids.add(update_id)
                updates.append(rec)
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")

    if updates:
        logging.info(f"Loaded {len(updates)} updates for {project_id} from batch files")
    return updates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Loading models (DeTeCtive may take a moment on GPU)...")
    try:
        sent_analyzer = KickstarterSentimentAnalyzer()
        ai_detector = DeTeCtiveDetector()
        tool = language_tool_python.LanguageTool("en-US")
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return

    if not os.path.exists(INPUT_CSV):
        logging.error(
            f"Input CSV not found: {INPUT_CSV}. "
            "Create data/my_file.csv with columns: id, project_url (or url, combined.url)"
        )
        return

    df = pd.read_csv(INPUT_CSV)
    url_col = next((col for col in ["project_url", "url", "combined.url"] if col in df.columns), None)
    if not url_col:
        logging.error("No URL column found.")
        return

    df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
    logging.info(f"Loaded {len(df)} Kickstarter projects from {INPUT_CSV}")

    projects_with_data = load_project_ids_with_data_from_summary(
        SCRAPED_UPDATES_DIR,
        "kickstarter_updates_summary_batch_*.csv",
        "updates_count",
    )
    processed_ids = load_processed_ids_from_checkpoint(CHECKPOINT_PATH)

    buffer = []
    batch_index = 0
    projects_in_buffer = []
    start_time = time.time()

    for index, row in df.iterrows():
        if (time.time() - start_time) / 3600 > MAX_RUNTIME_HOURS:
            logging.info(
                "Approaching SLURM 6-day limit. Saving and exiting gracefully so work is checkpointed."
            )
            break

        project_id = str(row.get("id", "unknown"))
        project_url = row.get(url_col, "")

        if project_id in processed_ids:
            continue
        if not project_url or not str(project_url).strip():
            record_failure(
                FAILURES_PATH,
                project_id,
                str(project_url)[:200],
                "skip",
                "Empty or missing project_url",
            )
            continue

        logging.info(f"Processing updates {project_id} ({index + 1}/{len(df)})")

        try:
            updates = []
            if project_id in projects_with_data and os.path.exists(SCRAPED_UPDATES_DIR):
                updates = load_updates_for_project(project_id, SCRAPED_UPDATES_DIR)

            if not updates:
                record_failure(
                    FAILURES_PATH,
                    project_id,
                    project_url,
                    "no_data",
                    "No updates found in scraped folder. Skipping scrape.",
                )
                continue

            analyzed_rows = []
            for update in updates:
                text = update.get("body") or update.get("update_body") or update.get("post_body") or ""
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""

                sent_scores = sent_analyzer.analyze_text(text)
                quality_score = grammar_quality(text, tool)
                ai_dict = _ai_scores_sentence_level(ai_detector, text)
                analyzed_rows.append(
                    {
                        **update,
                        **sent_scores,
                        "text_quality": quality_score if quality_score is not None else 0.0,
                        "project_status": row.get("state", "unknown"),
                        "ai_detector": "detective",
                        **ai_dict,
                    }
                )

            buffer.extend(analyzed_rows)
            projects_in_buffer.append(project_id)

            if len(projects_in_buffer) >= BATCH_SIZE_PROJECTS:
                batch_index += 1
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                batch_path = os.path.join(OUTPUT_DIR, f"updates_detective_batch_{batch_index}_{ts}.csv")
                pd.DataFrame(buffer).to_csv(batch_path, index=False)
                logging.info(f"Saved DeTeCtive updates batch {batch_index} ({len(buffer)} rows)")
                append_to_checkpoint(CHECKPOINT_PATH, projects_in_buffer, ensure_dir=False)
                buffer = []
                projects_in_buffer = []

        except Exception as e:
            record_failure(FAILURES_PATH, project_id, project_url, "analysis", str(e))
            logging.error(f"Failed {project_id}: {e}")

    if buffer:
        batch_index += 1
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        batch_path = os.path.join(OUTPUT_DIR, f"updates_detective_batch_{batch_index}_{ts}.csv")
        pd.DataFrame(buffer).to_csv(batch_path, index=False)
        append_to_checkpoint(CHECKPOINT_PATH, projects_in_buffer, ensure_dir=False)

    merge_batch_files(BATCH_GLOB, MERGED_OUTPUT_PATH, id_col="id")
    logging.info("DeTeCtive updates pipeline complete.")


if __name__ == "__main__":
    main()

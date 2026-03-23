"""
Analysis pipeline for Kickstarter updates.
- Loads from scraper batch files (or rescrapes failed projects)
- Runs sentiment + text quality + AI detection (sentence-by-sentence) on updates only
- Saves in batches to data/analysis/, merges to all_updates_analyzed.csv at end
- Checkpoint Option A + failures CSV
"""

import os
import sys
import pandas as pd
import logging
import glob
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
except Exception:
    pass

try:
    from analysis.analyze_kickstarter_comments import KickstarterSentimentAnalyzer
    from modeling.deberta_detector import DeBERTaDetector
    from processing.text_quality_analysis import grammar_quality
    import language_tool_python
    from pipelines.pipeline_helpers import (
        load_project_ids_with_data_from_summary,
        load_processed_ids_from_checkpoint,
        append_to_checkpoint,
        record_failure,
        merge_batch_files,
    )
except ImportError as e:
    logging.error(f"Import Error: {e}")
    logging.error("Make sure you are running this from the project root.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_updates.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

INPUT_CSV = "data/my_file.csv"  # No fallback. Input row 'id' = project id. Output row 'id' = update post id.
OUTPUT_DIR = "data/analysis"
SCRAPED_UPDATES_DIR = "data/scraped_updates_only"
BATCH_SIZE_PROJECTS = 50


def _ai_scores_sentence_level(ai_detector, text: str) -> dict:
    """
    Split text into sentences, run DeBERTa on each, return aggregates.
    Returns dict with ai_scores_sentences, ai_sentences, ai_score_mean, ai_score_median, ai_score_max.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            'ai_scores_sentences': [],
            'ai_sentences': [],
            'ai_score_mean': 0.0,
            'ai_score_median': 0.0,
            'ai_score_max': 0.0,
        }
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        try:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
    sentences = [s for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return {
            'ai_scores_sentences': [],
            'ai_sentences': [],
            'ai_score_mean': 0.0,
            'ai_score_median': 0.0,
            'ai_score_max': 0.0,
        }
    scores = ai_detector.predict_batch(sentences)
    arr = np.array(scores)
    return {
        'ai_scores_sentences': list(scores),
        'ai_sentences': sentences,
        'ai_score_mean': float(np.mean(arr)) if len(arr) else 0.0,
        'ai_score_median': float(np.median(arr)) if len(arr) else 0.0,
        'ai_score_max': float(np.max(arr)) if len(arr) else 0.0,
    }


def load_updates_for_project(project_id: str, scraped_dir: str) -> list:
    """
    Load updates from kickstarter_updates_full batch files.
    Aggregate across all files, dedupe by id.
    """
    updates = []
    batch_files = glob.glob(os.path.join(scraped_dir, "kickstarter_updates_full*.csv"))
    seen_ids = set()
    for f in batch_files:
        try:
            loaded_df = pd.read_csv(f)
            if 'project_id' not in loaded_df.columns:
                continue
            subset = loaded_df[loaded_df['project_id'].astype(str) == project_id]
            if len(subset) == 0:
                continue
            for _, r in subset.iterrows():
                rec = r.to_dict()
                uid = rec.get('id')
                if uid is not None and uid in seen_ids:
                    continue
                if uid is not None:
                    seen_ids.add(uid)
                updates.append(rec)
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")
    if updates:
        logging.info(f"Loaded {len(updates)} updates for {project_id} from batch files")
    return updates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Loading models (DeBERTa may take a moment on GPU)...")
    try:
        sent_analyzer = KickstarterSentimentAnalyzer()
        ai_detector = DeBERTaDetector()
        tool = language_tool_python.LanguageTool('en-US')
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return

    # Input: data/my_file.csv only (no fallback)
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV not found: {INPUT_CSV}. Create data/my_file.csv with columns: id, project_url (or url, combined.url)")
        return
    input_csv = INPUT_CSV

    df = pd.read_csv(input_csv)
    url_col = next((c for c in ['project_url', 'url', 'combined.url'] if c in df.columns), None)
    if not url_col:
        logging.error("No URL column found.")
        return

    df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
    logging.info(f"Loaded {len(df)} Kickstarter projects from {input_csv}")

    projects_with_data = load_project_ids_with_data_from_summary(
        SCRAPED_UPDATES_DIR, "kickstarter_updates_summary_batch_*.csv", "updates_count"
    )
    processed_ids = load_processed_ids_from_checkpoint(
        os.path.join(OUTPUT_DIR, "updates_processed_ids.txt")
    )
    failures_path = os.path.join(OUTPUT_DIR, "updates_failures.csv")

    buffer = []
    batch_index = 0
    projects_in_buffer = []
    
    # Time Tracking to exit before 6 days
    import time
    start_time = time.time()
    MAX_RUNTIME_HOURS = 139  # ~5.8 days to be safe

    for index, row in df.iterrows():
        
        # Check if we should gracefully exit
        if (time.time() - start_time) / 3600 > MAX_RUNTIME_HOURS:
            logging.info("Approaching SLURM 6-day limits. Saving and exiting gracefully so work is safely checkpointed.")
            break

        project_id = str(row.get('id', 'unknown'))
        project_url = row.get(url_col, '')

        if project_id in processed_ids:
            continue  # Already done; no failure record (would bloat on restarts)
        if not project_url or not str(project_url).strip():
            record_failure(failures_path, project_id, str(project_url)[:200], "skip", "Empty or missing project_url")
            continue

        logging.info(f"Processing updates {project_id} ({index + 1}/{len(df)})")

        try:
            updates = []
            if project_id in projects_with_data and os.path.exists(SCRAPED_UPDATES_DIR):
                updates = load_updates_for_project(project_id, SCRAPED_UPDATES_DIR)

            if not updates:
                record_failure(failures_path, project_id, project_url, "no_data", "No updates found in scraped folder. Skipping scrape.")
                continue

            analyzed_rows = []
            for update in updates:
                text = update.get('body') or update.get('update_body') or update.get('post_body') or ''
                if not isinstance(text, str):
                    text = str(text) if text is not None else ''
                sent_scores = sent_analyzer.analyze_text(text)
                q = grammar_quality(text, tool)
                q = q if q is not None else 0.0
                ai_dict = _ai_scores_sentence_level(ai_detector, text)
                analyzed_rows.append({
                    **update,
                    **sent_scores,
                    'text_quality': q,
                    'project_status': row.get('state', 'unknown'),
                    **ai_dict,
                })

            for r in analyzed_rows:
                buffer.append(r)
            projects_in_buffer.append(project_id)

            if len(projects_in_buffer) >= BATCH_SIZE_PROJECTS:
                batch_index += 1
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                batch_path = os.path.join(OUTPUT_DIR, f"updates_batch_{batch_index}_{ts}.csv")
                pd.DataFrame(buffer).to_csv(batch_path, index=False)
                logging.info(f"Saved updates batch {batch_index} ({len(buffer)} rows)")
                append_to_checkpoint(
                    os.path.join(OUTPUT_DIR, "updates_processed_ids.txt"),
                    projects_in_buffer,
                    ensure_dir=False
                )
                buffer = []
                projects_in_buffer = []

        except Exception as e:
            record_failure(failures_path, project_id, project_url, "analysis", str(e))
            logging.error(f"Failed {project_id}: {e}")

    if buffer:
        batch_index += 1
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        batch_path = os.path.join(OUTPUT_DIR, f"updates_batch_{batch_index}_{ts}.csv")
        pd.DataFrame(buffer).to_csv(batch_path, index=False)
        append_to_checkpoint(
            os.path.join(OUTPUT_DIR, "updates_processed_ids.txt"),
            projects_in_buffer,
            ensure_dir=False
        )

    # Dedupe by update 'id' (unique per update post), not project_id
    merge_batch_files(
        os.path.join(OUTPUT_DIR, "updates_batch_*.csv"),
        os.path.join(OUTPUT_DIR, "all_updates_analyzed.csv"),
        id_col='id'  # update post id from API; project_id is for grouping
    )
    logging.info("Updates pipeline complete.")


if __name__ == "__main__":
    main()

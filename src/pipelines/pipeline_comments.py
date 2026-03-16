"""
Analysis pipeline for Kickstarter comments.
- Loads from scraper batch files (or rescrapes failed projects)
- Runs sentiment + text quality (no AI detection)
- Saves in batches to data/analysis/, merges to all_comments_analyzed.csv at end
- Checkpoint Option A + failures CSV
"""

import os
import sys
import pandas as pd
import logging
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scrapers.scrape_comments import KickstarterCommentsScraper
    from analysis.analyze_kickstarter_comments import KickstarterSentimentAnalyzer
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
        logging.FileHandler("pipeline_comments.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

INPUT_CSV = "data/my_file.csv"  # No fallback. Input row 'id' = project id. Output row 'id' = comment id.
OUTPUT_DIR = "data/analysis"
SCRAPED_DIR = "data/scraped"
BATCH_SIZE_PROJECTS = 50


def load_comments_for_project(project_id: str, scraped_dir: str) -> list:
    """
    Load comments from batch files. Prefer raw files (exclude *_sentiment_*).
    Aggregate across all matching files, dedupe by id.
    Returns [] if no data (caller should rescrape for Failed/no-data projects).
    """
    comments = []
    all_files = glob.glob(os.path.join(scraped_dir, "kickstarter_comments*.csv"))
    raw_files = [f for f in all_files if "_sentiment_" not in f]
    files_to_try = raw_files if raw_files else all_files

    seen_ids = set()
    for scraped_file in files_to_try:
        try:
            loaded_df = pd.read_csv(scraped_file)
            if 'project_id' not in loaded_df.columns or 'body' not in loaded_df.columns:
                continue
            subset = loaded_df[loaded_df['project_id'].astype(str) == project_id]
            if len(subset) == 0:
                continue
            for _, r in subset.iterrows():
                rec = r.to_dict()
                cid = rec.get('id')
                if cid is not None and cid in seen_ids:
                    continue
                if cid is not None:
                    seen_ids.add(cid)
                comments.append(rec)
        except Exception as e:
            logging.warning(f"Failed to read {scraped_file}: {e}")

    if comments:
        logging.info(f"Loaded {len(comments)} comments for {project_id} from batch files")
    return comments


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Loading models...")
    try:
        scraper = KickstarterCommentsScraper()
        sent_analyzer = KickstarterSentimentAnalyzer()
        tool = language_tool_python.LanguageTool('en-US')
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return

    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV not found: {INPUT_CSV}. Create data/my_file.csv with columns: id, project_url (or url).")
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
        SCRAPED_DIR, "kickstarter_summary_batch_*.csv", "comments_count"
    )
    processed_ids = load_processed_ids_from_checkpoint(
        os.path.join(OUTPUT_DIR, "comments_processed_ids.txt")
    )
    failures_path = os.path.join(OUTPUT_DIR, "comments_failures.csv")

    buffer = []
    batch_index = 0
    projects_in_buffer = []

    for index, row in df.iterrows():
        project_id = str(row.get('id', 'unknown'))
        project_url = row.get(url_col, '')

        if project_id in processed_ids:
            continue  # Already done; no failure record (would bloat on restarts)
        if not project_url or not str(project_url).strip():
            record_failure(failures_path, project_id, str(project_url)[:200], "skip", "Empty or missing project_url")
            continue

        logging.info(f"Processing {project_id} ({index + 1}/{len(df)})")

        try:
            comments = []
            if project_id in projects_with_data and os.path.exists(SCRAPED_DIR):
                comments = load_comments_for_project(project_id, SCRAPED_DIR)

            if not comments:
                try:
                    for c in scraper.fetch_comments(project_url):
                        c['project_id'] = project_id
                        comments.append(c)
                except Exception as e:
                    record_failure(failures_path, project_id, project_url, "scrape", str(e))
                    logging.warning(f"Scrape failed for {project_id}: {e}")
                    continue

            if not comments:
                record_failure(failures_path, project_id, project_url, "no_data", "No comments after load/scrape")
                continue

            analyzed_rows = []
            for comment in comments:
                text = comment.get('body') or comment.get('comment') or comment.get('text') or ''
                if not isinstance(text, str):
                    text = str(text) if text is not None else ''
                sent_scores = sent_analyzer.analyze_text(text)
                q = grammar_quality(text, tool)
                q = q if q is not None else 0.0
                analyzed_rows.append({
                    **comment,
                    **sent_scores,
                    'text_quality': q,
                    'project_status': row.get('state', 'unknown')
                })

            for r in analyzed_rows:
                buffer.append(r)
            projects_in_buffer.append(project_id)

            if len(projects_in_buffer) >= BATCH_SIZE_PROJECTS:
                batch_index += 1
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                batch_path = os.path.join(OUTPUT_DIR, f"comments_batch_{batch_index}_{ts}.csv")
                pd.DataFrame(buffer).to_csv(batch_path, index=False)
                logging.info(f"Saved batch {batch_index} ({len(buffer)} rows)")
                append_to_checkpoint(
                    os.path.join(OUTPUT_DIR, "comments_processed_ids.txt"),
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
        batch_path = os.path.join(OUTPUT_DIR, f"comments_batch_{batch_index}_{ts}.csv")
        pd.DataFrame(buffer).to_csv(batch_path, index=False)
        append_to_checkpoint(
            os.path.join(OUTPUT_DIR, "comments_processed_ids.txt"),
            projects_in_buffer,
            ensure_dir=False
        )

    # Dedupe by comment 'id' (unique per comment), not project_id
    merge_batch_files(
        os.path.join(OUTPUT_DIR, "comments_batch_*.csv"),
        os.path.join(OUTPUT_DIR, "all_comments_analyzed.csv"),
        id_col='id'  # comment id from API; project_id is for grouping
    )
    logging.info("Comments pipeline complete.")


if __name__ == "__main__":
    main()

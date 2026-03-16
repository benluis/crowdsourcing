import os
import sys
import pandas as pd
import time
import logging
import glob
from datetime import datetime

# Add source directories to path so we can import your existing tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scrapers.scrape_updates import KickstarterUpdatesScraper
    from analysis.analyze_kickstarter_comments import KickstarterSentimentAnalyzer
    from processing.ai_text_detection import AITextDetector
    from processing.text_quality_analysis import simple_text_quality
except ImportError as e:
    logging.error(f"Import Error: {e}")
    logging.error("Make sure you are running this from the project root or src/pipelines folder.")
    sys.exit(1)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_updates.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    # 1. CONFIGURATION
    INPUT_CSV = "data/kickstarter_projects.csv"
    OUTPUT_DIR = "data/analyzed_updates"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. LOAD RESOURCES
    logging.info("Loading models (this takes a moment)...")
    try:
        scraper = KickstarterUpdatesScraper()
        sent_analyzer = KickstarterSentimentAnalyzer()
        ai_detector = AITextDetector()
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return

    # 3. LOAD PROJECT LIST
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV not found at {INPUT_CSV}")
        # Fallback: try to find any CSV in data/raw or data
        csv_files = glob.glob("data/*.csv") + glob.glob("data/raw/*.csv")
        if csv_files:
            INPUT_CSV = csv_files[0]
            logging.info(f"Falling back to {INPUT_CSV}")
        else:
            return

    df = pd.read_csv(INPUT_CSV)
    
    # Find the URL column
    url_col = next((col for col in ['project_url', 'url', 'combined.url'] if col in df.columns), None)
    if not url_col:
        logging.error("No URL column found in input CSV.")
        return

    logging.info(f"Loaded {len(df)} projects to process from {INPUT_CSV}")

    # 4. PROCESS LOOP
    for index, row in df.iterrows():
        project_id = str(row.get('id', 'unknown'))
        project_url = row.get(url_col)
        
        # --- CHECKPOINT: Skip if already done ---
        output_file = os.path.join(OUTPUT_DIR, f"{project_id}_updates.csv")
        if os.path.exists(output_file):
            continue  # Skip silently to keep logs clean
            
        logging.info(f"Processing Updates for {project_id} ({index + 1}/{len(df)})")

        try:
            # A. SCRAPE
            updates = []
            try:
                # fetch_updates_with_body is the method in your updates scraper
                for update in scraper.fetch_updates_with_body(project_url):
                    updates.append(update)
            except Exception as e:
                logging.warning(f"Scrape error for {project_id}: {e}")
            
            if not updates:
                logging.warning(f"No updates found for {project_id} (or scrape failed).")
                # Save an empty file so we don't retry endlessly
                pd.DataFrame({'status': ['no_updates'], 'project_id': [project_id]}).to_csv(output_file, index=False)
                continue

            # B. ANALYZE (In-Memory)
            analyzed_rows = []
            for update in updates:
                text = update.get('body', '')
                
                # 1. Sentiment
                sent_scores = sent_analyzer.analyze_text(text)
                
                # 2. AI Detection
                ai_scores = ai_detector.calculate_ai_score(text)
                
                # 3. Text Quality (Simple version for speed)
                quality_score = simple_text_quality(text)

                # Merge all data
                combined = {
                    **update,
                    **sent_scores,
                    **ai_scores,
                    'text_quality': quality_score,
                    'project_status': row.get('state', 'unknown')
                }
                analyzed_rows.append(combined)

            # C. SAVE
            result_df = pd.DataFrame(analyzed_rows)
            result_df.to_csv(output_file, index=False)
            logging.info(f"Saved {len(result_df)} analyzed updates for {project_id}")

        except Exception as e:
            logging.error(f"Failed project {project_id}: {e}")
            # Optional: Save a 'failed' marker file if you want to skip retries

if __name__ == "__main__":
    main()

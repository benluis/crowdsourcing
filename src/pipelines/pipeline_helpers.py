"""
Shared helpers for the analysis pipelines.
- Load project IDs from scraper summary files
- Checkpoint (Option A): load/append processed IDs
- Record failures
- Merge batch files into master CSV
"""

import os
import glob
import pandas as pd
import logging
from datetime import datetime


def load_project_ids_with_data_from_summary(summary_dir: str, summary_glob: str, count_col: str) -> set:
    """
    Load project IDs that have Success + (count_col > 0) from scraper summary files.
    Used to know which projects have scraped data we can load from batch files.

    Args:
        summary_dir: Directory containing summary files (e.g. data/scraped, data/scraped_updates_only)
        summary_glob: Glob pattern (e.g. kickstarter_summary_batch_*.csv)
        count_col: Column name for count (comments_count or updates_count)

    Returns:
        Set of project_id strings that have valid scraped data
    """
    result = set()
    if not os.path.exists(summary_dir):
        return result

    pattern = os.path.join(summary_dir, summary_glob)
    files = glob.glob(pattern)
    logging.info(f"Loading project IDs from {len(files)} summary files in {summary_dir}")

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'status' not in df.columns:
                continue
            df = df[df['status'] == 'Success']
            if count_col in df.columns:
                df = df[df[count_col].fillna(0) > 0]
            id_col = 'id' if 'id' in df.columns else 'project_id'
            if id_col in df.columns:
                result.update(df[id_col].astype(str).tolist())
        except Exception as e:
            logging.warning(f"Could not read summary {f}: {e}")

    logging.info(f"Found {len(result)} project IDs with Success + data")
    return result


def load_processed_ids_from_checkpoint(checkpoint_path: str) -> set:
    """Load project IDs that have already been processed (Option A checkpoint)."""
    result = set()
    if not os.path.exists(checkpoint_path):
        return result
    try:
        with open(checkpoint_path, 'r') as f:
            for line in f:
                pid = line.strip()
                if pid:
                    result.add(pid)
    except Exception as e:
        logging.warning(f"Could not read checkpoint {checkpoint_path}: {e}")
    return result


def append_to_checkpoint(checkpoint_path: str, project_ids: list, ensure_dir: bool = True) -> None:
    """Append project IDs to checkpoint file (one per line)."""
    if ensure_dir:
        d = os.path.dirname(checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)
    try:
        with open(checkpoint_path, 'a') as f:
            for pid in project_ids:
                f.write(f"{pid}\n")
    except Exception as e:
        logging.error(f"Failed to append to checkpoint {checkpoint_path}: {e}")


def record_failure(failures_path: str, project_id: str, project_url: str, error_stage: str, error_msg: str) -> None:
    """Append a failure record to the failures CSV. No retries."""
    d = os.path.dirname(failures_path)
    if d:
        os.makedirs(d, exist_ok=True)
    row = {
        'project_id': project_id,
        'project_url': project_url or '',
        'error_stage': error_stage,
        'error_message': str(error_msg)[:500],
        'timestamp': datetime.now().isoformat()
    }
    df = pd.DataFrame([row])
    write_header = not os.path.exists(failures_path)
    df.to_csv(failures_path, mode='a', header=write_header, index=False)


def merge_batch_files(batch_glob: str, output_path: str, id_col: str = 'id') -> int:
    """
    Glob batch files, concatenate, dedupe by id_col, save to output_path.
    Returns number of rows in final output.

    id_col: Column for deduping ROWS (comment id or update id). NOT project_id.
    Scraper output: each row has 'id' (comment/update) + 'project_id' (project).
    We dedupe by 'id' so the same comment/update doesn't appear twice.
    """
    files = glob.glob(batch_glob)
    if not files:
        logging.warning(f"No batch files found for {batch_glob}")
        return 0

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            logging.warning(f"Could not read {f}: {e}")

    if not dfs:
        return 0

    combined = pd.concat(dfs, ignore_index=True)

    if id_col in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=[id_col], keep='first')
        dupes = before - len(combined)
        if dupes:
            logging.info(f"Deduped {dupes} duplicate rows by {id_col}")

    combined.to_csv(output_path, index=False)
    logging.info(f"Merged {len(files)} batches -> {output_path} ({len(combined)} rows)")
    return len(combined)

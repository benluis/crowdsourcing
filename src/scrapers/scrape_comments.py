import os
import sys
import time
import logging
import pandas as pd
import glob
import subprocess
import random

try:
    import resource
except ImportError:
    resource = None

from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Generator
from datetime import datetime

from scrapers.ks_session import (
    CloudflareBlockedError,
    create_kickstarter_session,
    fetch_page,
    get_csrf_token,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class KickstarterCommentsScraper:
    CSRF_SOURCE_URL = "https://www.kickstarter.com"

    def __init__(
        self,
        *,
        request_delay: float = 6.0,
        request_delay_jitter: float = 2.0,
        reset_interval: int = 60,
    ):
        self.scraper, self._http_backend = create_kickstarter_session()
        self.graph_url = "https://www.kickstarter.com/graph"
        self.requests_made = 0
        self.request_delay = request_delay
        self.request_delay_jitter = request_delay_jitter
        self.reset_interval = reset_interval
        self.current_project_url = None

    def reset_session(self):
        """
        Re-initializes the scraper session to clear cookies and start fresh.
        Useful after consecutive failures or rate limits.
        """
        logging.info("Resetting scraper session (clearing cookies)...")
        self.scraper, self._http_backend = create_kickstarter_session()

        # Re-fetch CSRF token if we are in the middle of a project
        if self.current_project_url:
            logging.info(f"Re-fetching CSRF token for {self.current_project_url} after session reset...")
            try:
                csrf_token = get_csrf_token(self.scraper, self.current_project_url)
                self.scraper.headers.update({
                    'x-csrf-token': csrf_token,
                    'content-type': 'application/json'
                })
            except (CloudflareBlockedError, Exception) as exc:
                logging.error("Failed to refresh CSRF token after session reset: %s", exc)

    def _extract_slug(self, url: str) -> str:
        """
        Extracts the project slug from a Kickstarter URL.
        Example: https://www.kickstarter.com/projects/creator/project-name -> creator/project-name
        """
        try:
            # Handle query parameters and fragments
            clean_url = url.split('?')[0].split('#')[0]
            if '/projects/' in clean_url:
                return clean_url.split('/projects/')[1].strip('/')
            return ""
        except Exception as e:
            logging.error(f"Error extracting slug from {url}: {e}")
            return ""

    def _make_request(self, payload: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        Helper to send GraphQL requests with Rate Limit (429) handling and Smart Backoff.
        """
        retry_count = 0
        
        while retry_count < max_retries:
            # Proactive Session Reset
            self.requests_made += 1
            if self.requests_made % self.reset_interval == 0:
                logging.info(f"Proactive session reset after {self.requests_made} requests.")
                self.reset_session()
                time.sleep(5)

            try:
                time.sleep(self.request_delay + random.uniform(0, self.request_delay_jitter))
                start_ts = time.time()
                response = self.scraper.post(self.graph_url, json=payload)
                latency = time.time() - start_ts
                
                # Log Latency (Performance Tracking)
                # We log this to detect "soft throttling" where requests get slower before failing
                if latency > 2.0:
                    logging.warning(f"[PERF] Slow Request: {latency:.2f}s")
                
                # 1. Handle Rate Limits (429) - IMMEDIATE PENALTY BOX
                if response.status_code == 429:
                    logging.warning(f"Rate limit hit (429). Immediate 60s cooldown and session reset...")
                    time.sleep(60)
                    self.reset_session()
                    retry_count += 1
                    continue
                
                # 2. Handle Other Errors
                if response.status_code != 200:
                    logging.error(f"API Error {response.status_code}: {response.text[:100]}")
                    
                    # Save HTML Snapshot for debugging
                    self._save_error_snapshot(response.text, f"error_{response.status_code}")
                    
                    return None # Non-recoverable error

                data = response.json()
                
                # 3. Handle GraphQL Errors
                if 'errors' in data:
                    # Sometimes GraphQL returns "You are doing this too much" as a 200 OK with error text
                    error_msg = str(data['errors'])
                    if "too many requests" in error_msg.lower() or "throttle" in error_msg.lower():
                        logging.warning(f"GraphQL Rate Limit detected. Immediate 60s cooldown and session reset...")
                        time.sleep(60)
                        self.reset_session()
                        retry_count += 1
                        continue
                    
                    logging.error(f"GraphQL Error: {error_msg}")
                    return None

                return data # Success!

            except Exception as e:
                logging.error(f"Network Exception: {e}")
                time.sleep(10)
                retry_count += 1
        
        logging.error(f"Max retries ({max_retries}) reached. Giving up on request.")
        return None

    def _save_error_snapshot(self, content: str, prefix: str):
        """Saves the HTML/JSON content of a failed request for debugging."""
        try:
            debug_dir = "data/debug_snapshots"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/{prefix}_{timestamp}.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Saved error snapshot to {filename}")
        except Exception as e:
            logging.error(f"Failed to save error snapshot: {e}")

    def _parse_csrf_from_html(self, html: str) -> Optional[str]:
        from scrapers.ks_session import parse_csrf_from_html
        return parse_csrf_from_html(html)

    def _fetch_page_with_retries(self, url: str, max_attempts: int = 3):
        try:
            return fetch_page(self.scraper, url, max_attempts=max_attempts)
        except CloudflareBlockedError:
            raise

    def _get_csrf_token(self, url: str) -> Optional[str]:
        try:
            return get_csrf_token(self.scraper, url)
        except CloudflareBlockedError:
            raise
        except Exception:
            return None

    def fetch_comments(self, project_url: str) -> Generator[Dict, None, None]:
        """
        Generator that yields comments (and nested replies) for a given project URL.
        """
        slug = self._extract_slug(project_url)
        if not slug:
            logging.warning(f"Invalid project URL: {project_url}")
            return

        self.current_project_url = project_url
        logging.info(f"Starting comments scrape for: {slug}")

        try:
            csrf_token = get_csrf_token(self.scraper, project_url)
        except CloudflareBlockedError:
            logging.error("Cloudflare blocked CSRF fetch for %s. Aborting.", slug)
            raise
        except Exception as exc:
            logging.error("Failed to obtain CSRF token for %s: %s. Aborting.", slug, exc)
            return

        self.scraper.headers.update({
            'x-csrf-token': csrf_token,
            'content-type': 'application/json'
        })

        cursor = None
        has_next_page = True
        total_fetched = 0

        # Query includes nested replies
        query_template = """
        query GetProjectComments($slug: String!, $cursor: String) {
          project(slug: $slug) {
            id
            name
            comments(first: 25, after: $cursor) {
              totalCount
              edges {
                node {
                  id
                  body
                  createdAt
                  author {
                    name
                    id
                  }
                  replies(last: 100) {
                    totalCount
                    edges {
                      node {
                        id
                        body
                        createdAt
                        author {
                          name
                          id
                        }
                      }
                    }
                  }
                }
              }
              pageInfo {
                hasNextPage
                endCursor
              }
            }
          }
        }
        """

        while has_next_page:
            variables = {"slug": slug, "cursor": cursor}
            payload = {
                "operationName": "GetProjectComments",
                "query": query_template,
                "variables": variables
            }

            # Use the new robust request method
            data = self._make_request(payload)
            if not data: 
                # Raise exception to trigger failure handling in main loop
                raise Exception("Max retries reached or API error") 

            project_data = data.get('data', {}).get('project')
            if not project_data:
                logging.warning(f"Project data is None/Empty for {project_url}. Likely deleted/hidden.")
                # Treat as empty success rather than error
                break 

            comments_data = project_data.get('comments')
            if not comments_data:
                 logging.warning(f"Comments data missing for {project_url}")
                 break
            
            edges = comments_data.get('edges', [])
            
            for edge in edges:
                node = edge['node']
                # Process timestamp
                created_val = node.get('createdAt')
                dt = datetime.fromtimestamp(created_val) if isinstance(created_val, (int, float)) else created_val

                # Top-level comment
                comment = {
                    'id': node['id'],
                    'project_slug': slug,
                    'author': node['author']['name'],
                    'author_id': node['author']['id'],
                    'body': node['body'],
                    'created_at': dt,
                    'scraped_at': datetime.now().isoformat(),
                    'parent_id': None
                }
                yield comment
                total_fetched += 1
                
                # Process replies
                replies_data = node.get('replies', {})
                if replies_data and replies_data.get('totalCount', 0) > 0:
                    for reply_edge in replies_data.get('edges', []):
                        r_node = reply_edge['node']
                        r_created_val = r_node.get('createdAt')
                        r_dt = datetime.fromtimestamp(r_created_val) if isinstance(r_created_val, (int, float)) else r_created_val
                        
                        reply_comment = {
                            'id': r_node['id'],
                            'project_slug': slug,
                            'author': r_node['author']['name'],
                            'author_id': r_node['author']['id'],
                            'body': r_node['body'],
                            'created_at': r_dt,
                            'scraped_at': datetime.now().isoformat(),
                            'parent_id': node['id']
                        }
                        yield reply_comment
                        total_fetched += 1

            page_info = comments_data['pageInfo']
            has_next_page = page_info['hasNextPage']
            cursor = page_info['endCursor']
            
            logging.info(f"Fetched {len(edges)} top-level comments (Total items: {total_fetched})...")

            time.sleep(self.request_delay + random.uniform(0, self.request_delay_jitter))

def load_processed_ids(output_dir: str) -> set:
    """
    Scans the output directory for existing summary files and collects
    all project IDs that have already been attempted (Success or Failed).
    """
    processed_ids = set()
    if not os.path.exists(output_dir):
        return processed_ids
    
    # Look for summary files
    summary_files = glob.glob(os.path.join(output_dir, "kickstarter_summary_batch_*.csv"))
    logging.info(f"Found {len(summary_files)} existing summary files. Loading processed IDs...")
    
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            # Only skip projects that were successfully scraped
            if 'status' in df.columns:
                df = df[df['status'] == 'Success']
            
            if 'id' in df.columns:
                processed_ids.update(df['id'].astype(str).tolist())
            elif 'project_id' in df.columns: # Backwards compatibility
                processed_ids.update(df['project_id'].astype(str).tolist())
        except Exception as e:
            logging.warning(f"Could not read summary file {f}: {e}")
            
    logging.info(f"Loaded {len(processed_ids)} unique processed IDs.")
    return processed_ids

def process_kickstarter_projects(csv_file_path):
    """
    Main logic: Scrape comments and updates for all projects in the CSV.
    Filters for Kickstarter URLs only.
    Saves results to consolidated CSV files every 5000 projects.
    """
    # SLURM Time Limit Handling
    # Set to 9.8 days (partition max is 10 days) to allow graceful exit before SLURM kill
    MAX_RUNTIME_SECONDS = 9.8 * 24 * 3600 
    start_time = time.time()
    
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file {csv_file_path} not found.")
        return

    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"Loaded {len(df)} total rows from {csv_file_path}")

        # Filter for Kickstarter URLs
        url_col = None
        for col in ['project_url', 'url', 'combined.url']:
            if col in df.columns:
                url_col = col
                break
        
        if not url_col:
            logging.error("Could not find a URL column (project_url, url, or combined.url). Aborting.")
            return

        # Filter logic
        initial_count = len(df)
        df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
        filtered_count = len(df)
        logging.info(f"Filtered to {filtered_count} Kickstarter projects (skipped {initial_count - filtered_count} non-Kickstarter rows)")
        
        # --- RESUME LOGIC ---
        output_dir = "data/scraped"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        processed_ids = load_processed_ids(output_dir)
        if processed_ids:
            # Filter out already processed IDs
            # Ensure ID column matches type
            df['id_str'] = df['id'].astype(str)
            df = df[~df['id_str'].isin(processed_ids)]
            remaining_count = len(df)
            logging.info(f"Resuming: Skipped {len(processed_ids)} already processed projects. Remaining: {remaining_count}")
        else:
            logging.info("No previous progress found. Starting fresh.")
        
        if len(df) == 0:
            logging.info("All projects have been processed! Exiting.")
            return

        scraper = KickstarterCommentsScraper()
        
        # Batching Configuration
        BATCH_SIZE = 1000

        # Buffers to hold data before saving
        comments_buffer = []
        failures_buffer = [] 
        batch_summary_buffer = [] # New: Track status of every project
        
        # Track processed count for batch naming
        processed_count = 0
        batch_index = 1
        consecutive_failures = 0

        for index, row in df.iterrows():
            # Check Runtime (Safety exit before hard kill)
            if time.time() - start_time > MAX_RUNTIME_SECONDS:
                logging.warning("Max runtime (10 days) reached. Saving current progress and exiting.")
                if comments_buffer or failures_buffer or batch_summary_buffer:
                    save_batch(comments_buffer, failures_buffer, batch_summary_buffer, batch_index, output_dir)
                return

            project_id = str(row.get('id', 'unknown'))
            project_url = row.get(url_col, '')

            if not project_url:
                continue
            
            # Safety brake: If we've failed 10 times in a row, pause for a minute
            if consecutive_failures >= 10:
                logging.warning("10 consecutive project failures detected. Pausing for 60 seconds to cool down...")
                time.sleep(60)
                scraper.reset_session() # Get a fresh session/cookies
                consecutive_failures = 0  # Reset counter to try again

            logging.info(f"[PROJECT_START] Processing project {project_id} ({processed_count + 1}/{filtered_count})")
            
            project_failed = False
            failure_reasons = []
            
            c_count = 0

            # 1. Scrape Comments
            try:
                for comment in scraper.fetch_comments(project_url):
                    comment['project_id'] = project_id 
                    comments_buffer.append(comment)
                    c_count += 1
                logging.info(f"[METRIC] Project {project_id}: Fetched {c_count} comments")
            except Exception as e:
                logging.error(f"[ERROR] Error scraping comments for {project_id}: {e}")
                project_failed = True
                failure_reasons.append(f"Comments: {str(e)}")
            
            # Record Summary
            status = "Failed" if project_failed else "Success"
            error_msg = "; ".join(failure_reasons) if failure_reasons else ""
            
            if project_failed:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            summary_entry = {
                'id': project_id, # Use 'id' so this CSV can be fed back into the scraper directly
                'project_url': project_url,
                'status': status,
                'comments_count': c_count,
                'error_message': error_msg,
                'timestamp': datetime.now().isoformat()
            }
            batch_summary_buffer.append(summary_entry)

            # If any exception occurred, log to failures buffer as well
            if project_failed:
                failures_buffer.append({
                    'id': project_id, # Use 'id' so this CSV can be fed back into the scraper directly
                    'project_url': project_url,
                    'error_reason': error_msg,
                    'timestamp': datetime.now().isoformat()
                })

            logging.info(f"[PROJECT_END] Finished {project_id} Status={status}")

            processed_count += 1
            
            # Check if we need to save a batch (based on project count OR data size)
            # Safety: Save if we have too many comments in memory to prevent OOM
            if processed_count % BATCH_SIZE == 0 or len(comments_buffer) > 50000:
                save_batch(comments_buffer, failures_buffer, batch_summary_buffer, batch_index, output_dir)
                
                # Log System Health (Memory Usage) - Unix only
                if resource:
                    try:
                        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        # Linux: KB, macOS: Bytes. Assuming Linux (HPC) -> KB -> MB
                        mem_mb = mem_usage / 1024
                        logging.info(f"[SYSTEM] Memory Usage: {mem_mb:.2f} MB")
                    except Exception:
                        pass

                # Clear buffers
                comments_buffer = []
                failures_buffer = []
                batch_summary_buffer = []
                batch_index += 1

        # Save any remaining data
        if comments_buffer or failures_buffer or batch_summary_buffer:
            save_batch(comments_buffer, failures_buffer, batch_summary_buffer, batch_index, output_dir)

    except Exception as e:
        logging.error(f"An error occurred while processing the CSV file: {e}")

def save_batch(comments, failures, summary, batch_index, output_dir):
    """Helper to save current buffers to CSV files"""
    # Include Job ID (if running on SLURM) to prevent overwrites
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Common suffix for this batch save
    suffix = f"batch_{batch_index}_{job_id}_{timestamp}.csv"
    
    # Always save summary if it exists
    if summary:
        s_path = os.path.join(output_dir, f"kickstarter_summary_{suffix}")
        pd.DataFrame(summary).to_csv(s_path, index=False)
        logging.info(f"Saved batch {batch_index} SUMMARY ({len(summary)} rows) to {s_path}")

    if comments:
        c_path = os.path.join(output_dir, f"kickstarter_comments_{suffix}")
        pd.DataFrame(comments).to_csv(c_path, index=False)
        logging.info(f"Saved batch {batch_index} comments ({len(comments)} rows) to {c_path}")
    
    if failures:
        f_path = os.path.join(output_dir, f"kickstarter_failures_{suffix}")
        pd.DataFrame(failures).to_csv(f_path, index=False)
        logging.warning(f"Saved batch {batch_index} FAILURES ({len(failures)} rows) to {f_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        process_kickstarter_projects(csv_path)
    else:
        print("Usage: python kickstarter_comments_scraper.py <path_to_csv>")

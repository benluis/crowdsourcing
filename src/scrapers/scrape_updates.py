import os
import sys
import time
import logging
import pandas as pd
import cloudscraper
import glob
import random

# 'resource' module is Unix-only. Use conditional import for local Windows testing.
try:
    import resource
except ImportError:
    resource = None

from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Generator
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class KickstarterUpdatesScraper:
    def __init__(self):
        # Create a CloudScraper instance to handle Cloudflare challenges
        self.scraper = cloudscraper.create_scraper()
        self.scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.graph_url = "https://www.kickstarter.com/graph"
        self.requests_made = 0
        self.reset_interval = 100  # Increased from 20 to reduce session churn
        self.current_project_url = None
        # 17 requests per minute = 60/17 ≈ 3.53 seconds per request
        self.current_delay = 3.6

    def reset_session(self):
        logging.info("Resetting scraper session (clearing cookies)...")
        self.scraper = cloudscraper.create_scraper()
        self.scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })

        if self.current_project_url:
            logging.info(f"Re-fetching CSRF token for {self.current_project_url} after session reset...")
            csrf_token = self._get_csrf_token(self.current_project_url)
            if csrf_token:
                self.scraper.headers.update({
                    'x-csrf-token': csrf_token,
                    'content-type': 'application/json'
                })
            else:
                logging.error("Failed to refresh CSRF token after session reset.")

    def _extract_slug(self, url: str) -> str:
        try:
            clean_url = url.split('?')[0].split('#')[0]
            if '/projects/' in clean_url:
                return clean_url.split('/projects/')[1].strip('/')
            return ""
        except Exception as e:
            logging.error(f"Error extracting slug from {url}: {e}")
            return ""

    def _make_request(self, payload: Dict, max_retries: int = 3) -> Optional[Dict]:
        retry_count = 0
        
        while retry_count < max_retries:
            self.requests_made += 1
            if self.requests_made % self.reset_interval == 0:
                logging.info(f"Proactive session reset after {self.requests_made} requests.")
                self.reset_session()
                time.sleep(5)

            # Fixed Rate Throttling Sleep
            time.sleep(self.current_delay)

            try:
                start_ts = time.time()
                response = self.scraper.post(self.graph_url, json=payload)
                latency = time.time() - start_ts
                
                if latency > 2.0:
                    logging.warning(f"[PERF] Slow Request: {latency:.2f}s")
                
                if response.status_code == 429:
                    logging.warning(f"Rate limit hit (429). Immediate 600s cooldown and session reset...")
                    time.sleep(600) # 10 minute penalty box
                    self.reset_session()
                    
                    # Reset delay to base
                    self.current_delay = 3.6
                    
                    retry_count += 1
                    continue
                
                if response.status_code != 200:
                    logging.error(f"API Error {response.status_code}: {response.text[:100]}")
                    return None

                data = response.json()
                
                if 'errors' in data:
                    error_msg = str(data['errors'])
                    if "too many requests" in error_msg.lower() or "throttle" in error_msg.lower():
                        logging.warning(f"GraphQL Rate Limit detected. Immediate 600s cooldown and session reset...")
                        time.sleep(600) # 10 minute penalty box
                        self.reset_session()
                        
                        # Reset delay to base
                        self.current_delay = 3.6
                        
                        retry_count += 1
                        continue
                    
                    logging.error(f"GraphQL Error: {error_msg}")
                    return None

                # Success - Maintain fixed rate
                self.current_delay = 3.6
                return data

            except Exception as e:
                logging.error(f"Network Exception: {e}")
                time.sleep(10)
                retry_count += 1
        
        logging.error(f"Max retries ({max_retries}) reached. Giving up on request.")
        return None

    def _get_csrf_token(self, url: str) -> Optional[str]:
        try:
            for attempt in range(3):
                response = self.scraper.get(url)
                if response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logging.warning(f"Rate limit (429) on page load. Sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                if response.status_code == 200:
                    break
            
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            meta = soup.find('meta', {'name': 'csrf-token'})
            if meta:
                return meta.get('content')
            return None
        except Exception as e:
            logging.error(f"Error fetching CSRF token from {url}: {e}")
            return None

    def fetch_updates_with_body(self, project_url: str) -> Generator[Dict, None, None]:
        slug = self._extract_slug(project_url)
        if not slug:
            return

        self.current_project_url = project_url
        logging.info(f"Starting updates scrape for: {slug}")
        
        csrf_token = self._get_csrf_token(project_url)
        if not csrf_token:
            return

        self.scraper.headers.update({
            'x-csrf-token': csrf_token,
            'content-type': 'application/json'
        })

        cursor = None
        has_next_page = True
        total_fetched = 0

        # Attempt to fetch body in list query (Optimized)
        query_template_list_optimized = """
        query GetProjectUpdates($slug: String!, $cursor: String) {
          project(slug: $slug) {
            posts(first: 25, after: $cursor) {
              totalCount
              edges {
                node {
                  id
                  title
                  publishedAt
                  number
                  author {
                    name
                    id
                  }
                  ... on FreeformPost {
                    body
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
        
        # Fallback query if optimized one fails
        query_template_list_fallback = """
        query GetProjectUpdates($slug: String!, $cursor: String) {
          project(slug: $slug) {
            posts(first: 25, after: $cursor) {
              totalCount
              edges {
                node {
                  id
                  title
                  publishedAt
                  number
                  author {
                    name
                    id
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

        # New query to fetch individual post body
        query_template_post = """
        query GetPost($id: ID!) {
          node(id: $id) {
            ... on FreeformPost {
              body
            }
          }
        }
        """
        
        use_fallback = False

        while has_next_page:
            variables = {"slug": slug, "cursor": cursor}
            
            # Try optimized query first unless we know it fails
            query_to_use = query_template_list_fallback if use_fallback else query_template_list_optimized
            
            payload = {
                "operationName": "GetProjectUpdates",
                "query": query_to_use,
                "variables": variables
            }

            data = self._make_request(payload)
            
            # If optimized query failed (likely due to schema validation), switch to fallback
            if not data and not use_fallback:
                logging.warning("Optimized query failed. Switching to fallback (slower) mode.")
                use_fallback = True
                continue
                
            if not data: 
                raise Exception("Max retries reached or API error") 

            project_data = data.get('data', {}).get('project')
            if not project_data: 
                break

            posts_data = project_data.get('posts')
            if not posts_data:
                break
                
            edges = posts_data.get('edges', [])
            
            for edge in edges:
                node = edge['node']
                pub_val = node.get('publishedAt')
                dt = datetime.fromtimestamp(pub_val) if isinstance(pub_val, (int, float)) else pub_val
                
                body_content = ""
                
                # Check if body was fetched in list query
                if 'body' in node and node['body']:
                     raw_body = node['body']
                     body_content = self._clean_body(raw_body)
                else:
                    # Fetch body for this specific post (Fallback)
                    post_id = node['id']
                    try:
                        post_payload = {
                            "operationName": "GetPost",
                            "query": query_template_post,
                            "variables": {"id": post_id}
                        }
                        post_data = self._make_request(post_payload)
                        if post_data and 'data' in post_data and 'node' in post_data['data']:
                            raw_body = post_data['data']['node'].get('body', '')
                            if raw_body:
                                body_content = self._clean_body(raw_body)
                                
                    except Exception as e:
                        logging.warning(f"Failed to fetch body for post {post_id}: {e}")

                update = {
                    'id': node['id'],
                    'project_slug': slug,
                    'title': node['title'],
                    'number': node['number'],
                    'body': body_content,
                    'author': node['author']['name'],
                    'author_id': node['author']['id'],
                    'published_at': dt,
                    'scraped_at': datetime.now().isoformat()
                }
                yield update
                total_fetched += 1

            page_info = posts_data['pageInfo']
            has_next_page = page_info['hasNextPage']
            cursor = page_info['endCursor']
            
            logging.info(f"Fetched {len(edges)} updates (Total: {total_fetched})...")
    
    def _clean_body(self, raw_body):
        """Helper to clean HTML body content"""
        if not raw_body:
            return ""
        try:
            soup = BeautifulSoup(raw_body, 'html.parser')
            # 1. Replace images with [IMAGE]
            for img in soup.find_all('img'):
                img.replace_with(' [IMAGE] ')
            
            # 2. Replace iframes/embeds with [EMBED]
            for iframe in soup.find_all('iframe'):
                iframe.replace_with(' [EMBED] ')
            
            # 3. Extract text with separator
            return soup.get_text(separator='\n', strip=True)
        except:
            return raw_body

def load_processed_ids(output_dir: str) -> set:
    processed_ids = set()
    if not os.path.exists(output_dir):
        return processed_ids
    
    summary_files = glob.glob(os.path.join(output_dir, "kickstarter_updates_summary_batch_*.csv"))
    logging.info(f"Found {len(summary_files)} existing summary files. Loading processed IDs...")
    
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            if 'status' in df.columns:
                df = df[df['status'] == 'Success']
            
            if 'id' in df.columns:
                processed_ids.update(df['id'].astype(str).tolist())
            elif 'project_id' in df.columns:
                processed_ids.update(df['project_id'].astype(str).tolist())
        except Exception as e:
            logging.warning(f"Could not read summary file {f}: {e}")
            
    logging.info(f"Loaded {len(processed_ids)} unique processed IDs.")
    return processed_ids

def process_kickstarter_updates(csv_file_path):
    MAX_RUNTIME_SECONDS = 9.8 * 24 * 3600 
    start_time = time.time()
    
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file {csv_file_path} not found.")
        return

    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"Loaded {len(df)} total rows from {csv_file_path}")

        url_col = None
        for col in ['project_url', 'url', 'combined.url']:
            if col in df.columns:
                url_col = col
                break
        
        if not url_col:
            logging.error("Could not find a URL column (project_url, url, or combined.url). Aborting.")
            return

        initial_count = len(df)
        df = df[df[url_col].astype(str).str.contains("kickstarter.com", case=False, na=False)]
        filtered_count = len(df)
        logging.info(f"Filtered to {filtered_count} Kickstarter projects")
        
        # Use a DIFFERENT output directory to avoid conflicts
        output_dir = "data/scraped_updates_only"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        processed_ids = load_processed_ids(output_dir)
        if processed_ids:
            df['id_str'] = df['id'].astype(str)
            df = df[~df['id_str'].isin(processed_ids)]
            remaining_count = len(df)
            logging.info(f"Resuming: Skipped {len(processed_ids)} already processed projects. Remaining: {remaining_count}")
        else:
            logging.info("No previous progress found. Starting fresh.")
        
        if len(df) == 0:
            logging.info("All projects have been processed! Exiting.")
            return
        
        # Shuffle the dataframe to avoid hitting the same "bad" projects in order if restarting
        # df = df.sample(frac=1).reset_index(drop=True)

        scraper = KickstarterUpdatesScraper()
        
        BATCH_SIZE = 5
        updates_buffer = []
        failures_buffer = [] 
        batch_summary_buffer = []
        
        processed_count = 0
        batch_index = 1
        consecutive_failures = 0

        for index, row in df.iterrows():
            if time.time() - start_time > MAX_RUNTIME_SECONDS:
                logging.warning("Max runtime reached. Saving current progress and exiting.")
                if updates_buffer or failures_buffer or batch_summary_buffer:
                    save_batch(updates_buffer, failures_buffer, batch_summary_buffer, batch_index, output_dir)
                return

            project_id = str(row.get('id', 'unknown'))
            project_url = row.get(url_col, '')

            if not project_url:
                continue
            
            if consecutive_failures >= 10:
                logging.warning("10 consecutive failures. Pausing...")
                time.sleep(60)
                scraper.reset_session()
                consecutive_failures = 0

            logging.info(f"[PROJECT_START] Processing project {project_id} ({processed_count + 1}/{filtered_count})")
            
            project_failed = False
            failure_reasons = []
            u_count = 0

            try:
                for update in scraper.fetch_updates_with_body(project_url):
                    update['project_id'] = project_id 
                    updates_buffer.append(update)
                    u_count += 1
                logging.info(f"[METRIC] Project {project_id}: Fetched {u_count} updates")
            except Exception as e:
                logging.error(f"[ERROR] Error scraping updates for {project_id}: {e}")
                project_failed = True
                failure_reasons.append(f"Updates: {str(e)}")
            
            status = "Failed" if project_failed else "Success"
            error_msg = "; ".join(failure_reasons) if failure_reasons else ""
            
            if project_failed:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            summary_entry = {
                'id': project_id,
                'project_url': project_url,
                'status': status,
                'updates_count': u_count,
                'error_message': error_msg,
                'timestamp': datetime.now().isoformat()
            }
            batch_summary_buffer.append(summary_entry)

            if project_failed:
                failures_buffer.append({
                    'id': project_id,
                    'project_url': project_url,
                    'error_reason': error_msg,
                    'timestamp': datetime.now().isoformat()
                })

            logging.info(f"[PROJECT_END] Finished {project_id} Status={status}")

            processed_count += 1
            
            if processed_count % BATCH_SIZE == 0 or len(updates_buffer) > 50000:
                save_batch(updates_buffer, failures_buffer, batch_summary_buffer, batch_index, output_dir)
                
                # Log System Health (Memory Usage) - Linux/Mac only
                if resource:
                    try:
                        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        # Linux: KB, macOS: Bytes. Assuming Linux (HPC) -> KB -> MB
                        mem_mb = mem_usage / 1024
                        logging.info(f"[SYSTEM] Memory Usage: {mem_mb:.2f} MB")
                    except:
                        pass

                updates_buffer = []
                failures_buffer = []
                batch_summary_buffer = []
                batch_index += 1

        if updates_buffer or failures_buffer or batch_summary_buffer:
            save_batch(updates_buffer, failures_buffer, batch_summary_buffer, batch_index, output_dir)

    except Exception as e:
        logging.error(f"An error occurred while processing the CSV file: {e}")

def save_batch(updates, failures, summary, batch_index, output_dir):
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"batch_{batch_index}_{job_id}_{timestamp}.csv"
    
    if summary:
        s_path = os.path.join(output_dir, f"kickstarter_updates_summary_{suffix}")
        pd.DataFrame(summary).to_csv(s_path, index=False)
        logging.info(f"Saved batch {batch_index} SUMMARY ({len(summary)} rows) to {s_path}")
    
    if updates:
        u_path = os.path.join(output_dir, f"kickstarter_updates_full_{suffix}")
        pd.DataFrame(updates).to_csv(u_path, index=False)
        logging.info(f"Saved batch {batch_index} updates ({len(updates)} rows) to {u_path}")
        
    if failures:
        f_path = os.path.join(output_dir, f"kickstarter_updates_failures_{suffix}")
        pd.DataFrame(failures).to_csv(f_path, index=False)
        logging.warning(f"Saved batch {batch_index} FAILURES ({len(failures)} rows) to {f_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        process_kickstarter_updates(csv_path)
    else:
        print("Usage: python kickstarter_updates_scraper.py <path_to_csv>")

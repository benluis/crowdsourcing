import os
import pandas as pd
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time


def clean_text(text):
    return text.replace('\xa0', ' ').strip()


def scrape_indiegogo_story(campaign_url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(campaign_url)
        time.sleep(1)  # Reduce sleep time for speed
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        story_section = soup.find('div', class_='routerContentStory-storyBody')
        story_content = clean_text(story_section.get_text(separator="\n")) if story_section else ""
        word_count = len(story_content.split()) if story_content else 0
        return story_content, word_count
    except Exception as e:
        logging.error(f"Error scraping story: {e}")
        return "", 0
    finally:
        driver.quit()


def process_project(row):
    project_id = str(row.get('id', 'unknown'))
    project_link = row.get('combined.url', '')
    funding_started_at = row.get('funding_started_at', None)
    if not project_link:
        logging.warning(f"Skipping project {project_id}: URL is missing.")
        return None

    story_url = f"https://www.indiegogo.com{project_link}"
    logging.info(f"Scraping story for project {project_id} from {story_url}")
    story_content, word_count = scrape_indiegogo_story(story_url)

    return {
        "id": project_id,
        "story_content": story_content,
        "funding_started_at": funding_started_at,
        "word_count": word_count
    }


def process_indiegogo_projects(csv_file_path, output_pkl_path):
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"CSV Columns: {df.columns.tolist()}")
        all_projects = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_project, [row for _, row in df.iterrows()]))
            all_projects = [res for res in results if res is not None]

        with open(output_pkl_path, 'wb') as f:
            pickle.dump(all_projects, f)
        logging.info(f"All project data saved to {output_pkl_path}")
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV file: {e}")


if __name__ == "__main__":
    csv_file_path = "C:/Users/Ben/Downloads/indiegogo/test.csv"
    output_pkl_path = "C:/Users/Ben/Downloads/indiegogo/campaign_data.pkl"
    process_indiegogo_projects(csv_file_path, output_pkl_path)

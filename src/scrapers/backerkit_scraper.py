from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os
import re
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def clean_text(text):
    """
    Cleans the scraped text by replacing non-breaking spaces and other unwanted characters.
    """
    return text.replace('\xa0', ' ').strip()

def scrape_funding_progress_on_indiegogo(soup, save_directory):
    """
    Scrapes the funding progress data from Indiegogo, based on the BeautifulSoup object.
    Adapts the data structure based on whether the project is ongoing or ended.
    """
    # Check if project is ongoing by inspecting the "ending_message" content
    ending_message = soup.find('div', class_='ending_message')
    ongoing = False
    if ending_message and "days to go" in ending_message.text.lower():
        ongoing = True  # The project is still ongoing

    # Locate the funding data
    funding_progress_data_tag = soup.find('div', {'id': 'fundingData'})
    if funding_progress_data_tag:
        funding_progress_data = funding_progress_data_tag['data-chart']
        funding_progress_data = funding_progress_data.replace('null', 'None')
        funding_progress_data_list = eval(funding_progress_data)

        # Determine the structure based on ongoing or ended status
        if ongoing:
            # Project is ongoing; we expect all columns
            logging.info("Project is ongoing. Using ongoing data format and removing last 2 rows.")
            expected_columns = ['Date', 'Funds Raised', 'Goal', 'Trend', 'Projection Low', 'Projection High', 'Tooltip']
            df_funding_progress = pd.DataFrame(funding_progress_data_list, columns=expected_columns)

        else:
            # Project has ended; only 3 columns are expected
            logging.info("Project has ended. Using ended data format without row removal.")
            actual_columns = ['Date', 'Funds Raised', 'Goal']
            df_funding_progress = pd.DataFrame(funding_progress_data_list, columns=actual_columns)

        # Process date and numeric columns
        df_funding_progress['Date'] = pd.to_datetime(df_funding_progress['Date'], errors='coerce')
        df_funding_progress['Funds Raised'] = pd.to_numeric(df_funding_progress['Funds Raised'], errors='coerce')
        df_funding_progress['Goal'] = pd.to_numeric(df_funding_progress['Goal'], errors='coerce')

        # Handle missing columns in ended projects by adding the expected columns
        if not ongoing:
            for col in ['Trend', 'Projection Low', 'Projection High', 'Tooltip']:
                df_funding_progress[col] = pd.NA

        # Filter out rows with "Trending:" or "Trend:" in 'Tooltip', if the column exists
        if 'Tooltip' in df_funding_progress.columns:
            df_funding_progress['Tooltip'] = df_funding_progress['Tooltip'].astype(str)
            df_funding_progress = df_funding_progress[
                ~df_funding_progress['Tooltip'].str.contains("Trending:", na=False)]
            df_funding_progress = df_funding_progress[~df_funding_progress['Tooltip'].str.contains("Trend:", na=False)]

        # Fill NaN values appropriately for all columns
        df_funding_progress.fillna(
            {'Funds Raised': 0, 'Goal': 0, 'Trend': 0, 'Projection Low': 0, 'Projection High': 0, 'Tooltip': ''},
            inplace=True)

        if ongoing:
            # Remove the last 2 rows if the project is ongoing
            df_funding_progress = df_funding_progress.iloc[:-2]

        # Save data to a CSV
        csv_funding_progress_path = os.path.join(save_directory, 'funding_progress_indiegogo.csv')
        df_funding_progress.to_csv(csv_funding_progress_path, index=False)
        logging.info(f"Funding progress data saved as a CSV file at {csv_funding_progress_path}")

def scrape_backerkit(campaign_url, save_directory):
    """
    Scrapes the funds raised, backers per day, and funding progress from the BackerKit campaign page and saves them in CSV files.
    """
    # Initialize Chrome WebDriver with options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    )

    # Start the WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # Open the campaign page in the browser
        driver.get(campaign_url)
        time.sleep(3)  # Wait for page to load

        # Parse the page with BeautifulSoup
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Ensure the save directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Scrape "Daily Funding on Indiegogo"
        script_tag_funding = soup.find('script', text=re.compile(r'new Chartkick\["ColumnChart"\]\("chart-1"'))
        if script_tag_funding:
            funding_data = re.search(r'new Chartkick\["ColumnChart"\]\("chart-1", (\[\[.*?\]\])',
                                     script_tag_funding.string)
            if funding_data:
                funding_data_list = eval(funding_data.group(1))
                df_funding = pd.DataFrame(funding_data_list, columns=['Date', 'Funds Raised'])
                df_funding['Date'] = pd.to_datetime(df_funding['Date'], errors='coerce')
                csv_funding_path = os.path.join(save_directory, 'daily_funding_indiegogo.csv')
                df_funding.to_csv(csv_funding_path, index=False)
                logging.info(f"Daily funding data saved as a CSV file at {csv_funding_path}")

        # Scrape "Daily Backers on Indiegogo"
        script_tag_backers = soup.find('script', text=re.compile(r'new Chartkick\["ColumnChart"\]\("chart-2"'))
        if script_tag_backers:
            backers_data = re.search(r'new Chartkick\["ColumnChart"\]\("chart-2", (\[\[.*?\]\])',
                                     script_tag_backers.string)
            if backers_data:
                backers_data_list = eval(backers_data.group(1))
                df_backers = pd.DataFrame(backers_data_list, columns=['Date', 'Backers'])
                df_backers['Date'] = pd.to_datetime(df_backers['Date'], errors='coerce')
                csv_backers_path = os.path.join(save_directory, 'daily_backers_indiegogo.csv')
                df_backers.to_csv(csv_backers_path, index=False)
                logging.info(f"Daily backers data saved as a CSV file at {csv_backers_path}")

        # Scrape "Funding Progress on Indiegogo" using the helper function
        scrape_funding_progress_on_indiegogo(soup, save_directory)

    except Exception as e:
        logging.error(f"Error while fetching the page: {e}")

    finally:
        # Close the browser
        driver.quit()

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
import os
import pickle  # Import pickle for serialization


def clean_text(text):
    """
    Cleans the scraped text by replacing non-breaking spaces and other unwanted characters.
    """
    cleaned_text = text.replace('\xa0', ' ').strip()  # Replace non-breaking spaces with regular spaces
    return cleaned_text


def scrape_indiegogo_story(campaign_url, save_directory):
    """
    Scrapes the story section from the Indiegogo campaign and saves it as a pickle file in the specified directory.
    """
    # Initialize Chrome WebDriver with options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (without opening a window)
    options.add_argument('--disable-gpu')  # Disable GPU rendering
    options.add_argument('--no-sandbox')  # Bypass OS security model
    options.add_argument('--disable-blink-features=AutomationControlled')  # Prevent automation detection
    options.add_argument('--window-size=1920,1080')  # Set the window size for headless mode
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    # Start the WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # Open the campaign page in the browser
        driver.get(campaign_url)

        # Wait for the page to load fully
        time.sleep(2)

        # Scroll down to load all content
        total_scrolls = 3
        for i in range(total_scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)  # Allow time for new content to load

        # Get the page source and parse it with BeautifulSoup
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Locate the 'Story' section
        story_section = soup.find('div', class_='routerContentStory-storyBody')

        if story_section:
            # Extract and clean the story content
            story_content = clean_text(story_section.get_text(separator="\n").strip())

            # Ensure the save directory exists
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Save the story content as a pickle file
            story_path = os.path.join(save_directory, 'story.pkl')
            with open(story_path, 'wb') as f:
                pickle.dump(story_content, f)  # Save story_content as a pickle file

            print(f"Story content saved as a pickle file at {story_path}")
        else:
            print("Story section not found or unable to load the page correctly.")

    except Exception as e:
        print(f"Error while fetching the page: {e}")

    finally:
        # Close the browser after fetching the content
        driver.quit()

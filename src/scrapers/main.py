import os
import pandas as pd
import logging
from indiegogo_scraper import scrape_indiegogo_story
from updates_scraper import scrape_indiegogo_updates
from backerkit_scraper import scrape_backerkit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_project_directory(project_id):
    """
    Create a directory for a project using the project_id.
    """
    project_directory = os.path.join(os.getcwd(), project_id)
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)
    return project_directory

def process_indiegogo_projects(csv_file_path):
    """
    Processes Indiegogo projects from a CSV file.
    Scrapes the story, updates, and fund data for each project.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        for index, row in df.iterrows():
            project_id = str(row.get('id', 'unknown'))  # The project ID
            project_link = row.get('combined.url', '')  # The project base URL

            if not project_link:
                logging.warning(f"Skipping project {project_id}: URL is missing.")
                continue

            # Construct URLs for scraping
            story_url = f"https://www.indiegogo.com{project_link}"
            updates_url = f"{story_url}#/updates/all"
            backerkit_url = f"https://www.backerkit.com{project_link}"  # Example BackerKit URL pattern

            # Create a directory for this project using the project_id
            project_directory = create_project_directory(project_id)
            funds_data_directory = os.path.join(project_directory, 'funds_data')  # Subfolder for funds data

            try:
                # Scrape the story and save it to the project directory
                logging.info(f"Scraping story for project {project_id} from {story_url}")
                scrape_indiegogo_story(story_url, project_directory)

                # Scrape the updates and save them to the project directory
                logging.info(f"Scraping updates for project {project_id} from {updates_url}")
                scrape_indiegogo_updates(updates_url, project_directory)

                # Scrape the fund data from BackerKit and save to the project directory
                logging.info(f"Scraping fund data for project {project_id} from {backerkit_url}")
                scrape_backerkit(backerkit_url, funds_data_directory)

            except Exception as e:
                logging.error(f"Error scraping project {project_id}: {e}")

    except FileNotFoundError:
        logging.error(f"CSV file {csv_file_path} not found.")
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV file: {e}")

if __name__ == "__main__":
    # Example CSV path or you can pass it as an argument
    csv_file_path = "C:/Users/Ben/Downloads/indiegogo/test.csv"  # Update the path to your actual CSV file

    # Uncomment below line to process the CSV file
    process_indiegogo_projects(csv_file_path)

    # Test scraping for one specific project directly without CSV
    #project_id = str(888888)  # Example project ID
    #project_directory = create_project_directory(project_id)
    #funds_data_directory = os.path.join(project_directory, 'funds_data')
    #scrape_backerkit(
    #    'https://www.backerkit.com/projects/livall-pikaboost-2-electrify-your-rides-with-ease',
    #    funds_data_directory
    #)
    #scrape_indiegogo_story(
    #    'https://www.indiegogo.com/projects/livall-pikaboost-2-electrify-your-rides-with-ease',
    #    project_directory
    #)
    #scrape_indiegogo_updates(
    #    'https://www.indiegogo.com/projects/livall-pikaboost-2-electrify-your-rides-with-ease#/updates/all',
    #    project_directory
    #)
    #check_patent(project_directory)

    #project_id = str(111111)  # Example project ID
    #project_directory = create_project_directory(project_id)
    #funds_data_directory = os.path.join(project_directory, 'funds_data')
    #scrape_backerkit(
    #    'https://www.backerkit.com/projects/pocket-4-modular-full-featured-handheld-ai-pc#/',
    #    funds_data_directory
    #)
    #scrape_indiegogo_story(
    #    'https://www.indiegogo.com/projects/pocket-4-modular-full-featured-handheld-ai-pc#/',
    #    project_directory
    #)
    #scrape_indiegogo_updates(
    #    'https://www.indiegogo.com/projects/pocket-4-modular-full-featured-handheld-ai-pc#/updates/all',
    #    project_directory
    #)
    #check_patent(project_directory)
import pandas as pd
import os
import re

def detect_patent_terms(description):
    """
    Detect patent-related terms in the description.
    Return True if a match is found and False otherwise.
    """
    patent_terms = re.compile(r'\b(patent|patents|patent pending|patented|patenting)\b', re.IGNORECASE)
    return bool(patent_terms.search(description))

def load_pkl(file_path):
    """
    Load a .pkl file using pandas.
    Raise an error if the file does not exist or fails to load.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    return pd.read_pickle(file_path)

def process_updates(data):
    """
    Add a patent detection column to the 'updates.pkl' DataFrame.
    """
    if 'post_content' not in data.columns:
        raise ValueError("The DataFrame does not contain the 'post_content' column required for updates.")

    data['is_patented'] = data['post_content'].apply(detect_patent_terms)
    return data

def process_stories(data):
    """
    Add a patent detection column to the 'stories.pkl' DataFrame.
    """
    if 'story_content' not in data.columns:
        raise ValueError("The DataFrame does not contain the 'story_content' column required for stories.")

    data['is_patented'] = data['story_content'].apply(detect_patent_terms)
    return data

def process_file(file_path):
    """
    Process a .pkl file to add a patent detection column and save the filtered DataFrame with 'is_patented' = True as a new .pkl file.
    """
    try:
        # Load the .pkl file
        data = load_pkl(file_path)

        # Check if the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The file does not contain a pandas DataFrame.")

        # Process based on file type
        if 'updates' in file_path:
            data = process_updates(data)
        elif 'stories' in file_path:
            data = process_stories(data)
        else:
            raise ValueError("Unknown file type. Expected 'updates' or 'stories' in file name.")

        # Filter rows where 'is_patented' is True
        patented_data = data[data['is_patented']]

        if patented_data.empty:
            print(f"No patented data found in {file_path}. Skipping file.")
            return

        # Create output file name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_pkl_path = os.path.join(os.path.dirname(file_path), f"{base_name}_patented.pkl")

        # Save the filtered DataFrame as a new .pkl file
        patented_data.to_pickle(output_pkl_path)

        print(f"Filtered data saved to {output_pkl_path}.")
    except Exception as e:
        print(f"Error processing the file {file_path}: {e}")

if __name__ == "__main__":
    # Paths to your .pkl files
    file_paths = [
        'C:/Users/Ben/Downloads/indiegogo/stories.pkl',
        'C:/Users/Ben/Downloads/indiegogo/updates.pkl'
    ]

    # Process each file
    for file_path in file_paths:
        process_file(file_path)

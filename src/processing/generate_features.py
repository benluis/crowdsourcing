import pandas as pd
import numpy as np
import os


def add_funding_duration(file_path, save_path=None):
    """
    Adds a funding duration column (in days) to a DataFrame loaded from a .csv file.

    Args:
        file_path (str): Path to the .csv file.
        save_path (str, optional): Path to save the updated .csv file. Defaults to None.

    Returns:
        pd.DataFrame: Updated DataFrame with funding duration.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, low_memory=False)

        # Convert funding start and end dates to datetime format
        df["funding_started_at"] = pd.to_datetime(df["funding_started_at"], errors='coerce')
        df["funding_ends_at"] = pd.to_datetime(df["funding_ends_at"], errors='coerce')

        # Calculate funding duration in days
        df["funding_duration_days"] = (df["funding_ends_at"] - df["funding_started_at"]).dt.days

        # Define save path if not provided
        if save_path is None:
            save_path = os.path.splitext(file_path)[0] + "_updated.csv"

        # Save the updated DataFrame
        df.to_csv(save_path, index=False)

        return df  # Return the processed dataframe

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def add_word_count(file_path, save_path=None):
    """
    Adds a word count column to a DataFrame loaded from a .pkl file.

    Args:
        file_path (str): Path to the .pkl file.
        save_path (str, optional): Path to save the updated .pkl file. Defaults to None.

    Returns:
        pd.DataFrame: Updated DataFrame with word count.
    """
    try:
        # Load the Pickle file
        df = pd.read_pickle(file_path)

        # Add Word Count column (check whether 'story_content' or 'post_content' exists)
        if "story_content" in df.columns:
            df["word_count"] = df["story_content"].astype(str).apply(lambda x: len(x.split()))
        elif "post_content" in df.columns:
            df["word_count"] = df["post_content"].astype(str).apply(lambda x: len(x.split()))
        else:
            df["word_count"] = np.nan  # If no relevant text column exists

        # Define save path if not provided
        if save_path is None:
            save_path = os.path.splitext(file_path)[0] + "_updated.pkl"

        # Save the updated DataFrame
        df.to_pickle(save_path)

        return df  # Return the processed dataframe

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

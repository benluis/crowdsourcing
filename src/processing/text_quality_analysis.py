import numpy
import numpy.core.numeric as numeric_core
import sys
import pandas as pd
import os
import pickle
import language_tool_python
from tqdm import tqdm
import matplotlib.pyplot as plt

# Fix numpy compatibility issue
sys.modules['numpy._core.numeric'] = numeric_core

def load_data(file_path):
    """Load the combined data from pickle file"""
    print(f"Loading: {file_path}")
    try:
        with open(file_path, 'rb') as file:
            stories = pickle.load(file)
        print(f"Loaded {len(stories)} records")
        print(f"Total columns: {len(stories.columns)}")
        print(f"Key columns available:")
        key_columns = ['story_content', 'launched_at', 'created_at', 'deadline', 'funds_raised_percent', 'goal', 'pledged']
        for col in key_columns:
            if col in stories.columns:
                print(f"  ✓ {col}")
            else:
                print(f"  ✗ {col} (missing)")
        return stories
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def calculate_word_count(df):
    """Calculate word count for story content"""
    print("Calculating word counts...")
    df['word_count'] = df['story_content'].apply(lambda x: len(str(x).split()))
    print("Word count calculation completed")
    return df

def grammar_quality(text, tool):
    """Calculate grammar quality score for a text"""
    try:
        matches = tool.check(str(text))
        token_count = len(str(text).split())
        if token_count > 0:
            return 1 - (len(matches) / token_count)
        return None
    except Exception as e:
        print(f"Error checking grammar for text: {e}")
        return None

def simple_text_quality(text):
    """Calculate a simple text quality score based on basic metrics"""
    text = str(text)
    if not text or text.strip() == "":
        return 0.0
    
    # Basic quality metrics
    word_count = len(text.split())
    sentence_count = len([s for s in text.split('.') if s.strip()])
    
    # Avoid division by zero
    if word_count == 0:
        return 0.0
    
    # Calculate metrics
    avg_word_length = sum(len(word.strip('.,!?')) for word in text.split()) / word_count
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Simple quality score (normalized between 0 and 1)
    # Longer words and moderate sentence length generally indicate better quality
    quality_score = min(1.0, (avg_word_length / 10.0) * (min(avg_sentence_length, 20) / 20.0))
    
    return quality_score

def analyze_text_quality(df, use_simple=False):
    """Analyze text quality using LanguageTool or simple analysis"""
    if use_simple:
        print("Using simple text quality analysis...")
        tqdm.pandas()
        df['text_quality'] = df['story_content'].progress_apply(simple_text_quality)
        print("Simple text quality analysis completed")
        return df
        
    print("Initializing LanguageTool...")
    try:
        tool = language_tool_python.LanguageTool('en-US')
        print("LanguageTool initialized successfully")
    except Exception as e:
        print(f"Error initializing LanguageTool: {e}")
        print("This usually means Java is not installed or not in PATH.")
        print("Using simple text quality analysis instead...")
        tqdm.pandas()
        df['text_quality'] = df['story_content'].progress_apply(simple_text_quality)
        print("Simple text quality analysis completed")
        return df
    
    print("Analyzing text quality (this may take a while)...")
    tqdm.pandas()
    
    # Apply grammar quality check with progress bar
    df['text_quality'] = df['story_content'].progress_apply(lambda x: grammar_quality(x, tool))
    
    print("Text quality analysis completed")
    return df

def prepare_data_for_analysis(df):
    """Prepare data for temporal analysis"""
    # Check if goal_reached column exists, if not create it
    if 'funds_raised_percent' in df.columns:
        df['goal_reached'] = (df['funds_raised_percent'] >= 100).astype(int)
        print("Goal reached column created")
    else:
        print("funds_raised_percent column not found - skipping goal_reached analysis")

    # Look for date columns and use the best available one
    date_columns = ['open_date', 'launched_at', 'created_at', 'deadline']
    open_date_col = None
    
    for col in date_columns:
        if col in df.columns:
            open_date_col = col
            break
    
    if open_date_col:
        df['open_date'] = pd.to_datetime(df[open_date_col], utc=True)
        print(f"Using '{open_date_col}' as open_date column for temporal analysis")
    else:
        print("No suitable date column found - skipping temporal analysis")
    
    return df

def calculate_monthly_trends(df):
    """Calculate monthly trends for text quality"""
    if 'open_date' not in df.columns or 'text_quality' not in df.columns:
        print("Cannot calculate monthly trends - missing required columns")
        return None, None, None
    
    # Filter out rows with NaT dates or missing text_quality
    df_clean = df.dropna(subset=['open_date', 'text_quality'])
    
    if df_clean.empty:
        print("No valid data for monthly trends after cleaning")
        return None, None, None
    
    # Group by month and compute the average text_quality (using 'ME' instead of deprecated 'M')
    monthly_avg = df_clean.groupby(pd.Grouper(key='open_date', freq='ME'))['text_quality'].mean()
    
    monthly_avg_goal_reached = None
    monthly_avg_not_reached = None
    
    if 'goal_reached' in df.columns:
        # Filter for goal reached and not reached, ensuring we have valid dates
        df_goal_reached = df_clean[df_clean['goal_reached'] == 1]
        df_goal_not_reached = df_clean[df_clean['goal_reached'] == 0]
        
        if not df_goal_reached.empty:
            monthly_avg_goal_reached = df_goal_reached.groupby(pd.Grouper(key='open_date', freq='ME'))['text_quality'].mean()
        
        if not df_goal_not_reached.empty:
            monthly_avg_not_reached = df_goal_not_reached.groupby(pd.Grouper(key='open_date', freq='ME'))['text_quality'].mean()
    
    return monthly_avg, monthly_avg_goal_reached, monthly_avg_not_reached

def plot_overall_trend(monthly_avg):
    """Plot overall text quality trend"""
    if monthly_avg is None or monthly_avg.empty:
        print("No data available for overall trend plot")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-')
    plt.title('Average Monthly Text Quality Trend')
    plt.xlabel('Month')
    plt.ylabel('Average Text Quality')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('text_quality_trend.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Overall trend plot saved as 'text_quality_trend.png'")

def plot_goal_reached_comparison(monthly_avg_goal_reached, monthly_avg_not_reached):
    """Plot comparison between goal reached and not reached"""
    if monthly_avg_goal_reached is None or monthly_avg_not_reached is None:
        print("No data available for goal reached comparison plot")
        return
    
    if monthly_avg_goal_reached.empty and monthly_avg_not_reached.empty:
        print("No data available for goal reached comparison plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    if not monthly_avg_goal_reached.empty:
        plt.plot(monthly_avg_goal_reached.index, monthly_avg_goal_reached.values, 
                marker='o', linestyle='-', label='Goal Reached')
    
    if not monthly_avg_not_reached.empty:
        plt.plot(monthly_avg_not_reached.index, monthly_avg_not_reached.values, 
                marker='s', linestyle='--', label='Goal Not Reached')

    plt.title('Average Monthly Text Quality Trend by Goal Reached Status')
    plt.xlabel('Month')
    plt.ylabel('Average Text Quality')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('text_quality_goal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Goal comparison plot saved as 'text_quality_goal_comparison.png'")

def save_results(df, output_file='text_quality_results.csv'):
    """Save results to CSV file"""
    # Create a copy without story content and html content for output
    columns_to_drop = []
    if 'story_content' in df.columns:
        columns_to_drop.append('story_content')
    if 'html_content' in df.columns:
        columns_to_drop.append('html_content')
    
    if columns_to_drop:
        df_output = df.drop(columns=columns_to_drop)
    else:
        df_output = df.copy()
    
    df_output.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total records: {len(df)}")
    
    if 'word_count' in df.columns:
        print(f"Average word count: {df['word_count'].mean():.2f}")
        print(f"Median word count: {df['word_count'].median():.2f}")
    
    if 'text_quality' in df.columns:
        valid_quality = df['text_quality'].dropna()
        if not valid_quality.empty:
            print(f"Average text quality: {valid_quality.mean():.4f}")
            print(f"Median text quality: {valid_quality.median():.4f}")
            print(f"Records with quality scores: {len(valid_quality)}")
    
    if 'goal_reached' in df.columns:
        goal_reached_count = df['goal_reached'].sum()
        print(f"Projects that reached goal: {goal_reached_count} ({goal_reached_count/len(df)*100:.1f}%)")

def main():
    """Main function to run the text quality analysis"""
    print("Starting Text Quality Analysis")
    print("=" * 50)
    
    # Check if intermediate file exists (to avoid rerunning 8+ hour analysis)
    intermediate_path = 'intermediate_with_text_quality.pkl'
    if os.path.exists(intermediate_path):
        print(f"Found intermediate file: {intermediate_path}")
        print("Loading previously calculated text quality scores...")
        df = load_data(intermediate_path)
        if df is not None and 'text_quality' in df.columns:
            print("Successfully loaded intermediate results with text quality scores!")
            # Skip to data preparation
            df = prepare_data_for_analysis(df)
        else:
            print("Intermediate file invalid, starting from scratch...")
            df = None
    else:
        df = None
    
    # If no valid intermediate file, do full analysis
    if df is None:
        # Load data
        stories_path = 'combined_standardized_with_stories_new.pkl'
        df = load_data(stories_path)
        
        if df is None:
            print("Failed to load data. Exiting.")
            return
        
        # Check if story_content column exists
        if 'story_content' not in df.columns:
            print("Error: 'story_content' column not found in the data")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Calculate word count
        df = calculate_word_count(df)
        
        # Analyze text quality (change use_simple=True for faster analysis)
        df = analyze_text_quality(df, use_simple=False)
        
        # Save intermediate results to preserve the text quality analysis
        print("Saving intermediate results with text quality scores...")
        with open('intermediate_with_text_quality.pkl', 'wb') as f:
            pickle.dump(df, f)
        print("Intermediate results saved to 'intermediate_with_text_quality.pkl'")
        
        # Prepare data for analysis
        df = prepare_data_for_analysis(df)
    
    # Calculate monthly trends
    monthly_avg, monthly_avg_goal_reached, monthly_avg_not_reached = calculate_monthly_trends(df)
    
    # Create plots
    plot_overall_trend(monthly_avg)
    plot_goal_reached_comparison(monthly_avg_goal_reached, monthly_avg_not_reached)
    
    # Save results
    save_results(df)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Files generated:")
    print("- text_quality_results.csv (main results)")
    print("- text_quality_trend.png (overall trend plot)")
    print("- text_quality_goal_comparison.png (goal comparison plot)")

if __name__ == "__main__":
    main() 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def analyze_data_quality(df):
    """Analyze data quality and filtering issues"""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows:,}")
    
    # Check launched_at column
    print(f"\nLAUNCHED_AT Column Analysis:")
    launched_at_null = df['launched_at'].isnull().sum()
    launched_at_not_null = df['launched_at'].notna().sum()
    
    print(f"  - Null values: {launched_at_null:,} ({launched_at_null/total_rows*100:.1f}%)")
    print(f"  - Non-null values: {launched_at_not_null:,} ({launched_at_not_null/total_rows*100:.1f}%)")
    
    # Try to convert dates and see what fails
    if launched_at_not_null > 0:
        print(f"  - Sample values: {df['launched_at'].dropna().head().tolist()}")
        
        # Convert to datetime and check for conversion errors
        df['launch_date_temp'] = pd.to_datetime(df['launched_at'], unit='s', errors='coerce')
        valid_dates = df['launch_date_temp'].notna().sum()
        invalid_dates = df['launched_at'].notna().sum() - valid_dates
        
        print(f"  - Valid dates after conversion: {valid_dates:,}")
        print(f"  - Invalid dates after conversion: {invalid_dates:,}")
        
        if valid_dates > 0:
            print(f"  - Date range: {df['launch_date_temp'].min()} to {df['launch_date_temp'].max()}")
        
        if invalid_dates > 0:
            print(f"  - Examples of problematic launched_at values:")
            problematic = df[df['launched_at'].notna() & df['launch_date_temp'].isna()]['launched_at'].head()
            for val in problematic:
                print(f"    {val}")
    
    # Check text_quality column
    print(f"\nTEXT_QUALITY Column Analysis:")
    text_quality_null = df['text_quality'].isnull().sum()
    text_quality_not_null = df['text_quality'].notna().sum()
    
    print(f"  - Null/missing values: {text_quality_null:,} ({text_quality_null/total_rows*100:.1f}%)")
    print(f"  - Non-null values: {text_quality_not_null:,} ({text_quality_not_null/total_rows*100:.1f}%)")
    
    if text_quality_not_null > 0:
        print(f"  - Min: {df['text_quality'].min():.6f}")
        print(f"  - Max: {df['text_quality'].max():.6f}")
        print(f"  - Mean: {df['text_quality'].mean():.6f}")
        print(f"  - Sample values: {df['text_quality'].dropna().head().tolist()}")
    
    # Combined filtering analysis
    print(f"\nCOMBINED FILTERING ANALYSIS:")
    
    # Create the same filtering conditions we use later
    df['launch_date_temp'] = pd.to_datetime(df['launched_at'], unit='s', errors='coerce')
    
    has_valid_date = df['launch_date_temp'].notna()
    has_text_quality = df['text_quality'].notna()
    both_valid = has_valid_date & has_text_quality
    
    print(f"  - Rows with valid dates: {has_valid_date.sum():,} ({has_valid_date.sum()/total_rows*100:.1f}%)")
    print(f"  - Rows with text quality: {has_text_quality.sum():,} ({has_text_quality.sum()/total_rows*100:.1f}%)")
    print(f"  - Rows with BOTH valid date AND text quality: {both_valid.sum():,} ({both_valid.sum()/total_rows*100:.1f}%)")
    
    # Break down what gets filtered out
    print(f"\nFILTERING BREAKDOWN:")
    only_missing_date = has_text_quality & ~has_valid_date
    only_missing_quality = has_valid_date & ~has_text_quality
    missing_both = ~has_valid_date & ~has_text_quality
    
    print(f"  - Rows removed due to invalid/missing date only: {only_missing_date.sum():,}")
    print(f"  - Rows removed due to missing text quality only: {only_missing_quality.sum():,}")
    print(f"  - Rows removed due to both missing: {missing_both.sum():,}")
    print(f"  - Total rows removed: {total_rows - both_valid.sum():,}")
    print(f"  - Final usable rows: {both_valid.sum():,}")
    
    # Clean up temporary column
    df.drop('launch_date_temp', axis=1, inplace=True)
    
    return both_valid

def load_and_process_data(file_path):
    """Load the data (CSV or pickle) and process it for analysis"""
    print(f"Loading data from: {file_path}")
    
    # Try to load as pickle first, then CSV
    try:
        if file_path.endswith('.pkl'):
            # Try pickle first
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                print(f"Loaded as pickle: {len(df)} records")
            except Exception as pickle_error:
                # If pickle fails, try as CSV
                print(f"Pickle loading failed: {pickle_error}")
                print("Trying as CSV...")
                df = pd.read_csv(file_path)
                print(f"Loaded as CSV: {len(df)} records")
        else:
            df = pd.read_csv(file_path)
            print(f"Loaded as CSV: {len(df)} records")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Check if required columns exist
    required_cols = ['launched_at', 'text_quality']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found!")
            return None
    
    # Perform detailed data quality analysis
    valid_mask = analyze_data_quality(df)
    
    # Convert Unix timestamp to datetime
    df['launch_date'] = pd.to_datetime(df['launched_at'], unit='s', errors='coerce')
    
    # Filter to only valid data
    df_clean = df[valid_mask & df['launch_date'].notna() & df['text_quality'].notna()].copy()
    
    print(f"\nFINAL DATASET:")
    print(f"Records remaining after all filtering: {len(df_clean)}")
    
    # Check date range
    if len(df_clean) > 0:
        print(f"Date range: {df_clean['launch_date'].min()} to {df_clean['launch_date'].max()}")
    
    return df_clean

def create_monthly_trends(df):
    """Create monthly aggregated trends"""
    if df is None or len(df) == 0:
        print("No data available for trend analysis")
        return None
    
    # Set launch_date as index for grouping
    df_indexed = df.set_index('launch_date')
    
    # Group by month and calculate mean text quality
    monthly_avg = df_indexed['text_quality'].resample('ME').mean()
    monthly_count = df_indexed['text_quality'].resample('ME').count()
    
    # Remove months with no data
    monthly_avg = monthly_avg.dropna()
    monthly_count = monthly_count[monthly_count > 0]
    
    print(f"Monthly data points: {len(monthly_avg)}")
    print(f"Date range in monthly data: {monthly_avg.index.min()} to {monthly_avg.index.max()}")
    
    return monthly_avg, monthly_count

def plot_text_quality_trend(monthly_avg, monthly_count, dataset_size=None):
    """Create and save the text quality trend plot"""
    if monthly_avg is None or len(monthly_avg) == 0:
        print("No data available for plotting")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Text Quality Trend
    ax1.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', linewidth=2, markersize=6, color='darkblue')
    
    # Add dataset size to title if provided
    title = 'Average Monthly Text Quality Trend Over Time'
    if dataset_size:
        title += f' (n={dataset_size:,} projects)'
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average Text Quality Score', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)  # Text quality is between 0 and 1
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add trend line
    if len(monthly_avg) > 1:
        z = np.polyfit(mdates.date2num(monthly_avg.index), monthly_avg.values, 1)
        p = np.poly1d(z)
        ax1.plot(monthly_avg.index, p(mdates.date2num(monthly_avg.index)), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.6f} per day)')
        ax1.legend()
    
    # Add statistics text box
    stats_text = f'Mean: {monthly_avg.mean():.4f}\nMedian: {monthly_avg.median():.4f}\nStd: {monthly_avg.std():.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Number of Projects per Month
    ax2.bar(monthly_count.index, monthly_count.values, alpha=0.7, color='skyblue', width=20)
    ax2.set_title('Number of Projects per Month', fontsize=14)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Projects', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates for second plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add total projects text
    total_projects = monthly_count.sum()
    ax2.text(0.02, 0.98, f'Total Projects: {total_projects:,}', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Create filename based on dataset size
    if dataset_size and dataset_size > 10000:
        filename = 'text_quality_trend_full_dataset.png'
    else:
        filename = 'text_quality_trend.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    print(f"Text quality trend plot saved as '{filename}'")

def print_summary_stats(df, monthly_avg):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total projects analyzed: {len(df):,}")
    print(f"Date range: {df['launch_date'].min().strftime('%Y-%m-%d')} to {df['launch_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nText Quality Statistics:")
    print(f"  Mean: {df['text_quality'].mean():.4f}")
    print(f"  Median: {df['text_quality'].median():.4f}")
    print(f"  Std Dev: {df['text_quality'].std():.4f}")
    print(f"  Min: {df['text_quality'].min():.4f}")
    print(f"  Max: {df['text_quality'].max():.4f}")
    
    if monthly_avg is not None and len(monthly_avg) > 1:
        # Calculate overall trend
        x_numeric = mdates.date2num(monthly_avg.index)
        z = np.polyfit(x_numeric, monthly_avg.values, 1)
        trend_direction = "increasing" if z[0] > 0 else "decreasing"
        print(f"\nOverall Trend:")
        print(f"  Direction: {trend_direction}")
        print(f"  Slope: {z[0]:.6f} per day")
        print(f"  Monthly change: {z[0] * 30:.6f}")

def main():
    """Main function"""
    try:
        print("Text Quality Trend Analysis")
        print("="*50)
        


        file_path = 'intermediate_with_text_quality.pkl'  # Full dataset
        
        print(f"File: {file_path}")
        df = load_and_process_data(file_path)
        
        if df is None:
            print("Failed to load data. Exiting.")
            return
        
        # Create monthly trends
        monthly_avg, monthly_count = create_monthly_trends(df)
        
        if monthly_avg is None:
            print("Failed to create monthly trends. Exiting.")
            return
        
        # Create and save plot
        print(f"Creating plot for dataset with {len(df)} records...")
        plot_text_quality_trend(monthly_avg, monthly_count, len(df))
        
        # Print summary statistics
        print_summary_stats(df, monthly_avg)
        
        print(f"\nAnalysis complete!")
        if len(df) > 10000:
            print(f"Generated file: text_quality_trend_full_dataset.png")
        else:
            print(f"Generated file: text_quality_trend.png")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
import os
import pickle
import pandas as pd
from tqdm import tqdm

# Load main CSV
print("Loading main CSV...")
df = pd.read_csv('combined_all_columns.csv', encoding='utf-8', low_memory=False)
print(f"Main CSV loaded: {len(df)} rows")
print(f"Main CSV columns: {df.columns.tolist()}")

# Check platform values
print(f"Platform values in CSV: {df['platform'].value_counts().to_dict()}")

# Check if we have project_id or id column
if 'project_id' in df.columns:
    id_col = 'project_id'
    print(f"Using 'project_id' column for matching")
else:
    id_col = 'id'
    print(f"Using 'id' column for matching")

# Convert id to string and clean up .0 suffix from floats
df[id_col] = df[id_col].astype(str).str.replace('.0', '', regex=False)
print(f"Main CSV {id_col} column type: {df[id_col].dtype}")

# Debug: Show sample IDs from CSV
print(f"Sample IDs from CSV (first 5): {df[id_col].head().tolist()}")

# --- LOAD ALL IG STORIES ---
print("Loading IG stories...")
ig_stories_list = []
ig_dir = 'IG'
for fname in tqdm(sorted(os.listdir(ig_dir))):
    if fname.endswith('.pkl'):
        with open(os.path.join(ig_dir, fname), 'rb') as f:
            batch_df = pickle.load(f)
            ig_stories_list.append(batch_df)

# Combine all IG stories
ig_stories = pd.concat(ig_stories_list, ignore_index=True)
print(f"IG stories loaded: {len(ig_stories)} stories")
print(f"IG stories columns: {ig_stories.columns.tolist()}")

# Convert id to string
ig_stories['id'] = ig_stories['id'].astype(str)
print(f"IG stories id column type: {ig_stories['id'].dtype}")

# Debug: Show sample IDs from IG
print(f"Sample IDs from IG (first 5): {ig_stories['id'].head().tolist()}")

# --- LOAD ALL KS STORIES (ROUND 1) ---
print("Loading KS stories round 1...")
ks_stories_list = []
ks_dir = 'KS/KS_Stories'
for fname in tqdm(sorted(os.listdir(ks_dir))):
    if fname.endswith('.pkl'):
        with open(os.path.join(ks_dir, fname), 'rb') as f:
            batch_df = pickle.load(f)
            ks_stories_list.append(batch_df)

# Combine all KS round 1 stories
ks_stories_round1 = pd.concat(ks_stories_list, ignore_index=True)
print(f"KS round 1 loaded: {len(ks_stories_round1)} stories")
print(f"KS round 1 columns: {ks_stories_round1.columns.tolist()}")

# Convert id to string
ks_stories_round1['id'] = ks_stories_round1['id'].astype(str)
print(f"KS round 1 id column type: {ks_stories_round1['id'].dtype}")

# Debug: Show sample IDs from KS
print(f"Sample IDs from KS (first 5): {ks_stories_round1['id'].head().tolist()}")

# --- LOAD KS ROUND 2 ---
print("Loading KS stories round 2...")
ks_stories_list = []
ks_dir = 'KS/KS_Stories_round2'
for fname in tqdm(sorted(os.listdir(ks_dir))):
    if fname.endswith('.pkl'):
        with open(os.path.join(ks_dir, fname), 'rb') as f:
            batch_df = pickle.load(f)
            ks_stories_list.append(batch_df)

ks_stories_round2 = pd.concat(ks_stories_list, ignore_index=True)
print(f"KS round 2 loaded: {len(ks_stories_round2)} stories")

# Convert id to string
ks_stories_round2['id'] = ks_stories_round2['id'].astype(str)

# --- LOAD KS ROUND 3 ---
print("Loading KS stories round 3...")
ks_stories_list = []
ks_dir = 'KS/KS_Stories_round3'
for fname in tqdm(sorted(os.listdir(ks_dir))):
    if fname.endswith('.pkl'):
        with open(os.path.join(ks_dir, fname), 'rb') as f:
            batch_df = pickle.load(f)
            ks_stories_list.append(batch_df)

ks_stories_round3 = pd.concat(ks_stories_list, ignore_index=True)
print(f"KS round 3 loaded: {len(ks_stories_round3)} stories")

# Convert id to string
ks_stories_round3['id'] = ks_stories_round3['id'].astype(str)

# --- CREATE STORY LOOKUP DICTIONARIES ---
print("Creating story lookup dictionaries...")

# Check which story column exists in IG
if 'text_content' in ig_stories.columns:
    ig_story_col = 'text_content'
elif 'story_content' in ig_stories.columns:
    ig_story_col = 'story_content'
else:
    print(f"Available IG columns: {ig_stories.columns.tolist()}")
    raise ValueError("No story content column found in IG data")

print(f"Using '{ig_story_col}' column for IG stories")

# Create IG story lookup
ig_story_lookup = {}
for _, row in ig_stories.iterrows():
    ig_story_lookup[row['id']] = row[ig_story_col]

print(f"IG lookup created with {len(ig_story_lookup)} entries")

# Create KS story lookup (round 1 first, then fill blanks with round 2, then round 3)
ks_story_lookup = {}

# Add round 1 stories
for _, row in ks_stories_round1.iterrows():
    ks_story_lookup[row['id']] = row['story_content']

# Fill blanks with round 2
for _, row in ks_stories_round2.iterrows():
    if row['id'] not in ks_story_lookup or not ks_story_lookup[row['id']]:
        ks_story_lookup[row['id']] = row['story_content']

# Fill remaining blanks with round 3
for _, row in ks_stories_round3.iterrows():
    if row['id'] not in ks_story_lookup or not ks_story_lookup[row['id']]:
        ks_story_lookup[row['id']] = row['story_content']

print(f"KS lookup created with {len(ks_story_lookup)} entries")

# --- ADD STORY CONTENT TO MAIN DATAFRAME ---
print("Adding story content to main dataframe...")

# Initialize story_content column
df['story_content'] = ''

# Add IG stories
ig_mask = df['platform'] == 'Indiegogo'
ig_ids = df[ig_mask][id_col].tolist()
print(f"Found {len(ig_ids)} Indiegogo projects")

ig_matched = 0
for project_id in ig_ids:
    if project_id in ig_story_lookup:
        df.loc[df[id_col] == project_id, 'story_content'] = ig_story_lookup[project_id]
        ig_matched += 1

print(f"Matched {ig_matched} IG stories")

# Add KS stories
ks_mask = df['platform'] == 'Kickstarter'
ks_ids = df[ks_mask][id_col].tolist()
print(f"Found {len(ks_ids)} Kickstarter projects")

ks_matched = 0
for project_id in ks_ids:
    if project_id in ks_story_lookup:
        df.loc[df[id_col] == project_id, 'story_content'] = ks_story_lookup[project_id]
        ks_matched += 1

print(f"Matched {ks_matched} KS stories")

# --- DIAGNOSTICS ---
total_rows = len(df)
ig_rows = len(df[df['platform'] == 'Indiegogo'])
ks_rows = len(df[df['platform'] == 'Kickstarter'])
ig_with_story = len(df[(df['platform'] == 'Indiegogo') & (df['story_content'].notna()) & (df['story_content'] != '')])
ks_with_story = len(df[(df['platform'] == 'Kickstarter') & (df['story_content'].notna()) & (df['story_content'] != '')])

print(f"\n=== DIAGNOSTICS ===")
print(f"Total rows: {total_rows}")
print(f"IG rows: {ig_rows}")
print(f"KS rows: {ks_rows}")
print(f"IG with story: {ig_with_story}")
print(f"KS with story: {ks_with_story}")

# Safe division to avoid ZeroDivisionError
if ig_rows > 0:
    print(f"IG story coverage: {ig_with_story/ig_rows*100:.1f}%")
else:
    print("IG story coverage: N/A (no IG rows found)")

if ks_rows > 0:
    print(f"KS story coverage: {ks_with_story/ks_rows*100:.1f}%")
else:
    print("KS story coverage: N/A (no KS rows found)")

# Verify all original columns are preserved
print(f"\nFinal columns count: {len(df.columns)}")
print(f"Original columns preserved: {len(df.columns) == len(pd.read_csv('combined_standardized.csv', nrows=0).columns) + 1}")

# Save to new CSV with error handling
print("Saving to CSV...")
output_filename = 'combined_standardized_with_stories_new.csv'
output_pkl = 'combined_standardized_with_stories_new.pkl'

try:
    # Try to save with proper error handling
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Done! Output saved to {output_filename}")
    df.to_pickle(output_pkl)
    print(f"Done! Output saved to {output_pkl}")
except PermissionError:
    # If permission denied, try with a timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'combined_standardized_with_stories_{timestamp}.csv'
    output_pkl = f'combined_standardized_with_stories_{timestamp}.pkl'
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Done! Output saved to {output_filename} (with timestamp)")
    df.to_pickle(output_pkl)
    print(f"Done! Output saved to {output_pkl} (with timestamp)")
except Exception as e:
    print(f"Error saving file: {e}")
    # Try to save to a different location
    try:
        df.to_csv('stories_output.csv', index=False, encoding='utf-8')
        print("Done! Output saved to stories_output.csv")
        df.to_pickle('stories_output.pkl')
        print("Done! Output saved to stories_output.pkl")
    except Exception as e2:
        print(f"Failed to save file: {e2}") 
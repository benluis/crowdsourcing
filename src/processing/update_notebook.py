import json
import os

notebook_path = 'analysis/did_analysis_systematic.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Target cell is index 1 (Step 1)
    if len(nb['cells']) < 2:
        print("Error: Notebook has fewer than 2 cells")
        exit(1)
        
    cell = nb['cells'][1]

    # Verify it's the correct cell
    source_str = "".join(cell['source'])
    if "# Step 1: Load Libraries" not in source_str:
        print("Error: Target cell not found at index 1")
        # Try finding it
        found = False
        for i, c in enumerate(nb['cells']):
            if "# Step 1: Load Libraries" in "".join(c['source']):
                cell = c
                found = True
                break
        if not found:
            print("Could not find Step 1 cell.")
            exit(1)

    # New code to insert
    new_code = [
        "\n",
        "# --- Create preparation_time variable ---\n",
        "print(\"Creating 'preparation_time' variable...\")\n",
        "# Convert timestamps to compatible datetime objects (UTC)\n",
        "# launched_at is typically Unix timestamp (seconds), created_at is ISO string\n",
        "launched_dt = pd.to_datetime(df['launched_at'], unit='s', utc=True, errors='coerce')\n",
        "created_dt = pd.to_datetime(df['created_at'], utc=True, errors='coerce')\n",
        "\n",
        "df['preparation_time'] = (launched_dt - created_dt).dt.days\n",
        "print(f\"✅ Created 'preparation_time' (Mean: {df['preparation_time'].mean():.1f} days)\")\n",
        "\n"
    ]

    # Find insertion point
    original_source = cell['source']
    insertion_index = -1
    
    for i, line in enumerate(original_source):
        if "# Display a summary of the new success indicator" in line:
            insertion_index = i
            break
    
    if insertion_index == -1:
        # Fallback: insert at the end
        updated_source = original_source + new_code
    else:
        updated_source = original_source[:insertion_index] + new_code + original_source[insertion_index:]
    
    cell['source'] = updated_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook updated successfully.")

except Exception as e:
    print(f"An error occurred: {e}")










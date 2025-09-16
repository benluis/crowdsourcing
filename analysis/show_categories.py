#!/usr/bin/env python3
"""
Category Analysis - Detailed Output
==================================

Shows comprehensive category distribution across Kickstarter and Indiegogo platforms.
Prints results to terminal AND saves to text file.

Usage:
    python show_categories.py
"""

import pandas as pd
import numpy as np
import json
from collections import Counter
import datetime

def show_categories():
    """Display comprehensive category analysis and save to file"""
    
    # Capture all output for saving to file
    output_lines = []
    
    def print_and_save(text=""):
        """Print to console and save to output list"""
        print(text)
        output_lines.append(text)
    
    print_and_save("🔍 === COMPREHENSIVE CATEGORY ANALYSIS ===")
    print_and_save(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_and_save()
    
    # Load the data
    print_and_save("📂 Loading data...")
    df = pd.read_pickle('processing/intermediate_with_text_quality.pkl')
    print_and_save(f"✅ Loaded {len(df):,} total projects")
    
    # Platform distribution
    print_and_save("\n📊 Platform Distribution:")
    platform_counts = df['platform'].value_counts()
    for platform, count in platform_counts.items():
        pct = count / len(df) * 100
        print_and_save(f"  {platform}: {count:,} ({pct:.1f}%)")
    
    # Data quality overview
    print_and_save("\n📋 Data Quality Overview:")
    print_and_save(f"  Total projects: {len(df):,}")
    
    # Check for date columns
    date_cols = [col for col in df.columns if 'created_at' in col or 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]  # Use first available date column
        try:
            date_min = df[date_col].min()
            date_max = df[date_col].max()
            print_and_save(f"  Date range ({date_col}): {date_min} to {date_max}")
        except:
            print_and_save(f"  Date column found but cannot parse: {date_col}")
    else:
        print_and_save("  No date columns found")
    
    # Check for quality metrics
    if 'text_quality' in df.columns:
        print_and_save(f"  Projects with text_quality: {df['text_quality'].notna().sum():,}")
    else:
        print_and_save("  No text_quality column found")
        
    if 'word_count' in df.columns:
        print_and_save(f"  Projects with word_count: {df['word_count'].notna().sum():,}")
        print_and_save(f"  Average word count: {df['word_count'].mean():.0f}")
    else:
        print_and_save("  No word_count column found")
    
    print_and_save("\n🏷️  CATEGORIES BY PLATFORM")
    print_and_save("=" * 60)
    
    # Kickstarter categories (using category_name)
    ks_data = df[df['platform'] == 'Kickstarter'].copy()
    ks_categories = ks_data['category_name'].dropna().value_counts()
    
    print_and_save(f"\n🔹 KICKSTARTER ({len(ks_data):,} projects)")
    print_and_save(f"   Column: category_name")
    print_and_save(f"   Unique categories: {len(ks_categories)}")
    print_and_save(f"   Categories with missing data: {ks_data['category_name'].isna().sum():,}")
    print_and_save(f"   ALL CATEGORIES (showing all {len(ks_categories)}):")
    
    for i, (cat, count) in enumerate(ks_categories.items(), 1):
        pct = count / len(ks_data) * 100
        print_and_save(f"   {i:2d}. {cat:<35} {count:>6,} ({pct:4.1f}%)")
    
    # Indiegogo categories (using category_parent_name)
    ig_data = df[df['platform'] == 'Indiegogo'].copy()
    ig_categories = ig_data['category_parent_name'].dropna().value_counts()
    
    print_and_save(f"\n🔹 INDIEGOGO ({len(ig_data):,} projects)")
    print_and_save(f"   Column: category_parent_name")
    print_and_save(f"   Unique categories: {len(ig_categories)}")
    print_and_save(f"   Categories with missing data: {ig_data['category_parent_name'].isna().sum():,}")
    print_and_save(f"   ALL CATEGORIES:")
    
    for i, (cat, count) in enumerate(ig_categories.items(), 1):
        pct = count / len(ig_data) * 100
        print_and_save(f"   {i:2d}. {cat:<35} {count:>6,} ({pct:4.1f}%)")
    
    # Show Indiegogo detailed tags
    print_and_save(f"\n🏷️  INDIEGOGO DETAILED TAGS (from tags column)")
    print_and_save("=" * 60)
    
    ig_tags_raw = ig_data['tags'].dropna()
    all_tags = []
    
    for tags_json in ig_tags_raw:
        try:
            tags_list = json.loads(tags_json)
            all_tags.extend(tags_list)
        except:
            continue
    
    tag_counts = Counter(all_tags)
    print_and_save(f"   Projects with tags: {len(ig_tags_raw):,}")
    print_and_save(f"   Total tag instances: {len(all_tags):,}")
    print_and_save(f"   Unique tags: {len(tag_counts)}")
    print_and_save(f"   ALL TAGS (showing all {len(tag_counts)}):")
    
    for i, (tag, count) in enumerate(tag_counts.most_common(), 1):
        pct = count / len(all_tags) * 100
        print_and_save(f"   {i:2d}. {tag:<30} {count:>4,} ({pct:4.1f}%)")
    
    # Suggest unified mapping
    print_and_save(f"\n🔗 SUGGESTED UNIFIED CATEGORIES")
    print_and_save("=" * 60)
    
    print_and_save("\nFor your DID analysis, consider these unified groups:")
    print_and_save("\n1. CREATIVE:")
    print_and_save("   Kickstarter: Illustration, Digital Art, Art, Animation, Design, Photography")
    print_and_save("   Indiegogo:   Film, Art, Photography, Dance & Theater, Web Series & TV Shows")
    
    print_and_save("\n2. PUBLISHING:")
    print_and_save("   Kickstarter: Comic Books, Fiction, Graphic Novels, Nonfiction, Anthologies")
    print_and_save("   Indiegogo:   Comics, Writing & Publishing, Podcasts/Blogs & Vlogs")
    
    print_and_save("\n3. TECHNOLOGY:")
    print_and_save("   Kickstarter: Product Design, Apps, Gadgets, Hardware")
    print_and_save("   Indiegogo:   Home, Productivity, Phones & Accessories, Energy & Green Tech")
    
    print_and_save("\n4. GAMES:")
    print_and_save("   Kickstarter: Playing Cards, Video Games, Toys")
    print_and_save("   Indiegogo:   Tabletop Games, Video Games")
    
    print_and_save("\n5. CREATIVE MEDIA:")
    print_and_save("   Kickstarter: Shorts, Drama, Comedy, Horror, Animation")
    print_and_save("   Indiegogo:   Film, Music, Audio")
    
    print_and_save("\n6. OTHER:")
    print_and_save("   Everything else...")
    
    print_and_save(f"\n📊 SAMPLE UNIFIED DISTRIBUTION")
    print_and_save("=" * 60)
    
    # Create simple unified mapping for demonstration
    ks_unified = ks_data['category_name'].map({
        'Illustration': 'Creative', 'Digital Art': 'Creative', 'Art': 'Creative',
        'Comic Books': 'Publishing', 'Fiction': 'Publishing', 'Graphic Novels': 'Publishing',
        'Product Design': 'Technology', 'Apps': 'Technology', 'Gadgets': 'Technology',
        'Playing Cards': 'Games', 'Video Games': 'Games', 'Toys': 'Games'
    }).fillna('Other')
    
    ig_unified = ig_data['category_parent_name'].map({
        'Film': 'Creative', 'Art': 'Creative', 'Photography': 'Creative',
        'Comics': 'Publishing', 'Writing & Publishing': 'Publishing',
        'Home': 'Technology', 'Productivity': 'Technology',
        'Tabletop Games': 'Games', 'Video Games': 'Games'
    }).fillna('Other')
    
    # Combine and show distribution
    all_unified = pd.concat([ks_unified, ig_unified])
    unified_counts = all_unified.value_counts()
    
    print_and_save("\nUnified category distribution (sample mapping):")
    for cat, count in unified_counts.items():
        pct = count / len(all_unified) * 100
        print_and_save(f"   {cat:<12} {count:>6,} ({pct:4.1f}%)")
    
    print_and_save(f"\n✅ Analysis complete!")
    print_and_save(f"💡 Use these insights to create your unified category mapping for the DID analysis.")
    
    # Save to file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'category_analysis_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"\n📁 Output saved to: {filename}")
    
    return df, filename


if __name__ == "__main__":
    show_categories()

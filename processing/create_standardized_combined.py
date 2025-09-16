import pandas as pd

# Read the data
ks = pd.read_csv('original/KS_Data.csv')
ig = pd.read_csv('original/IG_Data.csv')

# Add platform identifier
ks['platform'] = 'Kickstarter'
ig['platform'] = 'Indiegogo'

# Standardize columns (mapping similar fields)
ks_std = pd.DataFrame()
ig_std = pd.DataFrame()

ks_std['platform'] = ks['platform']
ig_std['platform'] = ig['platform']

ks_std['id'] = ks.get('id', pd.NA)
ig_std['id'] = ig.get('project_id', pd.NA)

ks_std['name'] = ks.get('name', pd.NA)
ig_std['name'] = ig.get('title', pd.NA)

ks_std['blurb'] = ks.get('blurb', pd.NA)
ig_std['blurb'] = ig.get('tagline', pd.NA)

ks_std['category'] = ks.get('category_parent_name', ks.get('category_name', pd.NA))
ig_std['category'] = ig.get('category', pd.NA)

ks_std['category_url'] = ks.get('category_url', pd.NA)
ig_std['category_url'] = ig.get('category_url', pd.NA)

ks_std['project_url'] = ks.get('project_url', pd.NA)
ig_std['project_url'] = ig.get('clickthrough_url', pd.NA)

ks_std['created_date'] = ks.get('created_at', pd.NA)
ig_std['created_date'] = ig.get('open_date', pd.NA)

ks_std['deadline'] = ks.get('deadline', pd.NA)
ig_std['deadline'] = ig.get('close_date', pd.NA)

ks_std['pledged_amount'] = ks.get('pledged', pd.NA)
ig_std['pledged_amount'] = ig.get('funds_raised_amount', pd.NA)

ks_std['goal_amount'] = ks.get('goal', pd.NA)
ig_std['goal_amount'] = ig.get('price_offered', pd.NA)

ks_std['backers_count'] = ks.get('backers_count', pd.NA)
ig_std['backers_count'] = ig.get('perks_claimed', pd.NA)

ks_std['currency'] = ks.get('currency', pd.NA)
ig_std['currency'] = ig.get('currency', pd.NA)

ks_std['image_url'] = ks.get('photo_full', pd.NA)
ig_std['image_url'] = ig.get('image_url', pd.NA)

ks_std['state'] = ks.get('state', pd.NA)
ig_std['state'] = ig.get('is_indemand', pd.NA)

ks_std['source_url'] = ks.get('source_url', pd.NA)
ig_std['source_url'] = ig.get('source_url', pd.NA)

# Combine standardized
combined_std = pd.concat([ks_std, ig_std], ignore_index=True)
combined_std.to_csv('combined_standardized.csv', index=False)
combined_std.to_pickle('combined_standardized.pkl')
print('Saved: combined_standardized.csv and combined_standardized.pkl (standardized columns only)')

# For all columns: union of all columns
ks_all = ks.copy()
ig_all = ig.copy()

# Rename IG columns to match KS where possible for easier union
ig_all = ig_all.rename(columns={
    'project_id': 'id',
    'title': 'name',
    'tagline': 'blurb',
    'category': 'category_parent_name',
    'clickthrough_url': 'project_url',
    'open_date': 'created_at',
    'close_date': 'deadline',
    'funds_raised_amount': 'pledged',
    'price_offered': 'goal',
    'perks_claimed': 'backers_count',
    'image_url': 'photo_full',
})

# Add missing columns to each so both have the same columns
all_cols = sorted(set(ks_all.columns) | set(ig_all.columns))
ks_all = ks_all.reindex(columns=all_cols)
ig_all = ig_all.reindex(columns=all_cols)

combined_all = pd.concat([ks_all, ig_all], ignore_index=True)
combined_all.to_csv('combined_all_columns.csv', index=False)
combined_all.to_pickle('combined_all_columns.pkl')
print('Saved: combined_all_columns.csv and combined_all_columns.pkl (all columns from both sources)') 
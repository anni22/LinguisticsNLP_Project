import pandas as pd

# Load the taxonomy data
taxonomy_df = pd.read_csv('taxonomy_all7500_tags.csv', header=None, names=['tag', 'cluster'])

# Load the language codes
lang_codes_df = pd.read_csv('languagecodes.csv', header=None, names=['tag', 'full_name'])

# Create a mapping from tag to full name
tag_to_name_map = pd.Series(lang_codes_df.full_name.values, index=lang_codes_df.tag).to_dict()

# Replace tags with full names
taxonomy_df['language'] = taxonomy_df['tag'].map(tag_to_name_map)

# Convert language names to lowercase
taxonomy_df['language'] = taxonomy_df['language'].str.lower()

# Select and save the result
result_df = taxonomy_df[['language', 'cluster']]
result_df.to_csv('taxonomy_all7500_fullnames_from_tags.csv', index=False, header=False, encoding='utf-8')

print("Conversion complete. File 'taxonomy_all7500_fullnames_from_tags.csv' created with lowercase language names.")
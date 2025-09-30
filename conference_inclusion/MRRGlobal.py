import pandas as pd
import ast

# Load data
df = pd.read_csv("AllVenues_LanguagePapers.csv")

# Fix empty language mentions by assuming English
def fix_empty_languages(langs_str):
    if pd.isna(langs_str):
        return ['english']
    langs = eval(langs_str)
    if not langs:
        return ['english']
    return langs

df['languages'] = df['languages'].apply(fix_empty_languages)

if isinstance(df.loc[0, 'languages'], str):
    df['languages'] = df['languages'].apply(ast.literal_eval)

# Get taxonomy
taxonomy_file = "languages_clean2.csv"

# Load the CSV
taxonomy = pd.read_csv(taxonomy_file, sep=';')

# Dictionary mapping: language -> class
lang_to_class = {
    str(lang).strip().lower(): str(cls).strip()
    for lang, cls in zip(taxonomy["Full Language Name"], taxonomy["Class"])
    if pd.notnull(lang) and pd.notnull(cls)
}

print(f"Loaded {len(lang_to_class)} languages into dictionary.")

# Get class of a language
def get_lang_class(lang):
    return lang_to_class.get(str(lang).lower())

# Create dataframe with all single language mentions, to which conference/year it belonged to and its class
records = []
for idx, row in df.iterrows():
    venue = row['venue']
    langs = row['languages']
    for lang in langs:
        lang_class = get_lang_class(lang)
        records.append({
            'venue': venue,
            'language': lang,
            'class': lang_class
        })

lang_df = pd.DataFrame(records)

# How often does a language appear in a conference?
freqs = lang_df.groupby(['venue', 'language']).size().reset_index(name='count')

# Add class column
freqs['class'] = freqs['language'].apply(get_lang_class)

mrr_results = []

for venue in freqs['venue'].unique():
    venue_data = freqs[freqs['venue'] == venue]

    # Assign rank (within a conference) to all languages in the venue by descending frequency
    venue_data = venue_data.sort_values('count', ascending=False).reset_index(drop=True)
    venue_data['global_rank'] = venue_data.index + 1

    for lang_class in venue_data['class'].unique():
        class_data = venue_data[venue_data['class'] == lang_class]

        if len(class_data) == 0:
            continue

        # Compute MRR using global ranks
        reciprocal_ranks = 1 / class_data['global_rank']
        mrr = reciprocal_ranks.mean()
        inverse_mrr = 1 / mrr

        mrr_results.append({
            'venue': venue,
            'class': lang_class,
            'mrr': mrr,
            'inverse_mrr': inverse_mrr
        })

mrr_df = pd.DataFrame(mrr_results)

# Reshape to have venues as rows, classes as columns
pivot_table = mrr_df.pivot(index='venue', columns='class', values='inverse_mrr')

# Sort rows and columns alphabetically/increasing
pivot_table = pivot_table.sort_index()
pivot_table = pivot_table[sorted(pivot_table.columns)]

# Output results
print("Class-wise Inverse Mean Reciprocal Rank (1/MRR) per Conference:")
print(pivot_table.round(1))
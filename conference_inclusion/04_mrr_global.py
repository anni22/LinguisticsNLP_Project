import ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

script_dir = Path(__file__).resolve().parent

df = pd.read_csv(script_dir / "AllVenues_LanguagePapers.csv.zip")

# Exclude papers with no usable text extracted
unprocessable = df["extraction_case"].isna() | df["extraction_case"].isin(
    ["no_abstract_keyword_found", "empty_text_extraction"]
)
df = df[~unprocessable].copy()

# Default to English if no language mentions found
def fix_empty_languages(langs_str):
    if pd.isna(langs_str):
        return ['english']
    langs = ast.literal_eval(langs_str)
    return langs if langs else ['english']

df['languages'] = df['languages'].apply(fix_empty_languages)

taxonomy_file = script_dir / "languages_clean.csv"
taxonomy = pd.read_csv(taxonomy_file, sep=';')

lang_to_class = {
    str(lang).strip().lower(): str(cls).strip()
    for lang, cls in zip(taxonomy["Full Language Name"], taxonomy["Class"])
    if pd.notnull(lang) and pd.notnull(cls)
}

print(f"Loaded {len(lang_to_class)} languages into dictionary.")

def get_lang_class(lang):
    return lang_to_class.get(str(lang).lower())

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

freqs = lang_df.groupby(['venue', 'language']).size().reset_index(name='count')
freqs['class'] = freqs['language'].apply(get_lang_class)

paper_counts = df.groupby('venue').size().rename('num_papers')

# Table 6: Raw inverse MRR per class (Joshi et al. 2020 formula)
raw_mrr_results = []
for venue in freqs['venue'].unique():
    venue_data = freqs[freqs['venue'] == venue].sort_values('count', ascending=False).reset_index(drop=True)
    venue_data['rank'] = venue_data.index + 1
    for lang_class in venue_data['class'].unique():
        class_data = venue_data[venue_data['class'] == lang_class]
        if len(class_data) == 0:
            continue
        mrr = (1 / class_data['rank']).mean()
        raw_mrr_results.append({'venue': venue, 'class': lang_class, 'inverse_mrr': 1 / mrr})

raw_mrr_df = pd.DataFrame(raw_mrr_results)
raw_pivot = raw_mrr_df.pivot(index='venue', columns='class', values='inverse_mrr')
raw_pivot = raw_pivot.sort_index()
raw_pivot = raw_pivot[sorted(raw_pivot.columns)]
raw_class_cols = list(raw_pivot.columns)
num_langs_per_venue_raw = freqs.groupby('venue')['language'].nunique()
raw_pivot['#Languages'] = num_langs_per_venue_raw.reindex(raw_pivot.index)
raw_pivot['#Papers'] = paper_counts.reindex(raw_pivot.index)

print("\nTable 6 -- Raw (unnormalized) Inverse Mean Reciprocal Rank per Conference (Joshi et al. 2020 formula, unmodified):")
print(raw_pivot.round(2))

raw_out_file = script_dir / "mrr_by_class_raw.csv"
raw_pivot.round(3).to_csv(raw_out_file)
print(f"Saved results to {raw_out_file}")

# Table 7: Class-wise share of language mentions per venue
class_counts = freqs.groupby(['venue', 'class'])['count'].sum().reset_index(name='class_mentions')
venue_totals = freqs.groupby('venue')['count'].sum().rename('total_mentions')
class_counts = class_counts.join(venue_totals, on='venue')
class_counts['share'] = class_counts['class_mentions'] / class_counts['total_mentions']

pivot_table = class_counts.pivot(index='venue', columns='class', values='share')
pivot_table = pivot_table.sort_index()
pivot_table = pivot_table[sorted(pivot_table.columns)]

class_cols = list(pivot_table.columns)
num_langs_per_venue = freqs.groupby('venue')['language'].nunique()
pivot_table['#Languages'] = num_langs_per_venue.reindex(pivot_table.index)
pivot_table['#Papers'] = paper_counts.reindex(pivot_table.index)

print("Class-wise Share of Language Mentions per Conference:")
print(pivot_table.round(3))

class_colors = {
    '0': '#968bdc',
    '1': '#e34948',
    '2': '#aef2ae',
    '3': '#ffe245',
    '4': '#ffb99d',
    '5': '#0068c9',
}

def _text_color_for(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 140 else '#0b0b0b'

venues_top_to_bottom = list(pivot_table.index)[::-1]
fig, ax = plt.subplots(figsize=(9, 0.5 * len(pivot_table) + 1.5))

for venue in venues_top_to_bottom:
    left = 0.0
    for cls in reversed(class_cols):
        val = 100 * pivot_table.loc[venue, cls]
        ax.barh(venue, val, left=left, color=class_colors[cls],
                edgecolor='white', linewidth=1, height=0.7)
        if cls in ('0', '5'):
            ax.text(left + val / 2, venue, f"{val:.0f}%", ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')
        left += val

ax.set_xlim(0, 100)
ax.set_xlabel('Share of language mentions (%)', fontsize=14, fontweight='bold')
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12, length=0)
ax.grid(axis='x', color='#e1e0d9', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#c3c2b7')

legend_order = list(reversed(class_cols))
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=class_colors[c]) for c in legend_order]
ax.legend(legend_handles, [f"Class {c}" for c in legend_order],
          loc='lower center', bbox_to_anchor=(0.5, 1.02),
          ncol=6, frameon=False, fontsize=12)

plt.tight_layout()
bar_file = script_dir / "mrr_by_class_stacked_bar.png"
plt.savefig(bar_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved stacked bar chart to {bar_file}")

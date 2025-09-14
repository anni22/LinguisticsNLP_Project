import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from math import log

# Load data
df = pd.read_csv("AllVenues_LanguagePapers25.csv")

# Fix empty language mentions by assuming English
def fix_empty_languages(languages_str):
    if pd.isna(languages_str):
        return ['english']
    languages = eval(languages_str)
    if not languages:
        return ['english']
    return languages

df['languages'] = df['languages'].apply(fix_empty_languages)

conferences = ["ACL", "CL", "COLING", "CONLL", "EACL", "EMNLP", "LREC",
               "NAACL", "SEMEVAL", "TACL", "WS"]

# For entropy calculation
def entropy(prob_dist):
    """Calculate entropy for a probability distribution array"""
    return -sum(p * log(p) for p in prob_dist if p > 0)

# Extract all languages from the dataset to get language index
all_languages = sorted(set(lang for langs in df['languages'] for lang in langs)) #should I maybe add all languages here that did not get mention once in any conference paper?!!
lang_to_idx = {lang: idx for idx, lang in enumerate(all_languages)}

# structure to hold entropy results: {conference: {year: entropy_value}}
entropy_results = defaultdict(dict)

# Group by conference and year
grouped = df.groupby(['venue', 'year'])

for (conf, year), group in grouped:
    if conf not in conferences:
        continue
    
    # Number of papers in this conf-year
    P = len(group)
    L = len(all_languages)
    
    # Build binary matrix M_{P x L}
    # Rows = papers, Columns = languages
    M = np.zeros((P, L), dtype=int)
    
    for i, langs in enumerate(group['languages']):
        for lang in langs:
            if lang in lang_to_idx:
                M[i, lang_to_idx[lang]] = 1
    
    # Sum across papers for each language: occurrence count S_j
    S = M.sum(axis=0)
    
    # Normalize to get probability distribution S' = S / total mentions
    total_mentions = S.sum()
    if total_mentions == 0:
        entropy_val = 0.0  # no languages mentioned means no entropy
    else:
        prob_dist = S / total_mentions
        entropy_val = entropy(prob_dist)
    
    entropy_results[conf][year] = entropy_val

# Convert results to DataFrame for easier plotting/analysis
entropy_df = []

for conf in conferences:
    years = sorted(entropy_results[conf].keys())
    for y in years:
        entropy_df.append({"conference": conf, "year": y, "entropy": entropy_results[conf][y]})

entropy_df = pd.DataFrame(entropy_df)


# Save entropy data for further comparison if needed
entropy_df.to_csv("Language_Occurrence_Entropy.csv", index=False)
print("Entropy data saved to Language_Occurrence_Entropy.csv")

unique_languages = sorted(set(lang for langs in df['languages'] for lang in langs))
print(f"Total unique languages mentioned (including assumed 'english'): {len(unique_languages)}")
print(f"Languages: {', '.join(unique_languages)}")


# Optional: Plot entropy over years per conference (like Joshi et al.'s Figure 4)
import matplotlib.cm as cm

plt.figure(figsize=(12, 8))

# Use tab20 colormap for distinct colors
colors = cm.get_cmap('tab20', len(conferences))

for idx, conf in enumerate(conferences):
    sub_df = entropy_df[entropy_df['conference'] == conf]
    if not sub_df.empty:
        plt.plot(sub_df['year'], sub_df['entropy'], marker='o', label=conf, color=colors(idx))

plt.xlabel("Year")
plt.ylabel("Language Occurrence Entropy")
plt.title("Language Occurrence Entropy by Conference (2020–2025)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Conference")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
output_file = "language_entropy_plot_all_papers.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_file}")
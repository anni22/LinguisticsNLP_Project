import ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from math import log

script_dir = Path(__file__).resolve().parent

df = pd.read_csv(script_dir / "AllVenues_LanguagePapers.csv.zip")

# Exclude papers with no usable text extracted
unprocessable = df["extraction_case"].isna() | df["extraction_case"].isin(
    ["no_abstract_keyword_found", "empty_text_extraction"]
)
df = df[~unprocessable].copy()

# Default to English if no language mentions found
def fix_empty_languages(languages_str):
    if pd.isna(languages_str):
        return ['english']
    languages = ast.literal_eval(languages_str)
    if not languages:
        return ['english']
    return languages

df['languages'] = df['languages'].apply(fix_empty_languages)

conferences = ["ACL", "NAACL", "EMNLP", "EACL", "COLING",
               "CL", "WS_other", "CONLL", "SEMEVAL", "LREC"]

subplot_labels = [f"({chr(97+i)})" for i in range(len(conferences))]

def entropy(prob_dist):
    terms = []
    for p in prob_dist:
        if p > 0:
            terms.append(p * log(p))
    return -sum(terms)

all_languages = sorted(set(lang for langs in df['languages'] for lang in langs))
lang_to_idx = {lang: idx for idx, lang in enumerate(all_languages)}

entropy_results = defaultdict(dict)

for (conf, year), group in df.groupby(['venue', 'year']):
    if conf not in conferences:
        continue
    
    P = len(group)
    L = len(all_languages)
    M = np.zeros((P, L), dtype=int)
    
    for i, langs in enumerate(group['languages']):
        for lang in langs:
            if lang in lang_to_idx:
                M[i, lang_to_idx[lang]] = 1
    
    S = M.sum(axis=0)
    total_mentions = S.sum()
    if total_mentions == 0:
        entropy_val = 0.0
    else:
        prob_dist = S / total_mentions
        entropy_val = entropy(prob_dist)
    
    entropy_results[conf][year] = entropy_val

# Convert results to DataFrame
entropy_df = []
for conf in conferences:
    years = sorted(entropy_results[conf].keys())
    for y in years:
        entropy_df.append({"conference": conf, "year": y, "entropy": entropy_results[conf][y]})

entropy_df = pd.DataFrame(entropy_df)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
axes = axes.flatten()

plot_color = "deepskyblue"

for idx, conf in enumerate(conferences):
    ax = axes[idx]
    sub_df = entropy_df[entropy_df['conference'] == conf]
    if not sub_df.empty:
        ax.plot(sub_df['year'], sub_df['entropy'],
                marker='o', markersize=2,
                color=plot_color, linewidth=1)
        ax.set_title(f"{subplot_labels[idx]} {conf}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(entropy_df['year'].min(), 2024)
        ax.set_ylim(entropy_df['entropy'].min(), entropy_df['entropy'].max())
        ax.set_xlabel("Year", fontsize=11)
        ax.tick_params(axis='both', labelsize=10)

for j in range(len(conferences), len(axes)):
    fig.delaxes(axes[j])

fig.text(0.01, 0.5, "Entropy", va="center", rotation="vertical", fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0.02, 0.05, 1, 1])
plt.savefig(script_dir / "language_entropy_subplots.png", dpi=300)
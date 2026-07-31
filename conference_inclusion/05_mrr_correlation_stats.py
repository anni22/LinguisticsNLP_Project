from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

script_dir = Path(__file__).resolve().parent

raw = pd.read_csv(script_dir / "mrr_by_class_raw.csv")
classes = [str(c) for c in range(6)]

rows = []
for c in classes:
    r_lang, p_lang = stats.pearsonr(raw[c], raw["#Languages"])
    r_pap, p_pap = stats.pearsonr(raw[c], raw["#Papers"])
    rows.append({
        "Class": c,
        "r(#Languages)": round(r_lang, 2),
        "p(#Languages)": round(p_lang, 4),
        "r(#Papers)": round(r_pap, 2),
        "p(#Papers)": round(p_pap, 4),
    })

stats_df = pd.DataFrame(rows)
print("Pearson correlation of raw inverse MRR with venue size (n=11 venues):")
print(stats_df.to_string(index=False))

print("\n" + "="*80)
print("LaTeX table (ready to paste into appendix):")
print("="*80)

latex_rows = []
for idx, row in stats_df.iterrows():
    latex_rows.append(
        f"  {row['Class']} & {row['r(#Languages)']:6.2f} & "
        f"{row['p(#Languages)']:7.4f} & {row['r(#Papers)']:6.2f} & {row['p(#Papers)']:7.4f} \\\\"
    )

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{r r r r r}")
print(r"\toprule")
print(r"Class & $r$ (\#Languages) & $p$ & $r$ (\#Papers) & $p$ \\")
print(r"\midrule")
for row in latex_rows:
    print(row)
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Pearson correlation between class-wise inverse MRR and venue size ($n=11$ venues). "
      r"Raw inverse MRR correlates strongly with both \#Languages and \#Papers for Classes 0--3 ($p<.005$).}")
print(r"\label{tab:mrr_correlations}")
print(r"\end{table}")

class_colors = {
    "0": "#968bdc", "1": "#e34948", "2": "#aef2ae",
    "3": "#ffe245", "4": "#ffb99d", "5": "#0068c9",
}

fig, ax = plt.subplots(figsize=(8, 5))

for c in classes:
    x = raw["#Languages"]
    y = raw[c]
    r, p = stats.pearsonr(x, y)

    ax.scatter(x, y, color=class_colors[c], s=80, alpha=0.7,
               label=f"Class {c} (r={r:.2f}, p={p:.3f})", edgecolor="white", linewidth=0.5)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=class_colors[c], linewidth=1.5, alpha=0.6)

ax.set_xlabel("Number of Unique Languages Mentioned (#Languages)", fontsize=14, fontweight='bold')
ax.set_ylabel("Raw Inverse MRR (1/MRR)", fontsize=14, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc="upper left", fontsize=12, framealpha=0.95)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
fig_file = script_dir / "mrr_correlation_scatter.png"
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
print(f"\nSaved scatter plot to {fig_file}")

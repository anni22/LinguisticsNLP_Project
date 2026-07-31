from pathlib import Path
import pandas as pd

script_dir = Path(__file__).resolve().parent
source_file = script_dir.parent / "languagetaxonomy" / "taxonomy_all7500_fullnames.csv"
out_file = script_dir / "languages_clean.csv"

# Words that coincide with common English words, names, places, institutions,
# or non-language taxonomy artifacts -- see paper Section "Taxonomy for
# Conference-Language Inclusion Analysis".
EXCLUDED_WORDS = {
    "even", "label", "mono", "male", "anal", "fore",
    "sake", "bench", "broken", "thompson", "she", "aka",
    "bit", "day", "con", "duke", "yale",
    "are", "gen", "fur", "mal", "pal", "dem", "neo", "una", "mum", "gun",
    "sur", "ami", "sie", "tha", "mer",
    "kim", "gao", "dan", "sam", "dai", "ali", "zou", "che", "cen", "rao",
    "kang", "mae", "sara", "gal", "laura", "abu",
    "colorado", "alabama", "miami", "notre", "roma", "seneca", "titan",
    "pinyin", "multiple languages",
    "median", "nasal",
    "bilin", "tal", "dia", "shi", "dong", "ding", "wan", "chung", "dom",
    "agi", "yao", "pare", "ere",
    "evant",
}

df = pd.read_csv(source_file, header=None, names=["Full Language Name", "Class"])
df["Full Language Name"] = df["Full Language Name"].str.strip().str.lower()
total = len(df)

short = df["Full Language Name"].str.len() <= 2
df = df[~short]
removed_short = int(short.sum())

is_excluded_word = df["Full Language Name"].isin(EXCLUDED_WORDS)
df = df[~is_excluded_word]
removed_words = int(is_excluded_word.sum())

df = df.sort_values("Full Language Name").reset_index(drop=True)
df.to_csv(out_file, sep=";", index=False)

print(f"Loaded {total} languages from {source_file.name}")
print(f"Removed {removed_short} language names of two letters or fewer")
print(f"Removed {removed_words} language names matching common words/names/places")
print(f"Saved {len(df)} languages to {out_file}")

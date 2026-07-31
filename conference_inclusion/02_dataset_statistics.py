from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent

main_venues = ["ACL", "NAACL", "EMNLP", "EACL", "TACL", "CONLL",
               "COLING", "LREC", "SEMEVAL", "WS_other", "CL"]
main_venues_set = set(main_venues) - {"WS_other"}

df = pd.read_csv(script_dir / "AllVenues_LanguagePapers.csv.zip")

paper_counts = df.groupby("venue").size()
unprocessable = df["extraction_case"].isna() | df["extraction_case"].isin(
    ["no_abstract_keyword_found", "empty_text_extraction"]
)
unprocessable_counts = df[unprocessable].groupby("venue").size()


from_workshop_counts = {}
for venue in main_venues_set:
    is_workshop = df['original_venue'].str.startswith('W', na=False)
    in_venue = df['venue'] == venue
    from_workshop_counts[venue] = (is_workshop & in_venue).sum()

# --------------------------------------------------------------
# Assemble the table
# --------------------------------------------------------------
def format_excluded(count, total):
    pct = round(100 * count / total, 2) if total else 0.0
    return f"{count} ({pct}%)"


rows = []
for venue in main_venues:
    total = int(paper_counts.get(venue, 0))
    unproc = int(unprocessable_counts.get(venue, 0))
    # WS_other papers are, by definition, workshops that could NOT be
    # resolved to a host venue, so this column doesn't apply to that row.
    from_ws = total if venue == "WS_other" else from_workshop_counts.get(venue, 0)

    rows.append({
        "Venue": venue,
        "Total Papers": total,
        "From WS": from_ws,
        "Excluded (unproc.)": format_excluded(unproc, total),
        "Used in analysis": total - unproc,
    })

stats_df = pd.DataFrame(rows)
total_papers = stats_df["Total Papers"].sum()
total_unproc = stats_df["Total Papers"].sum() - stats_df["Used in analysis"].sum()
total_row = {
    "Venue": "Total",
    "Total Papers": total_papers,
    "From WS": stats_df["From WS"].sum(),
    "Excluded (unproc.)": format_excluded(total_unproc, total_papers),
    "Used in analysis": stats_df["Used in analysis"].sum(),
}
stats_df = pd.concat([stats_df, pd.DataFrame([total_row])], ignore_index=True)

print("Dataset statistics across all conferences:")
print(stats_df.to_string(index=False))

out_file = script_dir / "dataset_statistics.csv"
stats_df.to_csv(out_file, index=False)
print(f"\nSaved results to {out_file}")

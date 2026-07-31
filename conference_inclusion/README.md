## Conference Language Inclusion

This folder contains the code and pipeline for **Section 3: Conference-Language Inclusion** of the paper. It analyzes language diversity across 11 NLP conferences (10 main venues + workshops) by extracting language mentions from paper abstracts and the first 4 pages of PDFs, then computing statistics on language representation across different resource classes. Follow the instructions below to reproduce all tables and figures from Section 3.

### Setup

```
pip install -r requirements.txt
```

Requires a local clone of the [ACL Anthology data repository](https://github.com/acl-org/acl-anthology). By default, scripts look for `acl-anthology/` in this directory. Override with:

```
ACL_ANTHOLOGY_DATA_DIR=/path/to/acl-anthology python 01_collect_data.py
```

### Reproducibility

The scripts were run with **ACL Anthology data as of August 19, 2025** (commit `55ffb4b4`).
To reproduce with the same data version:

```bash
cd acl-anthology
git checkout 55ffb4b4
```

Without pinning the Anthology version, future runs may produce different paper counts or workshop attributions as the Anthology is continuously updated.

### Run Order

1. `00_preprocess_taxonomy.py` — generates `languages_clean.csv`
2. `01_collect_data.py` — generates `AllVenues_LanguagePapers.csv.zip`
3. `02_dataset_statistics.py` — generates `dataset_statistics.csv` (Table 5)
4. `03_analysis_entropy.py` — generates `language_entropy_subplots.png` (Figure 4)
5. `04_mrr_global.py` — generates `mrr_by_class_raw.csv` (Table 7) and `mrr_by_class_stacked_bar.png` (Figure 6)
6. `05_mrr_correlation_stats.py` — generates `mrr_correlation_scatter.png` (Figure 7). Depends on step 5.

Steps 3–5 are independent and can run in any order after step 2. Step 6 depends on step 5.

### Outputs

- **Table 5** — `dataset_statistics.csv`: Papers per venue, workshops resolved, unprocessable papers
- **Figure 4** — `language_entropy_subplots.png`: Language entropy over time
- **Table 7** — `mrr_by_class_raw.csv`: Raw inverse MRR by class
- **Figure 6** — `mrr_by_class_stacked_bar.png`: Class-wise share of language mentions per venue
- **Figure 7** — `mrr_correlation_scatter.png`: Correlation analysis showing why MRR is size-dependent

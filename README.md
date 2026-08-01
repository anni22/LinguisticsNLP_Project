# Revisiting Joshi et al. (2020): Methodological Gaps in the Analysis of Language Resource Distribution and NLP Conference Inclusion

This repository contains code and data for a critical reassessment of Joshi et al. (2020)'s influential analysis of language resource distribution in NLP research, accepted to [KONVENS 2026](https://www.inf.uni-hamburg.de/en/inst/ab/lt/konvens2026/) in Hamburg.

## Citation

```bibtex
@inproceedings{kuznetcova_spirgath2026revisiting,
  title={Revisiting {J}oshi et al. (2020): Methodological Gaps in the Analysis of Language Resource Distribution and {NLP} Conference Inclusion},
  author={Kuznetcova, Polina and Spirgath, Annika},
  booktitle={Proceedings of KONVENS 2026},
  year={2026},
  address={Hamburg, Germany},
  note={Accepted}
}
```

## Contribution

While Joshi et al. (2020) provides an influential taxonomy of languages by resource availability, the original data sets and collection procedures were never made publicly available, leaving the widely-cited six-level taxonomy without a verifiable empirical foundation.

This work identifies and addresses five key methodological limitations:

1. **Standardized language identification** via ISO639-3 (resolving naming inconsistencies that caused duplicate/missing entries)
2. **Principled distinction** between macrolanguages and individual varieties (e.g., Arabic vs. Egyptian Arabic)
3. **Reproducible data collection pipeline** from ELRA and LDC catalogs
4. **Targeted language-mention detection** (distinguishing genuine research from mere mentions)
5. **Size-independent metric** (replacing inverse MRR with share-of-mentions for fair cross-venue comparison)

## Repository Structure

- **`language_taxonomy/`** — Language classification: six-level taxonomy with standardized naming
  - See [language_taxonomy/README.md](language_taxonomy/README.md)

- **`conference_inclusion/`** — Conference-language inclusion analysis with corrected methodology
  - Reproducible pipeline for 2024 data across 10 major NLP venues
  - See [conference_inclusion/README.md](conference_inclusion/README.md)

- **`Revisiting_Joshi_et_al_2020.pdf`** — Full paper accepted to KONVENS 2026


## References

Joshi, P., Santy, S., Bugliarello, A., Bisk, Y., & Schwartz, R. (2020). The State and Fate of Linguistic Diversity and Multilingualism in NLP. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 6282–6293.

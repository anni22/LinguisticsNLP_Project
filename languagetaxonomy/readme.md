## Files Overview
### Python Scripts

*   `dashboard.py`: The main Streamlit application that provides an interactive dashboard for exploring language resource distribution and taxonomy comparisons. This dashboard includes visualizations of language clusters based on resource counts and comparisons between different taxonomies.
*   `convert_tags_to_names.py`: This script probably handles the conversion of ISO 639-3 codes to full language names.
*   `scrape_ldc_catalogs.py`: This script is designed to scrape data from LDC (Linguistic Data Consortium) catalogs to gather information about available linguistic resources.
*   `lang2tax_MED.py`: This script was used in an attempt to standardise language names in Joshi et al's (2020) taxonomy using Minimum Edit Distance (MED) approach.

### Data Files
*   `resource_distribution_with_speakers.csv`: Contains data on the distribution of linguistic resources (labelled and unlabelled) for languages, including speaker counts. This file is used for clustering and data visualisation.
*   `resource_distribution_with_speakers_and_all_languages.csv`: Similar to the above, but includes data for all 7500 languages.
*   `taxonomy_all7500_fullnames.csv`: Contains the taxonomy for all 7500 languages, with full language names and their assigned clusters.
*   `taxonomy_all7500_tags.csv`: Contains the taxonomyfor all 7500 languages, using ISO tags and their assigned clusters.
*   `taxonomy_original_fullnames.csv`: Contains the taxonomy for 333 languages with resources, with full language names and their assigned clusters.
*   `taxonomy_original_tags.csv`: Contains the taxonomyfor for 333 languages with resources, using ISO tags and their assigned clusters.
*   `DatasetsPerLanguage.xlsx`: Contains detailed information about data processing and language name standardisation.
*   `lang2tax_MED_suggestions.txt`: Results of the unsuccessful attempt to standardide languge names from the Joshi et al.'s (2020) taxonomy using MED.

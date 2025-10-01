## Files Overview
### Python Scripts

*   `All_Data_Final.py`: This is the main python script that downloads all conference papers from Anthology, extracts the text from the PDF if necessary and measures language mentions with our refined measurement algorithm (see paper section 3.2.1). It saves the data in the file `AllVenues_LanguagePapers.csv`.
*   `analysis_entropy.py`: This script calculates the language occurrence entropy values for all conferences and all years and plots the results (see figure 4 in the paper).
*   `MRRGlobal.py`: This script calculates the MRR values for all conferences and classes and outputs the results to the terminal.

### Data Files

*   `AllVenues_LanguagePapers.csv`: This is the data file resulting from the `All_Data_Final.py` script. It shows all conference papers through 2024, their abstracts if available, all language mentions and the place where the language mention was found according to the algorithm (either abstract_only, abstract_plus4pages, abstract_missing_snippet_only or abstract_missing_snippet_plus_4pages)

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# The full URL of the page with the search tool:
WEBSITE_URL = "https://catalog.ldc.upenn.edu/search" 

# The CSS Selector for the search button:
# Right-click the search button on the website -> Inspect -> 
# Right-click the highlighted HTML -> Copy -> Copy selector ('input[type="submit"]').
SEARCH_BUTTON_SELECTOR = 'input[type="submit"]' 

def get_languages(driver):
    """Navigates to the page and scrapes all language names and their values."""
    print("Fetching the list of languages...")
    driver.get(WEBSITE_URL)
    languages = {}
    try:
        select_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'q_languages_id_in'))
        )
        # All <option> elements within the dropdown:
        options = select_element.find_elements(By.TAG_NAME, 'option')
        for option in options:
            lang_name = option.text
            lang_value = option.get_attribute('value')
            if lang_name and lang_value:
                # Store as { 'Language Name': 'id_value' }
                languages[lang_name] = lang_value
        print(f"Successfully found {len(languages)} languages to process.")
        return languages
    except Exception as e:
        print(f"Error: Could not find the language dropdown. Please check the URL. Details: {e}")
        return None

def get_catalogs_for_language(driver, lang_name, lang_value):
    """Searches for a single language and scrapes its catalog numbers."""
    try:
        # Navigate to the search page:
        driver.get(WEBSITE_URL)
        
        # Find a language option by its value and click it:
        lang_option_selector = f'#q_languages_id_in > option[value="{lang_value}"]'
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, lang_option_selector))
        ).click()
        # Find the search button and and click it:
        driver.find_element(By.CSS_SELECTOR, SEARCH_BUTTON_SELECTOR).click()

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'catalog-id'))
        )
        
        # Find all elements containing catalog numbers:
        catalog_elements = driver.find_elements(By.CSS_SELECTOR, 'td.catalog-id a')
        catalogs = [elem.text for elem in catalog_elements]
        
        print(f"Found {len(catalogs)} catalogs for '{lang_name}'")
        return catalogs

    except Exception:
        print(f"-> No catalogs found or an error occurred for '{lang_name}'")
        return []

def main():
    """Main function to orchestrate the scraping process."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') 
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Step 1: Get all languages from the dropdown:
    languages = get_languages(driver)
    if not languages:
        driver.quit()
        return

    # Step 2: Loop through each language and scrape its catalogs, or datasets:
    all_results = []
    for lang_name, lang_value in languages.items():
        catalogs = get_catalogs_for_language(driver, lang_name, lang_value)
        
        # We store each dataset on a new row:
        if catalogs:
            for catalog_id in catalogs:
                all_results.append({'Language': lang_name, 'Catalog_Number': catalog_id})
        else:
            # In case no datasets are found:
            all_results.append({'Language': lang_name, 'Catalog_Number': 'N/A'})

    # Step 3: Close the browser:
    driver.quit()

    # Step 4: Convert results to a pandas DataFrame and save to CSV:
    if all_results:
        df = pd.DataFrame(all_results)
        output_filename = 'language_catalogs.csv'
        df.to_csv(output_filename, index=False)

# Step 5: Create a summary DataFrame with counts:
        print("\nCreating a summary file with catalog counts...")

        # Calculates the count for each language. 
        # Handles cases with no catalogs (where Catalog_Number is 'N/A').
        counts = df.groupby('Language')['Catalog_Number'].apply(
            lambda x: 0 if (x.iloc[0] == 'N/A') else x.count()
        ).reset_index(name='Catalog_Count')

        # Sort in descending order:
        counts_df = counts.sort_values(by='Catalog_Count', ascending=False)

        summary_filename = 'language_catalog_counts.csv'
        counts_df.to_csv(summary_filename, index=False)

        print(f"Summary saved to '{summary_filename}'")
        print(f"\n All data has been saved to '{output_filename}'")
    else:
        print("\nCould not retrieve any data.")

if __name__ == "__main__":
    main()
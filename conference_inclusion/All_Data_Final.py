from acl_anthology import Anthology
import os
import pandas as pd
from tqdm import tqdm
import re
import fitz
import requests
import io
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

# -------------------
# Paths
# -------------------
acl_path = "/Users/annikaspirgath/Master Heidelberg/LinguisticsNLP/Final_Project/GitHub/LinguisticsNLP_Project/acl-anthology"
data_dir = os.path.join(acl_path, "data")
taxonomy_file = os.path.join(
    "/Users/annikaspirgath/Master Heidelberg/LinguisticsNLP/Final_Project/GitHub/LinguisticsNLP_Project",
    "languages_clean2.csv"
)

# -------------------
# Load Anthology
# -------------------
anthology = Anthology(datadir=data_dir)

# -------------------
# Venues to include
# -------------------
main_venues = {"ACL", "NAACL", "EMNLP", "EACL", "TACL", "CL",
               "CONLL", "COLING", "LREC", "SEMEVAL"}

def get_venue(p):
    if not p.venue_ids:
        return None
    vid = p.venue_ids[0].upper()
    if vid in main_venues:
        return vid
    elif vid.startswith("W"):  # all workshops
        return "WS"
    else:
        return None

# ----------------------
# Load language taxonomy
# ----------------------
languages = []
df = pd.read_csv(taxonomy_file, sep=';')
languages = [str(lang).strip().lower() for lang in df["Full Language Name"] if pd.notnull(lang)]
languages = list(set(languages))

print(f"✅ Loaded {len(languages)} unique languages from taxonomy.")

# ----------------------------
# Define Patterns and Keywords
# ----------------------------

lang_pattern = re.compile(
    r'\b(' + '|'.join(map(re.escape, languages)) + r')\b',
    flags=re.IGNORECASE
) 

patterns = {
    "typological": re.compile(r"\btypological\b", re.IGNORECASE),
    "multilingual": re.compile(r"\bmultilingual\b", re.IGNORECASE),
    "num_languages": re.compile(r"(\d+)\s+languages", re.IGNORECASE),
    "cross_lingual": re.compile(r"\bcross[- ]lingual\b", re.IGNORECASE),
    "low_resource_languages": re.compile(r"\blow[- ]resource languages\b", re.IGNORECASE)
}

def mentions_any_pattern(text):
    if not isinstance(text, str):
        return False
    return any(regex.search(text) for regex in patterns.values())

# -----------------------------------------
# Setup persistent session with retry logic
# -----------------------------------------
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# -------------------------------------
# Layout detection + extraction helpers
# -------------------------------------
def detect_layout(page, threshold=100):
    """Detect whether a page is single- or double-column."""
    blocks = page.get_text("blocks")
    if not blocks:
        return "unknown"

    x0s = [b[0] for b in blocks]
    if not x0s:
        return "unknown"

    unique_x0s = sorted(set(int(x) for x in x0s))
    if len(unique_x0s) <= 1:
        return "single"

    diffs = [b - a for a, b in zip(unique_x0s, unique_x0s[1:])]
    max_gap = max(diffs) if diffs else 0

    return "double" if max_gap > threshold else "single"

def extract_first_page_text(doc):
    """Return text from first page depending on detected layout."""
    try:
        page = doc[0]
        layout = detect_layout(page)
        blocks = page.get_text("blocks")
        if not blocks:
            return None, layout

        if layout == "double":
            x0s = [b[0] for b in blocks]
            median_x0 = sorted(x0s)[len(x0s)//2]
            left_blocks = [b[4] for b in blocks if b[0] <= median_x0]
            return (" ".join(left_blocks) if left_blocks else None), layout
        else:
            blocks.sort(key=lambda b: (b[1], b[0]))
            return " ".join([b[4] for b in blocks]), layout
    except Exception:
        return None, "unknown"

def extract_text_pymupdf(pdf_bytes, max_pages=4):
    """Extract text from first max_pages, column-aware."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            layout = detect_layout(page)
            blocks = page.get_text("blocks")
            if not blocks:
                continue
            if layout == "double":
                x0s = [b[0] for b in blocks]
                median_x0 = sorted(x0s)[len(x0s)//2]
                blocks.sort(key=lambda b: (b[1], b[0]))
                left = " ".join([b[4] for b in blocks if b[0] <= median_x0])
                right = " ".join([b[4] for b in blocks if b[0] > median_x0])
                page_text = left + " " + right
            else:
                blocks.sort(key=lambda b: (b[1], b[0]))
                page_text = " ".join([b[4] for b in blocks])
            text_pages.append(page_text)
        return " ".join(text_pages) if text_pages else None
    except Exception:
        return None

def fetch_and_parse_pdf(paper_id, url):
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        pdf_bytes = io.BytesIO(r.content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return paper_id, pdf_bytes, doc
    except Exception:
        return paper_id, None, None

# ---------------------
# Clean extracted text
# ---------------------

def clean_text_for_language_matching(text):
    
    # Remove hyphen + line break (merge words)
    text = re.sub(r'-[\r\n]+', '', text)
    # Replace remaining line breaks with space
    text = re.sub(r'[\r\n]+', ' ', text)

    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower()

def extract_abstract_snippet(text, window=150):
    """Find 'abstract' and return ~window words after it."""
    m = re.search(r'\babstract\b', text, re.IGNORECASE)
    if not m:
        return None
    after_text = text[m.end():]
    words = after_text.split()
    return " ".join(words[:window])

# -----------------------
# Get all relevant Papers
# -----------------------
print("\nFiltering papers across all target conferences...")
papers_to_process = []
papers_to_fetch = []

for p in anthology.papers():
    if not p.year:
        continue
    year = int(p.year)

    if year > 2024:
        continue

    if not getattr(p.type, "value", None) == "paper":
        continue

    venue = get_venue(p)
    if not venue:
        continue

    if venue == "WS" and year < 1990:
        continue

    abstract_text = str(p.abstract) if p.abstract else ""
    needs_full_text = (not abstract_text.strip()) or mentions_any_pattern(abstract_text)
    papers_to_process.append((p, venue, abstract_text, needs_full_text))
    if needs_full_text and getattr(p, "pdf", None):
        papers_to_fetch.append((p.id, f"https://aclanthology.org/{p.pdf.name}.pdf"))

print(f"Selected papers: {len(papers_to_process)}")
print(f"Papers needing full text download: {len(papers_to_fetch)}")

# ----------------------
# Download PDFs parallel
# ----------------------
pdf_docs = {}
pdf_bytes_map = {}

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(fetch_and_parse_pdf, pid, url) for pid, url in papers_to_fetch]
    for future in tqdm(futures, desc="Downloading PDFs", unit="pdf"):
        pid, pdf_bytes, doc = future.result()
        if pdf_bytes:
            pdf_bytes_map[pid] = pdf_bytes
        if doc:
            pdf_docs[pid] = doc

# ---------------
# Main Processing
# ---------------
stats = defaultdict(lambda: {
    "papers": 0,
    "missing_abs": 0,
    "missing_id": 0,
    "fulltext_requested": 0,
    "fulltext_missing": 0,         
    "no_abstract_keyword_found": 0,
    "empty_text_extraction": 0
})

papers_data = [] 

for p, venue, abstract_text, needs_full_text in tqdm(papers_to_process, desc="Processing papers"):
    stats[venue]["papers"] += 1

    if not abstract_text.strip():
        stats[venue]["missing_abs"] += 1
    if not p.id:
        stats[venue]["missing_id"] += 1

    extraction_case = None
    full_text_checked = abstract_text
    layout_detected = "unknown"

    if needs_full_text and p.id in pdf_docs:
        stats[venue]["fulltext_requested"] += 1
        doc = pdf_docs[p.id]
        pdf_bytes = pdf_bytes_map[p.id]

        if abstract_text.strip():
            if mentions_any_pattern(abstract_text):
                pages_text = extract_text_pymupdf(pdf_bytes, max_pages=4)
                if pages_text is None:
                    full_text_checked = abstract_text
                    extraction_case = "empty_text_extraction"
                else:
                    full_text_checked = abstract_text + "\n" + pages_text
                    extraction_case = "abstract_plus_4pages"
                try:
                    layout_detected = detect_layout(doc[0])
                except Exception:
                    layout_detected = "unknown"
            else:
                full_text_checked = abstract_text
                extraction_case = "abstract_only"
        else:
            first_page_text, layout_detected = extract_first_page_text(doc)
            if first_page_text is None:
                full_text_checked = ""
                extraction_case = "empty_text_extraction"
            else:
                snippet = extract_abstract_snippet(first_page_text, window=150)
                if snippet:
                    if mentions_any_pattern(snippet):
                        pages_text = extract_text_pymupdf(pdf_bytes, max_pages=4)
                        if pages_text is None:
                            full_text_checked = snippet
                            extraction_case = "empty_text_extraction"
                        else:
                            full_text_checked = snippet + "\n" + pages_text
                            extraction_case = "abstract_missing_snippet_plus_4pages"
                    else:
                        full_text_checked = snippet
                        extraction_case = "abstract_missing_snippet_only"
                else:
                    full_text_checked = ""
                    extraction_case = "no_abstract_keyword_found"

        if not full_text_checked.strip():
            stats[venue]["fulltext_missing"] += 1
            if extraction_case == "no_abstract_keyword_found":
                stats[venue]["no_abstract_keyword_found"] += 1
                reason = "no_abstract_keyword_found"
            else:
                stats[venue]["empty_text_extraction"] += 1
                reason = "empty_text_extraction"
          

    elif abstract_text.strip():
        extraction_case = "abstract_only"

    text_clean = clean_text_for_language_matching(full_text_checked)
    matches = set(lang_pattern.findall(f" {text_clean} "))

    papers_data.append({
        "id": p.id,
        "title": p.title,
        "year": int(p.year),
        "venue": venue,
        "abstract": abstract_text,
        "languages": sorted(set(m.lower() for m in matches)),
        "mentions_language": len(matches) > 0,
        "extraction_case": extraction_case,
        "layout_detected": layout_detected
    })

# -------------------
# Save results
# -------------------
df = pd.DataFrame(papers_data)
out_file = "AllVenues_LanguagePapers.csv"
df.to_csv(out_file, index=False)

print(f"\nSaved results to {out_file}")
print(f"Papers mentioning languages: {df['mentions_language'].sum()} / {len(df)}")

print("\nSummary stats:")
for venue, s in stats.items():
    print(f"{venue:<10} papers={s['papers']:<6} "
          f"missing_abs={s['missing_abs']:<6} "
          f"no_abs_keyword={s['no_abstract_keyword_found']:<6} "
          f"empty_text={s['empty_text_extraction']:<6} "
          f"missing_id={s['missing_id']:<6} "
          f"fulltext_req={s['fulltext_requested']:<6} "
          f"fulltext_missing={s['fulltext_missing']:<6}")

print("\nProcessing finished for all venues.")
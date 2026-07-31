from acl_anthology import Anthology
import os
import zipfile
from pathlib import Path
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
from flashtext import KeywordProcessor

# -------------------
# Paths
# -------------------
script_dir = Path(__file__).resolve().parent
acl_path = os.environ.get(
    "ACL_ANTHOLOGY_DATA_DIR", str(script_dir / "acl-anthology")
)
data_dir = os.path.join(acl_path, "data")
taxonomy_file = script_dir / "languages_clean.csv"

# Downloaded PDFs are cached here (keyed by the paper's globally-unique
# full_id) so re-running the script after a crash/interruption doesn't
# re-download everything from scratch.
pdf_cache_dir = script_dir / "pdf_cache"
pdf_cache_dir.mkdir(exist_ok=True)

# -------------------
# Load Anthology
# -------------------
anthology = Anthology(datadir=data_dir)

# -------------------
# Venues to include
# -------------------
main_venues = {"ACL", "NAACL", "EMNLP", "EACL", "TACL", "CL",
               "CONLL", "COLING", "LREC", "SEMEVAL"}

def resolve_host_venue(volume):
    """Resolve a workshop volume to whichever of the 10 main_venues it was
    co-located with, if any."""
    candidates = set()
    try:
        events = volume.get_events()
    except Exception:
        return None
    for event in events:
        try:
            colocated_volumes = event.volumes()
        except Exception:
            continue
        for other in colocated_volumes:
            if other.full_id == volume.full_id:
                continue
            if getattr(other, "is_workshop", False):
                continue
            try:
                other_venues = other.venues()
            except Exception:
                continue
            for venue in other_venues:
                vid = venue.id.upper()
                if vid in main_venues:
                    candidates.add(vid)
    if not candidates:
        return None
    if len(candidates) > 1:
        workshop_stats["ambiguous_host"] += 1
    return sorted(candidates)[0]

workshop_stats = defaultdict(int)

def get_venue(p):
    if not p.venue_ids:
        return None
    vid = p.venue_ids[0].upper()
    if vid in main_venues:
        return vid

    volume = p.parent
    if not getattr(volume, "is_workshop", False):
        # Not one of our 10 main venues and not a workshop either
        # (e.g. a standalone venue outside our scope) -> excluded.
        return None

    workshop_stats["total_workshop_papers"] += 1
    host = resolve_host_venue(volume)
    if host is not None:
        workshop_stats["resolved_to_host"] += 1
        return host

    workshop_stats["fell_back_to_ws_other"] += 1
    return "WS_other"

# ----------------------
# Load language taxonomy
# ----------------------
df = pd.read_csv(taxonomy_file, sep=';')
languages = sorted(set(str(lang).strip().lower() for lang in df["Full Language Name"] if pd.notnull(lang)))

print(f"Loaded {len(languages)} unique languages from taxonomy.")

# ----------------------------
# Define Patterns and Keywords
# ----------------------------

# Using flashtext library which is far faster thanregex. It also prefers
# longest matches, preventing "german" from shadowing "german sign language".
keyword_processor = KeywordProcessor(case_sensitive=False)
for lang in languages:
    keyword_processor.add_keyword(lang)

patterns = {
    "typological": re.compile(r"\btypological\b", re.IGNORECASE),
    "multilingual": re.compile(r"\bmulti[- ]?lingual\b", re.IGNORECASE),
    "num_languages": re.compile(r"(\d+)\s+languages", re.IGNORECASE),
    "cross_lingual": re.compile(r"\bcross[- ]?lingual\b", re.IGNORECASE),
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

def extract_text_pymupdf(doc, max_pages=4):
    """Extract text from first max_pages, column-aware."""
    try:
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
    """Download (or load from local cache) a PDF and return its parsed
    fitz.Document."""
    cache_path = pdf_cache_dir / f"{paper_id.replace('/', '_')}.pdf"
    try:
        if cache_path.exists():
            content = cache_path.read_bytes()
        else:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            content = r.content
            cache_path.write_bytes(content)
        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
        return paper_id, doc
    except Exception:
        return paper_id, None

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

# Strip ACL Anthology proceedings headers so they don't consume the word
# budget meant for paper content.
PROCEEDINGS_HEADER_RE = re.compile(
    r'proceedings of.*?(?:association for computational linguistics|'
    r'european language resources association|'
    r'international committee on computational linguistics)',
    re.IGNORECASE | re.DOTALL
)

def extract_generic_snippet(text, skip_words=20, window=150):
    """Fallback for papers without explicit 'abstract' keyword (common in
    older CL papers). Strips header, skips title/author lines, takes next
    ~150 words. Lower precision than extract_abstract_snippet, but
    acceptable for keyword signal detection."""
    text = PROCEEDINGS_HEADER_RE.sub(' ', text)
    words = text.split()
    snippet_words = words[skip_words:skip_words + window]
    if not snippet_words:
        return None
    return " ".join(snippet_words)

# -----------------------
# Get all relevant Papers
# -----------------------
# Paper.id is per-volume only; Paper.full_id is globally unique.
# Use full_id as dict keys to avoid silent collisions.
print("\nFiltering papers across all target conferences...")
papers_to_process = []
fetch_url_by_id = {}

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

    if venue == "WS_other" and year < 1990:
        continue

    abstract_text = str(p.abstract) if p.abstract else ""
    needs_full_text = (not abstract_text.strip()) or mentions_any_pattern(abstract_text)
    papers_to_process.append((p, venue, abstract_text, needs_full_text))
    if needs_full_text and getattr(p, "pdf", None):
        fetch_url_by_id[p.full_id] = f"https://aclanthology.org/{p.pdf.name}.pdf"

print(f"Selected papers: {len(papers_to_process)}")
print(f"Papers needing full text download: {len(fetch_url_by_id)}")

# --------------------------------------------------------------
# Download + process in bounded batches
# --------------------------------------------------------------

BATCH_SIZE = 500

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
out_file = script_dir / "AllVenues_LanguagePapers.csv"

num_batches = (len(papers_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(num_batches), desc="Batches"):
    batch = papers_to_process[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]

    batch_fetch = [
        (p.full_id, fetch_url_by_id[p.full_id])
        for p, _, _, needs_full_text in batch
        if needs_full_text and p.full_id in fetch_url_by_id
    ]

    pdf_docs = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_and_parse_pdf, pid, url) for pid, url in batch_fetch]
        for future in tqdm(futures, desc=f"Downloading PDFs (batch {batch_idx + 1}/{num_batches})",
                            unit="pdf", leave=False):
            pid, doc = future.result()
            if doc:
                pdf_docs[pid] = doc

    for p, venue, abstract_text, needs_full_text in batch:
        stats[venue]["papers"] += 1

        if not abstract_text.strip():
            stats[venue]["missing_abs"] += 1
        if not p.full_id:
            stats[venue]["missing_id"] += 1

        extraction_case = None
        full_text_checked = abstract_text
        layout_detected = "unknown"

        if needs_full_text and p.full_id in pdf_docs:
            stats[venue]["fulltext_requested"] += 1
            doc = pdf_docs[p.full_id]

            if abstract_text.strip():
                if mentions_any_pattern(abstract_text):
                    pages_text = extract_text_pymupdf(doc, max_pages=4)
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
                    used_generic_snippet = False
                    if not snippet:
                        # No explicit "abstract" keyword found -- fall
                        # back to a generic snippet instead of excluding
                        # the paper outright (see extract_generic_snippet).
                        snippet = extract_generic_snippet(first_page_text, skip_words=20, window=150)
                        used_generic_snippet = snippet is not None
                    if snippet:
                        if mentions_any_pattern(snippet):
                            pages_text = extract_text_pymupdf(doc, max_pages=4)
                            if pages_text is None:
                                full_text_checked = snippet
                                extraction_case = "empty_text_extraction"
                            elif used_generic_snippet:
                                full_text_checked = snippet + "\n" + pages_text
                                extraction_case = "generic_snippet_plus_4pages"
                            else:
                                full_text_checked = snippet + "\n" + pages_text
                                extraction_case = "abstract_missing_snippet_plus_4pages"
                        else:
                            full_text_checked = snippet
                            extraction_case = "generic_snippet_only" if used_generic_snippet else "abstract_missing_snippet_only"
                    else:
                        full_text_checked = ""
                        extraction_case = "no_abstract_keyword_found"

            if not full_text_checked.strip():
                stats[venue]["fulltext_missing"] += 1
                if extraction_case == "no_abstract_keyword_found":
                    stats[venue]["no_abstract_keyword_found"] += 1
                else:
                    stats[venue]["empty_text_extraction"] += 1

        elif abstract_text.strip():
            extraction_case = "abstract_only"

        text_clean = clean_text_for_language_matching(full_text_checked)
        matches = set(keyword_processor.extract_keywords(text_clean))


        original_venue = p.venue_ids[0].upper() if p.venue_ids else None

        papers_data.append({
            "id": p.full_id,
            "title": p.title,
            "year": int(p.year),
            "venue": venue,
            "original_venue": original_venue,
            "abstract": abstract_text,
            "languages": sorted(matches),
            "mentions_language": len(matches) > 0,
            "extraction_case": extraction_case,
            "layout_detected": layout_detected
        })

    for doc in pdf_docs.values():
        doc.close()
    pdf_docs.clear()

    # Checkpoint: overwrite the output file after every batch so progress
    # survives a crash/OOM instead of only being saved at the very end.
    pd.DataFrame(papers_data).to_csv(out_file, index=False)

# -------------------
# Save results
# -------------------
df = pd.DataFrame(papers_data)

print(f"Papers mentioning languages: {df['mentions_language'].sum()} / {len(df)}")

# Create zipped CSV (downstream scripts read from this)
zip_file = out_file.with_suffix(".csv.zip")
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr(out_file.name, df.to_csv(index=False))
print(f"Saved zipped results to {zip_file}")

if out_file.exists():
    out_file.unlink()

print("\nSummary stats:")
for venue, s in stats.items():
    print(f"{venue:<10} papers={s['papers']:<6} "
          f"missing_abs={s['missing_abs']:<6} "
          f"no_abs_keyword={s['no_abstract_keyword_found']:<6} "
          f"empty_text={s['empty_text_extraction']:<6} "
          f"missing_id={s['missing_id']:<6} "
          f"fulltext_req={s['fulltext_requested']:<6} "
          f"fulltext_missing={s['fulltext_missing']:<6}")

print("\nWorkshop -> host-conference resolution:")
print(f"  Total workshop papers seen:  {workshop_stats['total_workshop_papers']}")
print(f"  Resolved to a host venue:    {workshop_stats['resolved_to_host']}")
print(f"  Fell back to WS_other:       {workshop_stats['fell_back_to_ws_other']}")
print(f"  Had >1 candidate host (tie-broken alphabetically): {workshop_stats['ambiguous_host']}")

print("\nProcessing finished for all venues.")
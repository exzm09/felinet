"""
Wikipedia Cat Breed scraper.
Discovers breed URLs from the "List of cat breeds" table, extracts clean text using trafilatura, validates with Pydantic and save as JSON.
Wikipedia is permissive for polite crawlers - use a 3-second delay.
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup
from felinet.schemas import ContentType, DataSource, SourceDocument

logger = logging.getLogger(__name__)

WIKI_BASE = "https://en.wikipedia.org"
BREED_LIST_URL = f"{WIKI_BASE}/wiki/List_of_cat_breeds"
CRAWL_DELAY = 3  # Wikipedia is more permissive than Cornell

def create_session() -> requests.Session:
    """
    Create a requests Session with proper headers.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "FeliNet/0.1 (respectful crawling)",
        "Accept": "text/html"
    })
    return session

def discover_breed_urls(session: requests.Session) -> list[dict]:
    """
    Stage 1: Visite the "List of cat breeds" page and collect breed names and URLs from the wikipedia.
    Returns a list of dicts: [{"name": "Abyssinian", "url": "https://..."}, ...]
    """
    logger.info(f"Discovering breeds from {BREED_LIST_URL}")

    response = session.get(BREED_LIST_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # There is exactly one wikipedia on this page
    table = soup.find("table", class_="wikitable")
    if not table:
        raise ValueError("Could not find the breed wikitable")
    
    breeds = []
    rows = table.find_all("tr")[1:] # Skip header row

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        # First cell has the breed name and link
        first_cell = cells[0]
        link = first_cell.find("a")
        if not link or not link.get("href"):
            continue
        href = link["href"]

        # Skip non-article links (ciation footnotes)
        if not href.startswith("/wiki/"):
            continue

        # Clean the breed name - remove footnote reference
        raw_name = first_cell.get_text(strip=True)
        clean_name = re.sub(r"\[\d+\]", "", raw_name).strip()

        breeds.append({
            "name": clean_name,
            "url": f"{WIKI_BASE}{href}"
        })
    logger.info(f"Discovered {len(breeds)} breed URLs")
    return breeds

def make_document_id(breed_name: str) -> str:
    """
    Generate a stable document ID from the breed name.
    "American Bobtail" -> "wiki_american-bobtail"
    """
    slug = breed_name.lower().replace(" ", "-")
    # Remove any characters that aren't letters, numbers, or hyphens
    slug = re.sub(f"[^a-z0-9-]", "", slug)
    return f"wiki-{slug}"

def scrape_wikipedia_breeds(
        output_dir: str = "data/raw/wikipedia",
        max_breeds: int | None = None
) -> list[SourceDocument]:
    """
    Main entry point: ran the full Wikipedia breeds scraping pipeline.

    Args:
        output_dir: where to save the JSON output
        max_breeds: if set, only scrape this many (useful for testing)

    Returns:
        List of validated SourceDocument objects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = create_session()

    breeds = discover_breed_urls(session)

    if max_breeds:
        breeds = breeds[:max_breeds]
        logger.info(f"Limited to {max_breeds} breeds for testing")

    # Stage 2 & 3: Extract + Validate
    documents: list[SourceDocument] = []
    failed: list[dict] = []

    for i, breed in enumerate(breeds):
        name = breed["name"]
        url = breed["url"]
        logger.info(f"[{i + 1}/{len(breeds)}] Scraping: {name}")

        # Fetch the page
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"   Failed to fetch: {e}")
            failed.append({
                "breed": name,
                "url": url,
                "reason": f"fetch failure: {e}"
            })
            time.sleep(CRAWL_DELAY)
            continue

        # Extract content with trafilatura
        content = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            include_links=False
        )

        if not content or len(content.strip()) < 50:
            logger.warning(f"   No usable content extracted")
            failed.append({
                "breed": name,
                "url": url,
                "reason": "extraction failure"
            })
            time.sleep(CRAWL_DELAY)
            continue

        # Validate with Pydantic
        try:
            doc = SourceDocument(
                id=make_document_id(name),
                source=DataSource.WIKIPEDIA,
                url=url,
                title=name,
                content=content.strip(),
                content_type=ContentType.BREED_PROFILE,
                metadata={
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "breed_name": name
                }
            )
            documents.append(doc)
            logger.info(f"  Finished {name} ({doc.metadata["word_count"]} words)")
        except Exception as e:
            failed.append({
                "breed": name, 
                "url": url,
                "reason": str(e)
            })
            logger.warning(f"   Validation failed: {e}")

        # Wait between requests
        if i < len(breeds) - 1:
            time.sleep(CRAWL_DELAY)

    # Save results
    docs_data = [doc.model_dump(mode="json") for doc in documents]

    output_file = output_path / "wikipedia_breeds.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone! {len(documents)} breeds saved to {output_file}")
    logger.info(f"{len(failed)} breeds failed")

    if failed:
        fail_file = output_path / "wikipedia_failues.json"
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"Failure log saved to {fail_file}")

    return documents

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    docs = scrape_wikipedia_breeds()
    print(f"\nScraped {len(docs)} breeds successfully")
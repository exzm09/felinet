"""
CFA (Cat Fanciers' Association) Breed Scraper.
Discovers breed URLs from the CFA breeds index page, ectract clean text using trafilatura, validates with Pydantic, and saves as JSON.
CFA is a smaller site - use a 5-sec dalay to be respectful.
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

CFA_BASE = "https://cfa.org"
BREEDS_INDEX_URL = f"{CFA_BASE}/breeds/"
CRAWL_DELAY = 5  # Respectful delay for a smaller organization site
# Full list of CFA-recognized breeds with their URL slugs
# Source: https://cfa.org/breeds/ (verified April 2026)
# CFA recognizes 45 breeds (42 Championship + 3 Miscellaneous/Provisional)
CFA_BREEDS = [
    {"name": "Abyssinian", "slug": "abyssinian"},
    {"name": "American Bobtail", "slug": "american-bobtail"},
    {"name": "American Curl", "slug": "american-curl"},
    {"name": "American Shorthair", "slug": "american-shorthair"},
    {"name": "American Wirehair", "slug": "american-wirehair"},
    {"name": "Balinese", "slug": "balinese"},
    {"name": "Bengal", "slug": "bengal"},
    {"name": "Birman", "slug": "birman"},
    {"name": "Bombay", "slug": "bombay"},
    {"name": "British Shorthair", "slug": "british-shorthair"},
    {"name": "Burmese", "slug": "burmese"},
    {"name": "Burmilla", "slug": "burmilla"},
    {"name": "Chartreux", "slug": "chartreux"},
    {"name": "Colorpoint Shorthair", "slug": "colorpoint-shorthair"},
    {"name": "Cornish Rex", "slug": "cornish-rex"},
    {"name": "Devon Rex", "slug": "devon-rex"},
    {"name": "Egyptian Mau", "slug": "egyptian-mau"},
    {"name": "European Burmese", "slug": "european-burmese"},
    {"name": "Exotic", "slug": "exotic"},
    {"name": "Havana Brown", "slug": "havana-brown"},
    {"name": "Japanese Bobtail", "slug": "japanese-bobtail"},
    {"name": "Khao Manee", "slug": "khao-manee"},
    {"name": "Korat", "slug": "korat"},
    {"name": "LaPerm", "slug": "laperm"},
    {"name": "Lykoi", "slug": "lykoi"},
    {"name": "Maine Coon Cat", "slug": "maine-coon-cat"},
    {"name": "Manx", "slug": "manx"},
    {"name": "Norwegian Forest Cat", "slug": "norwegian-forest-cat"},
    {"name": "Ocicat", "slug": "ocicat"},
    {"name": "Oriental", "slug": "oriental"},
    {"name": "Persian", "slug": "persian"},
    {"name": "Ragamuffin", "slug": "ragamuffin"},
    {"name": "Ragdoll", "slug": "ragdoll"},
    {"name": "Russian Blue", "slug": "russian-blue"},
    {"name": "Scottish Fold", "slug": "scottish-fold"},
    {"name": "Selkirk Rex", "slug": "selkirk-rex"},
    {"name": "Siamese", "slug": "siamese"},
    {"name": "Siberian", "slug": "siberian"},
    {"name": "Singapura", "slug": "singapura"},
    {"name": "Somali", "slug": "somali"},
    {"name": "Sphynx", "slug": "sphynx"},
    {"name": "Tonkinese", "slug": "tonkinese"},
    {"name": "Toybob", "slug": "toybob"},
    {"name": "Turkish Angora", "slug": "turkish-angora"},
    {"name": "Turkish Van", "slug": "turkish-van"},
]

def create_session() -> requests.Session:
    """
    Create a requests Session with proper headers.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "FeliNet/0.1 (respectful crawling)",
            "Accept": "text/html"
        }
    )
    return session

def discover_breed_urls(session: requests.Session) -> list[dict]:
    """
    Return the full list of CFA breed URLs.
    Uses a hardcoded breed list because the CFA breeds page lazy-load content, making dynamic discovery ureliable without a headless browser.
    Returns:
        List of dicts: [{"name": "Abyssinian", "url": "https://cfa.org/abyssinian/"}, ...]
    """
    breeds = [{
        "name": breed["name"],
        "url": f"{CFA_BASE}/breed/{breed['slug']}/"
    }
    for breed in CFA_BREEDS]
    logger.info(f"Loaded {len(breeds)} CFA breed URLs from static list")
    return breeds

def make_document_id(breed_name: str) -> str:
    """
    Generate a stable document ID from the breed name.
    "American Bobtail" -> "cfa_american-bobtail"
    """
    slug = breed_name.lower().replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    return f"cfa_{slug}"

def scrape_cfa_breeds(
        output_dir: str = "data/raw/cfa",
        max_breeds: int | None = None
) -> list[SourceDocument]:
    """
    Main entry point: run the full CFA breeds scraping pipeline.
    Args:
        output_dir: where to save the JSON output
        max_breeds: if set, only scrape this many (useful for testing)

    Returns:
        List of validated SourceDocument objects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = create_session()

    # Stage 1: Discover
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

        # Extract contengt with trafilatura
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
            time.sleep(30)
            continue

        # Validate with Pydantic
        try:
            doc = SourceDocument(
                id=make_document_id(name),
                source=DataSource.CFA,
                url=url,
                title=f"{name} - CFA Breed Profile",
                content=content.strip(),
                content_type=ContentType.BREED_PROFILE,
                metadata={
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "breed_name": name,
                    "registry": "CFA"
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

    output_file = output_path / "cfa_breeds.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone! {len(documents)} breeds saved to {output_file}")
    logger.info(f"{len(failed)} breeds failed")

    if failed:
        fail_file = output_path / "cfa_failures.json"
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
    docs = scrape_cfa_breeds()
    print(f"\nScraped {len(docs)} breeds successfully")
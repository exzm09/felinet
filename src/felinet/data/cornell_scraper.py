"""
Discover -> Extract -> Validate & Save
Cornell Feline Health Center Scraper
Discover articale URLs from the index page, extracts clean test using trafilatura, validates with Pydantic, and save as JSON.
Respoects robots.txt: 8-second delay between requests (> 6s requirement).
"""
import json
import logging
import time
from pathlib import Path
# Combinw a base URL with a relative link handling edge cases (trailing slashes, absolute vs. relative paths)
from urllib.parse import urljoin

import requests
import trafilatura
from bs4 import BeautifulSoup

from felinet.schemas import ContentType, DataSource, SourceDocument

# Visibility
logger = logging.getLogger(__name__)

BASE_URL = "https://www.vet.cornell.edu"
TOPICS_INDEX_URL = (
    f"{BASE_URL}/departments-centers-and-institutes/"
    "cornell-feline-health-center/health-information/feline-health-topics"
)
CRAWL_DELAY = 7

def create_session() -> requests.Session:
    """
    Create a requests Session with proper headers.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "FeliNet/0.1 (student ML project; respectful crawling)",
        "Accept": "text/html"
    })
    return session

def discover_article_urls(session: requests.Session) -> list[str]:
    """
    Visit the index page and collect all article URLs.
    The index page lists articles in three URL patterns:
    1. /feline-health-topics/[slug]  - "pretty" article URLs
    2. /node/[id]                    - CMS internal URLs (redirect to pretty URLs)
    3. /health-information/[slug]    - some articles sit one level up
    Capture all three patterns and skip non-article links (nagivation, links, external sites, etc.)
    """
    logger.info(f"Discovering articles from {TOPICS_INDEX_URL}")
    response = session.get(TOPICS_INDEX_URL, timeout=30)
    # Throws an error if the request fails instead of silently returning bad data.
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    # URLs that are not articles (navigation or section pages)
    skip_urls = {
        TOPICS_INDEX_URL,
        f"{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center",
        f"{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/health-information",
        f"{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/about-center",
        f"{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/health-studies",
        f"{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/professional-education",
        f"{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/givingmembership",
    }

    # Paths that indicate non-article pages
    skip_patterns = [
        "/camuti-consultation-service",
        "/catwatch",
        "/client-information-brochures",
    ]
    # Filter out the index page itself and any anchor links.
    article_urls = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(BASE_URL, href)

        # Skip external links, anchors, videos, emails, phone numbers
        if not full_url.startswith(f"{BASE_URL}/"):
            continue
        if any(x in full_url for x in ["#", "youtube.com", "mailto:", "tel:"]):
            continue

        # Skip known non-article pages
        if full_url in skip_urls:
            continue
        if any(pattern in full_url for pattern in skip_patterns):
            continue

        # Pattern 1
        if "/feline-health-topics/" in full_url and full_url != TOPICS_INDEX_URL:
            article_urls.add(full_url)
        # Pattern 2
        # Content Management System internal IDs on the feline health topics page, every /node/ link points to a feline health article
        elif "/node/" in full_url:
            article_urls.add(full_url)
        # Pattern 3
        # Some articles live directly under /health-information/ without being under /feline-health-topics/. Skip known section pages.
        elif "cornell-feline-health-center/health-information/" in full_url:
            path_after = full_url.split("health-information/")[-1]
            # Must have content after and not be a section
            if path_after and "/" not in path_after:
                article_urls.add(full_url)
    urls = sorted(article_urls) # for reproducibility
    logger.info(f"Discovered {(len(urls))} article URLs")
    return urls

def extract_article(session: requests.Session, url: str) -> dict | None:
    """
    Download one article and extract clean text.
    Return a dict with title, content, url, or None if extraction fails.

    trafilatura does the heavy lifting; it identifies the main content area of the page and strips nevigation, footers, sidebars, etc.
    """
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None
    
    html = response.text

    # Return the main article text stripping user comments is any exist and keep data tables.
    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False
    )
    if not content or len(content.strip()) < 50:
        logger.warning(f"No usable content extracted from {url}")
        return None
    
    # Extract title from the HTML <title> tag or <h1>
    soup = BeautifulSoup(html, "lxml")
    title = None

    # Try <h1> first - usually more specific than <title>
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=None)

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            # <title> often includes site name
            title = title_tag.get_text(strip=True).split("|")[0].strip()
    
    if not title:
        logger.warning(f"No title found for {url}")
        return None
    
    return {
        "url": url,
        "title": title,
        "content": content.strip()
    }

def classify_content_type(title: str) -> ContentType:
    """
    Simple keyword-based classification of article content type.
    Could upgrade to ML-based classification later.
    """
    title_lower = title.lower()
    
    # Order matters: check more specific categories first
    if any(kw in title_lower for kw in ["poison", "toxic", "hazard", "plant"]):
        return ContentType.TOXICOLOGY
    if any(kw in title_lower for kw in ["feed", "nutrition", "diet", "obesity", "food"]):
        return ContentType.NUTRITION
    if any(kw in title_lower for kw in ["behavior", "aggression", "litter", "scratch", "lick", "soiling", "destructive"]):
        return ContentType.BEHAVIOR
    if any(kw in title_lower for kw in ["breed", "persian", "siamese", "maine coon"]):
        return ContentType.BREED_PROFILE
    if any(kw in title_lower for kw in ["disease", "virus", "cancer", "diabetes", "kidney", "asthma", "infection", "fiv", "leukemia", "heart", "liver", "urinary"]):
        return ContentType.DISEASE

    # Default: general health article
    return ContentType.ARTICLE

def make_document_id(url: str) -> str:
    """
    Generate a stable document ID from the URL.
    Takes the last part of the URL path (slug) and prefixed it with 'cornell_'.
    It should be stable because same URL always produced the same ID - important for deduplication and DVC tracking.
    """
    slug = url.strip("/").split("/")[-1]
    return f"cornell_{slug}"

def scrape_cornell(
        output_dir: str = "data/raw/cornell",
        max_articles: int | None = None
) -> list[SourceDocument]:
    """
    Main entry point: run the full Cornell scraping pipeline.
    Args:
        output_dir: where to save the JSON output
        max_articles: if set, only scrape this many (useful for testing)

    Returns:
        List of validated SourceDocument objects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = create_session()

    # Stage 1 - Discover
    urls = discover_article_urls(session)

    if max_articles:
        urls = urls[:max_articles]
        logger.info(f"Limited to {max_articles} articles for testing")

    # Stage 2 & 3 - Extract & Validate
    documents: list[SourceDocument] = []
    failed: list[dict] = []
    seen_urls: set[str] = set() # Track final URLs to avoid duplicates

    for i, url in enumerate(urls):
        logger.info(f"[{i + 1} / {len(urls)}] Scraping: {url}")

        # Fetch the page - requests follows redirects automatically. Check the final URL to detect dupliates.
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"    Failed to fetch: {e}")
            failed.append({"url": url, "reason": f"fetch failure: {e}"})
            time.sleep(CRAWL_DELAY)
            continue

        # response.url is the FINAL url after any redirects.
        final_url = response.url
        if final_url in seen_urls:
            logger.info(f"  Duplicate (already scraped ad {final_url})")
            time.sleep(CRAWL_DELAY)
            continue
        seen_urls.add(final_url)

        # Extract
        # Return the main article text stripping user comments is any exist and keep data tables.
        html = response.text
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False
        )
        if not content or len(content.strip()) < 50:
            logger.warning(f"No usable content extracted from {url}")
            failed.append({"url": final_url, "reason": "extraction failure"})
            time.sleep(CRAWL_DELAY)
            continue
        
        # Extract title from the HTML <title> tag or <h1>
        soup = BeautifulSoup(html, "lxml")
        title = None

        # Try <h1> first - usually more specific than <title>
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=None)

        if not title:
            title_tag = soup.find("title")
            if title_tag:
                # <title> often includes site name
                title = title_tag.get_text(strip=True).split("|")[0].strip()
        
        if not title:
            logger.warning(f"No title found for {url}")
            failed.append({"url": final_url, "reason": "no title"})
            time.sleep(CRAWL_DELAY)
            continue

        # Validate with Pydantic
        try:
            doc = SourceDocument(
                id=make_document_id(url),
                source=DataSource.CORNELL,
                url=url,
                title=title,
                content=content.strip(),
                content_type=classify_content_type(title),
                metadata={
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "original_url": url,
                    "was_redirect": url != final_url
                }
            )
            documents.append(doc)
            logger.info(f"  Finished {doc.title} ({doc.metadata['word_count']} words)")
        except Exception as e:
            failed.append({"url": url, "reason": str(e)})
            logger.warning(f"   Validation failure: {e}")

        # Wait between requests
        if i < len(urls) - 1: # Don't wait after the last one
            time.sleep(CRAWL_DELAY)

    # Save results
    # .model_dump() converts Pydantic object to dicts; mode = "json" ensures datetimes become strings
    docs_data = [doc.model_dump(mode="json") for doc in documents]

    output_file = output_path / "cornell_articles.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone! {len(documents)} articles saved to {output_file}")
    logger.info(f"{len(url)} urls found (with duplication)")
    logger.info(f"{len(failed)} articles failed")

    # Save failure log for debugging
    if failed:
        fail_file = output_path / "cornell_failures.json"
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"Failure log saved to {fail_file}")
    return documents

# Run
if __name__ == "__main__":
    # Show INFO-level messages with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Start with 5 articles to test, then remove the limite for the full run
    docs = scrape_cornell()
    print(f"\nScraped {len(docs)} documents successfully")
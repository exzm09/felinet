"""
Corpus loader - reads persisted SourceDocuments from disk.
Decouples downstream stages (chunking, embedding, indexing) from the scraping pipeline. Scrape once, load many times.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from felinet.schemas import SourceDocument

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path("data/processed/felinet_corpus.json")

def load_corpus(
        path: Path | str = DEFAULT_CORPUS_PATH
) -> list[SourceDocument]:
    """
    Load the persisted corpus from disk and return a list of SourceDocuments.
    Expects a JSON file containing a list of document objects that validate against the SourceDocument schema. 
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found at {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = [SourceDocument.model_validate(item) for item in raw]
    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents
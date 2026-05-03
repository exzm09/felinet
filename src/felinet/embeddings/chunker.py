"""
Chunking module for FeliNet corpus.
Splits SourceDocuments into DocumentChunks suitable for embedding and retrieval.
Uses recursive character splitting with token-aware sizing.
"""
from __future__ import annotations
import hashlib
import logging

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from felinet.schemas import (
    DocumentChunk, 
    SourceDocument,
    ContentType,
    DataSource,
    ChunkingConfig
)

logger = logging.getLogger(__name__)

# Use the cl100k base tokenizer as a proxy for token counting. 
# all-MinLM uses a difference tokenizer (BERT WordPiece), but the counts are close enough for sizing decicions.
_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """
    Return the number of tokens in text using cl100k_base.
    """
    return len(_ENCODER.encode(text))

# Splitter factory
def make_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter from a ChunkingConfig.
    RecursiveCharacterTextSplitter works by trying separators in priority order: paragraph, sentence, word, character.
    Keeps chunks semantically alignment.
    Embedding models have token limits instead of character limits.
    Parameters
    ----------
    config : ChunkingConfig
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=count_tokens,
        separators=config.separators,
        is_separator_regex=False
    )

def _chunk_id(
        document_id: str,
        index: int,
        text: str
):
    """
    Deterministic chunk IS: hash of (document_id, index, text context).
    If rerun the chunker on the same corpus, we get the same IDs, making it safe to upsert into Qdrant without creating duplicates, and making evluation reproducible.
    """
    h = hashlib.sha256(f"{document_id}|{index}|{text}".encode("utf-8")).hexdigest()
    return f"{document_id}__{index}__{h[:12]}"

# Core chunking logic
PIPELINE_VERSION = "0.1.0"

def chunk_document(
        document: SourceDocument,
        splitter: RecursiveCharacterTextSplitter
) -> list[DocumentChunk]:
    """
    Split a single SourceSucoment into a list of DocumentChunks.
    Parameters
    ----------
    document : SourceDocument
        A validated raw document from corpus.
    splitter : RecursiveCharacterTextSplitter
        Pre-configured splitter.

    Returns
    -------
    list[DocumentChunk]
        Chunks ready for embedding. The embedding field is None (get filled in future).
    """
    raw_chunks: list[str] = splitter.split_text(document.content)
    chunks: list[DocumentChunk] = []
    for idx, text in enumerate(raw_chunks):
        chunk = DocumentChunk(
            id = _chunk_id(document.id, idx, text),
            document_id=document.id,
            source=document.source,
            content_type=document.content_type,
            content=text,
            chunk_index=idx,
            token_count=count_tokens(text),
            pipeline_version=PIPELINE_VERSION,
            # Build a dict with title and url. Then merge in whatever is in document.metadata, and if that's None, treat it as empty.
            metadata={
                "title": document.title,
                "url": document.url,
                "total_chunks": len(raw_chunks),
                **(document.metadata or {})
            }
        )
        chunks.append(chunk)
    return chunks

def chunk_corpus(
        documents: list[SourceDocument],
        config: ChunkingConfig | None = None
) -> list[DocumentChunk]:
    """
    Chunk an entire corpus.

    Parameters
    ----------
    documents : list[SourceDocument]
        Loaded, validated corpus.
    config : ChunkingConfig, optional
        Chunking settings. If None, uses defaults (512 tokens, 50 overlap).

    Returns
    -------
    list[DocumentChunk]
        Flat list of all chunks across all documents.
    """
    if config is None:
        config = ChunkingConfig()
    splitter = make_splitter(config)
    all_chunks: list[DocumentChunk] = []

    for doc in documents:
        chunks = chunk_document(doc, splitter)
        all_chunks.extend(chunks)
        logger.info(f"Chunked {doc.title[:50]} ({doc.source.value}) into {len(chunks)} chunks")
    logger.info(f"Total chunks across corpus: {len(all_chunks)} ({len(documents)})")
    return all_chunks
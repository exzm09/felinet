"""
Felinet Reranker - Cross-encoder reranking for retrieval results.
"""

from __future__ import annotations
import logging
import time
from sentence_transformers import CrossEncoder

from felinet.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Module-level cache so the model is loaded once and reused.

_reranker_model: CrossEncoder | None = None
_reranker_mode_name: str | None = None

def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Load a cross-encoder reranker model (cached after first all).
    Parameters
    ----------
    model_name : str
        HuggingFace model name for the cross-encoder.

    Returns
    -------
    CrossEncoder
        The loaded model, ready to score (query, passage) pairs.
    """
    global _reranker_mode_name, _reranker_model
    if _reranker_model is not None and _reranker_mode_name == model_name:
        return _reranker_model
    logger.info(f"Loading reranker model: {model_name}")
    start = time.time()
    _reranker_model = CrossEncoder(model_name)
    _reranker_mode_name = model_name
    load_time = time.time() - start
    logger.info(f"Reranker loaded in {load_time:.1f}s")

    return _reranker_model

def rerank(
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> list[RetrievedChunk]:
    """
    Rerank retrieved chunks using a cross-encoder model.
    Parameters
    ----------
    query : str
        The user's original question.
    chunks : list[RetrievedChunk]
        Candidate chunks from hybrid search (typically 30).
    top_k : int
        How many chunks to keep after reranking (default: 5).
    model_name : str
        Which cross-encoder to use.

    Returns
    -------
    list[RetrievedChunk]
        The top-k chunks, re-scored and sorted by cross-encoder relevance.
        The `score` field is REPLACED with the cross-encoder score
        (so downstream code sees the more accurate score).
    """
    if not chunks:
        return []
    
    model = load_reranker(model_name)

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [[query, chunk.content] for chunk in chunks]

    # Score all pairs in one batch - must faster than scoring one at a time.
    logger.info(f"Reranking {len(pairs)} chunks...")
    start = time.time()
    scores = model.predict(pairs)
    rerank_time = time.time() - start

    # Pair each chunk with its new cross-encoder score
    scored_chunks = list(zip(chunks, scores))

    # Sort by cross-encoder score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Take top-k and build new RetrievedChunk objects with updated scores
    reranked = []
    for chunk, ce_score in scored_chunks[:top_k]:
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                score=float(ce_score),
                document_title=chunk.document_title,
                url=chunk.url
            )
        )
    logger.info(f"Reranking done in {rerank_time:.2f}s | "
                f"top score: {reranked[0].score:.2f} | "
                f"kept {len(reranked)}/{len(chunks)} chunks")
    return reranked
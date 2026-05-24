"""
FeliNet Hybrid Retriever - BM25 + Dense search with Reciprocal Rank Fusion

Add keyword-search (BM25) alongside the dense vector search, then merge the result lists using RRF.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from felinet.embeddings.vector_store import get_client, search
from felinet.schemas import RAGConfig, RetrievedChunk, DataSource

logger = logging.getLogger(__name__)

# BM25 Index - build an in-memory keyword search index over chunks

class BM25Index:
    """
    1. Load all chunk texts from corpus JSON file.
    2. Tokenize each chunk - split it into individual words.
    3. BM25 builds frequency tables: which word appear in which chunks, how often, and how rare each word is acorss the whole corpus.
    4. When a query comes in, tokenize it the same way and BM25 scores every chunk based on word overlap.
    """

    def __init__(
            self,
            bm25: BM25Okapi,
            chunk_ids: list[str],
            chunk_texts: list[str],
            chunk_metadata: list[dict]
    ):
        """
        Parameters
        ----------
        bm25 : BM25Okapi
            The fitted BM25 model (from rank_bm25 library).
        chunk_ids : list[str]
            Chunk IDs in the same order as the BM25 index.
        chunk_texts : list[str]
            Raw text of each chunk (needed to return results).
        chunk_metadata : list[dict]
            Metadata for each chunk (source, document_id, title, url, etc.).
        """
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self.chunk_texts = chunk_texts
        self.chunk_metadata = chunk_metadata

    @classmethod
    def from_corpus(cls, corpus_path: str | Path) -> "BM25Index":
        """
        Build a BM25 index from chunked corpus JSON
        Parameters
        ----------
        corpus_path : str or Path
            Path to the chunked corpus JSON file.
            Expected format: a JSON list of dicts, each with at least
            "id", "content", and "source".

        Returns
        -------
        BM25Index
            Ready to search.
        """
        path = Path(corpus_path)
        logger.info(f"Building BM25 index from {path}")

        with open(path) as f:
            chunks = json.load(f)

        # Extract the pieces needed storing metadata so that BM25 results have the same fields as Qdrant results, which makes RRF seamless.
        chunk_ids = []
        chunk_texts = []
        chunk_metadata = []
        for chunk in chunks:
            cid = chunk.get("chunk_id", chunk.get("id", ""))
            chunk_ids.append(cid)
            chunk_texts.append(chunk["content"])
            chunk_metadata.append({
                "source": chunk.get("source", "unknown"),
                "document_id": chunk.get("document_id", ""),
                "content_type": chunk.get("content_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "title": chunk.get("title", ""),
                "url": chunk.get("url")
            })
        # Tokenize: split each chunk into lowercase words
        tokenized_chunks = [text.lower().split() for text in chunk_texts]

        # Build BM25 model - computes all frequency tables internally
        bm25 = BM25Okapi(tokenized_chunks)

        logger.info(f"BM25 index built: {len(chunk_ids)} chunks indexed")
        return cls(bm25, chunk_ids, chunk_texts, chunk_metadata)
    
    def search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Search the BM25 index for chunks matching the query keywords
        Parameters
        ----------
        query : str
            The user's question (e.g., "What causes kidney disease in cats?").
        top_k : int
            How many results to return (default: 30).

        Returns
        -------
        list[dict]
            Each dict has: chunk_id, content, source, score, title, url.
            Sorted by BM25 score (highest first).
            These fields intentionally mirror what vector_store.search() returns
            so that RRF fusion can treat both result lists the same way.
        """
        # Tokenize the query the same way tokenized the corpus
        tokenized_query = query.lower().split()

        # Get BM25 scores for ALL chunks
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of the top-k highest-scoring chunks
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break

            results.append({
                "chunk_id": self.chunk_ids[idx],
                "content": self.chunk_texts[idx],
                "source": self.chunk_metadata[idx]["source"],
                "score": float(scores[idx]),
                "title": self.chunk_metadata[idx]["title"],
                "url": self.chunk_metadata[idx]["url"]
            })

        if results:
            logger.info(f"BM25 search: '{query[:50]}...' -> {len(results)} results "
                        f"(top score: {results[0]['score']:.2f})")
            
        else:
            logger.info(f"BM25 search: '{query[:50]}...' -> 0 result")
        return results
    
# Reciprocal Rank Fusion - merges two ranked lists into one

def reciprocal_rank_fusion(
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0
) -> list[dict]:
    """
    Merge dense vector and BM25 (keyword) results using RRF.

    Parameters
    ----------
    dense_results : list[dict]
        Results from Qdrant. Must have "chunk_id" key.
    bm25_results : list[dict]
        Results from BM25. Must have "chunk_id" key.
    k : int
        RRF constant (default: 60).
    dense_weight : float
        Multiplier for dense search contribution.
    bm25_weight : float
        Multiplier for BM25 contribution.

    Returns
    -------
    list[dict]
        Fused results sorted by RRF score (highest first).
        Each dict has: chunk_id, content, source, rrf_score,
        dense_rank, bm25_rank, dense_score, bm25_score, title, url.
    """
    fused_scores: dict[str, dict] = {}
    # Process dense results
    for rank, result in enumerate(dense_results):
        cid = result.get("chunk_id", str(result.get("id", "")))

        rrf_contribution = dense_weight / (k + rank + 1)

        fused_scores[cid] = {
            "chunk_id": cid,
            "content": result.get("content", ""),
            "source": result.get("source", ""),
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "rrf_score": rrf_contribution,
            "dense_rank": rank + 1,
            "dense_score": result.get("score", 0.0),
            "bm25_rank": None,
            "bm25_score": 0.0
        }

    # Process BM25 results
    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        rrf_contribution = bm25_weight / (k + rank + 1)

        if cid in fused_scores:
            # Chunk appeared in BOTH lists - add BM25's RRF contribution
            fused_scores[cid]["rrf_score"] += rrf_contribution
            fused_scores[cid]["bm25_rank"] = rank + 1
            fused_scores[cid]["bm25_score"] = result.get("score", 0.0)

        else:
            # Chunk only appeared in BM25 results
            fused_scores[cid] = {
                "chunk_id": cid,
                "content": result.get("content", ""),
                "source": result.get("source", ""),
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "rrf_score": rrf_contribution,
                "dense_rank": None,
                "dense_score": 0.0,
                "bm25_rank": rank + 1,
                "bm25_score": result.get("score", 0.0)
            }
    # Sort by RRF score (highest = most relevant)
    fused = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Log diagnostics
    both = sum(1 for r in fused if r["dense_rank"] and r["bm25_rank"])
    d_only = sum(1 for r in fused if r["dense_rank"] and not r["bm25_rank"])
    b_only = sum(1 for r in fused if not r["dense_rank"] and r["bm25_rank"])
    logger.info(f"RRF fusion: {len(fused)} unqiue chunks | "
                f"in both {both} | dense only: {d_only} | bm25 only: {b_only}")
    
    return fused

# Hybrid Search - main function
def hybrid_search(
        query: str,
        query_vector: list[float],
        bm25_index: BM25Index,
        config: RAGConfig,
        qdrant_url: str = "http://localhost:6333"
) -> list[RetrievedChunk]:
    """
    Run hybrid search: BM25 + dense retrieval -> RRF fusion -> top chunks.
    1. BM25 searches for keyword matches -> top N candidates
    2. Qdrant searches for vector similarity -> top N candidates
    3. RRF merges both lists -> sorted by combined score
    4. Take the top K and convert to RetrievedChunk objects
     query : str
        The user's question (needed for BM25 tokenization).
    query_vector : list[float]
        The embedded query vector (for dense search in Qdrant).
    bm25_index : BM25Index
        Pre-built BM25 index (built once at startup, reused for all queries).
    config : RAGConfig
        Pipeline configuration (collection name, weights, top_k, etc.).
    qdrant_url : str
        Where Qdrant is running.

    Returns
    -------
    list[RetrievedChunk]
        Top chunks after hybrid fusion, as Pydantic RetrievedChunk objects.
        Same type that retrieve_chunks() returns - so the rest of the
        pipeline (format_context, generate_answer) doesn't need to change.
    """
    top_k_search = config.retrieval.top_k_initial
    top_k_final = config.retrieval.top_k_reranked


    # Lane 1: Dense search (Qdrant)
    client = get_client(url=qdrant_url)
    dense_results = search(
        client=client,
        query_vector=query_vector,
        collection_name=config.collection_name,
        top_k=top_k_search
    )
    logger.info(f"Dense search returned {len(dense_results)} results")

    # Lane 2: BM25 search
    bm25_results = bm25_index.search(query, top_k=top_k_search)

    # Merge with RRF
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=60,
        dense_weight=config.retrieval.dense_weight,
        bm25_weight=config.retrieval.bm25_weight
    )

    # Take top results and convert to RetrievedChunk
    # If reranker is enabled, return MORE candidates
    num_to_return = top_k_search if config.retrieval.use_reranker else top_k_final
    top_fused = fused[:num_to_return]

    retrieved = []
    for result in top_fused:
        try:
            source_enum = DataSource(result["source"])
        except ValueError:
            source_enum = DataSource.CORNELL    # fallback for unknown sources

        retrieved.append(
            RetrievedChunk(
                chunk_id=result["chunk_id"],
                content=result["content"],
                source=source_enum,
                score=result["rrf_score"],
                document_title=result.get("title") or None,
                url=result.get("url") or None
            )
        )
    logger.info(f"Hybrid search complete: {len(retrieved)} chunks | "
                f"dense_weight={config.retrieval.dense_weight} | "
                f"bm25_weight={config.retrieval.bm25_weight}")
    
    return retrieved
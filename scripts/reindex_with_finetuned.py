"""
Re-embed corpus with the fine-tuned model.
Replace all the old embeddings in Qdrant vector database with new ones from fine-tuned model preventing embedding drift
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from felinet.data.loader import load_corpus
from felinet.embeddings.chunker import chunk_corpus
from felinet.embeddings.vector_store import create_collection, get_client, search, upsert_chunks
from felinet.schemas import ChunkingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def reindex_corpus(
    model_path: str = "models/felinet-embedding-v1",
    collection_name: str = "felinet_chunks",
    corpus_path: str = "data/processed/felinet_corpus.json",
    batch_size: int = 64,
    test_limit: int = 0,
) -> dict:
    """
    Re-embed the entire corpus with the fine-tuned model and update Qdrant.

    Parameters
    ----------
    model_path : str
        Path to fine-tuned model folder.
    collection_name : str
        Qdrant collection name.
    corpus_path : str
        Path to corpus JSON file.
    batch_size : int
        How many chunks to embed at once. 64 is good for 16GB RAM.
    test_limit : int
        If > 0, only process this many chunks (for testing).

    Returns
    -------
    dict
        Stats about the re-indexing.
    """
    start_time = time.time()
    # Step 1: Load the fine-tuned model
    logger.info(f"Loading fine-tuned model from {model_path}")
    model = SentenceTransformer(model_path)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info("Model loaded successfully.")

    # Step 2: Load and chunk corpus
    logger.info(f"Loading corpus from: {corpus_path}")
    documents = load_corpus(corpus_path)
    logger.info(f"Loaded {len(documents)} documents")

    config = ChunkingConfig()
    chunks = chunk_corpus(documents, config)
    logger.info(f"Chunked into {len(chunks)} chunks")

    if test_limit > 0:
        chunks = chunks[:test_limit]
        logger.info(f"TEST MODE: limited to {test_limit} chunks")

    # Step 3: Embed all chunks with the fine-tuned model
    logger.info(f"Embedding {len(chunks)} chunks (batch_size={batch_size})...")
    texts = [chunk.content for chunk in chunks]

    # Embed in batches to avoid memory issues
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for consine similarity
        )
        all_embeddings.append(batch_embeddings)

        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    embeddings = np.vstack(all_embeddings)
    logger.info(f"All {len(embeddings)} chunks embedded.")

    # Step 4: Update chunks with new embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.tolist()
        chunk.embedding_model = "felinet-embedding-v1"

    # Step 5: Recreate Qdrant collection and upsert
    logger.info(f"Recreating Qdrant collection: {collection_name}")
    client = get_client()
    create_collection(
        client=client, collection_name=collection_name, vector_size=embedding_dim, recreate=True
    )

    logger.info(f"Upserting {len(chunks)} chunks to Qdrant...")
    upsert_chunks(
        client=client, chunks=chunks, collection_name=collection_name, batch_size=batch_size
    )

    elapsed = time.time() - start_time

    # Step 6: Quick verification
    logger.info("Running verification query...")
    test_query = "What are symptoms of kidney disease in cats?"

    query_embedding = model.encode(test_query, normalize_embeddings=True).tolist()

    results = search(
        client=client, query_vector=query_embedding, collection_name=collection_name, top_k=3
    )

    print("\n" + "=" * 60)
    print("RE-INDEXING COMPLETE")
    print("=" * 60)
    print("  Model: felinet-embedding-v1 (fine-tuned)")
    print(f"  Chunks indexed: {len(chunks)}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"\n  Verification query: '{test_query}'")
    for i, result in enumerate(results):
        score = result.get("score", 0)
        text = result.get("content", "")[:100]
        print(f"  #{i+1} (score={score:.4f}): {text}...")
    print("=" * 60)

    return {
        "chunks_indexed": len(chunks),
        "embedding_dim": embedding_dim,
        "model": "felinet-embedding-v1",
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed corpus with fine-tuned model and update Qdrant"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/felinet-embedding-v1",
        help="Path to fine-tuned model folder",
    )
    parser.add_argument(
        "--test", type=int, default=0, help="If > 0, only process this many chunks (for testing)"
    )
    parser.add_argument(
        "--collection", type=str, default="felinet_chunks", help="Qdrant collection name"
    )
    args = parser.parse_args()

    reindex_corpus(
        model_path=args.model,
        collection_name=args.collection,
        test_limit=args.test,
    )


if __name__ == "__main__":
    main()

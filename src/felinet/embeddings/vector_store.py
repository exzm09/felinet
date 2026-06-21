"""
Qdrant vector store interface for FeliNet.

Handles collection creation, upserting embedded chunks, and basic search.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from felinet.schemas import DocumentChunk

logger = logging.getLogger(__name__)

_MEMORY_CLIENT: QdrantClient | None = None


def get_client(url: str = "http://localhost:6333") -> QdrantClient:
    """
    Return a Qdrant client. QDRANT_MODE env var picks the behavior:
      unset / "server" -> connect to a running Qdrant at `url`
      "memory"         -> in-memory Qdrant, loaded once from bundled parquet (HF Space)
      "cloud"          -> Qdrant Cloud via QDRANT_URL + QDRANT_API_KEY
    """
    mode = os.getenv("QDRANT_MODE", "server")
    if mode == "memory":
        return _get_memory_client()
    if mode == "cloud":
        return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    return QdrantClient(url=url)


def _get_memory_client() -> QdrantClient:
    """
    Build the in-memory DB only on the first call, then hand back the same one.
    """
    global _MEMORY_CLIENT
    if _MEMORY_CLIENT is None:
        _MEMORY_CLIENT = _build_memory_client()
    return _MEMORY_CLIENT


def _coerce_id(value):
    """
    Qdrant point ids must be an int or a UUID string.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value)


def _build_memory_client() -> QdrantClient:
    import pandas as pd

    path = os.getenv("QDRANT_EXPORT_PATH", "data/qdrant_export.parquet")
    collection = os.getenv("QDRANT_COLLECTION", "felinet_chunks")

    df = pd.read_parquet(path)
    dim = len(df.iloc[0]["vector"])

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=_coerce_id(row["id"]),
                vector=[float(x) for x in row["vector"]],
                payload=json.loads(row["payload"]),
            )
            for _, row in df.iterrows()
        ],
    )
    return client


def create_collection(
    client: QdrantClient,
    collection_name: str = "felinet_chunks",
    vector_size: int = 384,
    recreate: bool = False,
) -> None:
    """
    Create (or recreate) a Qdrant collection.
    Parameters
    ----------
    collection_name : str
        Name of the collection.
    vector_size : int
        Dimension of embeddings. Must match the model output:
        - all-MiniLM-L6-v2 = 384
        - bge-base-en-v1.5 = 768
    recreate : bool
        If True, deletes the existing collection first. Useful during
        development when changing chunking strategy or re-embed.
    """
    if recreate:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deteled existing collection '{collection_name}'")
        except Exception:
            pass  # collection did not exist

        # Check if the collection already exists
        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            logger.info(f"Collection {collection_name} already exists, skipping creation")
            return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection {collection_name} (dim={vector_size}, distance=cosine)")


def upsert_chunks(
    client: QdrantClient,
    chunks: list[DocumentChunk],
    collection_name: str = "felinet_chunks",
    batch_size: int = 100,
) -> int:
    """
    Upsert embedded chunks into Qdrant.
    Parameters
    ----------
    chunks : list[DocumentChunk]
        Chunks WITH embeddings (embedding must not be None).
    batch_size : int
        How many points to send per API call. 100 is a safe default.
        Larger batches are faster but use more memory.

    Returns
    -------
    int
        Number of points upserted.
    """
    # Filter out any chunks without embeddings (defensive)
    embedded = [c for c in chunks if c.embedding is not None]
    if len(embedded) < len(chunks):
        logger.warning(f"{len(chunks) - len(embedded)} chunks skipped (no embedding)")

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id)),
            vector=chunk.embedding,
            payload={
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "source": chunk.source.value,
                "content": chunk.content,
                "content_type": chunk.content_type.value,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "title": chunk.metadata.get("title", ""),
                "url": chunk.metadata.get("url", ""),
            },
        )
        for chunk in embedded
    ]

    # Upsert in batches
    total = 0  # Total points upserted
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)
        logger.info(f"Upserted batch {i} - {i + len(batch)}")
    logger.info(f"Upsert complete: {total} points in {collection_name}")
    return total


def search(
    client: QdrantClient,
    query_vector: list[float],
    collection_name: str = "felinet_chunks",
    top_k: int = 5,
) -> list[dict]:
    """
    Search for the most similar chunks to a query vector.
    Parameters
    ----------
    query_vector : list[float]
        The embedded query (same dimension as stored vectors).
    top_k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Each dict has 'id', 'score', and all payload fields.
        Score is cosine similarity: 1.0 = perfect match, 0.0 = unrelated.
    """
    results = client.query_points(
        collection_name=collection_name, query=query_vector, limit=top_k, with_payload=True
    )
    hits = []
    for point in results.points:
        hits.append(
            {
                "id": point.id,
                "score": point.score,
                **point.payload,  # spreads all key-value
            }
        )
    return hits

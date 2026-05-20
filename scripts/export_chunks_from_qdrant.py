"""
Export chunks from Qdrant to a local JSON file.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

def export_chunks(
        collection_name: str = "felinet_chunks",
        qdrant_url: str = "http://localhost:6333",
    output_path: str = "data/processed/felinet_chunks.json",
    batch_size: int = 100
) -> int:
    """
    Scroll through all points to Qdrant and save thier payload as JSON.
    Parameters
    ----------
    collection_name : str
        Name of the Qdrant collection.
    qdrant_url : str
        Qdrant server URL.
    output_path : str
        Where to save the JSON file.
    batch_size : int
        How many points to fetch per scroll request.

    Returns
    -------
    int
        Number of chunks exported.
    """
    client = QdrantClient(url=qdrant_url)
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"    Collection '{collection_name}' not found in Qdrant.")
        print(f"    Available collections: {collections}")
        return 0
    
    # Get collection info
    info = client.get_collection(collection_name)
    total_points = info.points_count
    print(f"Collection '{collection_name}' has {total_points} points")

    # Scroll through all points
    chunks = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:
            chunk_data = {
                "id": str(point.id),
                "chunk_id": point.payload.get("chunk_id", str(point.id)),
                "content": point.payload.get("content", ""),
                "source": point.payload.get("source", "unknown"),
                "document_id": point.payload.get("document_id", ""),
                "content_type": point.payload.get("content_type", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "token_count": point.payload.get("token_count", 0),
                "title": point.payload.get("title", ""),
                "url": point.payload.get("url", "")
            }
            chunks.append(chunk_data)
        logger.info(f"Exported {len(chunks)}/{total_points} chunks...")

        if next_offset is None:
            break
        offset = next_offset

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(chunks, f, indent=2)

        print(f"\nExported {len(chunks)} chunks to {output_file}")
    return len(chunks)

def main():
    parser = argparse.ArgumentParser(description="Export Qdrant chunks to JSON")
    parser.add_argument("--output", type=str, default="data/processed/felinet_chunks.json", help="Output JSON file path")
    parser.add_argument("--collection", type=str, default="felinet_chunks", help="Qdrant collection name")
    parser.add_argument("--url", type=str, default="http://localhost:6333", help="Qdrant server URL")
    args = parser.parse_args()

    export_chunks(
        collection_name=args.collection,
        qdrant_url=args.url,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
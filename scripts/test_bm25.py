"""
Build and test the BM25 index.
"""
from __future__ import annotations
import logging
import argparse
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build and test BM25 index")
    parser.add_argument(
        "--chunks", type=str, default="data/processed/felinet_chunks.json",
        help="Path to chunked corpus JSON"
    )
    args = parser.parse_args()

    # Step 1: Check that the chunks file exists
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"\n Chunks file not found: {chunks_path}")
        print("\nNeed a JSON file containing your chunked corpus.")
        print("Run python scripts/export_chunks_from_qdrant.py to export from Qdrant")
        return
    
    # Step 2: Build the BM25 index
    from felinet.rag.retriever import BM25Index

    print(f"\nBuilding BM25 index from {chunks_path}...")
    start = time.time()
    index = BM25Index.from_corpus(str(chunks_path))
    build_time = time.time() - start

    # Step 3: Run test queries
    test_queries = [
        "What are the symptoms of kidney disease in cats?",
        "Is chocolate toxic to cats?",
        "What is FIP?",
        "How to care for a Persian cat?",
        "What vaccinations do kittens need?",
    ]
    print(f"\n{'=' * 70}")
    print("BM25 Test Queries")
    print(f"{'=' * 70}")

    for query in test_queries:
        print(f"\n Query: {query}")
        results = index.search(query, top_k=5)

        if not results:
            print("   (no results)")
            continue

        for i, r in enumerate(results):
            preview = r["content"][:100].replace("\n", " ")
            print(f"   [{i+1}] score={r['score']:.2f} | source={r['source']}")
            print(f"       {preview}...")

    print(f"\n{'=' * 70}")
    print(" BM25 index is working.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
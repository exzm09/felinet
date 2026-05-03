"""
Index the FeliNet corpus: load -> chunk -> embed -> upsert to Qdrant.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from felinet.data.loader import load_corpus
from felinet.embeddings.chunker import chunk_corpus
from felinet.embeddings.embedder import embed_chunks, load_embedding_model
from felinet.embeddings.vector_store import (
    create_collection,
    get_client,
    upsert_chunks
)
from felinet.schemas import ChunkingConfig

def main():
    # Load corpus
    print("Loading corpus...")
    documents = load_corpus()
    print(f"    -> {len(documents)} documents loaded\n")

    # Chunk
    print("Chunking...")
    config = ChunkingConfig()
    chunks = chunk_corpus(documents, config)
    token_counts = [c.token_count for c in chunks]
    print(f"    -> {len(chunks)} chunks (mean {sum(token_counts)/len(token_counts):.0f} tokens)\n")

    # Embed
    print("Embedding...")
    t0 = time.time()
    model = load_embedding_model("all-MiniLM-L6-v2")
    chunks = embed_chunks(chunks, model)
    elapsed = time.time() - t0
    print(f"    Embeded in {elapsed:.1f}s")
    print(f"    Vector dimensions: {len(chunks[0].embedding)}\n")

    # Upsert to Qdrant
    print("Upserting to Qdrant...")
    client = get_client()
    create_collection(client, recreate=True)    # fresh index each run during dev
    n = upsert_chunks(client, chunks)
    print(f"     -> {n} points in collection 'felinet_chunks'\n")
    print("Done! Visit http://localhost:6333/dashboard to inspect collection.")

if __name__ == "__main__":
    main()
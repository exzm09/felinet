"""
Test basic retrieval: embed a question, find similar chunks in Qdrant.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/ "src"))

from felinet.embeddings.embedder import load_embedding_model
from felinet.embeddings.vector_store import get_client, search

# Test queries

TEST_QUERIES = [
    "What causes kidney disease in cats?",
    "What are the symptoms of feline leukemia?",
    "How much should I feed my kitten?",
    "Are lilies toxic to cats?",
    "What is the personality of a Siamese cat?",
]

def main():
    model = load_embedding_model("all-MiniLM-L6-v2")
    client = get_client()

    for query in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        # Embed the query with the same model used for indexing
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        # Search Qdrant
        results = search(client, query_vector, top_k=3)
        for i, hit in enumerate(results, 1):
            print(f"\n  [{i}] Score: {hit['score']:.4f}")
            print(f"      Source: {hit['source']}")
            print(f"      Title:  {hit['title']}")
            print(f"      Text:   {hit['content'][:200]}...")

if __name__ == "__main__":
    main()

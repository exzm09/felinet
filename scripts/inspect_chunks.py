"""Quick visual sanity check on the chunker output."""
from felinet.embeddings.chunker import chunk_corpus
from felinet.data.loader import load_corpus


def main():
      documents = load_corpus()
      print(f"Loaded {len(documents)} documents\n")

      chunks = chunk_corpus(documents)

      token_counts = [c.token_count for c in chunks]
      print(f"Total chunks: {len(chunks)}")
      print(f"Avg chunks per document: {len(chunks) / len(documents):.1f}")
      print(f"Token count - min: {min(token_counts)}, "
            f"max: {max(token_counts)}, "
            f"mean: {sum(token_counts) / len(token_counts):.1f}")

      # Show a sample chunk
      print("\n=== Sample chunk (first chunk of first document) ===")
      print(f"ID:     {chunks[0].id}")
      print(f"Source: {chunks[0].source.value}")
      print(f"Tokens: {chunks[0].token_count}")
      print(f"Title:  {chunks[0].metadata.get('title', 'N/A')}")
      print(f"Text preview:\n{chunks[0].content[:400]}")
      print("...")

if __name__ == "__main__":
    main()
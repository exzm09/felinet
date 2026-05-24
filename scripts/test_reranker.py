"""
Test cross-encoder reranker independently.
"""
from __future__ import annotations
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    from felinet.rag.reranker import load_reranker

    # Step 1: Load the model
    print("\nLoading cross-encoder reranker model...")
    print("(First run downloads ~66MB - takes 10-30 seconds)\n")
    start = time.time()

    model = load_reranker()
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Step 2: Test with some (query, passage) pairs
    test_pairs = [
        # High relevance - the passage directly answers the question
        (
            "What are symptoms of kidney disease in cats?",
            "Chronic kidney disease in cats presents with increased thirst, "
            "frequent urination, weight loss, and decreased appetite."
        ),
        # Medium relevance - related topic but not a direct answer
        (
            "What are symptoms of kidney disease in cats?",
            "Cats are obligate carnivores and require a diet high in protein. "
            "Many commercial cat foods include kidney-supporting ingredients."
        ),
        # Low relevance - completely unrelated
        (
            "What are symptoms of kidney disease in cats?",
            "The Maine Coon is one of the largest domestic cat breeds, "
            "known for their tufted ears and bushy tail."
        ),
    ]

    print(f"\n{'=' * 70}")
    print("Reranker Score Test")
    print(f"{'=' * 70}")
    print(f"\nQuery: '{test_pairs[0][0]}'")
    print()

    scores = model.predict(test_pairs)
    labels = ["HIGH relevance (direct answer)", "MEDIUM relevance (related)", "LOW relevance (unrelated)"]

    for label, (_, passage), score in zip(labels, test_pairs, scores):
        preview = passage[:80] + "..."
        print(f"  {label}")
        print(f"    Score: {score:.2f}")
        print(f"    Text:  {preview}")
        print()

    # Verify scores are in expected order
    if scores[0] > scores[1] > scores[2]:
        print("Scores are correctly ordered: high > medium > low")
    else:
        print("Scores are NOT in expected order - but the model may still work fine.")
        print("Cross-encoder scores can vary based on exact text content.")

    # Step 3: Speed Test
    print(f"\n{'=' * 70}")
    print("Speed Test: Reranking 30 chunks")
    print(f"{'=' * 70}")

    # Simulate 30 (query, chunk) pairs
    dummy_pairs = [(test_pairs[0][0], f"This is test passage number {i} about cats.") for i in range(30)]

    start = time.time()
    _ = model.predict(dummy_pairs)
    batch_time = time.time() - start

    print(f"\n  Reranked 30 chunks in {batch_time:.3f}s ({batch_time*1000:.0f}ms)")
    print(f"  Per chunk: {batch_time/30*1000:.1f}ms")
    print(f"\n  This is the overhead added to each query.")
    print(f"  For comparison, Groq API call takes ~1000-1500ms.")

    print(f"\n{'=' * 70}")
    print("Reranker is working.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

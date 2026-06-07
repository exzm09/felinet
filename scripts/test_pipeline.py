"""
Quick smoke test for the RAG pipeline.

Test on:
1. Can we load the embedding model?
2. Can we connect to Qdrant and retieve chunks?
3. Can we call Groq and get a response?
4. Does Langfuse capture the trace?
"""

import logging
import time

from felinet.embeddings.embedder import load_embedding_model
from felinet.rag.pipeline import query_rag
from felinet.schemas import RAGConfig

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = RAGConfig()
    model = load_embedding_model(config.embedding_model)

    # 10 diverse test questions covering different content types
    test_questions = [
        "What are the symptoms of feline kidney disease?",
        "What breeds are hypoallergenic?",
        "Is chocolate toxic to cats?",
        "How often should I take my cat to the vet?",
        "What is the temperament of a Maine Coon?",
        "Why does my cat knead on blankets?",
        "What vaccines does my kitten need?",
        "How can I tell if my cat is in pain?",
        "What should I feed a senior cat?",
        "Are lilies dangerous for cats?",
    ]

    print("\n" + "=" * 70)
    print("FeliNet RAG Pipeline - Smoke Test")
    print("=" * 70)

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i}/10 ---")
        print(f"Q: {question}")

        try:
            response = query_rag(
                query=question,
                config=config,
                embedding_model=model,
            )
            # Show first 300 chars of answer
            preview = response.answer[:300]
            if len(response.answer) > 300:
                preview += "..."

            print(f"A: {preview}")
            print(f"   Chunks used: {len(response.retrieved_chunks)}")
            print(
                f"   Top source:  {response.retrieved_chunks[0].source.value if response.retrieved_chunks else 'none'}"
            )
            print(f"   Latency:     {response.latency_ms:.0f}ms")
            print(f"   Trace ID:    {response.trace_id}")

        except Exception as e:
            import traceback

            print(f"   ERROR: {e}")
            traceback.print_exc()
        time.sleep(10)

    print("\n" + "=" * 70)
    print("Smoke test complete!")
    print("=" * 70)

    # Flush Langfuse to make sure all traces are sent
    """try:
        from langfuse import Langfuse
        Langfuse().flush()
        print("Langfuse traces flushed successfully.")
    except Exception:
        pass"""


if __name__ == "__main__":
    main()

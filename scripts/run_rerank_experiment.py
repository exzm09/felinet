"""
FeliNet Week 8 - Reranking + Context Engineering Experiment Runner.
Compares:
    1. Hybrid search without reranking.
    2. Hybrid search with cross-encoder reranking.
    3. Hybrid + reranking + improved prompt template.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import time
from statistics import mean
import mlflow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

def load_golden_dataset(
        path: str = "data/eval/golden_eval_dataset.json",
        limit: int | None = None
) -> list[dict]:
    """
    Load evaluation test cases from the golden dataset.
    """
    with open(path) as f:
        data = json.load(f)
    cases = data["test_cases"]
    if limit:
        cases = cases[:limit]
    return cases

def run_single_query(query: str, pipeline_fn) -> dict | None:
    """
    Run a single query through the pipeline with error handling.
    """
    try:
        response = pipeline_fn(query)
        return {
            "query": query,
            "answer": response.answer,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "source": c.source.value if hasattr(c.source, "value") else str(c.source),
                    "score": c.score
                }
                for c in response.retrieved_chunks
            ],
            "latency_ms": response.latency_ms,
            "error": None
        }
    except Exception as e:
        logger.error(f"FAILED: {e}")
        return {
            "query": query,
            "answer": "",
            "retrieved_chunks": [],
            "latency_ms": 0,
            "error": str(e)
        }
    
def score_results(cases: list[dict], results: list[dict]) -> dict:
    """
    Score retrieval quality: source accuracy + latency stats.
    """
    successful = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    if not successful:
        return {"source_accuracy": 0.0, "avg_latency_ms": 0, "error_rate": 1.0}
    
    correct = 0
    for case, result in zip(cases, results):
        if result["error"]:
            continue
        expected_source = case.get("expected_source", "")
        actual_sources = [c["source"] for c in result["retrieved_chunks"]]
        if expected_source in actual_sources:
            correct += 1

    latencies = [r["latency_ms"] for r in successful]

    return {
        "source_accuracy": correct / len(successful) if successful else 0.0,
        "avg_latency_ms": mean(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "total_cases": len(results),
        "successful_cases": len(successful),
        "error_rate": len(errors) / len(results) if results else 0
    }

def run_experiment(
        cases: list[dict],
        use_reranker: bool = False,
        use_improved_prompt: bool = False,
        chunks_path: str = "data/processed/felinet_chunks.json",
        delay: float = 12.0
) -> dict:
    """
    Run evaluation with configurable reranking and prompt settings.
    Parameters
    ----------
    use_reranker : bool
        If True, apply cross-encoder reranking after hybrid search.
    use_improved_prompt : bool
        If True, use the upgraded prompt template with few-shot examples.
    """
    from felinet.schemas import RAGConfig
    from felinet.rag.pipeline import query_rag
    from felinet.rag.retriever import BM25Index
    from felinet.embeddings.embedder import load_embedding_model

    config = RAGConfig()
    config.retrieval.use_reranker = use_reranker

    # If using improved prompt, override the system prompt
    if use_improved_prompt:
        config.generation.system_prompt = IMPROVED_SYSTEM_PROMPT
    emb_model = load_embedding_model(config.embedding_model)
    bm25_index = BM25Index.from_corpus(chunks_path)

    def pipeline_fn(query: str):
        return query_rag(
            query=query,
            config=config,
            embedding_model=emb_model,
            retrieval_mode="hybrid",
            bm25_index=bm25_index
        )
    results = []
    for i, case in enumerate(cases):
        logger.info(f"[{i + 1}/{len(cases)}] {case['query'][:60]}...")
        result = run_single_query(case["query"], pipeline_fn)
        results.append(result)
        if i < len(cases) - 1:
            time.sleep(delay)

    scores = score_results(cases, results)
    return {**scores, "use_reranker": use_reranker, "use_improved_prompt": use_improved_prompt}

IMPROVED_SYSTEM_PROMPT = """You are FeliNet, a feline health and breed knowledge assistant.

RULES:
1. Answer ONLY using the provided context. Never use outside knowledge.
2. Cite sources by number: [1], [2], etc. Every factual claim needs a citation.
3. If the context doesn't contain enough information, say: "Based on the available sources, I don't have enough information to fully answer this question."
4. If sources disagree, mention both viewpoints and cite each.
5. For health questions, add: "Consult your veterinarian for advice specific to your cat."
6. Keep answers concise but thorough - aim for 3-5 sentences for simple questions, more for complex ones.
7. Stay on topic. If the question is not about cats, politely say you can only help with feline-related topics.

EXAMPLE:
Question: What are the symptoms of kidney disease in cats?
Good answer: Common signs of chronic kidney disease (CKD) in cats include increased thirst and urination, weight loss, decreased appetite, and vomiting [1]. As the disease progresses, cats may develop mouth ulcers, bad breath, and lethargy [1][2]. CKD is more common in older cats and is often detected through blood work showing elevated BUN and creatinine levels [2]. Consult your veterinarian for advice specific to your cat.
"""

def main():
    parser = argparse.ArgumentParser(description="Run reranking experiments")
    parser.add_argument("--limit", type=int, default=None, help="Limit test cases")
    parser.add_argument("--delay", type=float, default=12.0, help="Seconds between API calls")
    parser.add_argument("--chunks", type=str, default="data/processed/felinet_chunks.json", help="Path to chunked corpus JSON")
    parser.add_argument("--dataset", type=str,
        default="data/eval/golden_eval_dataset.json",
        help="Path to golden evaluation dataset")
    args = parser.parse_args()

    cases = load_golden_dataset(args.dataset, args.limit)
    logger.info(f"Loaded {len(cases)} test cases")

    mlflow.set_experiment("felinet-rag-pipeline")

    # Experiment 1: Hybrid without reranking
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Hybrid search, NO reranking (Week 7 baseline)")
    print("=" * 70)

    with mlflow.start_run(run_name="week8_hybrid_no_rerank"):
        mlflow.log_param("retrieval_mode", "hybrid")
        mlflow.log_param("use_reranker", False)
        mlflow.log_param("use_improved_prompt", False)
        mlflow.log_param("num_test_cases", len(cases))

        scores_1 = run_experiment(cases, use_reranker=False, use_improved_prompt=False, chunks_path=args.chunks, delay=args.delay)

        mlflow.log_metric("source_accuracy", scores_1["source_accuracy"])
        mlflow.log_metric("avg_latency_ms", scores_1["avg_latency_ms"])
        mlflow.log_metric("error_rate", scores_1["error_rate"])

        print(f"\n  Source accuracy: {scores_1['source_accuracy']:.1%}")
        print(f"  Avg latency:    {scores_1['avg_latency_ms']:.0f}ms")

    # Experiment 2: Hybrid with reranking
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Hybrid search + Cross-encoder reranking")
    print("=" * 70)

    with mlflow.start_run(run_name="week8_hybrid_reranked"):
        mlflow.log_param("retrieval_mode", "hybrid")
        mlflow.log_param("use_reranker", True)
        mlflow.log_param("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        mlflow.log_param("use_improved_prompt", False)
        mlflow.log_param("num_test_cases", len(cases))

        scores_2 = run_experiment(
            cases, use_reranker=True, use_improved_prompt=False,
            chunks_path=args.chunks, delay=args.delay,
        )

        mlflow.log_metric("source_accuracy", scores_2["source_accuracy"])
        mlflow.log_metric("avg_latency_ms", scores_2["avg_latency_ms"])
        mlflow.log_metric("error_rate", scores_2["error_rate"])

        print(f"\n  Source accuracy: {scores_2['source_accuracy']:.1%}")
        print(f"  Avg latency:    {scores_2['avg_latency_ms']:.0f}ms")

    # Experiment 3: Hybrid + Reranking + Improved prompt
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Hybrid + reranking + improved prompt")
    print("=" * 70)

    with mlflow.start_run(run_name="week8_hybrid_reranked_improved_prompt"):
        mlflow.log_param("retrieval_mode", "hybrid")
        mlflow.log_param("use_reranker", True)
        mlflow.log_param("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        mlflow.log_param("use_improved_prompt", True)
        mlflow.log_param("num_test_cases", len(cases))

        scores_3 = run_experiment(
            cases, use_reranker=True, use_improved_prompt=True,
            chunks_path=args.chunks, delay=args.delay,
        )

        mlflow.log_metric("source_accuracy", scores_3["source_accuracy"])
        mlflow.log_metric("avg_latency_ms", scores_3["avg_latency_ms"])
        mlflow.log_metric("error_rate", scores_3["error_rate"])

        print(f"\n  Source accuracy: {scores_3['source_accuracy']:.1%}")
        print(f"  Avg latency:    {scores_3['avg_latency_ms']:.0f}ms")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Configuration':<45} {'Accuracy':>10} {'Latency':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    print(f"  {'Hybrid (no reranking)':<45} {scores_1['source_accuracy']:>9.1%} {scores_1['avg_latency_ms']:>8.0f}ms")
    print(f"  {'Hybrid + reranking':<45} {scores_2['source_accuracy']:>9.1%} {scores_2['avg_latency_ms']:>8.0f}ms")
    print(f"  {'Hybrid + reranking + improved prompt':<45} {scores_3['source_accuracy']:>9.1%} {scores_3['avg_latency_ms']:>8.0f}ms")

    improvement = scores_3["source_accuracy"] - scores_1["source_accuracy"]
    print(f"Overall improvement (baseline -> best): {improvement:+.1%}")
    print(f"All experiments complete. (Run `mlflow ui --port 5000` to compare)")


if __name__ == "__main__":
    main()
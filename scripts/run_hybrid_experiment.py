"""
Hybrid Search Experiment Runner
1. Run evaluation on dense-only retrieval
2. Run the same evaluation on hybrid retrieval (BM25 + dense + RRF)
3. Optionally runs a grid search over alph (BM25 weight vs dense weight)
4. Logs everything to MLflow for side-by-side comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from statistics import mean

import mlflow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)


def load_golden_dataset(
    path: str = "data/golden_eval_dataset.json",
    limit: int | None = None,
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
                    "score": c.score,
                }
                for c in response.retrieved_chunks
            ],
            "latency_ms": response.latency_ms,
            "error": None,
        }
    except Exception as e:
        logger.error(f" FAILED: {e}")
        return {
            "query": query,
            "answer": "",
            "retrieved_chunks": [],
            "latency_ms": 0,
            "error": str(e),
        }


def score_results(cases: list[dict], results: list[dict]) -> dict:
    """
    Score retrieval quality by checking if the expected source was retrieved.
    """
    successful = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    if not successful:
        return {"source_accuracy": 0.0, "avg_latency_ms": 0, "error_rate": 1.0}

    # Check if retrieved from the expected source
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
        "error_rate": len(errors) / len(results) if results else 0,
    }


def run_experiment(
    cases: list[dict],
    mode: str,
    config_overrides: dict | None = None,
    chunks_path: str = "data/processed/felinet_chunks.json",
    delay: float = 12.0,
) -> dict:
    """
    Run the evaluation in either dense or hybrid mode.

    Parameters
    ----------
    cases : list[dict]
        Golden test cases.
    mode : str
        "dense" for current baseline, "hybrid" for BM25 + dense + RRF.
    config_overrides : dict, optional
        Override config values (e.g., {"bm25_weight": 0.7}).
    chunks_path : str
        Path to the chunked corpus JSON (for building BM25 index).
    delay : float
        Seconds between Groq API calls to respect rate limits.
    """
    from felinet.embeddings.embedder import load_embedding_model
    from felinet.rag.pipeline import query_rag
    from felinet.schemas import RAGConfig

    # Build config with any overrides
    config = RAGConfig()
    if config_overrides:
        if "bm25_weight" in config_overrides:
            config.retrieval.bm25_weight = config_overrides["bm25_weight"]
        if "dense_weight" in config_overrides:
            config.retrieval.dense_weight = config_overrides["dense_weight"]

    # Load embedding model once, reuse for all queries
    emb_model = load_embedding_model(config.embedding_model)

    # For hybrid mode, build BM25 index once
    bm25_index = None
    if mode == "hybrid":
        from felinet.rag.retriever import BM25Index

        bm25_index = BM25Index.from_corpus(chunks_path)

    # Define the pipeline function
    def pipeline_fn(query: str):
        return query_rag(
            query=query,
            config=config,
            embedding_model=emb_model,
            retrieval_mode=mode,
            bm25_index=bm25_index,
        )

    # Run all cases
    results = []
    for i, case in enumerate(cases):
        logger.info(f"  [{i + 1}/{len(cases)}] {case['query'][:50]}...")
        result = run_single_query(case["query"], pipeline_fn)
        results.append(result)

        # Rate limit delay
        if i < len(cases) - 1:
            time.sleep(delay)

    scores = score_results(cases, results)
    return {**scores, "mode": mode}


def main():
    parser = argparse.ArgumentParser(description="Run hybrid search experiments")
    parser.add_argument("--limit", type=int, default=None, help="Limit test cases")
    parser.add_argument("--delay", type=float, default=12.0, help="Seconds between API calls")
    parser.add_argument("--grid-search", action="store_true", help="Run alpha grid search")
    parser.add_argument(
        "--chunks",
        type=str,
        default="data/processed/felinet_chunks.json",
        help="Path to chunked corpus JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/golden_eval_dataset.json",
        help="Path to golden evaluation dataset",
    )
    args = parser.parse_args()

    cases = load_golden_dataset(args.dataset, args.limit)
    logger.info(f"Loaded {len(cases)} test cases")

    mlflow.set_experiment("felinet-rag-pipeline")

    # Experiment 1: Dense only baseline
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Dense-only retrieval (baseline)")
    print("=" * 70)

    with mlflow.start_run(run_name="week7_dense_baseline"):
        mlflow.log_param("retrieval_mode", "dense")
        mlflow.log_param("num_test_cases", len(cases))

        dense_scores = run_experiment(cases, mode="dense", delay=args.delay)

        mlflow.log_metric("souce_accuracy", dense_scores["source_accuracy"])
        mlflow.log_metric("avg_latency_ms", dense_scores["avg_latency_ms"])
        mlflow.log_metric("error_rate", dense_scores["error_rate"])

        print(f"\n  Source accuracy: {dense_scores['source_accuracy']:.1%}")
        print(f"  Avg latency:    {dense_scores['avg_latency_ms']:.0f}ms")
        print(f"  Error rate:     {dense_scores['error_rate']:.1%}")

    # Experiment 2: Hybrid search (default 0.5/0.5)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Hybrid search (BM25 + Dense, default weights)")
    print("=" * 70)

    with mlflow.start_run(run_name="week7_hybrid_default"):
        mlflow.log_param("retrieval_mode", "hybrid")
        mlflow.log_param("bm25_weight", 0.5)
        mlflow.log_param("dense_weight", 0.5)
        mlflow.log_param("rrf_k", 60)
        mlflow.log_param("num_test_cases", len(cases))

        hybrid_scores = run_experiment(
            cases, mode="hybrid", chunks_path=args.chunks, delay=args.delay
        )

        mlflow.log_metric("source_accuracy", hybrid_scores["source_accuracy"])
        mlflow.log_metric("avg_latency_ms", hybrid_scores["avg_latency_ms"])
        mlflow.log_metric("error_rate", hybrid_scores["error_rate"])
        print(f"\n  Source accuracy: {hybrid_scores['source_accuracy']:.1%}")
        print(f"  Avg latency:    {hybrid_scores['avg_latency_ms']:.0f}ms")
        print(f"  Error rate:     {hybrid_scores['error_rate']:.1%}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Dense vs Hybrid")
    print("=" * 70)
    improvement = hybrid_scores["source_accuracy"] - dense_scores["source_accuracy"]
    print(f"\n  Dense accuracy:  {dense_scores['source_accuracy']:.1%}")
    print(f"  Hybrid accuracy: {hybrid_scores['source_accuracy']:.1%}")
    print(f"  Improvement:     {improvement:+.1%}")
    print(f"\n  Dense latency:   {dense_scores['avg_latency_ms']:.0f}ms")
    print(f"  Hybrid latency:  {hybrid_scores['avg_latency_ms']:.0f}ms")

    # Optional grid search
    if args.grid_search:
        print("\n" + "=" * 70)
        print("GRID SEARCH: Tuning BM25 vs Dense weights")
        print("=" * 70)

        # alpha = BM25 weight, dense weight = 1 - alpha
        alphas = [0.3, 0.5, 0.7]
        for alpha in alphas:
            print(f"\n  --- Alpha={alpha} (BM25={alpha}, Dense={1-alpha}) ---")
            with mlflow.start_run(run_name=f"week7_hybrid_search_{alpha}"):
                mlflow.log_param("retrieval_mode", "hybrid")
                mlflow.log_param("bm25_weight", alpha)
                mlflow.log_param("dense_weight", 1 - alpha)
                mlflow.log_param("rrf_k", 60)
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("num_test_cases", len(cases))

                scores = run_experiment(
                    cases,
                    mode="hybrid",
                    config_overrides={"bm25_weight": alpha, "dense_weight": 1 - alpha},
                    chunks_path=args.chunks,
                    delay=args.delay,
                )

                mlflow.log_metric("source_accuracy", scores["source_accuracy"])
                mlflow.log_metric("avg_latency_ms", scores["avg_latency_ms"])
                mlflow.log_metric("error_rate", scores["eror_rate"])
                print(f"    Source accuracy: {scores['source_accuracy']:.1%}")
                print(f"    Avg latency:    {scores['avg_latency_ms']:.0f}ms")

    print("\nAll experiments complete - Run `mlflow ui --port 5000` to compare.")


if __name__ == "__main__":
    main()

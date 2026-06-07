"""
FeliNet RAG Evaluation Framework.
Scores the RAG pipeline using DeepEval's RAGAS-style metrics:
 - Faithfulness: Does the answer stick to the retrieval context?
 - Answer Relevancy: Does the answer address the question asked?
 - Contextual Precision: Are the retrieval chunks relevant to the query?
 - Contextual Recall: Did retrieval find all the information needed?
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Step 1: Load the golden dataset


def load_golen_dataset(
    path: str | Path = "data/eval/golden_eval_dataset.json",
    category: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Load test cases from the golden evaluation dataset.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    category : str, optional
        Filter by content_type (e.g., "disease", "breed_profile").
    limit : int, optional
        Only return the first N cases (useful for quick iteration).
    """
    with open(path) as f:
        data = json.load(f)

    cases = data["test_cases"]
    if category:
        cases = [c for c in cases if c["content_category"] == category]
        logger.info(f"Filtered to {len(cases)} cases with content_type = {category}")

    if limit:
        cases = cases[:limit]
        logger.info(f"Limited to {limit} cases")
    return cases


# Step 2: Run the RAG pipeline on each test case


def run_pipeline_on_cases(cases: list[dict], delay_seconds: float = 12.0) -> list[dict]:
    """
    Run the RAG pipeline on each test case and collect results

    Each result contains:
        - query: the original question
        - expected_answer: what the answer should be
        - actual_answer: what the RAG pipeline returned
        - retrieval_contexts: the text of each retrieved chunk
        - expected_source: which source should have been retrieved
        - actual_sources: which sources were actually retrieved
        - latency_ms: how long the pipeline took

    Parameters
    ----------
    delay_seconds : float
        Seconds to wait between API calls to respect Groq rate limits.
        Groq free tier = 12K tokens/min, so 12s between calls is safe.
    """
    from felinet.embeddings.embedder import load_embedding_model
    from felinet.rag.pipeline import query_rag
    from felinet.schemas import RAGConfig

    config = RAGConfig()
    model = load_embedding_model(config.embedding_model)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Evaluating {i + 1}/{len(cases)}: {case['query'][:60]}...")

        try:
            respense = query_rag(query=case["query"], config=config, embedding_model=model)

            result = {
                "id": case["id"],
                "query": case["query"],
                "expected_answer": case["expected_answer"],
                "actual_answer": respense.answer,
                "retrieved_contexts": [chunk.content for chunk in respense.retrieved_chunks],
                "expected_source": case["expected_source"],
                "actual_sources": [chunk.source.value for chunk in respense.retrieved_chunks],
                "content_type": case["content_type"],
                "difficulty": case["difficulty"],
                "latency_ms": respense.latency_ms,
            }
            results.append(result)

            # Show a preview
            preview = respense.answer[:100].replace("\n", " ")
            logger.info(f"  Answer: {preview}...")
            logger.info(f"  Latency: {respense.latency_ms:.0f}ms")

        except Exception as e:
            logger.error(f" FAILED: {e}")
            results.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_answer": case["expected_answer"],
                    "actual_answer": f"ERROR: {e}",
                    "retrieved_contexts": [],
                    "expected_source": case["expected_source"],
                    "actual_sources": [],
                    "content_type": case["content_type"],
                    "difficulty": case["difficulty"],
                    "latency_ms": 0,
                }
            )

        # Rate limit delay (skip after last case)
        if i < len(cases) - 1:
            logger.info(f"  Waiting {delay_seconds}s for rate limit...")
            time.sleep(delay_seconds)
    return results


# Step 3: Score results using DeepEval metrics


def score_with_deepeval(results: list[dict]) -> dict:
    """
    Score RAG pipeline results using DeepEval's LLM-as-judge metrics.

    DeepEval uses an LLM (by default GPT-4) to evaluate each response.
    This costs ~$0.01-0.05 per test case depending on context length.
    """
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevanceMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    # Build DeepEval test cases
    test_cases = []
    for r in results:
        if r["actual_answer"].startswith("ERROR:"):
            continue
        tc = LLMTestCase(
            input=r["query"],
            actual_output=r["actual_answer"],
            expected_output=r["expected_answer"],
            retrieval_context=r["retrieved_contexts"],
        )
        test_cases.append(tc)

    # Define metrics with thresholds acting as quality gates - the pipeline must score above these to be considered "good enough" for production
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevanceMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.6),
        ContextualRecallMetric(threshold=0.6),
    ]

    # Run evaluation
    logger.info(f"Running DeepEval on {len(test_cases)} test cases with {len(metrics)} metrics...")
    eval_result = evaluate(test_cases=test_cases, metrics=metrics)
    return eval_result


# Step 4: Compute simple (non-LLM) metrics as a free alternative
def score_without_llm_judge(results: list[dict]) -> dict:
    """
    Compute retrieval quality metrics that do not require an LLM judge. (Solid baseline)

    Metrics:
        - source_accuracy: How often did retrieval return chunks from the expected scourse?
        - avg_latency_ms: Average end-to-end latency.
        - avg_rate: What fraction of queries failed completely?
        - avg_top_score: Average similarity score of the top retrieved chunk.
    """
    valid_results = [r for r in results if not r["actual_answer"].startswith("ERROR:")]
    failed = len(results) - len(valid_results)

    # Source accuracy: did we retrieve from the right source?
    source_hits = 0
    for r in valid_results:
        if r["expected_source"] in r["actual_sources"]:
            source_hits += 1

    # Latency stats
    latencies = [r["latency_ms"] for r in valid_results if r["latency_ms"] > 0]

    # Breakdown by content type
    by_category = {}
    for r in valid_results:
        cat = r["content_type"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "source_hits": 0, "latencies": []}
        by_category[cat]["total"] += 1
        if r["expected_source"] in r["actual_sources"]:
            by_category[cat]["source_hits"] += 1
        by_category[cat]["latencies"].append(r["latency_ms"])

    # Breakdown by difficulty
    by_difficulty = {}
    for r in valid_results:
        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "source_hits": 0}
        by_difficulty[diff]["total"] += 1
        if r["expected_source"] in r["actual_sources"]:
            by_difficulty[diff]["source_hits"] += 1

    scores = {
        "total_cases": len(results),
        "successful_cases": len(valid_results),
        "error_rate": failed / len(results) if results else 0,
        "source_accuracy": source_hits / len(valid_results) if valid_results else 0,
        "avg_latency_ms": mean(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "by_category": {
            cat: {
                "source_accuracy": d["source_hits"] / d["total"] if d["total"] else 0,
                "avg_latency_ms": mean(d["latencies"]) if d["latencies"] else 0,
                "count": d["total"],
            }
            for cat, d in by_category.items()
        },
        "by_difficulty": {
            diff: {
                "source_accuracy": d["source_hits"] / d["total"] if d["total"] else 0,
                "count": d["total"],
            }
            for diff, d in by_difficulty.items()
        },
    }

    return scores


# Step 5: Log results to MLflow


def log_to_mlflow(scores: dict, run_name: str = "baseline_eval") -> None:
    """
    Log evaluation to MLflow for tracking over time for comparison.
    """
    try:
        import mlflow

        mlflow.set_experiment("felinet-rag-evaluation")

        with mlflow.start_run(run_name=run_name):
            # Log top-level metrics
            mlflow.log_metric("source_accuracy", scores["source_accuracy"])
            mlflow.log_metric("avg_latency_ms", scores["avg_latency_ms"])
            mlflow.log_metric("p95_latency_ms", scores["p95_latency_ms"])
            mlflow.log_metric("error_rate", scores["error_rate"])
            mlflow.log_metric("total_cases", scores["total_cases"])

            # Log per-category metrics
            for cat, cat_scores in scores.get("by_category", {}).items():
                mlflow.log_metric(f"avg_accuracy_{cat}", cat_scores["source_accuracy"])
                mlflow.log_metric(f"avg_latency_{cat}", cat_scores["avg_latency_ms"])

            # Log per-difficulty metrics
            for diff, diff_scores in scores.get("by_difficulty", {}).items():
                mlflow.log_metric(f"source_accuracy_{diff}", diff_scores["source_accuracy"])

            # Log pipeline config as parameters
            from felinet.schemas import RAGConfig

            config = RAGConfig()
            mlflow.log_param("embedding_model", config.embedding_model)
            mlflow.log_param("top_k", config.retrieval.top_k_reranked)
            mlflow.log_param("chunk_size", config.chunking.chunk_size)
            mlflow.log_param("llm_model", config.generation.model_name)

            logger.info(f"Results logged to MLflow run: {run_name}")

    except ImportError:
        logger.warning("MLflow not installed - skipping logging")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# Step 6: Pretty print results
def print_results(scores: dict) -> None:
    """Print evaluation results in a readable format."""

    print("\n" + "=" * 70)
    print("FeliNet RAG Evaluation Results — Baseline")
    print("=" * 70)

    print(f"\n  Total cases:      {scores['total_cases']}")
    print(f"  Successful:       {scores['successful_cases']}")
    print(f"  Error rate:       {scores['error_rate']:.1%}")
    print(f"\n  Source accuracy:  {scores['source_accuracy']:.1%}")
    print(f"  Avg latency:      {scores['avg_latency_ms']:.0f}ms")
    print(f"  P95 latency:      {scores['p95_latency_ms']:.0f}ms")

    print("\n  By content type:")
    for cat, cat_scores in scores.get("by_category", {}).items():
        print(
            f"    {cat:20s}  accuracy={cat_scores['source_accuracy']:.1%}  "
            f"latency={cat_scores['avg_latency_ms']:.0f}ms  "
            f"n={cat_scores['count']}"
        )

    print("\n  By difficulty:")
    for diff, diff_scores in scores.get("by_difficulty", {}).items():
        print(
            f"    {diff:10s}  accuracy={diff_scores['source_accuracy']:.1%}  "
            f"n={diff_scores['count']}"
        )

    print("\n" + "=" * 70)


# Main


def main():
    parser = argparse.ArgumentParser(description="Run FeliNet RAG evaluation")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate first N cases (default: all)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by content type (disease, breed_profile, nutrition, behavior, toxicology)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/golden_eval_dataset.json",
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=12.0,
        help="Seconds between API calls (default: 12 for Groq free tier)",
    )
    parser.add_argument(
        "--use-deepeval",
        action="store_true",
        help="Use DeepEval LLM-as-judge metrics (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline_eval",
        help="MLflow run name (default: baseline_eval)",
    )

    args = parser.parse_args()

    # Load test cases

    cases = load_golen_dataset(path=args.dataset, category=args.category, limit=args.limit)
    print(f"\nLoaded {len(cases)} test cases")

    # Run pipeline
    results = run_pipeline_on_cases(cases, delay_seconds=args.delay)

    # Score
    if args.use_deepeval:
        print("\nRunning DeepEval LLM-as-judge scoring...")
        deepeval_result = score_with_deepeval(results)
        print(deepeval_result)
    else:
        print("\nComputing retrieval metrics (no LLM judge)...")

    # Always compute free metrics
    scores = score_without_llm_judge(results)
    print_results(scores)

    # Log to MLflow
    log_to_mlflow(scores, run_name=args.run_name)

    # Save raw results for later analysis
    output_path = Path("data/eval/results")
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"{args.run_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nRaw results saved to {results_file}")


if __name__ == "__main__":
    main()

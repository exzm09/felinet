"""
Scheduled monitoring flows: daily eval + weekly drift
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
from prefect import flow, task

from felinet.monitoring.alerts import alert
from felinet.monitoring.drift import batch_drift_report, evidently_html_report
from felinet.monitoring.query_logger import load_queries

logger = logging.getLogger(__name__)
GOLDEN = Path("data/eval/golden_eval_dataset.json")
FAITHFULNESS_FLOOR = 0.99


# Daily
@task
def evaluate_faithfulness(n: int = 10) -> float:
    """
    Run ~10 golden cases through the live pipeline and score faithfulness
    """
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from felinet.embeddings.embedder import load_embedding_model
    from felinet.rag.pipeline import query_rag
    from felinet.schemas import RAGConfig

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["test_cases"][:n]
    config = RAGConfig()
    model = load_embedding_model(config.embedding_model)
    metric = FaithfulnessMetric(threshold=FAITHFULNESS_FLOOR, model="gpt-4o-mini")

    scores = []
    for case in golden:
        resp = query_rag(query=case["query"], config=config, embedding_model=model)
        tc = LLMTestCase(
            input=case["query"],
            actual_output=resp.answer,
            retrieval_context=[c.content for c in resp.retrieved_chunks],
        )

        metric.measure(tc)
        scores.append(metric.score)
    return sum(scores) / len(scores) if scores else 0.0


@flow(name="daily-eval")
def daily_eval_flow() -> None:
    mean_faithfulness = evaluate_faithfulness()
    with mlflow.start_run(run_name="daily_eval"):
        mlflow.log_metric("mean_faithfulness", mean_faithfulness)
    logger.info(f"Daily faithfulness = {mean_faithfulness:.3f}")
    if mean_faithfulness < FAITHFULNESS_FLOOR:
        alert(f"Faithfulness dropped to {mean_faithfulness:.3f} (floor {FAITHFULNESS_FLOOR})")


# Weekly
@task
def embed_texts(texts: list[str]) -> list[list[float]]:
    from felinet.embeddings.embedder import load_embedding_model
    from felinet.schemas import RAGConfig

    model = load_embedding_model(RAGConfig().embedding_model)
    return model.encode(texts, normalize_embeddings=True).tolist()


@flow(name="weekly-drift")
def weekly_drift_flow() -> None:
    # Reference = golden questions
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    ref_texts = [c["query"] for c in golden["test_cases"]]

    # Current = what users actually asked in the last 7 days
    current_rows = load_queries(since_days=7)
    cur_texts = [r["query"] for r in current_rows]

    if len(cur_texts) < 20:
        logger.warning(
            f"Only {len(cur_texts)} recent queries - too few for reliable dirft. Skipping"
        )
        return

    ref_emb = embed_texts(ref_texts)
    cur_emb = embed_texts(cur_texts)

    report = batch_drift_report(ref_emb, cur_emb)
    logger.info(f"Drift report: {report}")
    with mlflow.start_run(run_name="weekly_drift"):
        mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})

    try:
        evidently_html_report(ref_emb, cur_emb)
    except Exception as e:  # never let the cosmetic report kill the flow
        logger.warning(f"Evidently report failed (non-fatal): {e}")

    if report["drift_detected"]:
        alert(
            f"Query drift detected! centroid_shift="
            f"{report['centroid_shift_cosine_distance']}, PSI={report['psi_on_corpus_similarity']}"
        )


# Schedule
if __name__ == "__main__":
    from prefect import serve

    serve(
        daily_eval_flow.to_deployment(name="daily-eval", cron="0 8 * * *"),  # 8am daily
        weekly_drift_flow.to_deployment(name="weekly-drift", cron="0 9 * * 1"),  # 9am Mondays
    )

"""
Run a controlled A/B test over the golden dataset.

For each golden question run BOTH variants, score each answer with DeepEval
(faithfulness + answer relevancy), and log:
  - one JSONL row per (variant, question) -> for the t-test later
  - aggregate per-variant means -> to MLflow
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from felinet.embeddings.embedder import load_embedding_model
from felinet.experiments.variants import VARIANTS, make_variant_config
from felinet.rag.pipeline import query_rag
from felinet.schemas import RAGConfig

load_dotenv()

GOLDEN_PATH = Path("data/eval/golden_eval_dataset.json")
RESULTS_DIR = Path("data/ab_results")
JUDGE_MODEL = "gpt-4o-mini"


def load_questions(limit: int | None = None) -> list[str]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    questions = [row["query"] for row in data["test_cases"]]
    return questions[:limit] if limit else questions


def score_answer(question: str, response) -> dict:
    """
    Run DeepEval on one answer; return both scores.
    a"""
    test_case = LLMTestCase(
        input=question,
        actual_output=response.answer,
        retrieval_context=[c.content for c in response.retrieved_chunks],
    )
    faith = FaithfulnessMetric(threshold=0.75, model=JUDGE_MODEL)
    rel = AnswerRelevancyMetric(threshold=0.70, model=JUDGE_MODEL)
    faith.measure(test_case)
    rel.measure(test_case)
    return {"faithfulness": faith.score, "relevancy": rel.score}


def main(limit: int = 10):  # START SMALL. Bump to 50 (None = all) once it works.
    questions = load_questions(limit=limit)
    model = load_embedding_model(RAGConfig().embedding_model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"prompt_ab_{stamp}.jsonl"

    rows = []
    with results_path.open("w", encoding="utf-8") as f:
        for variant in ("A", "B"):
            config = make_variant_config(variant)
            for q in questions:
                resp = query_rag(query=q, config=config, embedding_model=model)
                scores = score_answer(q, resp)
                row = {
                    "variant": variant,
                    "question": q,
                    "faithfulness": scores["faithfulness"],
                    "relevancy": scores["relevancy"],
                    "latency_ms": resp.latency_ms,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                print(f"[{variant}] {q[:50]}... faith={scores['faithfulness']:.2f}")

    # Aggregate + log to MLflow
    mlflow.set_experiment("felinet_ab_prompt")
    with mlflow.start_run(run_name=f"prompt_ab_{stamp}"):
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("n_questions", len(questions))
        for variant in ("A", "B"):
            v = [r for r in rows if r["variant"] == variant]
            mlflow.log_metric(f"faithfulness_{variant}", sum(r["faithfulness"] for r in v) / len(v))
            mlflow.log_metric(f"relevancy_{variant}", sum(r["relevancy"] for r in v) / len(v))
            mlflow.log_param(f"prompt_{variant}", VARIANTS[variant][:250])
        mlflow.log_artifact(str(results_path))

    print(f"\nSaved results to {results_path}")
    print(f"Now run:  python scripts/analyze_ab_test.py {results_path} faithfulness")


if __name__ == "__main__":
    main(limit=10)

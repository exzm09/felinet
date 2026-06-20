"""
A/B router for live traffic.

For each incoming query we randomly send it to variant A or B, run the normal
RAG pipeline with that variant's config, and log the outcome to a JSONL file.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer

from felinet.experiments.variants import make_variant_config
from felinet.rag.pipeline import query_rag
from felinet.schemas import RAGResponse

AB_LOG_PATH = Path("data/ab_logs/live_ab.jsonl")


def assign_variant(split: float = 0.5) -> str:
    """
    Randomly return 'A' or 'B'.
    """
    return "A" if random.random() < split else "B"


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def query_rag_ab(
    query: str,
    embedding_model: SentenceTransformer | None = None,
    split: float = 0.5,
    log_path: Path = AB_LOG_PATH,
    **kwargs,
) -> RAGResponse:
    """
    Run the RAG pipeline through a randomly chosen variant and log the result.
    Returns the normal RAGResponse, so callers don't even need to know an A/B
    test is running underneath.
    """
    variant = assign_variant(split)
    config = make_variant_config(variant)

    response = query_rag(query=query, config=config, embedding_model=embedding_model, **kwargs)

    _append_jsonl(
        log_path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "variant": variant,
            "query": query,
            "trace_id": response.trace_id,
            "latency_ms": response.latency_ms,
            "model_used": response.model_used,
            "n_chunks": len(response.retrieved_chunks),
        },
    )
    return response

"""
User feedback storage + Langfuse linking.

When a user clicks thumbs up/down we:
  1. Append a row to a local JSONL file.
  2. Attach a score to the matching Langfuse trace, so feedback shows up next to
     the trace we already captured (retrieved chunks, latency, tokens).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FEEDBACK_LOG_PATH = Path("data/feedback/feedback.jsonl")


def record_feedback(
    trace_id: str | None,
    query: str,
    answer: str,
    liked: bool,
    comment: str | None = None,
    log_path: Path = FEEDBACK_LOG_PATH,
) -> None:
    """
    Store one feedback event.
    liked=True  -> thumbs up (score 1.0)
    liked=False -> thumbs down (score 0.0)
    """
    rating = 1.0 if liked else 0.0

    # 1. Local durable log (always works, even if Langfuse is off)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "trace_id": trace_id,
                    "query": query,
                    "answer": answer,
                    "rating": rating,
                    "comment": comment,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # 2. Push the score to Langfuse (best-effort; never crash the app over feedback)
    if trace_id and os.getenv("LANGFUSE_ENABLED", "true").lower() != "false":
        try:
            from langfuse import Langfuse

            client = Langfuse()
            client.score(trace_id=trace_id, name="user_feedback", value=rating, comment=comment)
            client.flush()  # make sure it's sent before the process exits
        except Exception as e:
            logger.warning(f"Could not send feedback to Langfuse: {e}")

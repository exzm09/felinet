"""
Append incoming queries to a JONSL log for offline drift analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_PATH = Path("data/monitoring/query_log.jsonl")


def log_query(query: str, extra: dict | None = None) -> None:
    """
    One line per query.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"query": query, "timestamp": datetime.now(timezone.utc).isoformat()}
    if extra:
        record.update(extra)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_queries(since_days: int | None = None) -> list[dict]:
    """
    Load all logged queries, optionally only those from the last N days
    """
    if not LOG_PATH.exists():
        return []
    rows = [
        json.loads(line)
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if since_days is None:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    recent = []
    for r in rows:
        try:
            if datetime.fromisoformat(r["timestamp"]) >= cutoff:
                recent.append(r)
        except (KeyError, ValueError):
            continue
    return recent

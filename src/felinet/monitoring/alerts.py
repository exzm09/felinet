"""
Lightweight alerting: console + file, optional Slack webhook.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
ALERT_LOG = Path("data/monitoring/alerts.log")


def alert(message: str) -> None:
    logger.warning(f"ALERT: {message}")
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}  {message}\n")

    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            requests.post(
                webhook, json={"text": f":rotating_light: FeliNet: {message}"}, timeout=10
            )
        except Exception as e:
            logger.warning(f"Slack alert failed: {e}")

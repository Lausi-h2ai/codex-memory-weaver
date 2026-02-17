"""Structured logging helpers for MCP tool telemetry."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def emit_tool_log(
    logger: logging.Logger,
    *,
    event: str,
    tool: str,
    correlation_id: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "tool": tool,
        "correlation_id": correlation_id,
        **fields,
    }
    logger.log(level, json.dumps(payload, default=str))

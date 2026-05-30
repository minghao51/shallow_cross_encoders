"""Structured logging configuration for the reranker toolkit.

Usage:
    from reranker.logging_config import configure_logging
    configure_logging()

Or rely on auto-configuration at import time via ``reranker/__init__.py``.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(
    level: str | None = None,
    json_format: bool = False,
) -> None:
    """Configure structlog-based structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
            Defaults to ``RERANKER_LOG_LEVEL`` env var or ``INFO``.
        json_format: If True, output JSON-formatted logs (for production).
            Defaults to False (coloured console output).
    """
    if level is None:
        level = os.environ.get("RERANKER_LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if json_format:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer()
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(),
                pad_level=False,
                force_colors=sys.stderr.isatty(),
            )
        )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    for logger_name in ("httpx", "urllib3", "transformers", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

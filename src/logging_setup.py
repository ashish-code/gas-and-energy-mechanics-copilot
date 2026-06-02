"""structlog setup: JSON logs with correlation IDs.

v1 shipped a stdlib ``basicConfig`` shim; this is the real thing. Every log line is a
structured JSON object carrying a ``correlation_id`` bound for the lifetime of a request
(one user query), so a full plan-execute-verify trace can be reconstructed from logs alone
— complementary to the LangSmith span tree.

Usage::

    from src.logging_setup import configure_logging, get_logger, bind_correlation_id

    configure_logging()                      # once, at process start
    bind_correlation_id()                    # once per query (returns the id)
    log = get_logger(__name__)
    log.info("planning", query=q, n_sub_questions=3)
"""

from __future__ import annotations

import logging
import sys
import uuid

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars

from src.config import settings

_configured = False


def configure_logging(*, json_logs: bool | None = None, level: str | None = None) -> None:
    """Configure structlog + stdlib logging. Idempotent."""
    global _configured
    if _configured:
        return

    json_logs = settings.log_json if json_logs is None else json_logs
    level = (level or settings.log_level).upper()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list = [
        merge_contextvars,  # injects correlation_id (and any other bound context)
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (boto3, httpx, langchain) through stderr at the same level.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation id to the current context (one per query). Returns the id."""
    cid = correlation_id or uuid.uuid4().hex[:12]
    bind_contextvars(correlation_id=cid)
    return cid


def clear_context() -> None:
    """Clear all context-local bindings (call at the end of a request)."""
    clear_contextvars()

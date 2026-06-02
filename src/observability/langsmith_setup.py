"""LangSmith tracing — env-gated, no-op when unconfigured.

LangChain/LangGraph emit traces automatically when the LANGCHAIN_* env vars are set. We
translate our typed settings into those vars at startup. With no API key (or tracing off),
this is a no-op and the app runs identically — local dev never requires LangSmith.
"""

from __future__ import annotations

import os

from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)


def setup_langsmith() -> bool:
    """Enable tracing if configured. Returns True when tracing is active."""
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        log.info("langsmith.disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    log.info("langsmith.enabled", project=settings.langsmith_project)
    return True

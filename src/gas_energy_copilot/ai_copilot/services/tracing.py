"""
Module: services/tracing.py

PURPOSE:
    Centralised Langfuse tracing for all LLM calls and agent pipeline runs.
    Every user query generates a "trace" (top-level unit of work) with nested
    "spans" for each agent step (routing, retrieval, synthesis, judging).
    Evaluation scores from TruLens and DeepEval are attached to traces so you
    can drill into low-scoring queries directly in the Langfuse dashboard.

ARCHITECTURE POSITION:

    HTTP Request → chat endpoint
                        │
                        ▼
                  langfuse_trace() ─────────────── creates Trace in Langfuse
                        │                               │
                        ├─ span("routing")          ────┤
                        ├─ span("retrieval")         ────┤ Nested spans
                        ├─ span("synthesis")         ────┤
                        └─ span("judging")           ────┘
                        │
                        ▼ crew.kickoff() ← LiteLLM callback auto-sends all LLM calls
                        │
                  score("trulens_groundedness", 0.82)  ← attached after eval
                  score("judge_confidence", 0.91)

HOW LANGFUSE INTEGRATES WITH CREWAI:
    CrewAI uses LiteLLM under the hood for all LLM calls. LiteLLM supports
    "success_callback" and "failure_callback" hooks. Langfuse registers itself
    as a LiteLLM callback by reading LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    from environment variables. This means ALL Bedrock calls made by CrewAI agents
    are automatically traced — no manual instrumentation of each agent needed.

    This module:
      1. Sets the environment variables from our typed_settings config (if enabled).
      2. Provides a context manager (langfuse_trace) for the top-level trace.
      3. Provides helper functions for scoring traces after evaluation.

WHY LANGFUSE over alternatives:
    - vs Arize Phoenix: Phoenix excels at embedding drift detection and ML monitoring
      dashboards. Langfuse excels at conversation tracing with RAG span hierarchy
      and is more natural for chat-style applications.
    - vs LangSmith: LangSmith is purpose-built for LangChain and has limited support
      for other frameworks. Langfuse is framework-agnostic.
    - vs OpenTelemetry: OTel is general-purpose infrastructure tracing (HTTP spans,
      DB calls). Langfuse adds LLM-specific semantics: prompt/completion content,
      token costs, model parameters, quality scores.
    - vs Weights & Biases Weave: W&B is strong for ML experiment tracking but is
      heavier and more complex than needed for production API tracing.

KEY CONCEPTS — Langfuse Object Model:
    Trace:  Top-level unit. One trace per user query. Has session_id for grouping
            all traces from the same chat session. Visible in Langfuse as a timeline.
    Span:   Child of a trace. Represents one agent step or tool call.
            Has start_time, end_time, input, output.
    Score:  Numeric or categorical quality score attached to a trace.
            Used to surface low-quality responses in the Langfuse dashboard.
    Generation: Special span type for LLM calls — captures prompt, completion,
                model name, token counts, and latency. Auto-created by LiteLLM callback.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import guard — langfuse is optional (only required if enabled)
# ---------------------------------------------------------------------------


def _try_import_langfuse():
    """
    Attempt to import the Langfuse client library.

    Why lazy import: langfuse is an optional dependency. If it is not installed
    and tracing is disabled (the default), we should not raise ImportError.
    If tracing is enabled without langfuse installed, we provide a clear error.

    Returns:
        Langfuse class if available, None otherwise.
    """
    try:
        from langfuse import Langfuse
        return Langfuse
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_langfuse_client = None
_tracing_enabled: bool | None = None  # None = not yet checked


def _get_client():
    """
    Return the Langfuse singleton client, or None if tracing is disabled.

    Initialises the client on first call by:
      1. Reading config to check if tracing is enabled.
      2. Setting LANGFUSE_* env vars so LiteLLM's callback picks them up.
      3. Creating the Langfuse client with project-level settings.

    Thread safety: Python's GIL makes module-level singleton assignment atomic
    for simple assignments. The double-check pattern here is sufficient for our
    single-process uvicorn deployment.
    """
    global _langfuse_client, _tracing_enabled

    if _tracing_enabled is not None:
        # Already initialised (either enabled or disabled)
        return _langfuse_client

    try:
        from gas_energy_copilot.ai_copilot.core.config import app_config
        config = app_config().langfuse
    except Exception as e:
        log.debug("Could not load langfuse config, tracing disabled", exc_info=e)
        _tracing_enabled = False
        return None

    if not config.enabled:
        log.debug("Langfuse tracing is disabled (set [app.langfuse] enabled=true to activate)")
        _tracing_enabled = False
        return None

    Langfuse = _try_import_langfuse()
    if Langfuse is None:
        log.warning(
            "Langfuse tracing is enabled in config but langfuse package is not installed. "
            "Run: uv add langfuse"
        )
        _tracing_enabled = False
        return None

    # Set environment variables for LiteLLM auto-callback.
    # LiteLLM reads LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY at the module level,
    # so these must be set before any LiteLLM import. We set them here rather than
    # requiring the user to set them manually.
    if config.public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.public_key)
    if config.secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", config.secret_key)
    os.environ.setdefault("LANGFUSE_HOST", config.host)

    try:
        _langfuse_client = Langfuse(
            public_key=config.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=config.secret_key or os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=config.host,
            flush_at=config.flush_at,
            flush_interval=config.flush_interval,
        )
        _tracing_enabled = True
        log.info("Langfuse tracing enabled", extra={"host": config.host, "project": config.project_name})
    except Exception as e:
        log.warning("Langfuse client init failed — tracing disabled", exc_info=e)
        _tracing_enabled = False

    return _langfuse_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def langfuse_trace(
    name: str,
    session_id: str,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager that wraps an agent pipeline run in a Langfuse trace.

    Everything that happens inside the `with` block (LLM calls, tool calls,
    agent steps) is automatically grouped under this trace in the Langfuse UI.

    Usage:
        with langfuse_trace("chat", session_id=session_id, metadata={"question": q}) as trace:
            result = crew.kickoff(inputs={...})
            # After the crew runs, attach evaluation scores:
            score_trace(trace, "answer_length", len(result.raw))

    If tracing is disabled (langfuse.enabled=false), this is a no-op context
    manager that yields None. The caller can safely call score_trace(None, ...) —
    score_trace handles None gracefully.

    Args:
        name:       Human-readable name for this trace (e.g., "chat", "eval_run").
        session_id: UUID grouping all traces from the same chat session.
                    In the Langfuse UI, you can filter by session_id to see
                    the full conversation history for a user.
        user_id:    Optional user identifier for per-user analytics.
        metadata:   Extra key-value pairs stored with the trace (question text,
                    query_type, etc.).

    Yields:
        Langfuse Trace object (or None if tracing is disabled).
        Use the yielded object to call `score_trace()`.
    """
    client = _get_client()
    if client is None:
        yield None
        return

    trace = client.trace(
        name=name,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata or {},
    )
    try:
        yield trace
    finally:
        # Flush is non-blocking (background thread). We call it to ensure traces
        # are sent even if the process exits soon after (e.g., one-off scripts).
        # In long-running servers, Langfuse also flushes periodically via flush_interval.
        client.flush()


def score_trace(
    trace: Any,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """
    Attach a numeric quality score to a Langfuse trace.

    Scores appear in the Langfuse dashboard as a column next to each trace,
    making it easy to sort by quality and identify low-scoring conversations.

    Common score names to use:
        "trulens_context_relevance"  → TruLens context relevance score [0,1]
        "trulens_groundedness"       → TruLens groundedness score [0,1]
        "trulens_answer_relevance"   → TruLens answer relevance score [0,1]
        "judge_confidence"           → Judge agent confidence score [0,1]
        "retrieval_count"            → Number of passages retrieved (integer)

    Args:
        trace:   Langfuse Trace object from langfuse_trace context manager.
                 If None (tracing disabled), this is a no-op.
        name:    Score identifier (shown as column header in Langfuse).
        value:   Numeric value (typically [0.0, 1.0] for quality scores).
        comment: Optional explanation of how this score was computed.

    Example:
        score_trace(trace, "judge_confidence", verdict.confidence,
                    comment=f"Judge verdict: {verdict.verdict}")
    """
    if trace is None:
        return

    try:
        trace.score(name=name, value=value, comment=comment)
    except Exception as e:
        log.debug("Failed to attach score to trace", extra={"score_name": name, "error": str(e)})


def flush_traces() -> None:
    """
    Block until all pending Langfuse events are sent.

    Call this at application shutdown (FastAPI lifespan shutdown handler)
    to ensure no traces are lost when the process exits.

    Example (in application.py lifespan):
        @asynccontextmanager
        async def lifespan(app):
            yield
            flush_traces()  # ensure all traces sent before shutdown
    """
    client = _get_client()
    if client is not None:
        try:
            client.flush()
            log.info("Langfuse traces flushed")
        except Exception as e:
            log.debug("Langfuse flush failed", exc_info=e)

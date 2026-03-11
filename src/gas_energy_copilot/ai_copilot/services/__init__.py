"""
Package: services

PURPOSE:
    Exposes module-level singletons for all long-lived service objects used by
    the API layer. Importing from here (rather than directly from submodules)
    keeps the rest of the codebase decoupled from initialization details.

SINGLETONS AVAILABLE:
    get_memory_manager()   — DynamoDBMemoryManager singleton (lazy init)
    langfuse_trace()       — context manager for Langfuse tracing
    score_trace()          — attach quality scores to a trace
    flush_traces()         — flush all pending Langfuse events (call at shutdown)

LAZY INITIALIZATION:
    All singletons initialize on first use, not at import time. This means:
    - Tests that don't need DynamoDB can import freely without AWS credentials.
    - Local dev runs without Langfuse credentials (tracing just no-ops).
    - Cold start time is not affected by service initialization.

USAGE:
    # In endpoint handlers:
    from gas_energy_copilot.ai_copilot.services import get_memory_manager, langfuse_trace

    memory = get_memory_manager()
    history = memory.get_history(session_id)

    with langfuse_trace("chat", session_id=session_id) as trace:
        result = crew.kickoff(...)
        score_trace(trace, "judge_confidence", 0.91)
"""

from gas_energy_copilot.ai_copilot.services.memory import get_memory_manager
from gas_energy_copilot.ai_copilot.services.tracing import (
    flush_traces,
    langfuse_trace,
    score_trace,
)

__all__ = [
    "get_memory_manager",
    "langfuse_trace",
    "score_trace",
    "flush_traces",
]

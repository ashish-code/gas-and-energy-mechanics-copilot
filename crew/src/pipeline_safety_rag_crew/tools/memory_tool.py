"""
Module: tools/memory_tool.py

PURPOSE:
    CrewAI-compatible tools that give agents explicit access to conversation memory.
    Rather than injecting history silently in middleware, we expose it as agent tools
    so that:
      1. Agents decide *when* to load history (router may not need it; analyst does).
      2. Memory reads/writes appear as tool calls in Langfuse traces — fully observable.
      3. Future agents can selectively store only the most important parts of a session.

ARCHITECTURE POSITION:

    FullPipelineCrew agents
         │
         ├─► GetConversationHistoryTool._run(session_id)
         │       ↓ calls
         │   DynamoDBMemoryManager.get_history(session_id)
         │       ↓ returns formatted context string
         │   Agent uses context in synthesis task
         │
         └─► AddToConversationTool._run(session_id, role, content)
                 ↓ calls
             DynamoDBMemoryManager.add_turn(session_id, role, content)
                 ↓ stored in DynamoDB
             Used after synthesis to persist the Q&A pair

WHY TOOLS OVER MIDDLEWARE:
    Memory could be loaded in the FastAPI endpoint before crew kickoff and injected
    as a string into the crew inputs dict. That approach works but hides the memory
    load from observability — you cannot see it in Langfuse traces or agent logs.
    Making it a tool makes it a first-class agent action that's logged, timed, and
    auditable. It also lets the router agent *choose* to skip memory for simple
    regulatory lookups, reducing latency.

CREWAI TOOL API:
    CrewAI tools inherit from `crewai.tools.BaseTool` and implement `_run(self, ...)`.
    The agent LLM decides whether to call a tool by matching the tool's `name` and
    `description` to its current task context. The `args_schema` (Pydantic model)
    tells the LLM the exact parameters to provide.

    Alternative: `@tool` decorator (simpler but less configurable — no Pydantic schema,
    no custom name, no async support). We use BaseTool for full control.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError:  # pragma: no cover
    from crewai_tools import BaseTool  # type: ignore[no-redef]

# Import the memory manager from the main package.
# The crew package is installed as an editable dependency of the main package,
# so this cross-package import works in both dev and Docker environments.
from gas_energy_copilot.ai_copilot.services.memory import get_memory_manager

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input schemas (Pydantic v2 models)
# ---------------------------------------------------------------------------
# CrewAI uses these schemas to generate the JSON argument spec shown to the LLM.
# Field descriptions should be written as if you are telling an LLM what to provide.


class GetHistoryInput(BaseModel):
    """Input schema for GetConversationHistoryTool."""

    session_id: str = Field(
        description=(
            "The unique identifier for the current chat session. "
            "This is provided in the task inputs as {session_id}."
        )
    )


class AddTurnInput(BaseModel):
    """Input schema for AddToConversationTool."""

    session_id: str = Field(
        description="The unique identifier for the current chat session ({session_id})."
    )
    role: str = Field(
        description="Who authored this turn: 'user' for the human question, 'assistant' for the agent answer.",
        pattern="^(user|assistant)$",
    )
    content: str = Field(
        description="The full text of the message to store in conversation history."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional metadata to store alongside the message. "
            "Use this to record agent name, judge confidence score, query_type, etc."
        ),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class GetConversationHistoryTool(BaseTool):
    """
    Retrieve the conversation history for the current session.

    This tool is used by the retrieval_specialist and regulatory_analyst agents
    to understand prior turns before answering. The router_agent typically does
    NOT call this tool — routing is done on the current query alone.

    Output format:
        A formatted string of prior turns:
            CONVERSATION HISTORY:
            User: <prior question>
            Assistant: <prior answer (truncated to 500 chars)>
            ...

        Returns empty string if this is the first turn in the session.

    When to call this tool:
        - At the start of the retrieval task, to check if the user is following
          up on a prior question ("that rule" → needs to know what rule was discussed).
        - Before synthesis, to ensure the answer acknowledges prior context if relevant.
    """

    name: str = "get_conversation_history"
    description: str = (
        "Retrieve the conversation history for the current chat session. "
        "Returns a formatted log of prior user questions and assistant answers. "
        "Use this to understand follow-up questions that reference prior turns."
    )
    args_schema: type[BaseModel] = GetHistoryInput

    def _run(self, session_id: str) -> str:
        """
        Load and format conversation history from DynamoDB.

        Args:
            session_id: UUID of the current chat session.

        Returns:
            Formatted conversation history string, or empty string if no history.
        """
        log.debug("get_conversation_history called", extra={"session_id": session_id})
        manager = get_memory_manager()
        turns = manager.get_history(session_id)

        if not turns:
            return "No prior conversation history for this session."

        formatted = manager.format_as_context(turns)
        log.debug("Loaded conversation context", extra={"session_id": session_id, "turns": len(turns)})
        return formatted


class AddToConversationTool(BaseTool):
    """
    Store a new message turn in the conversation history.

    This tool is called by the final agent in the pipeline (typically the judge
    or the regulatory_analyst, depending on the crew configuration) after
    producing the final answer. It persists both the user question and the
    assistant answer so they are available in future turns.

    Best practice:
        Store both turns in one crew task at the end of the pipeline:
            tool.run(session_id=..., role="user", content=user_question)
            tool.run(session_id=..., role="assistant", content=final_answer,
                     metadata={"query_type": "regulatory_lookup", "confidence": 0.92})

    Why store metadata:
        Metadata makes it possible to filter/analyse conversations later.
        For example: "Show me all sessions where judge_confidence < 0.5" →
        these are cases where the system was uncertain and may need review.
    """

    name: str = "add_to_conversation"
    description: str = (
        "Store a message in the conversation history for the current session. "
        "Call this after generating the final answer to persist the Q&A pair. "
        "Provide role='user' for the question and role='assistant' for the answer."
    )
    args_schema: type[BaseModel] = AddTurnInput

    def _run(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Persist a conversation turn to DynamoDB.

        Args:
            session_id: UUID of the current chat session.
            role:       "user" | "assistant"
            content:    Message text to persist.
            metadata:   Optional extra data (scores, agent name, etc.).

        Returns:
            Confirmation string for the agent to acknowledge.
        """
        log.debug("add_to_conversation called", extra={"session_id": session_id, "role": role})
        manager = get_memory_manager()
        success = manager.add_turn(session_id, role, content, metadata or {})

        if success:
            return f"Successfully stored {role} turn in conversation history (session={session_id[:8]}...)."
        else:
            return "Warning: Could not store turn in conversation history (DynamoDB unavailable)."

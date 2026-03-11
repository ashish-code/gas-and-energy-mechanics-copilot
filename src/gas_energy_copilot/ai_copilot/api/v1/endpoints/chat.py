"""
Module: api/v1/endpoints/chat.py

PURPOSE:
    FastAPI endpoint definitions for the chat interface. Exposes two endpoints:

    POST /v1/chat        — Standard request/response. Returns the full answer
                           after the entire pipeline completes (~10-30 seconds).
    POST /v1/chat/stream — Server-Sent Events (SSE) streaming. Returns agent
                           progress events as they happen, then the final answer.

    Both endpoints support multi-turn conversation via an optional `session_id`
    field. When provided, conversation history is loaded from DynamoDB before
    the crew runs, and the Q&A pair is stored after completion.

ARCHITECTURE POSITION:

    Browser / Streamlit UI
         │  POST /v1/chat  or  POST /v1/chat/stream
         ▼
    chat.py (this file)
         │
         ├─ validate request (empty question check)
         ├─ generate or use provided session_id
         ├─ open Langfuse trace
         ├─ choose crew: FullPipelineCrew (default) or PipelineSafetyRAGCrew
         ├─ crew.kickoff(inputs={question, session_id})
         ├─ parse JudgeVerdict from crew output
         ├─ resolve_final_answer (apply judge verdict)
         ├─ attach TruLens/judge scores to Langfuse trace
         └─ return ChatResponse (or SSE EventSourceResponse for /stream)

ENDPOINT DESIGN CHOICES:

    Standard endpoint (POST /v1/chat):
      - Uses async def with run_in_executor to run the synchronous CrewAI kickoff
        without blocking the event loop. FastAPI's default async handler would block
        all other requests during the LLM calls.
      - Returns a single JSON response with the answer and metadata.

    Streaming endpoint (POST /v1/chat/stream):
      - Uses Server-Sent Events (SSE) via sse-starlette library.
      - WHY SSE over WebSockets: SSE is unidirectional (server→client), HTTP/1.1
        compatible, auto-reconnects, and requires zero client-side protocol upgrade.
        WebSockets are better for bidirectional real-time communication (e.g., chat
        apps where the client also pushes events) — overkill for our use case.
      - CrewAI's `step_callback` fires synchronously on each agent step. We use
        asyncio.Queue as a thread-safe bridge between the sync crew thread and the
        async SSE generator.
      - Events emitted: agent_start, agent_step, agent_finish, final_answer, error.

BACKWARD COMPATIBILITY:
    The original POST /v1/chat endpoint is preserved exactly. The new fields
    (session_id, use_full_pipeline) are optional with backward-compatible defaults.
    Existing clients that don't send session_id continue to work identically.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncGenerator

import structlog.stdlib
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gas_energy_copilot.ai_copilot.core.config import app_config
from gas_energy_copilot.ai_copilot.services.tracing import langfuse_trace, score_trace
from pipeline_safety_rag_crew.crew import (
    FullPipelineCrew,
    JudgeVerdict,
    PipelineSafetyRAGCrew,
    resolve_final_answer,
)

router = APIRouter()
log = structlog.stdlib.get_logger()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """
    Incoming chat request payload.

    Fields:
        question:          The user's question about pipeline safety regulations.
                           Must be non-empty.
        session_id:        UUID identifying the conversation session.
                           If not provided, a new UUID is generated automatically.
                           Clients should persist this value and send it with every
                           subsequent turn to enable multi-turn memory.
        use_full_pipeline: When True (default), uses FullPipelineCrew with router,
                           memory, and judge agents. When False, uses the simpler
                           2-agent PipelineSafetyRAGCrew (faster, no quality gate).
                           Set to False for development/testing.
    """

    question: str = Field(description="The pipeline safety question to answer.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional UUID for multi-turn conversation tracking. "
            "Generated automatically if not provided. "
            "Send the same session_id in subsequent requests to enable memory."
        ),
    )
    use_full_pipeline: bool = Field(
        default=True,
        description=(
            "If True, use the full 4-agent pipeline (router + retrieval + synthesis + judge). "
            "If False, use the simple 2-agent pipeline (retrieval + synthesis only). "
            "Default: True."
        ),
    )


class ChatResponse(BaseModel):
    """
    Chat response payload.

    Fields:
        answer:      The final answer text from the regulatory analyst (or judge's
                     revision if the original answer needed correction).
        session_id:  Echo of the session_id used for this request. Clients should
                     store this and send it with the next request.
        verdict:     Judge's verdict: "approved", "needs_revision", "rejected",
                     or "unknown" (when full pipeline is not used).
        confidence:  Judge's confidence score [0.0, 1.0], or None if not available.
        issues:      List of issues found by the judge (empty if approved).
    """

    answer: str
    session_id: str
    verdict: str = "unknown"
    confidence: float | None = None
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Standard request/response endpoint (backward compatible)
# ---------------------------------------------------------------------------


@router.post(
    "/chat",
    summary="Ask a pipeline safety question",
    description=(
        "Run the multi-agent CrewAI pipeline and return a cited regulatory answer. "
        "Supports multi-turn conversation via session_id. "
        "Use POST /v1/chat/stream for streaming responses."
    ),
    response_model=ChatResponse,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Standard (non-streaming) chat endpoint.

    Runs the full CrewAI pipeline in a thread pool executor to avoid blocking the
    event loop. The call typically takes 10-30 seconds depending on the number of
    agent steps and Bedrock latency.

    Why run_in_executor:
        CrewAI's kickoff() is synchronous (it does not use async/await). Calling
        it directly in an `async def` handler would block the FastAPI event loop,
        preventing other requests from being served while the crew runs.
        asyncio.get_event_loop().run_in_executor(None, func) runs the sync function
        in the default ThreadPoolExecutor, keeping the event loop free.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # Generate a session_id if not provided by the client.
    # Using uuid4 (random UUID) rather than uuid5 (deterministic) because we want
    # each new chat session to be unique even if the first question is the same.
    session_id = request.session_id or str(uuid.uuid4())

    log.info(
        "chat_request",
        question=request.question[:120],
        session_id=session_id[:8],
        full_pipeline=request.use_full_pipeline,
    )

    try:
        with langfuse_trace(
            name="chat",
            session_id=session_id,
            metadata={
                "question": request.question,
                "full_pipeline": request.use_full_pipeline,
            },
        ) as trace:
            # Run the synchronous crew kickoff in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # use default ThreadPoolExecutor
                _run_crew,
                request.question,
                session_id,
                request.use_full_pipeline,
            )

            answer_text, metadata = result

            # Attach quality scores to the Langfuse trace for dashboarding
            if metadata.get("confidence") is not None:
                score_trace(trace, "judge_confidence", metadata["confidence"])

            log.info(
                "chat_response_ok",
                session_id=session_id[:8],
                verdict=metadata.get("verdict"),
                confidence=metadata.get("confidence"),
            )

            return ChatResponse(
                answer=answer_text,
                session_id=session_id,
                verdict=metadata.get("verdict", "unknown"),
                confidence=metadata.get("confidence"),
                issues=metadata.get("issues", []),
            )

    except Exception as exc:
        log.error("chat_error", error=str(exc), session_id=session_id[:8])
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Streaming SSE endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/chat/stream",
    summary="Ask a pipeline safety question (streaming)",
    description=(
        "Server-Sent Events (SSE) endpoint. Returns agent progress events as they happen, "
        "then the final answer. Use this for responsive UIs that show typing indicators. "
        "Each SSE event is a JSON object with 'type' and 'data' fields."
    ),
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    SSE streaming chat endpoint.

    Emits events for each agent step so the client can show a typing indicator
    or incremental progress. Event types:
      - {"type": "agent_start",  "data": "router_agent starting..."}
      - {"type": "agent_step",   "data": "<intermediate thought or tool call>"}
      - {"type": "agent_finish", "data": "synthesis_task complete"}
      - {"type": "final_answer", "data": "<full answer text>", "session_id": "..."}
      - {"type": "error",        "data": "<error message>"}

    IMPLEMENTATION NOTE — asyncio.Queue bridge:
        CrewAI's step_callback fires synchronously in the crew's thread.
        The SSE generator is an async function in the event loop thread.
        We use asyncio.Queue as a thread-safe channel:
          - The crew thread puts events into the queue via queue.put_nowait().
          - The async generator consumes events via await queue.get().
        This pattern bridges the sync/async boundary without polling or sleep().
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    session_id = request.session_id or str(uuid.uuid4())
    log.info("chat_stream_request", question=request.question[:120], session_id=session_id[:8])

    return StreamingResponse(
        _sse_event_generator(request.question, session_id, request.use_full_pipeline),
        media_type="text/event-stream",
        headers={
            # Prevent proxy buffering — critical for SSE to work through nginx/ALB
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx directive
        },
    )


async def _sse_event_generator(
    question: str,
    session_id: str,
    use_full_pipeline: bool,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted events.

    SSE format (RFC 8895):
        data: {"type": "agent_step", "data": "..."}\n\n

    The double newline (\n\n) signals the end of one event to the SSE client.
    The client (browser EventSource or curl -N) parses each event as a separate message.
    """
    import json

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    sentinel = object()  # unique sentinel to signal "crew done"

    def step_callback(step_output: Any) -> None:
        """
        Callback called by CrewAI on each agent step.
        Runs in the crew's thread — must be thread-safe (queue.put_nowait is).
        """
        queue.put_nowait({
            "type": "agent_step",
            "data": str(step_output)[:500],  # truncate very long step outputs
        })

    def run_crew_with_callback():
        """Runs the crew synchronously in a thread, feeding events into the queue."""
        try:
            if use_full_pipeline:
                crew_instance = FullPipelineCrew(session_id=session_id)
                c = crew_instance.crew()
                c.step_callback = step_callback
                result = c.kickoff(inputs={"question": question, "session_id": session_id})
            else:
                c = PipelineSafetyRAGCrew().crew()
                c.step_callback = step_callback
                result = c.kickoff(inputs={"question": question})

            # Parse judge verdict if available
            verdict = None
            if use_full_pipeline and result.json_dict:
                try:
                    verdict = JudgeVerdict.model_validate(result.json_dict)
                except Exception:
                    pass

            answer_text, metadata = resolve_final_answer(result.raw, verdict)
            queue.put_nowait({
                "type": "final_answer",
                "data": answer_text,
                "session_id": session_id,
                "verdict": metadata.get("verdict", "unknown"),
                "confidence": metadata.get("confidence"),
            })
        except Exception as e:
            queue.put_nowait({"type": "error", "data": str(e)})
        finally:
            queue.put_nowait(sentinel)  # signal completion to the async generator

    # Emit a start event immediately so the client knows the pipeline is running
    yield f"data: {json.dumps({'type': 'agent_start', 'data': 'Pipeline starting...'})}\n\n"

    # Start the crew in the thread pool (does not block the event loop)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_crew_with_callback)

    # Consume events from the queue and yield them as SSE
    while True:
        event = await queue.get()
        if event is sentinel:
            break
        yield f"data: {json.dumps(event)}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Internal helper: run crew synchronously (called via run_in_executor)
# ---------------------------------------------------------------------------


def _run_crew(
    question: str,
    session_id: str,
    use_full_pipeline: bool,
) -> tuple[str, dict[str, Any]]:
    """
    Run the CrewAI crew synchronously and return (answer_text, metadata).

    This function runs in a thread pool (via run_in_executor) so it can use
    blocking I/O without affecting the event loop.

    Args:
        question:         The user's question.
        session_id:       UUID for conversation memory.
        use_full_pipeline: Whether to use FullPipelineCrew (4 agents) or
                          PipelineSafetyRAGCrew (2 agents).

    Returns:
        Tuple of (answer_text, metadata_dict) where metadata contains
        verdict, confidence, issues, etc. from the JudgeVerdict.
    """
    if use_full_pipeline:
        crew_instance = FullPipelineCrew(session_id=session_id)
        result = crew_instance.crew().kickoff(
            inputs={"question": question, "session_id": session_id}
        )

        # Parse the JudgeVerdict from the final task's JSON output
        verdict: JudgeVerdict | None = None
        if result.json_dict:
            try:
                verdict = JudgeVerdict.model_validate(result.json_dict)
            except Exception as e:
                log.warning("Failed to parse JudgeVerdict", error=str(e))

        # synthesis_task is the third task (index 2); its raw output is the answer
        # before judge revision. We pass it to resolve_final_answer for gating.
        # If tasks_output is unavailable, fall back to result.raw.
        synthesis_raw = (
            result.tasks_output[-2].raw
            if result.tasks_output and len(result.tasks_output) >= 2
            else result.raw
        )
        return resolve_final_answer(synthesis_raw, verdict)

    else:
        # Simple 2-agent crew — no routing, no judging, faster
        result = PipelineSafetyRAGCrew().crew().kickoff(inputs={"question": question})
        return result.raw, {"verdict": "unknown", "confidence": None, "issues": []}

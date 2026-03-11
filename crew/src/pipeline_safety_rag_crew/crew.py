"""
Module: crew.py

PURPOSE:
    Defines the CrewAI crew classes that orchestrate multi-agent pipelines for
    answering pipeline safety regulatory questions. Two crew classes are provided:

    1. PipelineSafetyRAGCrew (original, 2-agent, backward compatible):
         retrieval_specialist → regulatory_analyst
       Fast, simple, no quality gating. Use for development and regression testing.

    2. FullPipelineCrew (new, 4-agent, production pipeline):
         router_agent → retrieval_specialist → regulatory_analyst → judge_agent
       Full pipeline with query classification, memory, and answer quality gating.

ARCHITECTURE POSITION:

    FastAPI chat endpoint
         │
         ├─ PipelineSafetyRAGCrew().crew().kickoff(inputs={"question": q})
         │       Simple 2-step pipeline for dev/backward compat
         │
         └─ FullPipelineCrew(session_id=sid).crew().kickoff(inputs={...})
                 Full 4-step pipeline with router, memory, and judge

CREWAI CONCEPTS (for readers new to CrewAI):
    @CrewBase:  Class decorator that auto-loads agents.yaml and tasks.yaml from
                the config/ directory adjacent to this file. Also registers all
                @agent and @task decorated methods.
    @agent:     Marks a method that returns an Agent() instance. The method name
                must match a top-level key in agents.yaml.
    @task:      Marks a method that returns a Task() instance. The method name
                must match a top-level key in tasks.yaml.
    @crew:      Marks a method that assembles and returns the Crew() object.
    Process.sequential: Tasks run in the order they are listed in the @crew
                method's tasks= argument. Each task's output is available to
                subsequent tasks via the `context` field in tasks.yaml.

WHY SEQUENTIAL OVER HIERARCHICAL:
    CrewAI supports two process modes:
      - Process.sequential: Tasks run in a fixed, deterministic order. Each task
        can access prior task outputs via context. Predictable, auditable, fast.
      - Process.hierarchical: A "manager" LLM dynamically decides which agent
        to call and in what order. More flexible but non-deterministic, harder
        to debug, and uses more tokens (manager needs full context).

    For regulatory guidance, determinism is critical. The four-stage pipeline
    (route → retrieve → synthesise → judge) has a natural fixed ordering, and
    sequential process maps directly to it without the overhead of a manager LLM.

OUTPUT SCHEMAS:
    The router and judge agents output structured JSON. We use Pydantic models
    as the output_json= parameter of their tasks, which:
      1. Instructs the LLM to produce JSON matching the schema.
      2. Provides CrewAI's task output parser with a typed result object.
      3. Enables downstream tasks to access structured fields (e.g., recommended_search_terms).

    Why Pydantic over plain dicts:
      - Validation at parse time (CrewAI raises if the LLM output is malformed).
      - Type-annotated fields are self-documenting for anyone reading the code.
      - IDE autocomplete when accessing verdict.confidence, decision.query_type, etc.
"""

from __future__ import annotations

import logging
from typing import Literal

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field

from pipeline_safety_rag_crew.tools.memory_tool import (
    AddToConversationTool,
    GetConversationHistoryTool,
)
from pipeline_safety_rag_crew.tools.rag_tool import RAGSearchTool
from pipeline_safety_rag_crew.tools.web_search_tool import RegulatoryWebSearchTool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schemas — used as output_json= for router and judge tasks
# ---------------------------------------------------------------------------


class RouterDecision(BaseModel):
    """
    Structured output from the router_agent task.

    This schema is used in two ways:
      1. As output_json= in the routing_task — CrewAI instructs the LLM to
         produce JSON matching this schema and validates the response.
      2. As a typed data structure accessed by subsequent tasks and the endpoint.

    Fields:
        query_type:              One of four categories that determine retrieval strategy.
        confidence:              Router's confidence in its classification [0.0, 1.0].
                                 Low confidence (< 0.6) may indicate an ambiguous query.
        reasoning:               1–2 sentence explanation of why this category was chosen.
                                 Useful for debugging misclassifications.
        recommended_search_terms: Keywords/phrases the retrieval specialist should search.
                                 Based on the query type: section numbers for regulatory_lookup,
                                 operator class terms for compliance_check, etc.
        requires_web_search:     True if the query likely involves recent events or
                                 enforcement actions not in the FAISS index snapshot.
    """

    query_type: Literal[
        "regulatory_lookup",
        "compliance_check",
        "incident_analysis",
        "general_engineering",
    ] = Field(description="The category of the user query.")

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Router's confidence in the classification.",
    )

    reasoning: str = Field(
        description="Brief explanation of why this category was chosen.",
    )

    recommended_search_terms: list[str] = Field(
        description="3–5 targeted search terms for the retrieval specialist.",
    )

    requires_web_search: bool = Field(
        default=False,
        description="True if real-time web search is needed (recent events/enforcement).",
    )


class JudgeVerdict(BaseModel):
    """
    Structured output from the judge_agent task.

    This schema is the quality gate for every answer produced by the pipeline.
    The chat endpoint reads `verdict` to decide what to return to the user:
      - "approved"       → return synthesised (or revised) answer as-is
      - "needs_revision" → return revised_answer (judge's corrected version)
      - "rejected"       → return a canned disclaimer ("cannot answer confidently")

    Fields:
        verdict:               The overall quality verdict.
        confidence:            Judge's confidence that the answer is accurate [0.0, 1.0].
                               Score attached to Langfuse trace for monitoring.
        issues:                List of specific problems found (empty if approved).
        revised_answer:        Corrected answer text if verdict == "needs_revision".
                               None if verdict is "approved" or "rejected".
        citations_verified:    CFR sections that were found in the retrieved context.
        hallucinations_detected: Claims in the answer with no grounding in context.
    """

    verdict: Literal["approved", "needs_revision", "rejected"] = Field(
        description="Overall quality verdict for the synthesised answer.",
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Judge's confidence in the answer accuracy.",
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found. Empty list if verdict is 'approved'.",
    )

    revised_answer: str | None = Field(
        default=None,
        description="Corrected answer text when verdict is 'needs_revision'.",
    )

    citations_verified: list[str] = Field(
        default_factory=list,
        description="CFR sections confirmed present in the retrieved context.",
    )

    hallucinations_detected: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by retrieved context.",
    )


# ---------------------------------------------------------------------------
# Original 2-agent crew (backward compatible)
# ---------------------------------------------------------------------------


@CrewBase
class PipelineSafetyRAGCrew:
    """
    Original two-agent crew: retrieve regulatory passages → synthesise answer.

    This crew is kept for backward compatibility and development use. It is simpler,
    faster (no routing or judging overhead), and uses fewer LLM tokens per request.

    Use this crew when:
      - Running quick tests or debugging retrieval quality.
      - Benchmarking retrieval speed without evaluation overhead.
      - In CI pipelines where response speed matters more than quality gating.

    For production, use FullPipelineCrew which adds routing and quality gating.

    Pipeline:
        retrieval_task ──► synthesis_task
        (retrieval_specialist)   (regulatory_analyst)
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def retrieval_specialist(self) -> Agent:
        """
        PHMSA Regulatory Retrieval Specialist.

        Searches the FAISS index via RAGSearchTool. Can issue multiple searches
        with different query reformulations to improve coverage.

        Tools: RAGSearchTool (FAISS semantic search over 49 CFR regulations)
        """
        return Agent(
            config=self.agents_config["retrieval_specialist"],  # type: ignore[index]
            tools=[RAGSearchTool()],
            verbose=True,
        )

    @agent
    def regulatory_analyst(self) -> Agent:
        """
        Pipeline Safety Regulatory Analyst.

        Reads the retrieved passages from retrieval_task context and synthesises
        a cited, plain-language answer. Has no tools — pure synthesis.
        """
        return Agent(
            config=self.agents_config["regulatory_analyst"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def retrieval_task(self) -> Task:
        """Search the FAISS index for relevant regulatory passages."""
        return Task(config=self.tasks_config["retrieval_task"])  # type: ignore[index]

    @task
    def synthesis_task(self) -> Task:
        """Synthesise retrieved passages into a cited answer."""
        return Task(
            config=self.tasks_config["synthesis_task"],  # type: ignore[index]
            output_file="output/answer.md",
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the 2-agent sequential crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


# ---------------------------------------------------------------------------
# Full 4-agent production pipeline
# ---------------------------------------------------------------------------


@CrewBase
class FullPipelineCrew:
    """
    Production-grade 4-agent sequential pipeline with routing, memory, and quality gating.

    This crew extends the original 2-agent pipeline with:
      1. Router agent — classifies the query and recommends targeted search terms.
      2. Memory tools — loads prior conversation turns so the analyst can handle
         follow-up questions without the user repeating themselves.
      3. Judge agent — verifies the synthesised answer for hallucinations and
         citation accuracy before it reaches the user.
      4. Web search tool — used by retrieval_specialist when the router flags
         requires_web_search=True (e.g., recent PHMSA enforcement actions).

    PIPELINE FLOW:
        routing_task          → classify query, generate search terms
             │
             ▼
        retrieval_task        → search FAISS (+ web if router says so)
        (uses router output)       read conversation history
             │
             ▼
        synthesis_task        → write cited answer
        (uses retrieved passages)  acknowledge prior conversation context
             │
             ▼
        judging_task          → verify claims against retrieved context
        (uses answer + context)    output RouterDecision or JudgeVerdict JSON

    USAGE:
        crew = FullPipelineCrew(session_id="uuid-xxx")
        result = crew.crew().kickoff(inputs={
            "question": "What are the MAOP requirements under §192.619?",
            "session_id": "uuid-xxx",
        })
        # result.json_dict contains the JudgeVerdict
        # result.raw contains the final answer text

    session_id:
        The UUID for the current chat session. Used by memory tools to load/store
        conversation history. If None, the crew runs statelessly (no memory).
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(self, session_id: str | None = None) -> None:
        """
        Initialise the full pipeline crew.

        Args:
            session_id: UUID of the current chat session for memory tool access.
                        If None, memory tools are still present but will receive
                        an empty/null session_id and will return no history.
        """
        super().__init__()
        self.session_id = session_id or ""

    # ── Agents ────────────────────────────────────────────────────────────────

    @agent
    def router_agent(self) -> Agent:
        """
        Pipeline Safety Query Classification Specialist.

        Fast, lightweight agent — no tools needed (classification is pure LLM reasoning).
        verbose=False keeps the logs clean since routing output is always structured JSON.

        Output: RouterDecision JSON (validated by Task's output_json=RouterDecision)
        """
        return Agent(
            config=self.agents_config["router_agent"],  # type: ignore[index]
            verbose=False,  # suppress step-by-step logs for classification
        )

    @agent
    def retrieval_specialist(self) -> Agent:
        """
        PHMSA Regulatory Retrieval Specialist — extended with memory and web search.

        Tools available:
          - RAGSearchTool:             Semantic search over 49 CFR FAISS index.
          - GetConversationHistoryTool: Read prior turns from DynamoDB.
          - RegulatoryWebSearchTool:   Web search (Tavily) for real-time regulatory data.

        The agent chooses which tools to use based on the task description and
        the routing decision from routing_task (available in context).
        """
        return Agent(
            config=self.agents_config["retrieval_specialist"],  # type: ignore[index]
            tools=[
                RAGSearchTool(),
                GetConversationHistoryTool(),
                RegulatoryWebSearchTool(),
            ],
            verbose=True,
        )

    @agent
    def regulatory_analyst(self) -> Agent:
        """
        Pipeline Safety Regulatory Analyst — extended with memory write access.

        The analyst reads prior conversation context (loaded by retrieval_specialist)
        and after producing the answer, stores both the question and answer in memory
        via AddToConversationTool so future turns have continuity.
        """
        return Agent(
            config=self.agents_config["regulatory_analyst"],  # type: ignore[index]
            tools=[AddToConversationTool()],
            verbose=True,
        )

    @agent
    def judge_agent(self) -> Agent:
        """
        Pipeline Safety Regulatory Accuracy Judge.

        No tools — the judge only reads the retrieved passages and synthesised answer
        from task context. It reasons over text entirely within its context window.

        Output: JudgeVerdict JSON (validated by Task's output_json=JudgeVerdict)
        """
        return Agent(
            config=self.agents_config["judge_agent"],  # type: ignore[index]
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task
    def routing_task(self) -> Task:
        """
        Query classification task.

        Output JSON is validated against RouterDecision schema.
        The output is automatically injected into the context of retrieval_task.
        """
        return Task(
            config=self.tasks_config["routing_task"],  # type: ignore[index]
            output_json=RouterDecision,
        )

    @task
    def retrieval_task(self) -> Task:
        """
        FAISS retrieval task — enhanced with routing context.

        Receives the RouterDecision from routing_task via context= in tasks.yaml.
        The retrieval_specialist reads recommended_search_terms and issues targeted
        follow-up searches in addition to the primary query.
        """
        return Task(config=self.tasks_config["retrieval_task"])  # type: ignore[index]

    @task
    def synthesis_task(self) -> Task:
        """
        Answer synthesis task — receives retrieval results via context.

        The regulatory_analyst synthesises retrieved passages into a cited answer
        and optionally stores the Q&A pair to memory via AddToConversationTool.
        """
        return Task(
            config=self.tasks_config["synthesis_task"],  # type: ignore[index]
            output_file="output/answer.md",
        )

    @task
    def judging_task(self) -> Task:
        """
        Quality gate task — receives both retrieval results and synthesised answer.

        Output JSON is validated against JudgeVerdict schema.
        The chat endpoint reads verdict to decide what to return to the user.
        """
        return Task(
            config=self.tasks_config["judging_task"],  # type: ignore[index]
            output_json=JudgeVerdict,
        )

    # ── Crew ──────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """
        Assemble the full 4-agent sequential crew.

        Task order matters — CrewAI runs them in the order listed here:
          routing_task → retrieval_task → synthesis_task → judging_task

        The `tasks` list is populated by the @CrewBase decorator from the @task
        decorated methods in definition order (Python 3.7+ guaranteed dict ordering).
        We pass them explicitly here for clarity rather than relying on implicit ordering.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


# ---------------------------------------------------------------------------
# Helper: interpret the judge verdict and return the best answer text
# ---------------------------------------------------------------------------


def resolve_final_answer(
    synthesis_raw: str,
    verdict: JudgeVerdict | None,
) -> tuple[str, dict]:
    """
    Apply the judge's verdict to determine what answer text to return to the user.

    This function is called by the chat endpoint after crew kickoff to map the
    structured JudgeVerdict into a user-facing answer and metadata.

    Verdict logic:
        "approved"       → return the synthesis answer as-is
        "needs_revision" → return the judge's revised_answer (cleaner version)
        "rejected"       → return a standard disclaimer (do not show hallucinated text)
        None             → verdict unavailable (simple crew or parsing failed); return synthesis

    Args:
        synthesis_raw: The raw answer text from regulatory_analyst (synthesis_task.raw).
        verdict:       The JudgeVerdict parsed from judging_task.json_dict, or None.

    Returns:
        Tuple of:
          - answer_text: The final answer string to show the user.
          - metadata: Dict with verdict, confidence, issues for the API response.

    Example:
        result = crew.kickoff(inputs={"question": q, "session_id": sid})
        verdict = JudgeVerdict.model_validate(result.json_dict) if result.json_dict else None
        answer, meta = resolve_final_answer(result.tasks_output[-2].raw, verdict)
    """
    if verdict is None:
        # Simple crew or failed verdict parse — return synthesis answer without gating
        return synthesis_raw, {"verdict": "unknown", "confidence": None, "issues": []}

    metadata = {
        "verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "issues": verdict.issues,
        "citations_verified": verdict.citations_verified,
        "hallucinations_detected": verdict.hallucinations_detected,
    }

    if verdict.verdict == "approved":
        log.info("Judge approved answer", extra={"confidence": verdict.confidence})
        return synthesis_raw, metadata

    elif verdict.verdict == "needs_revision":
        log.info(
            "Judge revised answer",
            extra={"confidence": verdict.confidence, "issues": verdict.issues},
        )
        revised = verdict.revised_answer or synthesis_raw
        return revised, metadata

    else:  # "rejected"
        log.warning(
            "Judge rejected answer — returning disclaimer",
            extra={
                "confidence": verdict.confidence,
                "hallucinations": verdict.hallucinations_detected,
            },
        )
        disclaimer = (
            "I was unable to produce a confident answer for this question based on the "
            "available regulatory documents. The retrieved passages may be insufficient "
            "or the question may be outside the scope of the indexed regulations (49 CFR "
            "Parts 192, 193, 195). Please consult the eCFR directly at ecfr.gov or "
            "contact a qualified PHMSA compliance engineer."
        )
        return disclaimer, metadata

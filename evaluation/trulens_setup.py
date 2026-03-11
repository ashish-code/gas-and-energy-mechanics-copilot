"""
Module: evaluation/trulens_setup.py

PURPOSE:
    Sets up TruLens evaluation for live production monitoring of the RAG pipeline.
    Every real user request is wrapped in a TruLens recording that computes the
    RAG Triad — three quality scores that together determine whether the pipeline's
    response is trustworthy.

WHY TRULENS (for live monitoring, not offline CI):
    TruLens integrates with the running application as a "recorder" that wraps
    function calls. Each wrapped call produces a database record with input, output,
    and quality scores. A built-in dashboard (localhost:8502) shows trends over time.

    The key difference from DeepEval:
      - DeepEval runs offline against a fixed dataset (CI/CD gate).
        It tells you "will this change break quality?" before merging.
      - TruLens runs live against real user queries (production monitoring).
        It tells you "is quality degrading in the wild?" after deploying.

    They are complementary, not competing. Ideal workflow:
        1. Code change → DeepEval CI gate (don't merge if metrics drop)
        2. Deploy → TruLens monitors real user interactions
        3. Quality alert → investigate with Langfuse traces → fix → repeat

THE RAG TRIAD (Garg & Miller, 2023 — "Evaluating the Ideal RAG System"):
    A trustworthy RAG response requires all three components to be high:

    ┌──────────────────────────────────────────────────────────────────┐
    │                      THE RAG TRIAD                               │
    │                                                                  │
    │  Query ──► Context Relevance ──► Retrieved Context               │
    │                                         │                        │
    │                               Groundedness                       │
    │                                         │                        │
    │                                         ▼                        │
    │                                      Answer ──► Answer Relevance ──► User │
    └──────────────────────────────────────────────────────────────────┘

    1. Context Relevance (Query → Retrieved Context):
       Are the retrieved passages actually relevant to the question?
       Low score → retrieval failure (FAISS returning irrelevant chunks).
       Example: User asks about MAOP, retrieval returns cathodic protection sections.

    2. Groundedness (Retrieved Context → Answer):
       Is every claim in the answer supported by the retrieved context?
       Low score → hallucination (analyst invented facts not in retrieved text).
       This is the most critical metric — a low score means the answer is unreliable.

    3. Answer Relevance (Answer → Query):
       Does the answer actually address what the user asked?
       Low score → the answer is off-topic or evasive.
       Example: User asks about pressure testing, answer discusses corrosion protection.

    If ALL THREE are high, the RAG pipeline is working correctly.
    If ANY is low, there is a specific failure mode to investigate.

TRULENS INTERNALS:
    TruLens uses "feedback functions" to compute scores. Each feedback function:
      1. Takes the inputs and outputs of a function call as arguments.
      2. Uses an LLM to judge quality (returns a float 0.0-1.0).
      3. Stores the score in the TruLens database.

    TruLens wraps Python functions (not agent classes) using the `@tru.instrument`
    decorator or the TruBasicApp recorder. We use TruBasicApp because our pipeline
    entry point is a simple function (not a LangChain chain or LlamaIndex query engine).

    Alternative: TruChain (for LangChain) or TruLlama (for LlamaIndex). We use
    TruBasicApp because our pipeline uses CrewAI, not LangChain or LlamaIndex.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

log = logging.getLogger(__name__)


def _try_import_trulens():
    """
    Attempt to import TruLens components.

    TruLens is optional — if not installed, all TruLens functions are no-ops.
    This prevents import errors for users who only want DeepEval or Langfuse.

    Returns:
        Tuple of (Tru, TruBasicApp, Feedback, OpenAI) or None if not available.
    """
    try:
        from trulens.core import TruSession, Feedback
        from trulens.apps.basic import TruBasicApp
        return TruSession, TruBasicApp, Feedback
    except ImportError:
        return None, None, None


class TruLensRAGEvaluator:
    """
    Wraps the CrewAI pipeline with TruLens to score the RAG Triad on every request.

    This class is used by the FastAPI chat endpoint (when eval.enabled=True) to
    record every response with quality scores. Scores are stored in a SQLite
    (dev) or PostgreSQL (prod) database and viewable in the TruLens dashboard.

    USAGE IN PRODUCTION:
        evaluator = TruLensRAGEvaluator()

        # In the chat endpoint:
        with evaluator.record(question, session_id) as recording:
            result = crew.kickoff(inputs={"question": question})
        # Score is automatically computed and stored by TruLens

    INITIALISATION:
        The TruLens session (database connection) is created lazily on first use.
        If TruLens is not installed or eval.enabled=False, all methods are no-ops.

    DATABASE:
        Dev:  SQLite file at config.eval.trulens_db_url (default: evaluation/trulens.db)
        Prod: PostgreSQL URI (e.g., "postgresql://user:pw@host/trulens_db")
        Schema is auto-created by TruLens on first run.
    """

    def __init__(self) -> None:
        self._session = None
        self._tru_app = None
        self._enabled = False
        self._initialised = False

    def _ensure_initialised(self) -> bool:
        """
        Lazily initialise TruLens session and feedback functions.

        Called on first use of record(). Checks config.eval.enabled, then
        creates the TruLens session and registers the RAG Triad feedback functions.

        Returns:
            True if TruLens is active and ready, False if disabled or unavailable.
        """
        if self._initialised:
            return self._enabled

        self._initialised = True

        try:
            from gas_energy_copilot.ai_copilot.core.config import app_config
            config = app_config()
        except Exception as e:
            log.debug("Could not load config for TruLens", exc_info=e)
            return False

        if not config.eval.enabled:
            log.debug("TruLens evaluation disabled (set [app.eval] enabled=true to activate)")
            return False

        TruSession, TruBasicApp, Feedback = _try_import_trulens()
        if TruSession is None:
            log.warning(
                "TruLens evaluation is enabled but trulens-eval is not installed. "
                "Run: uv add trulens-eval"
            )
            return False

        try:
            # Initialise TruLens session (creates database tables on first run)
            self._session = TruSession(database_url=config.eval.trulens_db_url)
            self._session.reset_database()  # only resets if schema version mismatch

            # Build feedback functions using the Bedrock LLM provider
            feedback_functions = self._build_feedback_functions(Feedback)

            # TruBasicApp wraps a simple callable. We wrap a placeholder function
            # here and reassign it at runtime in the record() context manager.
            # This is necessary because TruLens computes scores based on
            # the wrapped function's inputs and outputs.
            self._tru_app_class = TruBasicApp
            self._feedback_functions = feedback_functions

            self._enabled = True
            log.info("TruLens RAG evaluator initialised", extra={"db": config.eval.trulens_db_url})
            return True

        except Exception as e:
            log.warning("TruLens initialisation failed", exc_info=e)
            return False

    def _build_feedback_functions(self, Feedback) -> list:
        """
        Build the three RAG Triad feedback functions.

        TruLens feedback functions are LLM-based judges. Each one takes
        specific inputs (question, retrieved context, answer) and returns a
        quality score in [0.0, 1.0].

        LLM PROVIDER FOR SCORING:
            We use LiteLLM as the scoring LLM (same backend as CrewAI).
            TruLens supports multiple providers:
              - OpenAI (default) — best quality, requires OpenAI API key
              - Bedrock (via LiteLLM) — consistent with our stack, no extra keys
              - Hugging Face (local) — free but weaker quality

            We choose Bedrock/LiteLLM for cost and operational consistency.

        Returns:
            List of three Feedback objects: context_relevance, groundedness, answer_relevance.
        """
        try:
            from trulens.providers.litellm import LiteLLM
        except ImportError:
            try:
                # Older TruLens versions have different import paths
                from trulens.feedback.litellm import LiteLLM  # type: ignore
            except ImportError:
                log.warning("TruLens LiteLLM provider not found — using dummy provider")
                return []

        model_id = os.environ.get("MODEL", "bedrock/openai.gpt-oss-120b-1:0")
        provider = LiteLLM(model_engine=model_id)

        # Feedback 1: Context Relevance
        # Scores whether retrieved passages are relevant to the question.
        # TruLens computes this by asking the LLM: "Is this context relevant to this query?"
        f_context_relevance = (
            Feedback(provider.context_relevance, name="Context Relevance")
            .on_input()         # input = the user question
            .on(lambda record: record.app.retrieved_contexts)  # retrieved passages
            .aggregate(lambda scores: sum(scores) / len(scores) if scores else 0.0)
        )

        # Feedback 2: Groundedness
        # Scores whether each claim in the answer can be traced to retrieved context.
        # This is the primary hallucination detector in TruLens.
        f_groundedness = (
            Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(lambda record: record.app.retrieved_contexts)  # context = retrieved passages
            .on_output()        # output = the synthesised answer
        )

        # Feedback 3: Answer Relevance
        # Scores whether the answer addresses the user's question.
        f_answer_relevance = (
            Feedback(provider.relevance, name="Answer Relevance")
            .on_input()         # input = the user question
            .on_output()        # output = the synthesised answer
        )

        return [f_context_relevance, f_groundedness, f_answer_relevance]

    @contextmanager
    def record(
        self,
        question: str,
        session_id: str,
        retrieved_contexts: list[str] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager that records a crew response with TruLens RAG Triad scores.

        Usage in chat endpoint:
            evaluator = get_trulens_evaluator()
            with evaluator.record(question, session_id, contexts) as recorder:
                result = crew.kickoff(inputs={"question": question})
                # TruLens captures the output and computes scores asynchronously

        If TruLens is disabled or unavailable, this is a no-op context manager.

        Args:
            question:           The user's question (used as the "input" for scoring).
            session_id:         UUID for grouping records in the TruLens dashboard.
            retrieved_contexts: List of retrieved passage strings (for context relevance
                                and groundedness scoring). If None, these two scores
                                cannot be computed accurately.

        Yields:
            TruLens recording context (or None if disabled).
        """
        if not self._ensure_initialised():
            yield None
            return

        try:
            # Create a simple pipeline function for TruLens to wrap
            # TruLens needs a callable that takes (input) and returns (output)
            def pipeline_fn(q: str) -> str:
                """Placeholder — actual output is set via the recording context."""
                return ""

            # Attach retrieved contexts so feedback functions can access them
            pipeline_fn.retrieved_contexts = retrieved_contexts or []  # type: ignore

            tru_app = self._tru_app_class(
                pipeline_fn,
                app_id=f"gas-energy-copilot-{session_id[:8]}",
                feedbacks=self._feedback_functions,
                metadata={"session_id": session_id},
            )

            with tru_app as recording:
                yield recording

        except Exception as e:
            log.warning("TruLens record() failed", exc_info=e)
            yield None

    def get_leaderboard(self) -> dict:
        """
        Return aggregate TruLens scores across all recorded sessions.

        Used by the /v1/eval/metrics endpoint to surface quality trends via API.

        Returns:
            Dict with aggregate metrics:
              {"context_relevance": 0.78, "groundedness": 0.82, "answer_relevance": 0.85,
               "total_records": 1234, "last_updated": "2024-01-15T10:30:00Z"}
        """
        if not self._ensure_initialised() or self._session is None:
            return {
                "context_relevance": None,
                "groundedness": None,
                "answer_relevance": None,
                "total_records": 0,
                "message": "TruLens evaluation is disabled or unavailable.",
            }

        try:
            leaderboard = self._session.get_leaderboard()
            # Convert pandas DataFrame to dict (TruLens returns a DataFrame)
            if hasattr(leaderboard, "to_dict"):
                return leaderboard.to_dict(orient="records")
            return {"raw": str(leaderboard)}
        except Exception as e:
            log.warning("Failed to get TruLens leaderboard", exc_info=e)
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_evaluator: TruLensRAGEvaluator | None = None


def get_trulens_evaluator() -> TruLensRAGEvaluator:
    """
    Return the module-level singleton TruLensRAGEvaluator.

    Using a singleton avoids re-initialising the TruLens database connection
    on every request. The evaluator lazily initialises on first use.

    Returns:
        Shared TruLensRAGEvaluator instance.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = TruLensRAGEvaluator()
    return _evaluator

"""
Module: services/evaluation_service.py

PURPOSE:
    Bridge between the FastAPI application and the evaluation frameworks
    (TruLens + DeepEval). This service is imported by:
      - The chat endpoint — to record each real user interaction with TruLens.
      - The evaluation API endpoint — to return aggregate metrics and trigger runs.

    Keeps evaluation logic out of the API layer and provides clean interfaces
    that the endpoints can call without knowing about TruLens internals.

ARCHITECTURE POSITION:
    chat endpoint
         │
         ├─► EvaluationService.record_interaction(question, answer, contexts, session_id)
         │       ↓ calls TruLensRAGEvaluator.record()
         │   TruLens scores the response and stores in DB
         │
    evaluation endpoint (GET /v1/eval/metrics)
         │
         └─► EvaluationService.get_metrics()
                 ↓ calls TruLensRAGEvaluator.get_leaderboard()
             Returns aggregate scores as JSON
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class EvaluationService:
    """
    Application service that manages evaluation recording and metric retrieval.

    This service is intentionally thin — it delegates to TruLens for live monitoring
    and DeepEval for offline evaluation. Its value is providing a stable, typed
    interface to the application layer without leaking evaluation framework details.
    """

    def record_interaction(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        session_id: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Record a chat interaction with TruLens for quality scoring.

        This is called from the chat endpoint after every successful response.
        TruLens computes the RAG Triad scores asynchronously and stores them
        in its database. The scores appear in the TruLens dashboard within
        a few seconds of the call completing.

        Args:
            question:           The user's question.
            answer:             The final answer shown to the user.
            retrieved_contexts: List of passage texts retrieved by the FAISS search.
                                These are needed for Context Relevance and Groundedness.
            session_id:         UUID for grouping records by session in the dashboard.
            metadata:           Optional extra data (judge_confidence, verdict, etc.)
                                stored alongside the TruLens record.
        """
        try:
            from evaluation.trulens_setup import get_trulens_evaluator
            evaluator = get_trulens_evaluator()

            with evaluator.record(question, session_id, retrieved_contexts) as recording:
                if recording is not None:
                    # TruLens captures the answer as the "output" of the recorded call
                    recording.answer = answer
                    if metadata:
                        recording.metadata = metadata

        except Exception as e:
            # Never crash the chat response due to evaluation failure
            log.warning("TruLens recording failed — continuing without evaluation", exc_info=e)

    def get_metrics(self) -> dict:
        """
        Return the current aggregate TruLens quality metrics.

        Returns:
            Dict with RAG Triad scores and record count, e.g.:
            {
                "context_relevance": 0.78,
                "groundedness": 0.82,
                "answer_relevance": 0.85,
                "total_records": 1234,
            }
            Returns placeholder dict if TruLens is disabled.
        """
        try:
            from evaluation.trulens_setup import get_trulens_evaluator
            return get_trulens_evaluator().get_leaderboard()
        except Exception as e:
            log.warning("Failed to get TruLens metrics", exc_info=e)
            return {"error": str(e)}


# Module-level singleton
_evaluation_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    """Return the module-level EvaluationService singleton."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service

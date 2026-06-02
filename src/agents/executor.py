"""EXECUTE node — run the full retrieval pipeline for each sub-question.

Haiku-tier orchestration: for every sub-question we call the M3 `Retriever` (multi-query +
HyDE + dense + sparse -> RRF -> MMR -> rerank -> parent expansion), then merge and dedupe the
parent sections across sub-questions so the synthesizer sees each unique section once.
"""

from __future__ import annotations

from src.agents.schemas import Plan
from src.logging_setup import get_logger
from src.retrieval.parent_doc import Evidence
from src.retrieval.pipeline import Retriever

log = get_logger(__name__)


def execute(plan: Plan, retriever: Retriever) -> tuple[dict[str, list[Evidence]], list[Evidence]]:
    """Retrieve evidence per sub-question; return (per_sub_question, deduped_flat)."""
    per_sq: dict[str, list[Evidence]] = {}
    merged: dict[str, Evidence] = {}  # parent_id -> best Evidence (highest score)

    for sq in plan.sub_questions:
        evidence = retriever.retrieve(sq.question, source=sq.source_hint)
        per_sq[sq.id] = evidence
        for ev in evidence:
            cur = merged.get(ev.parent_id)
            if cur is None or ev.score > cur.score:
                merged[ev.parent_id] = ev

    flat = sorted(merged.values(), key=lambda e: e.score, reverse=True)
    log.info("executor.done", sub_questions=len(per_sq), unique_evidence=len(flat))
    return per_sq, flat

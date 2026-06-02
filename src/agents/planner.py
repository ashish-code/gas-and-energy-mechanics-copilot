"""PLAN node — Sonnet decomposes the query into 1-5 sub-questions, or refuses.

Refusal is a first-class outcome: if nothing in the query is answerable from the corpus
(49 CFR Parts 192/193/195, PHMSA enforcement actions, NTSB pipeline accident reports), the
planner returns `in_scope=False` with a one-sentence reason. A visible refusal is the most
important trust signal in the demo.
"""

from __future__ import annotations

from src.agents.schemas import Plan
from src.config import settings
from src.llm import chat_model
from src.logging_setup import get_logger

log = get_logger(__name__)

_SYSTEM = """You are the planner for a retrieval system over a fixed corpus of US federal \
pipeline-safety material:
  - 49 CFR Parts 192 (gas pipelines), 193 (LNG facilities), 195 (hazardous-liquid pipelines)
  - PHMSA enforcement actions (NOPV/CAO: operators, violations, cited regs, civil penalties)
  - NTSB pipeline accident reports (probable cause, findings, safety recommendations)

Decompose the user's question into 1-5 atomic, self-contained sub-questions that can each be \
answered by retrieving from this corpus. Multi-part or comparison questions MUST be split \
(e.g. "compare Part 192 and Part 195 testing" -> one sub-question per Part). Set a source_hint \
when a sub-question clearly targets one source.

If the question cannot be answered from this corpus at all (e.g. general science, other \
jurisdictions, current events), set in_scope=false, give a one-sentence refusal_reason, and \
return no sub-questions. Do NOT invent sub-questions for out-of-scope queries."""


def plan(query: str) -> Plan:
    """Produce a structured Plan (or a refusal) for the query."""
    model = chat_model(settings.model_planner, temperature=0.0).with_structured_output(Plan)
    result: Plan = model.invoke([("system", _SYSTEM), ("human", query)])  # type: ignore[assignment]
    # Clamp to the configured maximum; ensure stable ids.
    result.sub_questions = result.sub_questions[: settings.max_sub_questions]
    for i, sq in enumerate(result.sub_questions, 1):
        sq.id = sq.id or f"sq{i}"
    log.info(
        "planner.done",
        in_scope=result.in_scope,
        n_sub_questions=len(result.sub_questions),
        refused=not result.in_scope,
    )
    return result

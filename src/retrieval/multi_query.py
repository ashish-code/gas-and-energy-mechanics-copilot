"""Multi-query expansion — Haiku generates 3 alternate phrasings per sub-question.

Each phrasing becomes its own retrieval lane (dense + sparse), and lanes are fused with RRF.
This widens recall over vocabulary mismatch (a question says "pressure test"; the reg says
"strength test" / "hydrostatic test") without any tuning.
"""

from __future__ import annotations

from src.config import settings
from src.llm import converse
from src.logging_setup import get_logger

log = get_logger(__name__)

_SYSTEM = (
    "You rewrite a question about US pipeline-safety regulations (49 CFR), PHMSA enforcement, "
    "and NTSB accident reports into alternate search phrasings. Vary terminology (e.g. "
    "'pressure test' vs 'strength/hydrostatic test', 'MAOP' vs 'maximum allowable operating "
    "pressure'). Output exactly 3 rephrasings, one per line, no numbering, no commentary."
)


def expand(question: str, n: int = 3) -> list[str]:
    """Return up to `n` alternate phrasings (empty list on failure -> caller uses original only)."""
    try:
        out = converse(settings.model_executor, f"Question: {question}", system=_SYSTEM, max_tokens=200)
        variants = [ln.strip(" -•\t") for ln in out.splitlines() if ln.strip()]
        return variants[:n]
    except Exception as e:  # noqa: BLE001
        log.warning("multi_query.failed", error=str(e))
        return []

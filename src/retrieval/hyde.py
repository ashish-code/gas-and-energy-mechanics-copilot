"""HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Haiku drafts one hypothetical regulation/finding paragraph that *would* answer the
sub-question; we embed that paragraph and use it as an extra dense query. A hypothetical
answer sits closer in embedding space to the real source passages than the bare question
does, lifting dense recall on terse or jargon-heavy queries.
"""

from __future__ import annotations

from src.config import settings
from src.llm import converse
from src.logging_setup import get_logger

log = get_logger(__name__)

_SYSTEM = (
    "You write a single concise paragraph (3-5 sentences) that plausibly answers a question "
    "about US pipeline-safety regulation, in the voice of a 49 CFR provision, a PHMSA "
    "enforcement finding, or an NTSB accident report. Be specific and use regulatory "
    "terminology. Do not hedge or say you are unsure — this is a hypothetical passage for retrieval."
)


def generate(question: str) -> str:
    """Return a hypothetical answer paragraph (falls back to the question on failure)."""
    try:
        return converse(settings.model_executor, f"Question: {question}", system=_SYSTEM, max_tokens=300).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("hyde.failed", error=str(e))
        return question

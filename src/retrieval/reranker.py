"""Cross-encoder reranking — Cohere Rerank 3.5 via the Bedrock Rerank API.

Takes the MMR-diversified top-20 and returns the top-10 by cross-encoder relevance. Keeping
the reranker on Bedrock (vs a self-hosted cross-encoder) is the choice that eliminates the
Streamlit Cloud RAM constraint. Cohere Rerank 3.5 is available in us-east-1 (verified at M0),
so the `bge-reranker-base` fallback in PLAN.md is documented but not built. On any rerank
error we degrade gracefully to the incoming (MMR) order so the demo never hard-fails.
"""

from __future__ import annotations

from src.bedrock_clients import bedrock_agent_runtime
from src.config import settings
from src.logging_setup import get_logger
from src.retrieval.base import Candidate

log = get_logger(__name__)


def _model_arn() -> str:
    return f"arn:aws:bedrock:{settings.aws_region}::foundation-model/{settings.model_reranker}"


def rerank(query: str, candidates: list[Candidate], *, top_k: int | None = None) -> list[Candidate]:
    """Rerank candidates by Cohere relevance; returns the top_k reordered candidates."""
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []
    docs = [c.contextualized_text or c.text for c in candidates]
    try:
        resp = bedrock_agent_runtime().rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query[:2000]}}],
            sources=[
                {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": d[:8000]}}}
                for d in docs
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(top_k, len(docs)),
                    "modelConfiguration": {"modelArn": _model_arn()},
                },
            },
        )
        out: list[Candidate] = []
        for r in resp["results"]:
            cand = candidates[r["index"]]
            cand.score = float(r["relevanceScore"])
            out.append(cand)
        return out
    except Exception as e:  # noqa: BLE001 — degrade to MMR order rather than fail the query
        log.warning("rerank.failed", error=str(e), n=len(candidates))
        return candidates[:top_k]

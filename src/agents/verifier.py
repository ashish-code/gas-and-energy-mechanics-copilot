"""VERIFY + SYNTHESIZE node.

1. Sonnet synthesizes a structured `DraftAnswer` from the evidence only — each `Claim` carries
   its citation, the evidence id it rests on, and a short verbatim quote.
2. Each claim is verified three ways:
     (a) mechanical — the cited evidence id exists in what we retrieved
     (b) mechanical — the quoted span actually appears in that evidence text
     (c) LLM-as-judge — Haiku decides whether the evidence span entails the claim (NLI)
   Claims failing any check go to `unsupported_claims` (surfaced, never silently dropped).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from src.agents.schemas import Claim, DraftAnswer, VerifiedAnswer, VerifiedClaim
from src.chunking.base import normalize_ws
from src.config import settings
from src.llm import chat_model
from src.logging_setup import get_logger
from src.retrieval.parent_doc import Evidence

log = get_logger(__name__)

_SYNTH_SYSTEM = """You answer questions about US pipeline-safety regulation STRICTLY from the \
provided evidence. Do not use outside knowledge. If the evidence is insufficient, say so in the \
summary rather than guessing.

Write a clear summary, then break it into atomic claims. For EACH claim provide:
  - citation: the human-readable source (e.g. "49 CFR 192.619", "PHMSA 52026001NOPV", "NTSB PAR-11/01")
  - chunk_id: the exact [evidence id] you used (copy it verbatim from the evidence block)
  - quote: a SHORT verbatim span (<=160 chars) copied exactly from that evidence supporting the claim
Every claim must be grounded in a specific evidence item. Prefer fewer, well-supported claims."""

_JUDGE_SYSTEM = (
    "You are a strict entailment judge. Given EVIDENCE and a CLAIM, decide whether the evidence "
    "alone entails (supports) the claim. Answer entailed=true only if a careful reader would agree "
    "the evidence establishes the claim. If it is unstated, contradicted, or only loosely related, "
    "answer entailed=false."
)


class _Verdict(BaseModel):
    entailed: bool = Field(description="does the evidence entail the claim?")
    reason: str = Field(default="", description="one short sentence")


def _evidence_block(evidence: list[Evidence]) -> str:
    parts = []
    for ev in evidence:
        cite = ev.metadata.get("citation") or " > ".join(ev.section_path) or ev.parent_id
        parts.append(f"[{ev.parent_id}] ({ev.source} — {cite})\n{ev.text}")
    return "\n\n---\n\n".join(parts)


def synthesize(query: str, evidence: list[Evidence]) -> DraftAnswer:
    model = chat_model(settings.model_synthesizer, temperature=0.0, max_tokens=2048).with_structured_output(
        DraftAnswer
    )
    prompt = f"Question: {query}\n\nEVIDENCE:\n{_evidence_block(evidence)}"
    draft: DraftAnswer = model.invoke([("system", _SYNTH_SYSTEM), ("human", prompt)])  # type: ignore[assignment]
    log.info("synth.done", claims=len(draft.claims))
    return draft


def _judge_entailment(claim: Claim, evidence_text: str) -> _Verdict:
    model = chat_model(settings.model_verifier, temperature=0.0, max_tokens=200).with_structured_output(_Verdict)
    prompt = f"EVIDENCE:\n{evidence_text[:6000]}\n\nCLAIM:\n{claim.text}"
    try:
        return model.invoke([("system", _JUDGE_SYSTEM), ("human", prompt)])  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        log.warning("judge.failed", error=str(e))
        return _Verdict(entailed=False, reason="verifier error")


def _verify_one(claim: Claim, by_id: dict[str, Evidence]) -> VerifiedClaim:
    ev = by_id.get(claim.chunk_id or "")
    citation_exists = ev is not None
    quote_matches = False
    entailed = False
    reason = ""

    if ev is not None:
        quote_matches = bool(claim.quote) and normalize_ws(claim.quote).lower() in normalize_ws(ev.text).lower()
        verdict = _judge_entailment(claim, ev.text)
        entailed, reason = verdict.entailed, verdict.reason
    else:
        reason = "cited evidence id not found in retrieved set"

    return VerifiedClaim(
        text=claim.text,
        citation=claim.citation,
        chunk_id=claim.chunk_id,
        quote=claim.quote,
        citation_exists=citation_exists,
        quote_matches=quote_matches,
        entailed=entailed,
        reason=reason,
    )


def verify(query: str, evidence: list[Evidence]) -> VerifiedAnswer:
    """Synthesize, then verify every claim. Returns the partitioned VerifiedAnswer."""
    draft = synthesize(query, evidence)
    by_id = {ev.parent_id: ev for ev in evidence}

    with ThreadPoolExecutor(max_workers=2) as pool:  # account is ~1 req/s; keep it light
        verified = list(pool.map(lambda c: _verify_one(c, by_id), draft.claims))

    supported = [v for v in verified if v.supported]
    unsupported = [v for v in verified if not v.supported]
    log.info("verify.done", total=len(verified), supported=len(supported), unsupported=len(unsupported))
    return VerifiedAnswer(
        summary=draft.summary,
        claims=supported,
        unsupported_claims=unsupported,
    )

"""Offline unit tests for the agent layer (no Bedrock).

The live plan-execute-verify path is validated end-to-end manually + in the e2e smoke; these
lock down the deterministic logic: schema invariants and the verifier's mechanical checks
(citation-exists + quote-matches), with the LLM-as-judge call mocked.
"""

from __future__ import annotations

from src.agents import verifier
from src.agents.schemas import Claim, Plan, SubQuestion, VerifiedClaim
from src.retrieval.parent_doc import Evidence


def _evidence(pid: str, text: str) -> Evidence:
    return Evidence(
        parent_id=pid, doc_id="d", source="ecfr", section_path=["49 CFR", "§ 192.619"],
        text=text, metadata={"citation": "49 CFR 192.619"}, score=0.9, matched_chunk_ids=[pid],
    )


def test_verified_claim_supported_requires_all_three() -> None:
    base = dict(text="t", citation="c", chunk_id="x", quote="q")
    assert VerifiedClaim(**base, citation_exists=True, quote_matches=True, entailed=True).supported
    assert not VerifiedClaim(**base, citation_exists=False, quote_matches=True, entailed=True).supported
    assert not VerifiedClaim(**base, citation_exists=True, quote_matches=False, entailed=True).supported
    assert not VerifiedClaim(**base, citation_exists=True, quote_matches=True, entailed=False).supported


def test_plan_holds_sub_questions() -> None:
    p = Plan(in_scope=True, sub_questions=[SubQuestion(id="sq1", question="q?", source_hint="ecfr")])
    assert p.in_scope and p.sub_questions[0].source_hint == "ecfr"


def test_verify_one_mechanical_checks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ev = {"p1": _evidence("p1", "The MAOP must not exceed the design pressure under §192.619.")}
    # Force the judge to say "entailed" so we isolate the mechanical checks.
    monkeypatch.setattr(verifier, "_judge_entailment", lambda c, t: verifier._Verdict(entailed=True))

    good = verifier._verify_one(Claim(text="MAOP capped by design pressure", citation="49 CFR 192.619",
                                      chunk_id="p1", quote="MAOP must not exceed the design pressure"), ev)
    assert good.citation_exists and good.quote_matches and good.entailed and good.supported

    bad_id = verifier._verify_one(Claim(text="x", citation="c", chunk_id="missing", quote="whatever"), ev)
    assert not bad_id.citation_exists and not bad_id.supported

    bad_quote = verifier._verify_one(Claim(text="x", citation="c", chunk_id="p1",
                                           quote="this span is not in the evidence"), ev)
    assert bad_quote.citation_exists and not bad_quote.quote_matches and not bad_quote.supported


def test_verify_partitions_supported_and_unsupported(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    evidence = [_evidence("p1", "Strength testing requires a minimum test pressure per §192.505.")]
    draft = verifier.DraftAnswer(
        summary="…",
        claims=[
            Claim(text="ok", citation="49 CFR 192.505", chunk_id="p1", quote="minimum test pressure"),
            Claim(text="bad", citation="49 CFR 192.505", chunk_id="p1", quote="not present span"),
        ],
    )
    monkeypatch.setattr(verifier, "synthesize", lambda q, e: draft)
    monkeypatch.setattr(verifier, "_judge_entailment", lambda c, t: verifier._Verdict(entailed=True))

    result = verifier.verify("q", evidence)
    assert len(result.claims) == 1 and result.claims[0].text == "ok"
    assert len(result.unsupported_claims) == 1 and result.unsupported_claims[0].text == "bad"

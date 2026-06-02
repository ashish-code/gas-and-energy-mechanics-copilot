"""Pydantic schemas for the plan-execute-verify graph.

Every LLM boundary is structured output — never free-text parsed. The planner emits a `Plan`
(or refuses), the synthesizer emits a `DraftAnswer` of atomic `Claim`s each carrying its own
citation + verbatim quote, and the verifier turns those into `VerifiedClaim`s partitioned
into supported vs. unsupported.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SourceName = Literal["ecfr", "phmsa_enforcement", "ntsb_accident"]


# --- PLAN -------------------------------------------------------------------
class SubQuestion(BaseModel):
    id: str = Field(description="stable short id, e.g. 'sq1'")
    question: str = Field(description="a single, self-contained, retrievable sub-question")
    source_hint: Optional[SourceName] = Field(
        default=None, description="optional: the most likely source for this sub-question"
    )


class Plan(BaseModel):
    in_scope: bool = Field(description="false if the query cannot be answered from the corpus")
    refusal_reason: Optional[str] = Field(
        default=None, description="if out of scope, a one-sentence explanation for the user"
    )
    sub_questions: list[SubQuestion] = Field(
        default_factory=list, description="1-5 sub-questions; empty when out of scope"
    )


# --- SYNTHESIZE -------------------------------------------------------------
class Claim(BaseModel):
    text: str = Field(description="one atomic factual assertion in the answer")
    citation: str = Field(description="human-readable source, e.g. '49 CFR 192.619' or 'PHMSA 52026001NOPV'")
    chunk_id: Optional[str] = Field(default=None, description="the evidence chunk/parent id this claim rests on")
    quote: str = Field(description="a short verbatim span copied from the evidence supporting the claim")


class DraftAnswer(BaseModel):
    summary: str = Field(description="the prose answer, synthesized from evidence only")
    claims: list[Claim] = Field(default_factory=list)


# --- VERIFY -----------------------------------------------------------------
class VerifiedClaim(BaseModel):
    text: str
    citation: str
    chunk_id: Optional[str] = None
    quote: str
    citation_exists: bool = Field(description="mechanical: the cited chunk/parent is in the retrieved evidence")
    quote_matches: bool = Field(description="mechanical: the quoted span actually appears in that evidence")
    entailed: bool = Field(description="LLM-as-judge: the evidence span entails the claim")
    reason: str = Field(default="", description="verifier's one-line justification")

    @property
    def supported(self) -> bool:
        return self.citation_exists and self.quote_matches and self.entailed


class VerifiedAnswer(BaseModel):
    summary: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    unsupported_claims: list[VerifiedClaim] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None

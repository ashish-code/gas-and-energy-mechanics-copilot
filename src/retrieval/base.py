"""Shared retrieval types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A retrieved chunk flowing through the fusion -> MMR -> rerank -> expand pipeline."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    contextualized_text: str
    section_path: list[str]
    parent_id: str | None
    chunk_index: int
    metadata: dict
    score: float = 0.0  # meaning depends on stage (cosine, RRF, rerank relevance)
    lanes: list[str] = field(default_factory=list)  # which retrieval lanes surfaced it

    @classmethod
    def from_row(cls, row: dict, score: float = 0.0, lane: str = "") -> "Candidate":
        return cls(
            chunk_id=row["id"],
            doc_id=row["doc_id"],
            source=row["source"],
            text=row["text"],
            contextualized_text=row.get("contextualized_text") or row["text"],
            section_path=row.get("section_path") or [],
            parent_id=row.get("parent_id"),
            chunk_index=row.get("chunk_index") or 0,
            metadata=row.get("metadata") or {},
            score=score,
            lanes=[lane] if lane else [],
        )

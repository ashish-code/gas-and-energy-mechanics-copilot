"""Small-to-big retrieval — expand reranked small chunks to their parent sections.

We embed and rerank paragraph-sized chunks (precise matching), then hand the *generator* the
full parent section (complete context). Reranked hits are grouped by `parent_id`, each parent
is reconstructed from all its sibling chunks (ordered by `chunk_index`), and parents are
deduplicated so the synthesizer sees ~5-7 unique, self-contained sections instead of
overlapping fragments.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.config import settings
from src.retrieval.base import Candidate
from src.store import fetch_parent_chunks


@dataclass
class Evidence:
    """A unique parent section assembled for generation."""

    parent_id: str
    doc_id: str
    source: str
    section_path: list[str]
    text: str  # full reconstructed parent section
    metadata: dict
    score: float  # best (rerank) score among the parent's matched child chunks
    matched_chunk_ids: list[str]


def expand_to_parents(
    conn: psycopg.Connection,
    reranked: list[Candidate],
    *,
    max_parents: int | None = None,
) -> list[Evidence]:
    """Group reranked candidates by parent, reconstruct each parent section, dedup."""
    max_parents = max_parents or settings.parent_sections_target

    # Preserve rerank order; first occurrence of a parent sets its rank + best score.
    order: list[str] = []
    best: dict[str, Candidate] = {}
    matched: dict[str, list[str]] = {}
    for c in reranked:
        pid = c.parent_id or c.chunk_id
        if pid not in best:
            best[pid] = c
            order.append(pid)
        matched.setdefault(pid, []).append(c.chunk_id)

    keep = order[:max_parents]
    siblings = fetch_parent_chunks(conn, keep)

    out: list[Evidence] = []
    for pid in keep:
        lead = best[pid]
        rows = siblings.get(pid)
        if rows:
            text = "\n\n".join(r["text"] for r in rows)
            meta = rows[0]["metadata"] or lead.metadata
            section_path = rows[0]["section_path"] or lead.section_path
            doc_id, source = rows[0]["doc_id"], rows[0]["source"]
        else:  # parent had no siblings recorded (e.g. single-chunk parent)
            text, meta = lead.text, lead.metadata
            section_path, doc_id, source = lead.section_path, lead.doc_id, lead.source
        out.append(
            Evidence(
                parent_id=pid,
                doc_id=doc_id,
                source=source,
                section_path=section_path,
                text=text,
                metadata=meta,
                score=lead.score,
                matched_chunk_ids=matched[pid],
            )
        )
    return out

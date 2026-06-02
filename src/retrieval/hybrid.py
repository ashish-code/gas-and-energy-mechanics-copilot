"""Reciprocal Rank Fusion (RRF) across all retrieval lanes.

RRF (Cormack et al., 2009) fuses ranked lists by summing 1/(k + rank) per item across lanes,
which needs no score calibration between heterogeneous lanes (cosine similarity vs BM25 vs
HyDE-cosine). With multi-query (orig + 3 variants) x {dense, sparse} + HyDE-dense, we fuse
up to 9 lanes into one ranking. k=60 is the canonical default.
"""

from __future__ import annotations

from src.config import settings
from src.retrieval.base import Candidate


def rrf_fuse(lanes: list[list[Candidate]], *, k: int | None = None, top_n: int | None = None) -> list[Candidate]:
    """Fuse ranked candidate lists by Reciprocal Rank Fusion."""
    k = k or settings.rrf_k
    top_n = top_n or settings.fused_top_k

    fused: dict[str, Candidate] = {}
    scores: dict[str, float] = {}
    for lane in lanes:
        for rank, cand in enumerate(lane):
            cid = cand.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in fused:
                fused[cid] = cand
            else:
                for ln in cand.lanes:
                    if ln not in fused[cid].lanes:
                        fused[cid].lanes.append(ln)

    ranked = sorted(fused.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    for c in ranked:
        c.score = scores[c.chunk_id]
    return ranked[:top_n]

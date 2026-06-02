"""Maximal Marginal Relevance (Carbonell & Goldstein, 1998), lambda=0.6.

Run on the fused top-50 before reranking: greedily select items that are relevant to the
query but dissimilar to those already chosen, so near-duplicate paragraphs (common across
multi-query lanes and overlapping reg sections) don't crowd out the cross-encoder's input.
Uses the stored chunk embeddings (cosine == dot, vectors are L2-normalized).
"""

from __future__ import annotations

import numpy as np

from src.config import settings
from src.retrieval.base import Candidate


def mmr_select(
    query_vec: list[float],
    candidates: list[Candidate],
    id_to_vec: dict[str, list[float]],
    *,
    lambda_mult: float | None = None,
    k: int | None = None,
) -> list[Candidate]:
    """Return up to k candidates diversified by MMR. Candidates lacking a vector are kept last."""
    lambda_mult = settings.mmr_lambda if lambda_mult is None else lambda_mult
    k = k or settings.mmr_top_k

    usable = [c for c in candidates if c.chunk_id in id_to_vec]
    missing = [c for c in candidates if c.chunk_id not in id_to_vec]
    if not usable:
        return candidates[:k]

    q = np.asarray(query_vec, dtype=np.float32)
    mat = np.asarray([id_to_vec[c.chunk_id] for c in usable], dtype=np.float32)
    relevance = mat @ q  # cosine to query

    selected: list[int] = []
    remaining = set(range(len(usable)))
    while remaining and len(selected) < k:
        best_i, best_score = None, -np.inf
        for i in remaining:
            if not selected:
                score = relevance[i]
            else:
                max_sim = max(float(mat[i] @ mat[j]) for j in selected)
                score = lambda_mult * relevance[i] - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)  # type: ignore[arg-type]
        remaining.discard(best_i)  # type: ignore[arg-type]

    out = [usable[i] for i in selected]
    if len(out) < k:
        out.extend(missing[: k - len(out)])
    return out

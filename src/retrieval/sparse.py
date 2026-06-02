"""Sparse retrieval — rank-bm25 (Okapi BM25), in-memory.

At ~1.5–2K chunks the BM25 index is ~20 MB, so we load it once at app startup from the
chunk table and keep it resident. Real BM25 beats Postgres tf-idf for hybrid retrieval and
keeps the exact-term matching (citations like "192.619", operator names) that dense
embeddings blur. We index `contextualized_text` for parity with the dense lane.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.retrieval.base import Candidate

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")  # keeps "192.619" as one token


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 over the chunk corpus."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self._bm25 = BM25Okapi([tokenize(r.get("contextualized_text") or r["text"]) for r in rows])

    def __len__(self) -> int:
        return len(self._rows)

    def search(self, query: str, k: int) -> list[Candidate]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Candidate.from_row(self._rows[i], score=float(scores[i]), lane="bm25") for i in top if scores[i] > 0]

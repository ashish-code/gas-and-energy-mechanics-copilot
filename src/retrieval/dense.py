"""Dense retrieval — Titan query embedding -> pgvector HNSW cosine via the match_chunks RPC."""

from __future__ import annotations

import psycopg

from src.config import settings
from src.embedding.bedrock_titan import TitanEmbedder
from src.retrieval.base import Candidate


class DenseRetriever:
    """Embeds a query and runs cosine kNN against the pgvector HNSW index."""

    def __init__(self, conn: psycopg.Connection, embedder: TitanEmbedder | None = None) -> None:
        self._conn = conn
        self._embedder = embedder or TitanEmbedder()

    def search(self, query: str, k: int | None = None, *, source: str | None = None) -> list[Candidate]:
        k = k or settings.dense_top_k
        qvec = self._embedder.embed_text(query)
        with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("select * from match_chunks(%s::vector, %s, %s)", (qvec, k, source))
            rows = cur.fetchall()
        return [Candidate.from_row(r, score=float(r["similarity"]), lane="dense") for r in rows]

    def search_with_vector(
        self, qvec: list[float], k: int | None = None, *, source: str | None = None
    ) -> list[Candidate]:
        """Dense search from a precomputed embedding (e.g. the HyDE document vector)."""
        k = k or settings.dense_top_k
        with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("select * from match_chunks(%s::vector, %s, %s)", (qvec, k, source))
            rows = cur.fetchall()
        return [Candidate.from_row(r, score=float(r["similarity"]), lane="dense_hyde") for r in rows]

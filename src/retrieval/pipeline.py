"""The full retrieval pipeline — one call runs every stage for one (sub-)question.

    multi-query (orig + 3 variants) + HyDE (1 hypothetical paragraph)
      -> per query form: dense (Titan->pgvector) + sparse (BM25)
      -> RRF fusion (k=60)            -> top-50
      -> MMR diversification (λ=0.6)  -> top-20
      -> Cohere Rerank via Bedrock    -> top-10
      -> small-to-big parent expansion + dedup -> ~5-7 Evidence

The agent EXECUTE node (M4) calls `Retriever.retrieve()` once per sub-question. The BM25
index is built once from the chunk table and held resident; the DB connection is reused.
"""

from __future__ import annotations

import psycopg

from src.config import settings
from src.embedding.bedrock_titan import TitanEmbedder
from src.logging_setup import get_logger
from src.retrieval import hybrid, hyde, mmr, multi_query, reranker
from src.retrieval.base import Candidate
from src.retrieval.dense import DenseRetriever
from src.retrieval.parent_doc import Evidence, expand_to_parents
from src.retrieval.sparse import BM25Index
from src.store import connect, fetch_all_chunks, fetch_embeddings

log = get_logger(__name__)

# Embed query forms with light concurrency — this account throttles above ~2 in-flight.
_QUERY_EMBED_WORKERS = 2


class Retriever:
    """Holds the resident BM25 index + DB connection + dense retriever."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn
        self._embedder = TitanEmbedder()
        self._dense = DenseRetriever(conn, self._embedder)
        rows = fetch_all_chunks(conn)
        self._bm25 = BM25Index(rows)
        log.info("retriever.ready", corpus=len(rows))

    @classmethod
    def open(cls) -> "Retriever":
        """Open a fresh DB connection and build the retriever (caller owns lifecycle)."""
        conn = psycopg.connect(settings.supabase_db_url)  # type: ignore[arg-type]
        from pgvector.psycopg import register_vector

        register_vector(conn)
        return cls(conn)

    def retrieve(self, question: str, *, source: str | None = None) -> list[Evidence]:
        # 1-2. Multi-query expansion + HyDE.
        variants = [question, *multi_query.expand(question)]
        hyde_doc = hyde.generate(question)

        # Embed all query forms (+ HyDE doc) together under the concurrency cap.
        embeds = self._embedder.embed_batch([*variants, hyde_doc], max_workers=_QUERY_EMBED_WORKERS)
        variant_vecs, hyde_vec = embeds[:-1], embeds[-1]
        query_vec = variant_vecs[0]  # original question, used for MMR relevance

        # 3. Dense + sparse lane per query form, plus a HyDE dense lane.
        lanes: list[list[Candidate]] = []
        for q, qv in zip(variants, variant_vecs):
            lanes.append(self._dense.search_with_vector(qv, settings.dense_top_k, source=source))
            lanes.append(self._bm25.search(q, settings.sparse_top_k))
        lanes.append(self._dense.search_with_vector(hyde_vec, settings.dense_top_k, source=source))

        # 4. RRF fusion -> top-50.
        fused = hybrid.rrf_fuse(lanes)
        if not fused:
            return []

        # 5. MMR diversification -> top-20 (needs stored vectors of the fused set).
        id_to_vec = fetch_embeddings(self._conn, [c.chunk_id for c in fused])
        diversified = mmr.mmr_select(query_vec, fused, id_to_vec)

        # 6. Cohere Rerank -> top-10.
        reranked = reranker.rerank(question, diversified)

        # 7-9. Small-to-big parent expansion + dedup -> ~5-7 unique sections.
        evidence = expand_to_parents(self._conn, reranked)
        log.info(
            "retriever.done",
            question=question[:80],
            lanes=len(lanes),
            fused=len(fused),
            reranked=len(reranked),
            evidence=len(evidence),
        )
        return evidence


def retrieve(question: str, *, source: str | None = None) -> list[Evidence]:
    """Convenience one-shot: open a retriever, run once, close. (Tests / CLI; the app caches.)"""
    with connect() as conn:
        return Retriever(conn).retrieve(question, source=source)

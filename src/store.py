"""Supabase Postgres access (psycopg + pgvector).

Supabase is just hosted Postgres, so we use psycopg directly with the connection string —
cleaner than the Supabase client for raw vector SQL. This module owns connection setup
(pgvector adapter registration), batched upserts for the ingestion pipeline, and the
chunk fetch used to build the in-memory BM25 index at app startup.

Dense cosine retrieval lives in `retrieval/dense.py` (it calls the `match_chunks` RPC).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from pgvector.psycopg import register_vector
import psycopg
from psycopg.types.json import Json

from src.chunking.base import Chunk, Document
from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)


def _dsn() -> str:
    if not settings.supabase_db_url:
        raise RuntimeError("SUPABASE_DB_URL is not set — add it to .env or Streamlit secrets.")
    return settings.supabase_db_url


def make_connection(*, autocommit: bool = False) -> psycopg.Connection:
    """A pgvector-enabled psycopg connection that is safe on Supabase's transaction pooler.

    `prepare_threshold=None` disables psycopg3's automatic server-side prepared statements —
    these break under pgbouncer transaction-mode pooling (backends are reassigned per
    transaction, so a prepared statement vanishes / collides), which manifests as hangs after
    a handful of identical queries. This is essential for both the app and the batch jobs.
    """
    conn = psycopg.connect(_dsn(), prepare_threshold=None, autocommit=autocommit, connect_timeout=15)
    register_vector(conn)
    return conn


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """A pgvector-enabled connection (context-managed). Explicit-commit (autocommit off)."""
    conn = make_connection(autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


def upsert_documents(conn: psycopg.Connection, docs: list[Document]) -> int:
    sql = """
        insert into documents (id, source, title, url, effective_date, metadata)
        values (%s, %s, %s, %s, %s, %s)
        on conflict (id) do update set
            title = excluded.title, url = excluded.url,
            effective_date = excluded.effective_date, metadata = excluded.metadata
    """
    rows = [
        (d.id, d.source, d.title, d.url, d.effective_date or None, Json(d.metadata))
        for d in docs
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    log.info("store.upsert_documents", n=len(rows))
    return len(rows)


def upsert_chunks(conn: psycopg.Connection, chunks: list[Chunk], *, batch_size: int = 200) -> int:
    sql = """
        insert into chunks
            (id, doc_id, source, text, contextualized_text, embedding,
             section_path, parent_id, page, chunk_index, metadata)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (id) do update set
            text = excluded.text,
            contextualized_text = excluded.contextualized_text,
            embedding = excluded.embedding,
            section_path = excluded.section_path,
            parent_id = excluded.parent_id,
            page = excluded.page,
            chunk_index = excluded.chunk_index,
            metadata = excluded.metadata
    """
    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            rows = [
                (
                    c.chunk_id, c.doc_id, c.source, c.text,
                    c.contextualized_text or c.text,
                    c.embedding, c.section_path or None, c.parent_id,
                    c.page, c.chunk_index, Json(c.metadata),
                )
                for c in batch
            ]
            cur.executemany(sql, rows)
            conn.commit()
            total += len(batch)
            log.info("store.upsert_chunks_batch", done=total, total=len(chunks))
    return total


def fetch_all_chunks(conn: psycopg.Connection) -> list[dict]:
    """Load chunks for the in-memory BM25 index (no embeddings)."""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "select id, doc_id, source, text, contextualized_text, section_path, "
            "parent_id, chunk_index, metadata from chunks"
        )
        return cur.fetchall()


def count_chunks(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("select source, count(*) from chunks group by source")
        return {src: n for src, n in cur.fetchall()}


def fetch_embeddings(conn: psycopg.Connection, ids: list[str]) -> dict[str, list[float]]:
    """Stored embeddings for the given chunk ids (for MMR diversification)."""
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("select id, embedding from chunks where id = any(%s)", (ids,))
        return {cid: list(vec) for cid, vec in cur.fetchall()}


def fetch_parent_chunks(conn: psycopg.Connection, parent_ids: list[str]) -> dict[str, list[dict]]:
    """All sibling chunks for each parent_id, ordered by chunk_index (small-to-big expansion)."""
    if not parent_ids:
        return {}
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "select id, doc_id, source, text, section_path, parent_id, chunk_index, metadata "
            "from chunks where parent_id = any(%s) order by parent_id, chunk_index",
            (parent_ids,),
        )
        out: dict[str, list[dict]] = {}
        for row in cur.fetchall():
            out.setdefault(row["parent_id"], []).append(row)
        return out

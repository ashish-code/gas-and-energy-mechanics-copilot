"""Backfill Contextual-Retrieval blurbs for chunks that fell back to raw text.

During a build, a throttled context-blurb call degrades gracefully (the chunk keeps its raw
text as `contextualized_text`). This re-generates the blurb and re-embeds just those chunks,
so the corpus reaches 100% contextual coverage. Idempotent; safe to re-run. Paced by the
global Bedrock rate limiter (botocore hook), so it won't throttle.

  uv run python scripts/backfill_context.py
"""

from __future__ import annotations

import psycopg

from src.chunking.base import Chunk
from src.chunking.contextual import _blurb
from src.embedding.bedrock_titan import TitanEmbedder
from src.logging_setup import bind_correlation_id, configure_logging, get_logger
from src.store import connect

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    bind_correlation_id()
    embedder = TitanEmbedder()

    with connect() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "select c.id, c.doc_id, c.source, c.text, c.section_path, c.metadata, d.title "
                "from chunks c join documents d on d.id = c.doc_id "
                "where c.contextualized_text = c.text"
            )
            rows = cur.fetchall()
        log.info("backfill.start", n=len(rows))

        done = fixed = 0
        for r in rows:
            chunk = Chunk(
                chunk_id=r["id"], doc_id=r["doc_id"], source=r["source"], text=r["text"],
                section_path=r["section_path"] or [], metadata=r["metadata"] or {},
            )
            blurb = _blurb(chunk, r["title"] or r["source"])
            if not blurb:
                done += 1
                continue
            ctext = f"{blurb}\n\n{chunk.text}"
            emb = embedder.embed_text(ctext)
            with conn.cursor() as cur:
                cur.execute(
                    "update chunks set contextualized_text = %s, embedding = %s where id = %s",
                    (ctext, emb, r["id"]),
                )
            conn.commit()
            done += 1
            fixed += 1
            if done % 50 == 0:
                log.info("backfill.progress", done=done, total=len(rows), fixed=fixed)

        log.info("backfill.done", processed=done, fixed=fixed)


if __name__ == "__main__":
    main()

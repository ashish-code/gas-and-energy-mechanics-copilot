"""Build the corpus index: fetch -> chunk -> contextualize -> embed -> upsert to Supabase.

Usage (after applying infra/supabase_schema.sql and setting SUPABASE_DB_URL):
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --sources ecfr ntsb   # subset
    uv run python scripts/build_index.py --phmsa-limit 100 --no-upsert --dump data/chunks/chunks.json

Idempotent: re-running upserts by content-hash chunk id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingestion.pipeline import contextualize_and_embed, gather
from src.logging_setup import bind_correlation_id, configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["ecfr", "phmsa", "ntsb"])
    ap.add_argument("--ecfr-date", default="2026-01-01")
    ap.add_argument("--ecfr-parts", nargs="+", default=None)
    ap.add_argument("--phmsa-limit", type=int, default=200)
    ap.add_argument("--ntsb-codes", nargs="+", default=None)
    ap.add_argument("--no-upsert", action="store_true", help="skip Supabase upsert")
    ap.add_argument("--dump", type=str, default=None, help="write chunks (incl. embeddings) to JSON")
    args = ap.parse_args()

    configure_logging()
    bind_correlation_id()
    log.info("build_index.start", sources=args.sources)

    documents, chunks = gather(
        ecfr_date=args.ecfr_date,
        ecfr_parts=args.ecfr_parts,
        phmsa_limit=args.phmsa_limit,
        ntsb_codes=args.ntsb_codes,
        sources=tuple(args.sources),
    )
    if not chunks:
        log.error("build_index.no_chunks")
        raise SystemExit(1)

    contextualize_and_embed(documents, chunks)

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps([c.model_dump() for c in chunks]))
        log.info("build_index.dumped", path=args.dump, chunks=len(chunks))

    if not args.no_upsert:
        from src.store import connect, count_chunks, upsert_chunks, upsert_documents

        with connect() as conn:
            upsert_documents(conn, documents)
            upsert_chunks(conn, chunks)
            counts = count_chunks(conn)
        log.info("build_index.done", upserted=len(chunks), by_source=counts)
    else:
        by_source: dict[str, int] = {}
        for c in chunks:
            by_source[c.source] = by_source.get(c.source, 0) + 1
        log.info("build_index.done_no_upsert", chunks=len(chunks), by_source=by_source)


if __name__ == "__main__":
    main()

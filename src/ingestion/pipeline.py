"""Ingestion orchestration: fetch -> chunk (source-router) -> contextualize -> embed.

Produces the (documents, chunks) pair ready to upsert. Each source is independent and
failure-isolated: if one source raises, we log and continue with the others (per-source
failure handling from PLAN.md / the contract).
"""

from __future__ import annotations

from src.chunking.base import Chunk, Document
from src.chunking.contextual import contextualize
from src.chunking.ecfr_xml import chunk_parts
from src.chunking.ntsb_semantic import chunk_reports
from src.chunking.phmsa_layout import chunk_cases
from src.embedding.bedrock_titan import TitanEmbedder
from src.ingestion import ecfr, ntsb, phmsa
from src.logging_setup import get_logger

log = get_logger(__name__)


def gather(
    *,
    ecfr_date: str = "2026-01-01",
    ecfr_parts: list[str] | None = None,
    phmsa_limit: int = 200,
    ntsb_codes: list[str] | None = None,
    sources: tuple[str, ...] = ("ecfr", "phmsa", "ntsb"),
) -> tuple[list[Document], list[Chunk]]:
    """Fetch + chunk all enabled sources. No embeddings yet."""
    documents: list[Document] = []
    chunks: list[Chunk] = []

    if "ecfr" in sources:
        try:
            parts = ecfr.fetch_all(ecfr_date, ecfr_parts)
            documents += [p.document for p in parts]
            chunks += chunk_parts(parts)
        except Exception as e:  # noqa: BLE001
            log.warning("ingest.ecfr_failed", error=str(e))

    if "phmsa" in sources:
        try:
            cases = phmsa.fetch_cases(limit=phmsa_limit)
            documents += [c.document for c in cases]
            chunks += chunk_cases(cases)
        except Exception as e:  # noqa: BLE001
            log.warning("ingest.phmsa_failed", error=str(e))

    if "ntsb" in sources:
        try:
            reports = ntsb.fetch_reports(ntsb_codes)
            documents += [r.document for r in reports]
            chunks += chunk_reports(reports)
        except Exception as e:  # noqa: BLE001
            log.warning("ingest.ntsb_failed", error=str(e))

    log.info("ingest.gathered", documents=len(documents), chunks=len(chunks))
    return documents, chunks


def contextualize_and_embed(documents: list[Document], chunks: list[Chunk]) -> list[Chunk]:
    """Prepend Contextual-Retrieval blurbs, then embed `contextualized_text` with Titan.

    Crash-proof: a chunk whose embedding fails (after retries) is dropped with a warning
    rather than aborting the whole build — at this corpus size a handful of drops is
    acceptable, a lost 90-minute build is not.
    """
    titles = {d.id: (d.title or d.source) for d in documents}
    contextualize(chunks, titles)

    embedder = TitanEmbedder()
    embedded: list[Chunk] = []
    failed = 0
    for i, c in enumerate(chunks, 1):
        try:
            c.embedding = embedder.embed_text(c.contextualized_text or c.text)
            embedded.append(c)
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warning("ingest.embed_failed", chunk_id=c.chunk_id, error=str(e))
        if i % 250 == 0:
            log.info("ingest.embed_progress", done=i, total=len(chunks), kept=len(embedded))
    log.info("ingest.embedded", kept=len(embedded), dropped=failed, total=len(chunks))
    return embedded

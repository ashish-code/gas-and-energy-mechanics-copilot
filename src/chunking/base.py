"""Unified chunk schema + shared chunking utilities.

All three source-specific chunkers (eCFR, PHMSA, NTSB) emit `Chunk` objects with this
one schema, so everything downstream — contextualization, embedding, upsert, retrieval —
is source-agnostic. Source-specific structure lives only in `metadata` and `section_path`.

Small-to-big: chunkers emit paragraph-/clause-sized *small* chunks and set `parent_id` to
the enclosing section/clause group. At generation time `parent_doc.py` reconstructs the
parent by concatenating sibling chunks sharing a `parent_id` (ordered by `chunk_index`),
so we never store parent text twice.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["ecfr", "phmsa_enforcement", "ntsb_accident"]

# Heuristic token estimate (avoids a tokenizer dependency at ingest time). English prose
# averages ~0.75 words/token; thresholds below are expressed in tokens via this factor.
_WORDS_PER_TOKEN = 0.75


def approx_tokens(text: str) -> int:
    """Rough token count from whitespace word count."""
    return round(len(text.split()) / _WORDS_PER_TOKEN)


def content_hash(*parts: str) -> str:
    """Stable content-hash id (first 16 hex of sha256 over the joined parts)."""
    h = hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def normalize_ws(text: str) -> str:
    """Collapse runs of whitespace; strip. Keeps single newlines out of chunk text."""
    return re.sub(r"\s+", " ", text).strip()


class Chunk(BaseModel):
    """One retrieval unit. Mirrors the Supabase `chunks` row."""

    chunk_id: str
    doc_id: str
    source: Source
    text: str  # raw chunk text
    contextualized_text: str = ""  # text with the Contextual-Retrieval blurb prepended
    embedding: list[float] | None = None  # 1024D Titan V2, filled by the embedding step
    section_path: list[str] = Field(default_factory=list)
    parent_id: str | None = None  # groups siblings for small-to-big expansion
    page: int | None = None
    char_span: tuple[int, int] | None = None
    chunk_index: int = 0  # order within the parent
    effective_date: str | None = None  # ISO date
    metadata: dict = Field(default_factory=dict)


class Document(BaseModel):
    """One source document. Mirrors the Supabase `documents` row."""

    id: str
    source: Source
    title: str | None = None
    url: str | None = None
    effective_date: str | None = None
    metadata: dict = Field(default_factory=dict)


def split_long_text(text: str, max_tokens: int, target_tokens: int) -> list[str]:
    """Recursively split an over-long block on the strongest available boundary.

    Boundary preference: blank line -> sentence -> hard word window. Used only when a
    single natural unit (a paragraph/clause) exceeds `max_tokens`; not fixed-size chunking
    of whole documents.
    """
    if approx_tokens(text) <= max_tokens:
        return [text]

    # Try paragraph, then sentence boundaries.
    for pattern in (r"\n\s*\n", r"(?<=[.;:])\s+"):
        pieces = re.split(pattern, text)
        if len(pieces) > 1:
            return _pack(pieces, max_tokens, target_tokens)

    # Last resort: hard word window at the target size.
    words = text.split()
    target_words = max(1, round(target_tokens * _WORDS_PER_TOKEN))
    return [" ".join(words[i : i + target_words]) for i in range(0, len(words), target_words)]


def _pack(pieces: list[str], max_tokens: int, target_tokens: int) -> list[str]:
    """Greedily pack pieces into ~target_tokens blocks, recursing on oversize pieces."""
    out: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        ptok = approx_tokens(piece)
        if ptok > max_tokens:
            if buf:
                out.append(" ".join(buf))
                buf, buf_tok = [], 0
            out.extend(split_long_text(piece, max_tokens, target_tokens))
            continue
        if buf_tok + ptok > target_tokens and buf:
            out.append(" ".join(buf))
            buf, buf_tok = [], 0
        buf.append(piece)
        buf_tok += ptok
    if buf:
        out.append(" ".join(buf))
    return out


def pack_units(
    units: list[str],
    *,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
) -> list[str]:
    """Pack a section's natural units (paragraphs/clauses) into section-natural chunks.

    Greedy: accumulate units until ~target_tokens, then emit. Oversize single units are
    recursively split. A trailing block under `min_tokens` is merged back into the
    previous block (we never drop regulatory text, unlike a naive <min-token drop).
    """
    blocks = _pack([u for u in units if u and u.strip()], max_tokens, target_tokens)
    if len(blocks) >= 2 and approx_tokens(blocks[-1]) < min_tokens:
        blocks[-2] = blocks[-2] + " " + blocks[-1]
        blocks.pop()
    return blocks

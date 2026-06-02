"""NTSB chunker — two-pass: structural sectioning + semantic chunking.

Pass 1 (structural): pdfplumber text -> detect major section headers (Abstract, Executive
Summary, numbered top-level sections, Conclusions/Findings, Probable Cause, Recommendations)
and slice the report into labeled sections. Front matter, table of contents, and appendices
are dropped.

Pass 2 (semantic): within each kept section, split into sentences and place chunk boundaries
at *semantic dips* — points where the cosine distance between adjacent sentence embeddings
(Titan) exceeds the section's 95th-percentile distance, i.e. the strongest topic shifts.
Resulting segments are packed to a section-natural size. Very long sections fall back to
paragraph packing (no per-sentence embedding) to bound build-time embedding cost.

Small-to-big: chunks within a section share `parent_id`, so generation expands to the section.
"""

from __future__ import annotations

import re

import numpy as np
import pdfplumber

from src.chunking.base import Chunk, approx_tokens, content_hash, normalize_ws, pack_units, split_long_text
from src.chunking.phmsa_layout import parse_cited_regs
from src.embedding.bedrock_titan import TitanEmbedder
from src.ingestion.ntsb import NtsbReport
from src.logging_setup import get_logger

log = get_logger(__name__)

TARGET_TOKENS = 400
MAX_TOKENS = 700
MIN_TOKENS = 30
DIP_PERCENTILE = 95  # spec: split at 95th-percentile similarity dips
# Semantic-dip chunking embeds every sentence (Titan), which is slow/throttled at scale.
# We apply it only to bounded sections (abstract, conclusions, probable cause,
# recommendations — the Q-relevant ones); long analysis/factual sections fall back to
# paragraph packing. A deliberate cost/quality trade-off, not tuning.
MAX_SENTENCES_SEMANTIC = 150

# Header -> canonical label. Sections not matched here that look like headers are "other".
_KEEP = {
    "abstract": "abstract",
    "executive summary": "executive_summary",
    "investigation and analysis": "analysis",
    "factual information": "factual",
    "analysis": "analysis",
    "conclusions": "conclusions",
    "findings": "findings",
    "probable cause": "probable_cause",
    "recommendations": "recommendations",
}
_DROP_HINTS = ("contents", "appendix", "appendixes", "abbreviation", "acronym", "about the ntsb", "board member")

_HEADER_RE = re.compile(
    r"^(?:(?:\d+\.?\s+)?(abstract|executive summary|investigation and analysis|factual information|"
    r"analysis|conclusions|findings|probable cause|recommendations))\s*$",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")


def _extract_text(report: NtsbReport) -> str:
    pages: list[str] = []
    with pdfplumber.open(report.pdf_path) as pdf:
        for pg in pdf.pages:
            pages.append(pg.extract_text() or "")
    return "\n".join(pages)


def _split_sections(full: str) -> list[tuple[str, str]]:
    """Return [(canonical_label, text)] for kept sections, in document order."""
    lines = full.splitlines()
    marks: list[tuple[int, str]] = []  # (line_index, canonical_label)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or len(s) > 60 or "...." in s or s.count(".") > 3:
            continue  # skip prose and TOC dot-leader lines
        m = _HEADER_RE.match(s)
        if m:
            label = _KEEP.get(m.group(1).lower())
            if label:
                marks.append((i, label))

    if not marks:
        return [("report_body", normalize_ws(full))]

    # Keep only the *last* occurrence run; the first hits are usually the TOC. Slice between marks.
    sections: list[tuple[str, str]] = []
    for idx, (line_i, label) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = normalize_ws("\n".join(lines[line_i + 1 : end]))
        if approx_tokens(body) >= MIN_TOKENS and not any(h in label for h in _DROP_HINTS):
            sections.append((label, body))

    # Deduplicate: a label can appear twice (TOC + real); keep the longest body per label.
    best: dict[str, tuple[str, str]] = {}
    for label, body in sections:
        if label not in best or len(body) > len(best[label][1]):
            best[label] = (label, body)
    # Preserve first-seen order of the kept (real) sections.
    seen: list[str] = []
    for label, _ in sections:
        if label not in seen:
            seen.append(label)
    return [best[label] for label in seen if label in best]


def _semantic_segments(text: str, embedder: TitanEmbedder) -> list[str]:
    """Place boundaries at 95th-percentile cosine dips between adjacent sentences."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 2:
        return [text]
    if len(sentences) > MAX_SENTENCES_SEMANTIC:
        # Too large to embed per-sentence economically: pack on paragraph/sentence boundaries.
        return split_long_text(text, MAX_TOKENS, TARGET_TOKENS)

    vecs = embedder.embed_array(sentences)  # (n, 1024), L2-normalized
    # cosine distance between consecutive sentences
    dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = float(np.percentile(dists, DIP_PERCENTILE))
    boundaries = {i + 1 for i, d in enumerate(dists) if d >= threshold}

    segments: list[str] = []
    cur: list[str] = []
    for i, sent in enumerate(sentences):
        if i in boundaries and cur:
            segments.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        segments.append(" ".join(cur))
    # Pack the semantic segments to a section-natural size.
    return pack_units(segments, target_tokens=TARGET_TOKENS, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS)


def chunk_report(report: NtsbReport, embedder: TitanEmbedder) -> list[Chunk]:
    full = _extract_text(report)
    if approx_tokens(full) < MIN_TOKENS:
        log.warning("ntsb.empty_pdf", code=report.document.metadata.get("report_code"))
        return []

    doc_id = report.document.id
    code = report.document.metadata.get("report_code", "")
    mode = report.document.metadata.get("mode")
    sections = _split_sections(full)
    chunks: list[Chunk] = []

    for sec_idx, (label, body) in enumerate(sections):
        parent_id = f"{doc_id}:sec:{sec_idx}"
        section_path = [code, label.replace("_", " ").title()]
        blocks = _semantic_segments(body, embedder)
        for idx, block in enumerate(blocks):
            cited = parse_cited_regs(block)
            chunks.append(
                Chunk(
                    chunk_id=content_hash(parent_id, str(idx), block),
                    doc_id=doc_id,
                    source="ntsb_accident",
                    text=block,
                    section_path=section_path,
                    parent_id=parent_id,
                    chunk_index=idx,
                    metadata={
                        "report_code": code,
                        "section": label,
                        "mode": mode,
                        "cited_regs": cited,
                        "mentions_phmsa": "phmsa" in block.lower(),
                        "url": report.document.url,
                    },
                )
            )

    log.info("ntsb.chunked", code=code, sections=len(sections), chunks=len(chunks))
    return chunks


def chunk_reports(reports: list[NtsbReport]) -> list[Chunk]:
    embedder = TitanEmbedder()
    out: list[Chunk] = []
    for r in reports:
        try:
            out.extend(chunk_report(r, embedder))
        except Exception as e:  # noqa: BLE001 — a malformed PDF shouldn't sink the batch
            log.warning("ntsb.report_failed", code=r.document.metadata.get("report_code"), error=str(e))
    return out

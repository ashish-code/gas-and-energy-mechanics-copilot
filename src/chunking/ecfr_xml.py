"""eCFR chunker — section-aware XML walker.

Exploits the eCFR XML hierarchy directly:
    DIV6 TYPE=SUBPART   -> subpart (N="C", HEAD="Subpart C—Design of Pipe Components")
      DIV8 TYPE=SECTION -> section (N="192.105", HEAD="§ 192.105 Design formula...")
        P / PSPACE / FP / TABLE -> the section's natural paragraph units

Small-to-big: each section's paragraph units are packed into section-natural chunks
(~350 tokens), all sharing `parent_id = <section>` so `parent_doc.py` can rebuild the
full section at generation time. Only TYPE=SECTION is ingested (appendices/notes skipped).
"""

from __future__ import annotations

import json

from lxml import etree

from src.chunking.base import Chunk, content_hash, normalize_ws, pack_units
from src.ingestion.ecfr import EcfrPart
from src.logging_setup import get_logger

log = get_logger(__name__)

TARGET_TOKENS = 350
MAX_TOKENS = 800  # spec: recursive-split a section/paragraph beyond this
MIN_TOKENS = 30   # spec: drop <20-30 tokens (we merge-forward instead of dropping)

# Block-level tags whose text forms a section's natural units, in document order.
_UNIT_TAGS = {"P", "PSPACE", "FP", "FP-1", "FP-2", "FP-DASH", "TABLE", "EXTRACT", "NOTE"}


def _text(el: etree._Element) -> str:
    return normalize_ws("".join(el.itertext()))


def _citation(section: etree._Element, part: str, num: str) -> str:
    """Pull '49 CFR 192.105' from hierarchy_metadata, else synthesize it."""
    meta = section.get("hierarchy_metadata")
    if meta:
        try:
            cit = json.loads(meta).get("citation")
            if cit:
                return cit
        except (ValueError, TypeError):
            pass
    return f"49 CFR {num or part}"


def chunk_part(part: EcfrPart) -> list[Chunk]:
    """Walk one part's XML into section-natural small chunks."""
    root = etree.fromstring(part.xml)
    doc_id = part.document.id
    part_num = str(part.document.metadata.get("part", ""))
    eff = part.document.effective_date
    chunks: list[Chunk] = []

    for section in root.iter("DIV8"):
        if section.get("TYPE") != "SECTION":
            continue
        num = (section.get("N") or "").strip()
        head_el = section.find("HEAD")
        head = _text(head_el) if head_el is not None else f"§ {num}"

        # Nearest enclosing subpart (DIV6), if any.
        subpart_letter = subpart_title = None
        anc = section.getparent()
        while anc is not None:
            if anc.tag == "DIV6" and anc.get("TYPE") == "SUBPART":
                subpart_letter = (anc.get("N") or "").strip()
                sp_head = anc.find("HEAD")
                subpart_title = _text(sp_head) if sp_head is not None else None
                break
            anc = anc.getparent()

        # Collect the section's natural units, in order, skipping the HEAD.
        units: list[str] = []
        for child in section.iter():
            if child is section or child.tag not in _UNIT_TAGS:
                continue
            t = _text(child)
            if t:
                units.append(t)
        if not units:
            continue

        section_path = ["49 CFR", f"Part {part_num}"]
        if subpart_letter:
            section_path.append(f"Subpart {subpart_letter}")
        section_path.append(f"§ {num}")

        citation = _citation(section, part_num, num)
        parent_id = f"{doc_id}:sec:{num}"
        blocks = pack_units(units, target_tokens=TARGET_TOKENS, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS)

        for idx, block in enumerate(blocks):
            chunks.append(
                Chunk(
                    chunk_id=content_hash(parent_id, str(idx), block),
                    doc_id=doc_id,
                    source="ecfr",
                    text=block,
                    section_path=section_path,
                    parent_id=parent_id,
                    chunk_index=idx,
                    effective_date=eff,
                    metadata={
                        "citation": citation,
                        "section": num,
                        "section_title": head,
                        "part": part_num,
                        "subpart": subpart_letter,
                        "subpart_title": subpart_title,
                        "url": f"https://www.ecfr.gov/current/title-49/part-{part_num}/section-{num}",
                    },
                )
            )

    log.info("ecfr.chunked", part=part_num, sections="walked", chunks=len(chunks))
    return chunks


def chunk_parts(parts: list[EcfrPart]) -> list[Chunk]:
    out: list[Chunk] = []
    for p in parts:
        out.extend(chunk_part(p))
    return out

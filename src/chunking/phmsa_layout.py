"""PHMSA enforcement chunker.

The spec calls for detecting a "Respondent / Findings / Order / Civil Penalty" structure
and preserving action_id / respondent / date / civil_penalty in metadata. We *synthesize*
that structure from PHMSA's structured fields (cleaner than OCR-ing enforcement letters):

    Respondent   <- operator_name
    Findings     <- violation_desc + proposed_subject (which embeds the cited 49 CFR regs)
    Order        <- type_of_case + dates
    Civil Penalty<- proposed / assessed / collected

Cited regulations are parsed out of `proposed_subject` (e.g. "[195.452(j)(3)]") so the
chunk's metadata carries the enforcement->regulation join key.
"""

from __future__ import annotations

import re

from src.chunking.base import Chunk, content_hash
from src.ingestion.phmsa import PhmsaCase
from src.logging_setup import get_logger

log = get_logger(__name__)

# Matches a 49 CFR section like 192.451, 195.452(j)(3), 192.911(a).
_CITE_RE = re.compile(r"\b(19[0-9])\.(\d+)((?:\([0-9a-zA-Z]+\))*)")


def parse_cited_regs(*texts: str | None) -> list[str]:
    """Extract distinct '49 CFR §<n>' citations from any text fields."""
    found: list[str] = []
    for t in texts:
        if not t:
            continue
        for m in _CITE_RE.finditer(t):
            cite = f"{m.group(1)}.{m.group(2)}{m.group(3)}"
            if cite not in found:
                found.append(cite)
    return found


def _money(v: float | None) -> str:
    return f"${v:,.0f}" if v else "$0"


def chunk_case(case: PhmsaCase) -> list[Chunk]:
    """Render one enforcement case as a structured chunk."""
    cited = parse_cited_regs(case.proposed_subject, case.violation_desc, case.incident_detail)

    lines = [
        f"PHMSA Enforcement Action {case.cpf_num} — {case.type_of_case or case.case_type}",
        f"Respondent (operator): {case.operator_name}"
        + (f" (Operator ID {case.operator_id})" if case.operator_id else ""),
        f"Region: {case.region or 'N/A'}",
        f"Opened: {case.opened_dt or 'N/A'}; Closed: {case.closed_dt or 'N/A'}"
        + (f"; Final order: {case.final_order_dt}" if case.final_order_dt else ""),
    ]
    findings = case.proposed_subject or case.violation_desc
    if findings:
        lines.append(f"Findings / violation subject: {findings}")
    if cited:
        lines.append("Cited regulations (49 CFR): " + ", ".join(f"§{c}" for c in cited))
    if case.incident_detail:
        lines.append(f"Related incident: {case.incident_detail}")
    lines.append(
        "Civil penalty — proposed "
        f"{_money(case.proposed_penalties)}, assessed {_money(case.assessed_penalties)}, "
        f"collected {_money(case.collected)}."
    )
    text = "\n".join(lines)

    doc_id = case.document.id
    section_path = ["PHMSA Enforcement", case.type_of_case or case.case_type, case.cpf_num]
    return [
        Chunk(
            chunk_id=content_hash(doc_id, text),
            doc_id=doc_id,
            source="phmsa_enforcement",
            text=text,
            section_path=section_path,
            parent_id=doc_id,
            chunk_index=0,
            effective_date=case.closed_dt or case.opened_dt,
            metadata={
                "action_id": case.cpf_num,
                "respondent": case.operator_name,
                "operator_id": case.operator_id,
                "case_type": case.case_type,
                "region": case.region,
                "date": case.closed_dt or case.opened_dt,
                "civil_penalty": case.assessed_penalties,
                "cited_regs": cited,
                "url": case.document.url,
            },
        )
    ]


def chunk_cases(cases: list[PhmsaCase]) -> list[Chunk]:
    out: list[Chunk] = []
    for c in cases:
        out.extend(chunk_case(c))
    log.info("phmsa.chunked", chunks=len(out), cases=len(cases))
    return out

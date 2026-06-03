"""Offline unit tests for ingestion chunkers (no network).

eCFR walker is driven by a tiny inline XML fragment; PHMSA chunker by a constructed case.
NTSB's chunker needs PDFs + Bedrock, so it's covered by the live build, not here.
"""

from __future__ import annotations

from src.chunking.base import Document
from src.chunking.ecfr_xml import chunk_part
from src.chunking.phmsa_layout import chunk_case
from src.ingestion.ecfr import EcfrPart
from src.ingestion.phmsa import PhmsaCase

_ECFR_XML = b"""<DIV5 N="192" TYPE="PART">
  <DIV6 N="C" TYPE="SUBPART"><HEAD>Subpart C\xe2\x80\x94Design of Pipe</HEAD>
    <DIV8 N="192.105" TYPE="SECTION"
          hierarchy_metadata='{"citation":"49 CFR 192.105"}'>
      <HEAD>\xc2\xa7 192.105 Design formula for steel pipe.</HEAD>
      <P>(a) The design pressure for steel pipe is determined by the formula P = (2 St/D) x F x E x T.</P>
      <P>(b) Definitions apply as set out in this section for each variable in the formula above.</P>
    </DIV8>
  </DIV6>
</DIV5>"""


def test_ecfr_walker_chunks_section_with_citation_and_path() -> None:
    doc = Document(id="ecfr-49-192", source="ecfr", effective_date="2026-01-01", metadata={"part": "192"})
    chunks = chunk_part(EcfrPart(document=doc, xml=_ECFR_XML))
    assert chunks, "expected at least one chunk"
    c = chunks[0]
    assert c.source == "ecfr"
    assert c.metadata["section"] == "192.105"
    assert c.metadata["citation"] == "49 CFR 192.105"
    assert c.section_path == ["49 CFR", "Part 192", "Subpart C", "§ 192.105"]
    assert c.parent_id == "ecfr-49-192:sec:192.105"
    assert "design pressure" in c.text.lower()


def test_phmsa_chunker_synthesizes_structure_and_cites() -> None:
    case = PhmsaCase(
        cpf_num="52026001NOPV", case_type="NOPV", type_of_case="Notice of Probable Violation",
        operator_name="ACME PIPELINE LLC", operator_id="123", region="Western",
        opened_dt="2026-03-25", closed_dt="2026-05-01", final_order_dt="2026-05-01",
        proposed_penalties=53900.0, assessed_penalties=53900.0, collected=53900.0,
        violation_desc="Integrity Management",
        proposed_subject="Integrity Management [195.452(j)(3)] - 1 item(s)",
        incident_detail=None,
    )
    chunks = chunk_case(case)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.source == "phmsa_enforcement"
    assert c.metadata["action_id"] == "52026001NOPV"
    assert c.metadata["respondent"] == "ACME PIPELINE LLC"
    assert c.metadata["civil_penalty"] == 53900.0
    assert "195.452(j)(3)" in c.metadata["cited_regs"]
    assert "ACME PIPELINE LLC" in c.text and "$53,900" in c.text

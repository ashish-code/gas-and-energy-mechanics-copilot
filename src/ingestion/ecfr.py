"""eCFR ingestion — fetch 49 CFR Parts 192 / 193 / 195 as XML.

Source: eCFR Versioner API (public, stable, no auth).
    GET /api/versioner/v1/full/{date}/title-49.xml?part={N}

Built fresh in v2 (v1 had no eCFR ingestion — it scraped Wikipedia; see PLAN.md §7).
This module only does I/O: it fetches the XML and emits one `Document` per Part. The
section-aware walker in `chunking/ecfr_xml.py` parses and chunks it.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.chunking.base import Document
from src.logging_setup import get_logger

log = get_logger(__name__)

ECFR_BASE = "https://www.ecfr.gov/api/versioner/v1"
TITLE = 49

# Parts to ingest, with the human title used on the Document row.
PARTS: dict[str, str] = {
    "192": "Transportation of Natural and Other Gas by Pipeline: Minimum Federal Safety Standards",
    "193": "Liquefied Natural Gas Facilities: Federal Safety Standards",
    "195": "Transportation of Hazardous Liquids by Pipeline",
}


@dataclass
class EcfrPart:
    """A fetched eCFR part: its Document plus the raw XML for the chunker."""

    document: Document
    xml: bytes


def part_url(part: str, date: str) -> str:
    return f"{ECFR_BASE}/full/{date}/title-{TITLE}.xml?part={part}"


def fetch_part(part: str, date: str, *, client: httpx.Client | None = None) -> EcfrPart:
    """Fetch one part's XML and build its Document record."""
    url = part_url(part, date)
    owns = client is None
    client = client or httpx.Client(timeout=60.0, headers={"User-Agent": "gas-energy-copilot/2.0"})
    try:
        log.info("ecfr.fetch", part=part, url=url)
        resp = client.get(url)
        resp.raise_for_status()
        xml = resp.content
    finally:
        if owns:
            client.close()

    doc = Document(
        id=f"ecfr-{TITLE}-{part}",
        source="ecfr",
        title=f"{TITLE} CFR Part {part} — {PARTS.get(part, '')}".rstrip(" —"),
        url=f"https://www.ecfr.gov/current/title-{TITLE}/part-{part}",
        effective_date=date,
        metadata={"title": TITLE, "part": part},
    )
    log.info("ecfr.fetched", part=part, bytes=len(xml))
    return EcfrPart(document=doc, xml=xml)


def fetch_all(date: str, parts: list[str] | None = None) -> list[EcfrPart]:
    """Fetch all configured parts (or a subset) using a shared HTTP client."""
    parts = parts or list(PARTS)
    out: list[EcfrPart] = []
    with httpx.Client(timeout=60.0, headers={"User-Agent": "gas-energy-copilot/2.0"}) as client:
        for part in parts:
            out.append(fetch_part(part, date, client=client))
    return out

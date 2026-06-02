"""NTSB pipeline accident report ingestion.

The spec's listing page (AccidentReports/Pages/pipeline.aspx) is dead — NTSB moved to the
CAROL system. The legacy PDF repository is still live and stable, so we ingest a curated
set of pipeline accident reports (PAR / PIR) by their canonical report codes:

    https://www.ntsb.gov/investigations/AccidentReports/Reports/{CODE}.pdf

Curated for variety across gas transmission, gas distribution, and hazardous-liquid
accidents. A "report not found" soft-404 returns a ~51 KB HTML page, so we validate the
%PDF magic bytes and skip anything that fails (per-document failure handling).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.chunking.base import Document
from src.logging_setup import get_logger

log = get_logger(__name__)

REPORT_BASE = "https://www.ntsb.gov/investigations/AccidentReports/Reports"
CORPUS_DIR = Path("data/corpora/ntsb")
_HEADERS = {"User-Agent": "Mozilla/5.0 gas-energy-copilot/2.0"}

# Curated reports: code -> (short title, mode). Codes verified to return real PDFs at M2.
CURATED: dict[str, tuple[str, str]] = {
    "PAR0202": ("Olympic Pipe Line gasoline rupture and fire, Bellingham, WA (1999)", "hazardous_liquid"),
    "PAR0301": ("El Paso natural gas transmission rupture and fire, Carlsbad, NM (2000)", "gas_transmission"),
    "PAR0401": ("Enbridge crude oil pipeline rupture near Cohasset, MN (2002)", "hazardous_liquid"),
    "PAR0901": ("Dixie hazardous-liquid (propane) pipeline rupture, Carmichael, MS (2007)", "hazardous_liquid"),
    "PAR1101": ("Pacific Gas & Electric gas transmission rupture and fire, San Bruno, CA (2010)", "gas_transmission"),
    "PAR1201": ("Enbridge crude oil pipeline rupture and release, Marshall, MI (2010)", "hazardous_liquid"),
    "PAR1401": ("Columbia Gas Transmission pipeline rupture, Sissonville, WV (2012)", "gas_transmission"),
    "PAR1501": ("NTSB pipeline accident report PAR-15/01", "gas_transmission"),
    "PAR1902": ("Columbia Gas overpressurization, Merrimack Valley, MA (2018)", "gas_distribution"),
    "PIR22002": ("Enbridge natural gas transmission pipeline rupture (PIR-22/02)", "gas_transmission"),
}


@dataclass
class NtsbReport:
    document: Document
    pdf_path: Path


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=15), reraise=True)
def _download(client: httpx.Client, code: str, dest: Path) -> bool:
    resp = client.get(f"{REPORT_BASE}/{code}.pdf")
    resp.raise_for_status()
    data = resp.content
    if data[:4] != b"%PDF":
        log.warning("ntsb.not_pdf", code=code, bytes=len(data))
        return False
    dest.write_bytes(data)
    return True


def fetch_reports(codes: list[str] | None = None) -> list[NtsbReport]:
    """Download the curated PDFs (cached on disk) and build Document records."""
    codes = codes or list(CURATED)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[NtsbReport] = []
    with httpx.Client(timeout=90.0, headers=_HEADERS, follow_redirects=True) as client:
        for code in codes:
            title, mode = CURATED.get(code, (f"NTSB report {code}", "pipeline"))
            dest = CORPUS_DIR / f"{code}.pdf"
            if not dest.exists() or dest.stat().st_size < 50_000:
                log.info("ntsb.download", code=code)
                if not _download(client, code, dest):
                    continue
            doc = Document(
                id=f"ntsb-{code}",
                source="ntsb_accident",
                title=title,
                url=f"{REPORT_BASE}/{code}.pdf",
                metadata={"report_code": code, "mode": mode},
            )
            out.append(NtsbReport(document=doc, pdf_path=dest))
    log.info("ntsb.fetched", n=len(out))
    return out

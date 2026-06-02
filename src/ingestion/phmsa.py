"""PHMSA enforcement ingestion.

Source: the PHMSA "Enforcement Data" app (https://primis.phmsa.dot.gov/enforcement-data/).
It is a Gatsby SPA whose data is served as statically-built page-data JSON backed by
PHMSA's enforcement Postgres. We use that JSON directly — no PDF scraping, no headless
browser — which is far more robust than parsing enforcement letters.

Two endpoints:
  * master list : /enforcement-data/page-data/cases/closed/NOPV/page-data.json
                  -> postgres.sc_cases : every case (cpfNum, operator, type, dates, penalties)
  * case detail : /enforcement-data/page-data/case/{cpfNum}/page-data.json
                  -> postgres.scCase : adds `proposedSubjectEt`, which embeds the cited
                     49 CFR section(s), e.g. "Integrity Management [195.452(j)(3)] - 1 item(s)"

We select recent NOPV/CAO cases (the action types carrying violations, penalties, and
cited regulations) so the corpus supports enforcement->regulation join questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.chunking.base import Document
from src.logging_setup import get_logger

log = get_logger(__name__)

BASE = "https://primis.phmsa.dot.gov/enforcement-data"
MASTER_URL = f"{BASE}/page-data/cases/closed/NOPV/page-data.json"  # holds ALL cases
CASE_URL = BASE + "/page-data/case/{cpf}/page-data.json"
CASE_PAGE = "https://primis.phmsa.dot.gov/enforcement-data/case/{cpf}"

# Action types with violations + penalties + cited regs (vs. warning letters / advisories).
RELEVANT_TYPES = ("NOPV", "CAO")
_HEADERS = {"User-Agent": "gas-energy-copilot/2.0"}


@dataclass
class PhmsaCase:
    """Merged master + detail record for one enforcement case."""

    cpf_num: str
    case_type: str
    type_of_case: str
    operator_name: str
    operator_id: str | None
    region: str | None
    opened_dt: str | None
    closed_dt: str | None
    final_order_dt: str | None
    proposed_penalties: float | None
    assessed_penalties: float | None
    collected: float | None
    violation_desc: str | None
    proposed_subject: str | None  # raw `proposedSubjectEt` (carries cited regs)
    incident_detail: str | None
    extra: dict = field(default_factory=dict)

    @property
    def document(self) -> Document:
        title = f"{self.type_of_case or self.case_type} — {self.operator_name} ({self.cpf_num})"
        return Document(
            id=f"phmsa-{self.cpf_num}",
            source="phmsa_enforcement",
            title=title,
            url=CASE_PAGE.format(cpf=self.cpf_num),
            effective_date=self.closed_dt or self.opened_dt,
            metadata={
                "cpf_num": self.cpf_num,
                "operator": self.operator_name,
                "case_type": self.case_type,
                "region": self.region,
            },
        )


def fetch_master(client: httpx.Client) -> list[dict]:
    """Fetch the full case list (postgres.sc_cases)."""
    log.info("phmsa.fetch_master")
    resp = client.get(MASTER_URL)
    resp.raise_for_status()
    cases = resp.json()["result"]["data"]["postgres"]["sc_cases"]
    log.info("phmsa.master_loaded", total=len(cases))
    return cases


def select_recent(
    master: list[dict],
    *,
    limit: int,
    types: tuple[str, ...] = RELEVANT_TYPES,
) -> list[dict]:
    """Most-recent closed cases of the relevant action types (by closed date)."""
    rows = [c for c in master if c.get("caseType") in types and c.get("closedDt")]
    rows.sort(key=lambda c: c.get("closedDt") or "", reverse=True)
    return rows[:limit]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
def _get_detail(client: httpx.Client, cpf: str) -> dict:
    resp = client.get(CASE_URL.format(cpf=cpf))
    resp.raise_for_status()
    return resp.json()["result"]["data"]["postgres"]["scCase"]


def _clean(v: object) -> str | None:
    """Normalize PHMSA's sentinel 'None'/empty strings to actual None."""
    return None if v in (None, "None", "") else str(v)


def _num(v: object) -> float | None:
    try:
        if v in (None, "None", ""):
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch_cases(limit: int = 200, *, types: tuple[str, ...] = RELEVANT_TYPES) -> list[PhmsaCase]:
    """Fetch master list, select recent cases, and enrich each with its detail record."""
    out: list[PhmsaCase] = []
    with httpx.Client(timeout=30.0, headers=_HEADERS) as client:
        selected = select_recent(fetch_master(client), limit=limit, types=types)
        log.info("phmsa.selected", n=len(selected), limit=limit, types=types)
        for i, row in enumerate(selected):
            cpf = row["cpfNum"]
            try:
                d = _get_detail(client, cpf)
            except Exception as e:  # noqa: BLE001 — one bad case shouldn't sink the batch
                log.warning("phmsa.detail_failed", cpf=cpf, error=str(e))
                d = {}
            out.append(
                PhmsaCase(
                    cpf_num=cpf,
                    case_type=row.get("caseType") or d.get("caseType") or "",
                    type_of_case=d.get("typeOfCase") or "",
                    operator_name=row.get("operatorName") or d.get("operatorName") or "Unknown",
                    operator_id=str(row.get("operatorId") or d.get("operatorId") or "") or None,
                    region=row.get("region") or d.get("region"),
                    opened_dt=row.get("openedDt") or d.get("openedDt"),
                    closed_dt=row.get("closedDt") or d.get("closedDt"),
                    final_order_dt=d.get("finalOrderDt"),
                    proposed_penalties=_num(row.get("proposedPenalties")),
                    assessed_penalties=_num(row.get("assessedPenalties")),
                    collected=_num(row.get("collected")),
                    violation_desc=_clean(d.get("violationDesc")),
                    proposed_subject=_clean(d.get("proposedSubjectEt")),
                    incident_detail=_clean(d.get("incidentDetail")),
                )
            )
            if (i + 1) % 25 == 0:
                log.info("phmsa.detail_progress", done=i + 1, total=len(selected))
    log.info("phmsa.cases_loaded", n=len(out))
    return out

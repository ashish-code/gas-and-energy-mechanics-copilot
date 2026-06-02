"""Warm up the demo before a live session.

Two modes:
  * local  : build the Copilot (loads the BM25 index, opens the DB connection, primes Bedrock)
             and run one cheap query so first-real-query latency is low.
  * url     : ping a deployed Streamlit URL to wake it from Streamlit Cloud cold-sleep.

  uv run python scripts/warm_up_demo.py                       # local warm-up
  uv run python scripts/warm_up_demo.py --url https://...     # wake a deployed app

Run ~5 minutes before the interview.
"""

from __future__ import annotations

import argparse

from src.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def warm_local() -> None:
    from src.agents.graph import Copilot

    copilot = Copilot()  # loads BM25 + opens DB + Bedrock clients
    state = copilot.ask("What does 49 CFR 192.505 require for strength testing?")
    answer = state.get("answer")
    log.info("warmup.local_done", refused=getattr(answer, "refused", None),
             claims=len(getattr(answer, "claims", []) or []))


def warm_url(url: str) -> None:
    import httpx

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        r = client.get(url)
        log.info("warmup.url_done", url=url, status=r.status_code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="deployed Streamlit URL to wake (otherwise local warm-up)")
    args = ap.parse_args()
    configure_logging()
    if args.url:
        warm_url(args.url)
    else:
        warm_local()


if __name__ == "__main__":
    main()

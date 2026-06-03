"""Run the full evaluation and write eval_report.md.

  uv run python scripts/run_eval.py                 # hand-curated 10 + ragas_set.json if present
  uv run python scripts/run_eval.py --hand-only     # skip the generated set
  uv run python scripts/run_eval.py --limit 5       # quick subset (latency is ~1 req/s)

Accepts defaults; does not tune. Writes per-set RAGAS scores + a per-tag breakdown for the
hand-curated multi-hop set.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.agents.graph import Copilot
from src.eval.hand_curated import HAND_CURATED_MULTI_HOP_QUESTIONS
from src.eval.run_metrics import build_samples, evaluate_samples
from src.logging_setup import bind_correlation_id, configure_logging, get_logger
from src.observability.langsmith_setup import setup_langsmith

log = get_logger(__name__)
REPORT = Path("eval_report.md")
RAGAS_SET = Path("data/golden_set/ragas_set.json")


def _fmt(scores: dict[str, float]) -> str:
    return "\n".join(f"| {k} | {v:.3f} |" for k, v in sorted(scores.items())) or "| (none) | — |"


def _native_table(native: list[dict]) -> list[str]:
    rows = ["| id | refused | sub-qs | evidence | verified | unsupported |", "|---|---|---|---|---|---|"]
    for n in native:
        rows.append(
            f"| {n['id']} | {'yes' if n['refused'] else 'no'} | {n['sub_questions']} | "
            f"{n['evidence']} | {n['verified_claims']} | {n['unsupported_claims']} |"
        )
    tot_v = sum(n["verified_claims"] for n in native)
    tot_u = sum(n["unsupported_claims"] for n in native)
    rate = tot_v / (tot_v + tot_u) if (tot_v + tot_u) else 0.0
    rows += ["", f"**Verified-claim rate: {tot_v}/{tot_v + tot_u} = {rate:.2f}** "
             f"(unsupported claims surfaced, not dropped)."]
    return rows


def _score_safely(samples: list[dict], *, with_reference: bool) -> dict[str, float]:
    """RAGAS scoring is token-heavy and can fail (truncation/throttle); never sink the report."""
    try:
        return evaluate_samples(samples, with_reference=with_reference)
    except Exception as e:  # noqa: BLE001
        log.warning("eval.ragas_failed", error=str(e))
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-only", action="store_true")
    ap.add_argument("--no-ragas", action="store_true", help="native verification stats only (skip RAGAS)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    configure_logging()
    bind_correlation_id()
    setup_langsmith()
    copilot = Copilot()

    lines = ["# Evaluation report", ""]

    # --- Hand-curated multi-hop set: agent runs (native stats) + optional RAGAS ---
    hand = HAND_CURATED_MULTI_HOP_QUESTIONS[: args.limit] if args.limit else HAND_CURATED_MULTI_HOP_QUESTIONS
    log.info("eval.hand_curated.start", n=len(hand))
    hand_samples, hand_native = build_samples(copilot, [{"id": q["id"], "question": q["question"]} for q in hand])

    lines += ["## Hand-curated multi-hop set", "", f"{len(hand)} questions.", ""]
    lines += ["### Native per-claim verification", ""] + _native_table(hand_native) + [""]
    if not args.no_ragas:
        hand_scores = _score_safely(hand_samples, with_reference=False)
        lines += ["### RAGAS", "", "| metric | score |", "|---|---|", _fmt(hand_scores), ""]

    by_tag: dict[str, int] = defaultdict(int)
    for q in hand:
        for t in q["tags"]:
            by_tag[t] += 1
    lines += ["### Question coverage by tag", "", "| tag | count |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in sorted(by_tag.items())] + [""]

    # --- Generated RAGAS set (with references) ---
    if not args.hand_only and not args.no_ragas and RAGAS_SET.exists():
        gen = json.loads(RAGAS_SET.read_text())
        gen = gen[: args.limit] if args.limit else gen
        log.info("eval.ragas_set.start", n=len(gen))
        gen_samples, _ = build_samples(copilot, gen)
        lines += ["## Generated test set (RAGAS, with references)", "", f"{len(gen)} questions.", ""]
        lines += ["| metric | score |", "|---|---|", _fmt(_score_safely(gen_samples, with_reference=True)), ""]
    else:
        lines += ["## Generated test set", "", "_Not run (no ragas_set.json, or --hand-only/--no-ragas)._", ""]

    REPORT.write_text("\n".join(lines))
    log.info("eval.done", report=str(REPORT))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()

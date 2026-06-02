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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    configure_logging()
    bind_correlation_id()
    setup_langsmith()
    copilot = Copilot()

    lines = ["# Evaluation report", ""]

    # --- Hand-curated multi-hop (10) ---
    hand = HAND_CURATED_MULTI_HOP_QUESTIONS[: args.limit] if args.limit else HAND_CURATED_MULTI_HOP_QUESTIONS
    log.info("eval.hand_curated.start", n=len(hand))
    hand_samples = build_samples(copilot, [{"question": q["question"]} for q in hand])
    hand_scores = evaluate_samples(hand_samples, with_reference=False)
    lines += ["## Hand-curated multi-hop set", "", f"{len(hand)} questions (no reference answers).", ""]
    lines += ["| metric | score |", "|---|---|", _fmt(hand_scores), ""]

    # Per-tag refusal/answer visibility (the demo signal).
    by_tag: dict[str, int] = defaultdict(int)
    for q in hand:
        for t in q["tags"]:
            by_tag[t] += 1
    lines += ["### Question coverage by tag", "", "| tag | count |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in sorted(by_tag.items())] + [""]

    # --- Generated RAGAS set (with references) ---
    if not args.hand_only and RAGAS_SET.exists():
        gen = json.loads(RAGAS_SET.read_text())
        gen = gen[: args.limit] if args.limit else gen
        log.info("eval.ragas_set.start", n=len(gen))
        gen_samples = build_samples(copilot, gen)
        gen_scores = evaluate_samples(gen_samples, with_reference=True)
        lines += ["## Generated test set (RAGAS, with references)", "", f"{len(gen)} questions.", ""]
        lines += ["| metric | score |", "|---|---|", _fmt(gen_scores), ""]
    else:
        lines += ["## Generated test set", "", "_Not run (no ragas_set.json; run generate_testset.py first)._", ""]

    REPORT.write_text("\n".join(lines))
    log.info("eval.done", report=str(REPORT))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()

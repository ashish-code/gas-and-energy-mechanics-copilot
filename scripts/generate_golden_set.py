"""Generate the RAGAS golden set (thin CLI over src.eval.generate_testset).

  uv run python scripts/generate_golden_set.py --size 40
"""

from __future__ import annotations

import argparse

from src.eval.generate_testset import generate
from src.logging_setup import bind_correlation_id, configure_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20, help="number of questions (40 to match the spec; slow)")
    args = ap.parse_args()
    configure_logging()
    bind_correlation_id()
    records = generate(args.size)
    print(f"Generated {len(records)} questions -> data/golden_set/ragas_set.json")


if __name__ == "__main__":
    main()

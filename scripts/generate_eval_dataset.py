"""
Script: scripts/generate_eval_dataset.py

PURPOSE:
    Command-line script to generate the synthetic QA evaluation dataset using RAGAS.
    Run this once after building the FAISS index (scripts/build_index.py).
    Re-run whenever the index is rebuilt with new regulatory content.

PREREQUISITES:
    1. FAISS index built: uv run python scripts/build_index.py
    2. AWS credentials configured: export AWS_PROFILE=your-profile
    3. Evaluation dependencies installed: uv sync (ragas, langchain-aws in pyproject.toml)

USAGE:
    # Generate 300 QA pairs (recommended for full evaluation):
    uv run python scripts/generate_eval_dataset.py --n 300

    # Quick test run with 20 pairs (fast, for validating the setup):
    uv run python scripts/generate_eval_dataset.py --n 20 --output evaluation/data/test_qa.json

    # Use a different Bedrock model:
    MODEL=bedrock/amazon.nova-lite-v1:0 uv run python scripts/generate_eval_dataset.py --n 50

OUTPUT:
    evaluation/data/synthetic_qa.json — array of QA pair objects:
    [
      {
        "question": "What is the maximum allowable operating pressure...",
        "ground_truth": "Under 49 CFR §192.619, the MAOP for steel pipelines...",
        "contexts": ["§192.619 states that...", "§192.621 further specifies..."],
        "evolution_type": "regulatory_lookup",
        "metadata": {"generated_by": "ragas", ...}
      },
      ...
    ]

TIME AND COST ESTIMATE:
    300 questions × ~3 LLM calls each = ~900 Bedrock calls
    At $0.003/1K tokens × ~500 tokens/call = ~$1.35 total cost
    Time: approximately 15-25 minutes (rate-limited by Bedrock API)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA evaluation dataset using RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="Number of QA pairs to generate (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/synthetic_qa.json"),
        help="Output path for the generated dataset (default: evaluation/data/synthetic_qa.json)",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/rag_index/chunks.parquet"),
        help="Path to the FAISS index chunks.parquet file (default: data/rag_index/chunks.parquet)",
    )
    parser.add_argument(
        "--max-source-chunks",
        type=int,
        default=500,
        help="Max chunks to sample for generation (default: 500; higher = more diverse but slower)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Gas & Energy Mechanics Copilot — Evaluation Dataset Generator")
    print("=" * 70)
    print(f"  Questions to generate: {args.n}")
    print(f"  Source chunks:         {args.chunks}")
    print(f"  Output path:           {args.output}")
    print(f"  Max source chunks:     {args.max_source_chunks}")
    print()
    print("This will make ~{:,} Bedrock API calls. Est. cost: ~${:.2f}".format(
        args.n * 3, args.n * 3 * 0.003 * 0.5  # rough estimate
    ))
    print("Press Ctrl+C to cancel.")
    print()

    try:
        from evaluation.dataset_generator import RegulatoryQADatasetGenerator

        generator = RegulatoryQADatasetGenerator(chunks_path=args.chunks)

        print(f"Generating {args.n} QA pairs...")
        dataset = generator.generate(
            n_questions=args.n,
            max_source_chunks=args.max_source_chunks,
        )

        print(f"\nGenerated {len(dataset)} QA pairs.")
        print("\nSample questions:")
        for item in dataset[:3]:
            print(f"  [{item.get('evolution_type', 'simple')}] {item['question'][:100]}")

        generator.save(dataset, args.output)

        print()
        print("=" * 70)
        print(f"✓ Dataset saved: {args.output}")
        print(f"  Total pairs:   {len(dataset)}")
        types = {}
        for item in dataset:
            t = item.get("evolution_type", "simple")
            types[t] = types.get(t, 0) + 1
        for t, count in sorted(types.items()):
            print(f"  {t}: {count} ({count / len(dataset) * 100:.0f}%)")
        print()
        print("Next steps:")
        print("  Run CI evaluation: uv run pytest tests/test_eval_metrics.py -m eval -v")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nHave you built the FAISS index? Run:")
        print("  uv run python scripts/build_index.py")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nInstall evaluation dependencies:")
        print("  uv add ragas langchain-core langchain-aws")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDataset generation cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()

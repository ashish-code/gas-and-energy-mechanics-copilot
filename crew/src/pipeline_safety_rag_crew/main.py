"""
Pipeline Safety RAG Crew — CLI entry point.

Run from the crew/ directory:

    # Default question
    uv run run_crew

    # Custom question
    uv run run_crew --question "What cathodic protection rules apply under §192.461?"

    # Or via Python module
    uv run python -m pipeline_safety_rag_crew.main -q "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Load .env if present (development convenience — install python-dotenv if needed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew

Path("output").mkdir(exist_ok=True)

DEFAULT_QUESTION = (
    "What are the pressure testing requirements for steel pipelines "
    "under 49 CFR §192.505?"
)


def run(question: str = DEFAULT_QUESTION) -> str:
    """Kick off the crew and return the final answer as a string."""
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70 + "\n")

    result = PipelineSafetyRAGCrew().crew().kickoff(inputs={"question": question})

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.raw)
    print("\nSaved → output/answer.md")
    return result.raw


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline Safety RAG Crew — ask questions about PHMSA regulations"
    )
    parser.add_argument(
        "-q", "--question",
        default=DEFAULT_QUESTION,
        help="Question to answer",
    )
    args = parser.parse_args()
    run(args.question)


if __name__ == "__main__":
    run_cli()

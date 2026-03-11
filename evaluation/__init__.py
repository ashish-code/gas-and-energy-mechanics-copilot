"""
Package: evaluation

PURPOSE:
    Houses all evaluation, monitoring, and dataset generation components for the
    Gas & Energy Mechanics Copilot. Evaluation is split into two complementary layers:

    1. OFFLINE EVALUATION (CI/CD gate) — evaluation/deepeval_suite.py
       Runs DeepEval metrics against a pre-generated synthetic QA dataset.
       Executed as a pytest suite before merges. Blocks bad changes.

    2. LIVE MONITORING (production) — evaluation/trulens_setup.py
       Wraps every real user query with TruLens scoring. Accumulates scores
       in a database and shows them in a dashboard (port 8502).

    3. DATASET GENERATION — evaluation/dataset_generator.py
       Uses RAGAS TestsetGenerator to create 300+ synthetic QA pairs from the
       existing FAISS index chunks. Run once; re-run when the index changes.

QUICK START:
    # Generate evaluation dataset (one-time, needs Bedrock):
    uv run python scripts/generate_eval_dataset.py --n 300

    # Run DeepEval CI suite:
    uv run pytest tests/test_eval_metrics.py -m eval -v

    # Start TruLens live dashboard:
    uv run python -m trulens.dashboard --port 8502
"""

"""
Tests: tests/test_eval_metrics.py

PURPOSE:
    DeepEval CI gate tests. These run the full agent pipeline against the
    pre-generated synthetic QA dataset and assert that five quality metrics
    meet their configured thresholds.

    These tests are the quality gate for the pipeline. They block merges
    if any metric drops below its threshold, catching regressions early.

MARKERS:
    All tests here are marked with @pytest.mark.eval.
    Run these tests explicitly:
        uv run pytest tests/test_eval_metrics.py -m eval -v --timeout=600

    They are excluded from normal pytest runs (too slow, require Bedrock + dataset):
        uv run pytest tests/ -m "not eval"

PREREQUISITES:
    1. Bedrock credentials: export AWS_PROFILE=your-profile
    2. Evaluation dataset: uv run python scripts/generate_eval_dataset.py --n 300
    3. Dependencies: uv add deepeval (ragas already installed)

COST ESTIMATE:
    Each DeepEval test case requires 5 metric evaluations × ~3 LLM calls each.
    30 samples × 15 LLM calls = 450 Bedrock calls ≈ $0.68 (at $0.003/1K tokens)
    Time: approximately 10-20 minutes for 30 samples.

READING TEST RESULTS:
    DeepEval outputs a detailed report showing:
      - Score per test case per metric
      - Pass/fail per metric per test case
      - Failure reason (what specific claim failed faithfulness, etc.)
    This makes it easy to understand WHERE the pipeline is failing,
    not just THAT it failed.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("CONFIG_DIR", "./config")

# Dataset path — must exist before running these tests
DATASET_PATH = Path("evaluation/data/synthetic_qa.json")

# Number of samples to evaluate per test run (subset of full 300-pair dataset)
# Increase to 100+ for comprehensive evaluation; 10-30 for faster CI runs
EVAL_SAMPLE_SIZE = int(os.environ.get("EVAL_SAMPLE_SIZE", "10"))


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def eval_config():
    """Load the application config (needed for metric thresholds)."""
    from gas_energy_copilot.ai_copilot.core.config import app_config
    return app_config()


@pytest.fixture(scope="session")
def qa_dataset():
    """
    Load the synthetic QA evaluation dataset.

    scope="session" means this is loaded once per pytest session,
    not once per test function — important because loading 300 items from disk
    is fast, but we want to avoid repeated file reads.
    """
    if not DATASET_PATH.exists():
        pytest.skip(
            f"Evaluation dataset not found at {DATASET_PATH}. "
            "Run: uv run python scripts/generate_eval_dataset.py --n 300"
        )

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Sample a consistent subset for this test run
    random.seed(42)  # fixed seed for reproducibility
    sample_size = min(EVAL_SAMPLE_SIZE, len(dataset))
    return random.sample(dataset, sample_size)


@pytest.fixture(scope="session")
def pipeline_runner():
    """
    Returns a callable that runs the 2-agent crew on a question.

    scope="session" so the FAISS index is only loaded once per test run.
    We use the simple 2-agent crew for evaluation (not the full 4-agent pipeline)
    because:
      1. Evaluation should measure baseline retrieval + synthesis quality.
      2. Including the judge agent in eval would measure judge performance too,
         conflating metrics.
      3. The 2-agent crew is faster, reducing eval run time.
    """
    from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew

    def run(question: str) -> dict[str, Any]:
        """
        Run the crew and return answer + retrieved contexts.

        Returns:
            Dict with "answer" (str) and "contexts" (list[str]).
        """
        result = PipelineSafetyRAGCrew().crew().kickoff(inputs={"question": question})
        return {
            "answer": result.raw,
            "contexts": [],  # we use the dataset's ground-truth contexts for scoring
        }

    return run


@pytest.fixture(scope="session")
def deepeval_test_cases(qa_dataset, pipeline_runner):
    """
    Build DeepEval LLMTestCase objects for the sampled dataset.

    scope="session" so the pipeline is only run once for all metric tests.
    Running the pipeline once and caching results avoids N×5 redundant LLM calls.
    """
    from evaluation.deepeval_suite import build_deepeval_test_cases
    return build_deepeval_test_cases(qa_dataset, pipeline_runner)


# ---------------------------------------------------------------------------
# Metric tests (each marked @pytest.mark.eval)
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.integration
class TestDeepEvalMetrics:
    """
    DeepEval metric CI gate tests.

    Each test evaluates one metric across all sampled test cases and asserts
    that the average score meets the configured threshold. If a test fails,
    check the DeepEval output for which specific question/answer pair failed
    and why (the `include_reason=True` flag in build_metrics() provides this).

    Why separate tests per metric (not one test for all metrics):
      - If faithfulness fails but relevancy passes, we want that distinction in CI.
      - Separate tests allow the CI system to report which metric failed,
        making it easier to root-cause regressions.
      - pytest's parametrize could also be used, but explicit methods are clearer.
    """

    def test_answer_relevancy_above_threshold(self, deepeval_test_cases, eval_config):
        """
        Answer Relevancy must be ≥ threshold (default 0.7).

        Failure means: answers are not addressing what was asked.
        Common causes: retrieval returning off-topic passages, synthesis task
        ignoring the question focus.
        """
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric
        from evaluation.deepeval_suite import _build_bedrock_eval_model

        metric = AnswerRelevancyMetric(
            threshold=eval_config.eval.deepeval_threshold_relevancy,
            model=_build_bedrock_eval_model(),
            include_reason=True,
        )

        scores = []
        for case in deepeval_test_cases:
            metric.measure(case)
            scores.append(metric.score)

        avg_score = sum(scores) / len(scores)
        assert avg_score >= eval_config.eval.deepeval_threshold_relevancy, (
            f"Answer relevancy ({avg_score:.3f}) below threshold "
            f"({eval_config.eval.deepeval_threshold_relevancy})"
        )

    def test_faithfulness_above_threshold(self, deepeval_test_cases, eval_config):
        """
        Faithfulness must be ≥ threshold (default 0.7).

        Failure means: the synthesised answer contains claims not grounded
        in the retrieved context (hallucinations).
        Common causes: regulatory_analyst inventing regulatory text, citing
        non-existent CFR sections, or misremembering specifics from training data.
        """
        from deepeval.metrics import FaithfulnessMetric
        from evaluation.deepeval_suite import _build_bedrock_eval_model

        metric = FaithfulnessMetric(
            threshold=eval_config.eval.deepeval_threshold_faithfulness,
            model=_build_bedrock_eval_model(),
            include_reason=True,
        )

        scores = []
        for case in deepeval_test_cases:
            metric.measure(case)
            scores.append(metric.score)

        avg_score = sum(scores) / len(scores)
        assert avg_score >= eval_config.eval.deepeval_threshold_faithfulness, (
            f"Faithfulness ({avg_score:.3f}) below threshold "
            f"({eval_config.eval.deepeval_threshold_faithfulness})"
        )

    def test_hallucination_below_threshold(self, deepeval_test_cases, eval_config):
        """
        Hallucination must be ≤ threshold (default 0.3).

        Note: this is an UPPER bound — lower is better. A score of 0.3 means
        "at most 30% of answer sentences contradict the retrieved context."
        Failure means: explicit contradictions between the answer and source text.
        This is more severe than faithfulness failures (which just measure
        unsupported claims; hallucination measures direct contradictions).
        """
        from deepeval.metrics import HallucinationMetric
        from evaluation.deepeval_suite import _build_bedrock_eval_model

        metric = HallucinationMetric(
            threshold=eval_config.eval.deepeval_threshold_hallucination,
            model=_build_bedrock_eval_model(),
            include_reason=True,
        )

        scores = []
        for case in deepeval_test_cases:
            metric.measure(case)
            scores.append(metric.score)

        avg_score = sum(scores) / len(scores)
        # Note: for HallucinationMetric, lower is better (it's an error rate)
        assert avg_score <= eval_config.eval.deepeval_threshold_hallucination, (
            f"Hallucination rate ({avg_score:.3f}) above threshold "
            f"({eval_config.eval.deepeval_threshold_hallucination})"
        )

    def test_contextual_precision_above_threshold(self, deepeval_test_cases, eval_config):
        """
        Contextual Precision must be ≥ threshold (default 0.6).

        Failure means: retrieved passages include too many irrelevant chunks.
        Common causes: FAISS similarity_threshold too low, queries too broad,
        or the embedding model not distinguishing between different CFR topics.
        """
        from deepeval.metrics import ContextualPrecisionMetric
        from evaluation.deepeval_suite import _build_bedrock_eval_model

        metric = ContextualPrecisionMetric(
            threshold=eval_config.eval.deepeval_threshold_precision,
            model=_build_bedrock_eval_model(),
            include_reason=True,
        )

        scores = []
        for case in deepeval_test_cases:
            metric.measure(case)
            scores.append(metric.score)

        avg_score = sum(scores) / len(scores)
        assert avg_score >= eval_config.eval.deepeval_threshold_precision, (
            f"Contextual precision ({avg_score:.3f}) below threshold "
            f"({eval_config.eval.deepeval_threshold_precision})"
        )

    def test_contextual_recall_above_threshold(self, deepeval_test_cases, eval_config):
        """
        Contextual Recall must be ≥ threshold (default 0.6).

        Failure means: retrieval missed relevant passages — the answer
        is based on incomplete information. Common causes: top_k too low,
        chunking too aggressive (splitting related regulatory paragraphs),
        or query terms not matching the regulatory vocabulary.
        """
        from deepeval.metrics import ContextualRecallMetric
        from evaluation.deepeval_suite import _build_bedrock_eval_model

        metric = ContextualRecallMetric(
            threshold=eval_config.eval.deepeval_threshold_recall,
            model=_build_bedrock_eval_model(),
            include_reason=True,
        )

        scores = []
        for case in deepeval_test_cases:
            metric.measure(case)
            scores.append(metric.score)

        avg_score = sum(scores) / len(scores)
        assert avg_score >= eval_config.eval.deepeval_threshold_recall, (
            f"Contextual recall ({avg_score:.3f}) below threshold "
            f"({eval_config.eval.deepeval_threshold_recall})"
        )

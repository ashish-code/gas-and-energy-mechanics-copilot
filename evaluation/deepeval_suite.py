"""
Module: evaluation/deepeval_suite.py

PURPOSE:
    Defines the DeepEval evaluation suite for the Gas & Energy Mechanics Copilot.
    This module implements the CI/CD quality gate: a set of metrics that must pass
    before any code change can be merged. Think of it as "unit tests for LLM quality."

    The suite runs the full agent pipeline against a pre-generated synthetic QA
    dataset and asserts that five quality metrics meet their configured thresholds.

WHY TWO EVALUATION FRAMEWORKS (DeepEval + TruLens):
    DeepEval and TruLens serve complementary purposes:

    DeepEval (THIS FILE — offline CI gate):
      - Runs as a pytest suite triggered on PR/merge.
      - Evaluates the pipeline against a fixed dataset of known QA pairs.
      - Blocks merges if quality degrades below thresholds.
      - Best for: catching regressions, testing prompt changes, comparing agents.
      - Analogy: unit tests / integration tests for your LLM pipeline.

    TruLens (trulens_setup.py — live production monitoring):
      - Wraps every real user request in production.
      - Scores each response in real time and shows trends in a dashboard.
      - Does NOT block anything — it observes and alerts.
      - Best for: catching quality drift, monitoring real-world distribution shift.
      - Analogy: production monitoring / observability dashboard.

DEEPEVAL METRIC EXPLANATIONS:

    1. AnswerRelevancyMetric (threshold ≥ 0.7):
       Does the answer actually address what was asked?
       Computed by: LLM judges whether the answer is topically relevant to the question.
       Low score indicates: the pipeline retrieved off-topic passages and synthesised
       an answer that doesn't match the question intent.

    2. FaithfulnessMetric (threshold ≥ 0.7):
       Are all claims in the answer grounded in the retrieved context?
       Computed by: LLM extracts all "statements" from the answer, then checks each
       against the retrieved passages. Unfounded statements = low score.
       This is the PRIMARY hallucination proxy for RAG systems.
       Low score indicates: the analyst agent invented facts not in the retrieved text.

    3. ContextualPrecisionMetric (threshold ≥ 0.6):
       Are all retrieved passages relevant to the question?
       Computed by: For each retrieved chunk, LLM judges whether it is relevant.
       Score = fraction of relevant chunks out of total retrieved.
       Low score indicates: FAISS is returning irrelevant passages (retrieval noise).
       Why threshold is 0.6 (lower than others): FAISS is approximate; some noise
       is expected. A threshold of 0.6 allows 40% irrelevant chunks.

    4. ContextualRecallMetric (threshold ≥ 0.6):
       Did retrieval find all the relevant information?
       Computed by: LLM checks whether the ground_truth answer could be derived
       from the retrieved passages. If key information is missing = low recall.
       Low score indicates: FAISS missed relevant passages (query terms too narrow).

    5. HallucinationMetric (threshold ≤ 0.3):
       Does the answer contain explicit contradictions with the retrieved context?
       Computed by: LLM checks each sentence against retrieved passages for
       contradictions (not just absence — actual contradictions).
       Note: This is an UPPER bound (lower is better) — the threshold says
       "at most 30% of answer sentences can contradict context."
       Complements FaithfulnessMetric (which catches unsupported claims).

DEEPEVAL + PYTEST INTEGRATION:
    DeepEval provides a pytest plugin. When you run:
        uv run pytest tests/test_eval_metrics.py -m eval
    DeepEval intercepts the test results and logs them to the DeepEval dashboard
    (if DEEPEVAL_API_KEY is set) or to a local JSON file.

    The `@pytest.mark.eval` marker excludes these from normal `pytest` runs
    since they're slow (each test case requires multiple LLM calls).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Path to the generated QA dataset
DATASET_PATH = Path("evaluation/data/synthetic_qa.json")


def build_deepeval_test_cases(dataset: list[dict], pipeline_runner) -> list:
    """
    Convert the synthetic QA dataset into DeepEval LLMTestCase objects.

    Each LLMTestCase represents one evaluation scenario. It contains:
      - input:              The user's question (from the dataset).
      - actual_output:      The answer produced by the full pipeline (from pipeline_runner).
      - expected_output:    The ground-truth answer (from RAGAS generation).
      - retrieval_context:  The passages retrieved by the retrieval_specialist agent.

    The pipeline_runner is a callable that runs the crew and returns a dict:
        {"answer": str, "retrieved_contexts": list[str]}

    This function is called from tests/test_eval_metrics.py, not here directly,
    to keep evaluation logic separate from test fixtures.

    Args:
        dataset:        List of QA pair dicts from RegulatoryQADatasetGenerator.load().
        pipeline_runner: Callable(question: str) → {"answer": str, "contexts": list[str]}

    Returns:
        List of deepeval.test_case.LLMTestCase objects.

    Raises:
        ImportError: If deepeval is not installed.
    """
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError:
        raise ImportError("deepeval is required. Run: uv add deepeval")

    test_cases = []
    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        reference_contexts = item.get("contexts", [])

        log.info(f"Running pipeline for test case {i + 1}/{len(dataset)}: {question[:80]}...")
        try:
            pipeline_result = pipeline_runner(question)
            actual_output = pipeline_result.get("answer", "")
            retrieval_context = pipeline_result.get("contexts", reference_contexts)
        except Exception as e:
            log.warning(f"Pipeline failed for question {i}: {e}")
            actual_output = f"[Pipeline error: {e}]"
            retrieval_context = reference_contexts

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=ground_truth,
            retrieval_context=retrieval_context,
            # context is the same as retrieval_context for RAG systems
            # (contrast with non-RAG where context is the full document)
            context=retrieval_context,
        )
        test_cases.append(test_case)

    return test_cases


def build_metrics(config) -> list:
    """
    Build the list of DeepEval metrics using thresholds from the app config.

    Metrics are instantiated here (not at test collection time) to allow threshold
    values to be read from settings.toml rather than being hardcoded.

    WHY BEDROCK FOR METRIC EVALUATION:
        DeepEval metrics use an LLM to judge the quality of answers. We use the
        same Bedrock model as the agent pipeline:
          - Consistent evaluation: the "judge" model matches the generator model.
          - Cost: Bedrock pay-per-token is cheaper than OpenAI API for batch eval.
          - No additional API keys needed.

        Alternative: GPT-4 (OpenAI) as the metric evaluator. GPT-4 is considered
        a stronger "judge" for LLM evaluation but requires an OpenAI API key and
        has higher per-token cost. For regulatory content, Bedrock 120B is sufficient.

    Args:
        config: ApplicationConfig with eval threshold settings.

    Returns:
        List of deepeval metric objects ready for use in assert_test().
    """
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            HallucinationMetric,
        )
        from deepeval.models import DeepEvalBaseLLM
    except ImportError:
        raise ImportError("deepeval is required. Run: uv add deepeval")

    # Use the same Bedrock model as the agent pipeline for evaluation
    # DeepEval supports custom model classes via DeepEvalBaseLLM interface
    model = _build_bedrock_eval_model()

    return [
        AnswerRelevancyMetric(
            threshold=config.eval.deepeval_threshold_relevancy,
            model=model,
            include_reason=True,   # include_reason=True logs WHY the score was given
        ),
        FaithfulnessMetric(
            threshold=config.eval.deepeval_threshold_faithfulness,
            model=model,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=config.eval.deepeval_threshold_precision,
            model=model,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=config.eval.deepeval_threshold_recall,
            model=model,
            include_reason=True,
        ),
        HallucinationMetric(
            threshold=config.eval.deepeval_threshold_hallucination,
            model=model,
            include_reason=True,
        ),
    ]


def _build_bedrock_eval_model():
    """
    Build a DeepEval-compatible LLM wrapper for Amazon Bedrock.

    DeepEval's metrics require a model object that implements DeepEvalBaseLLM.
    By default, DeepEval uses OpenAI GPT-4. We override this with a custom
    Bedrock wrapper that uses LiteLLM (the same backend as CrewAI).

    IMPLEMENTATION APPROACH — LiteLLM wrapper:
        LiteLLM provides a unified interface to 100+ LLM providers including
        Bedrock. DeepEval accepts any callable that returns a string as a "model".
        We use LiteLLM's completion() function wrapped in a class.

    Alternative approaches:
        1. Set OPENAI_API_KEY and use OpenAI's GPT-4 — simpler but requires
           a separate API key and has higher cost.
        2. DeepEval's `model="gpt-4"` default — same as above.
        3. Local Ollama model — free but much weaker for evaluation quality.

    Returns:
        A DeepEvalBaseLLM-compatible model object using Bedrock.
    """
    try:
        from deepeval.models import DeepEvalBaseLLM
        import litellm
    except ImportError:
        raise ImportError("deepeval and litellm are required. Run: uv add deepeval litellm")

    model_id = os.environ.get("MODEL", "bedrock/openai.gpt-oss-120b-1:0")

    class BedrockEvalModel(DeepEvalBaseLLM):
        """
        Custom DeepEval model adapter for Amazon Bedrock via LiteLLM.

        DeepEvalBaseLLM requires two methods:
          - generate(prompt: str) → str
          - a_generate(prompt: str) → Coroutine[str]  (async version)

        We use LiteLLM's completion() for both (a_generate wraps generate).
        """

        def load_model(self):
            return model_id

        def generate(self, prompt: str) -> str:
            """Synchronous LLM call via LiteLLM → Bedrock."""
            response = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # deterministic evaluation — same input → same judgment
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""

        async def a_generate(self, prompt: str) -> str:
            """Async LLM call (DeepEval calls this in async evaluation runs)."""
            response = await litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""

        def get_model_name(self) -> str:
            return model_id

    return BedrockEvalModel()


def run_batch_evaluation(
    dataset: list[dict],
    pipeline_runner,
    config,
    output_path: Path | None = None,
) -> dict[str, float]:
    """
    Run the full DeepEval evaluation suite on a dataset and return aggregated scores.

    This is the programmatic API for running evaluation (as opposed to the pytest
    fixtures in tests/test_eval_metrics.py). Use this for ad-hoc evaluation runs,
    CI scripts that need metric values as numbers, or the /v1/eval/run API endpoint.

    Args:
        dataset:        QA pairs from RegulatoryQADatasetGenerator.load().
        pipeline_runner: Callable(question) → {"answer": str, "contexts": list[str]}
        config:         ApplicationConfig with eval settings.
        output_path:    Optional path to save results JSON.

    Returns:
        Dict mapping metric name → average score across all test cases.
        Example: {"answer_relevancy": 0.82, "faithfulness": 0.75, ...}
    """
    try:
        from deepeval import evaluate
    except ImportError:
        raise ImportError("deepeval is required. Run: uv add deepeval")

    test_cases = build_deepeval_test_cases(dataset, pipeline_runner)
    metrics = build_metrics(config)

    # evaluate() runs all metrics on all test cases in parallel (where possible)
    results = evaluate(test_cases=test_cases, metrics=metrics)

    # Aggregate scores per metric
    aggregated: dict[str, list[float]] = {}
    for test_result in results.test_results:
        for metric_data in test_result.metrics_data:
            name = metric_data.name
            score = metric_data.score
            aggregated.setdefault(name, []).append(score)

    summary = {name: sum(scores) / len(scores) for name, scores in aggregated.items()}

    if output_path:
        import json as json_module
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json_module.dump(summary, f, indent=2)
        log.info(f"Evaluation results saved to {output_path}")

    return summary

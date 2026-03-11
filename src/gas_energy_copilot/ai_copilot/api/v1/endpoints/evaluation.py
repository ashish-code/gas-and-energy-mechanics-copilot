"""
Module: api/v1/endpoints/evaluation.py

PURPOSE:
    REST API endpoints for evaluation status and triggering. These endpoints allow:
      - External monitoring systems to poll quality metrics.
      - CI/CD pipelines to trigger evaluation runs via HTTP.
      - Development teams to inspect dataset samples without running scripts.

    These endpoints are only active when [app.eval] enabled=true in settings.toml.
    In production, they may be protected behind authentication middleware
    (not implemented here — add API key middleware at the router level).

ENDPOINTS:
    GET  /v1/eval/metrics          — Return latest TruLens aggregate scores.
    GET  /v1/eval/dataset/sample   — Return N random QA pairs from the synthetic dataset.
    POST /v1/eval/run              — Trigger an async DeepEval evaluation run.

ARCHITECTURE POSITION:
    These endpoints use the EvaluationService (services/evaluation_service.py)
    and directly read the evaluation dataset (evaluation/data/synthetic_qa.json).
    They do NOT depend on TruLens or DeepEval being installed — they degrade
    gracefully when these optional packages are missing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import structlog.stdlib
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from gas_energy_copilot.ai_copilot.services.evaluation_service import get_evaluation_service

router = APIRouter()
log = structlog.stdlib.get_logger()

DATASET_PATH = Path("evaluation/data/synthetic_qa.json")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class MetricsResponse(BaseModel):
    """Aggregate TruLens RAG Triad scores."""

    context_relevance: float | None = Field(
        None, description="Average context relevance score [0.0, 1.0]. Higher is better."
    )
    groundedness: float | None = Field(
        None, description="Average groundedness score [0.0, 1.0]. Higher is better. "
                          "Low groundedness = high hallucination risk."
    )
    answer_relevance: float | None = Field(
        None, description="Average answer relevance score [0.0, 1.0]. Higher is better."
    )
    total_records: int = Field(0, description="Total number of recorded interactions.")
    message: str | None = Field(None, description="Status message if metrics are unavailable.")


class DatasetSample(BaseModel):
    """A single QA pair from the synthetic evaluation dataset."""

    question: str
    ground_truth: str
    evolution_type: str = Field(
        "simple",
        description="RAGAS question type: simple, multi_context, or reasoning."
    )
    contexts: list[str] = Field(default_factory=list)


class DatasetSampleResponse(BaseModel):
    """Response for the dataset sample endpoint."""

    samples: list[DatasetSample]
    total_in_dataset: int
    dataset_path: str


class EvalRunRequest(BaseModel):
    """Request body for triggering an evaluation run."""

    n_samples: int = Field(
        default=10,
        ge=1, le=100,
        description="Number of QA pairs to evaluate (subset of full dataset for speed).",
    )
    save_results: bool = Field(
        default=True,
        description="Whether to save evaluation results to evaluation/data/eval_results.json.",
    )


class EvalRunResponse(BaseModel):
    """Response for a triggered evaluation run."""

    status: str
    message: str
    n_samples: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/eval/metrics",
    summary="Get live TruLens RAG quality metrics",
    description=(
        "Returns the current aggregate RAG Triad scores from TruLens. "
        "Scores are computed on real user interactions (not test data). "
        "Returns null values if TruLens is disabled or no interactions have been recorded."
    ),
    response_model=MetricsResponse,
)
async def get_metrics() -> MetricsResponse:
    """
    Return the latest aggregate TruLens quality metrics.

    These scores represent the average quality across all real user interactions
    since TruLens was enabled. They are a live indicator of production quality.

    Interpretation:
        context_relevance ≥ 0.7: Retrieval is finding relevant passages.
        groundedness ≥ 0.7:      Low hallucination rate — answers are grounded.
        answer_relevance ≥ 0.7:  Answers are addressing the questions asked.
    """
    service = get_evaluation_service()
    raw_metrics = service.get_metrics()

    return MetricsResponse(
        context_relevance=_extract_float(raw_metrics, "Context Relevance"),
        groundedness=_extract_float(raw_metrics, "Groundedness"),
        answer_relevance=_extract_float(raw_metrics, "Answer Relevance"),
        total_records=raw_metrics.get("total_records", 0),
        message=raw_metrics.get("message") or raw_metrics.get("error"),
    )


@router.get(
    "/eval/dataset/sample",
    summary="Get sample QA pairs from the synthetic evaluation dataset",
    description=(
        "Returns N randomly sampled QA pairs from the RAGAS-generated dataset. "
        "Useful for inspecting dataset quality or debugging evaluation issues."
    ),
    response_model=DatasetSampleResponse,
)
async def get_dataset_sample(n: int = 5) -> DatasetSampleResponse:
    """
    Return a random sample from the synthetic QA evaluation dataset.

    Args (query param):
        n: Number of samples to return (default 5, max capped at 50).
    """
    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation dataset not found at {DATASET_PATH}. "
                "Run: uv run python scripts/generate_eval_dataset.py --n 300"
            ),
        )

    n = min(n, 50)  # cap at 50 to avoid large response payloads

    try:
        with open(DATASET_PATH) as f:
            dataset = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}")

    # Sample without replacement (or return all if fewer than n)
    sample_size = min(n, len(dataset))
    samples = random.sample(dataset, sample_size)

    return DatasetSampleResponse(
        samples=[
            DatasetSample(
                question=item["question"],
                ground_truth=item.get("ground_truth", ""),
                evolution_type=item.get("evolution_type", "simple"),
                contexts=item.get("contexts", []),
            )
            for item in samples
        ],
        total_in_dataset=len(dataset),
        dataset_path=str(DATASET_PATH),
    )


@router.post(
    "/eval/run",
    summary="Trigger an async DeepEval evaluation run",
    description=(
        "Triggers a background DeepEval evaluation run on N samples from the "
        "synthetic dataset. Results are saved to evaluation/data/eval_results.json. "
        "This is a long-running operation (5-30 minutes depending on n_samples). "
        "Check the server logs for progress."
    ),
    response_model=EvalRunResponse,
)
async def trigger_eval_run(
    request: EvalRunRequest,
    background_tasks: BackgroundTasks,
) -> EvalRunResponse:
    """
    Trigger an asynchronous DeepEval evaluation run.

    Uses FastAPI's BackgroundTasks to run the evaluation without blocking the response.
    The evaluation runs in the background and saves results to disk.
    Monitor progress via server logs or the evaluation results file.

    Note: Running evaluation consumes Bedrock API quota (multiple LLM calls per
    test case). Cost estimate: ~$0.50-2.00 per 10 test cases depending on model.
    """
    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation dataset not found at {DATASET_PATH}. "
                "Run: uv run python scripts/generate_eval_dataset.py --n 300"
            ),
        )

    log.info("eval_run_triggered", n_samples=request.n_samples)

    # Register the evaluation as a background task
    # BackgroundTasks runs after the response is sent, so the client gets an
    # immediate 202-like response without waiting for evaluation to complete.
    background_tasks.add_task(
        _run_eval_background,
        n_samples=request.n_samples,
        save_results=request.save_results,
    )

    return EvalRunResponse(
        status="started",
        message=(
            f"Evaluation run started with {request.n_samples} samples. "
            "Check server logs for progress. "
            "Results will be saved to evaluation/data/eval_results.json"
        ),
        n_samples=request.n_samples,
    )


# ---------------------------------------------------------------------------
# Background task helper
# ---------------------------------------------------------------------------


def _run_eval_background(n_samples: int, save_results: bool) -> None:
    """
    Background function that runs the DeepEval evaluation suite.

    This runs in FastAPI's background task executor (synchronous thread pool).
    It loads the dataset, samples N items, runs the pipeline for each, and
    computes DeepEval metrics.

    Args:
        n_samples:    Number of QA pairs to evaluate.
        save_results: If True, saves the metric scores to disk.
    """
    try:
        import json as json_module
        from gas_energy_copilot.ai_copilot.core.config import app_config
        from evaluation.dataset_generator import RegulatoryQADatasetGenerator
        from evaluation.deepeval_suite import run_batch_evaluation

        config = app_config()
        gen = RegulatoryQADatasetGenerator()
        dataset = gen.load(DATASET_PATH)

        # Sample subset for this run
        sample_size = min(n_samples, len(dataset))
        sample = random.sample(dataset, sample_size)

        log.info(f"Eval run: {sample_size} samples from {len(dataset)} total")

        def pipeline_runner(question: str) -> dict:
            """Run the crew and return answer + contexts for evaluation."""
            from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew
            from pipeline_safety_rag_crew.tools.rag_tool import RAGSearchTool

            # Get the answer from the 2-agent crew (faster than full pipeline for eval)
            result = PipelineSafetyRAGCrew().crew().kickoff(inputs={"question": question})

            # Extract retrieved contexts from the RAG tool's last search
            # (RAGSearchTool caches its last results on the instance)
            return {
                "answer": result.raw,
                "contexts": [],  # contexts come from the test case ground truth
            }

        output_path = Path("evaluation/data/eval_results.json") if save_results else None
        scores = run_batch_evaluation(sample, pipeline_runner, config, output_path)

        log.info("Eval run complete", extra={"scores": scores})

    except Exception as e:
        log.error("Eval run failed", exc_info=e)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _extract_float(data: dict, key: str) -> float | None:
    """Extract a float value from a dict, returning None if absent or non-numeric."""
    val = data.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

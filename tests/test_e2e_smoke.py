"""End-to-end smoke test skeleton.

M1: assert config + logging import and basic invariants hold (no network, no AWS).
Later milestones fill in the live end-to-end path (build a tiny index -> query the
agent graph) behind the `live` marker so CI can run the offline subset.
"""

from __future__ import annotations

import pytest


def test_config_imports_and_has_resolved_model_ids() -> None:
    """src.config loads and carries the M0-resolved, inference-profile model IDs."""
    from src.config import settings

    # Claude 4.x on Bedrock is inference-profile only: IDs must be the `us.` form.
    assert settings.model_planner.startswith("us.anthropic.claude-sonnet")
    assert settings.model_synthesizer == settings.model_planner
    assert settings.model_executor.startswith("us.anthropic.claude-haiku")
    assert settings.model_verifier == settings.model_executor
    assert settings.model_embedding == "amazon.titan-embed-text-v2:0"
    assert settings.embedding_dim == 1024
    assert settings.model_reranker == "cohere.rerank-v3-5:0"
    assert settings.aws_region == "us-east-1"


def test_retrieval_defaults_match_spec() -> None:
    """The retrieval pipeline knobs match the architecture defaults (not tuned)."""
    from src.config import settings

    assert settings.dense_top_k == 30
    assert settings.sparse_top_k == 30
    assert settings.rrf_k == 60
    assert settings.mmr_lambda == 0.6
    assert settings.rerank_top_k == 10


def test_logging_configures_and_binds_correlation_id() -> None:
    """structlog configures idempotently and binds a correlation id."""
    from src.logging_setup import bind_correlation_id, clear_context, configure_logging, get_logger

    configure_logging(json_logs=True, level="INFO")
    cid = bind_correlation_id()
    assert isinstance(cid, str) and len(cid) >= 8
    log = get_logger(__name__)
    log.info("smoke", phase="m1")
    clear_context()


@pytest.mark.skip(reason="Live end-to-end wired in M4 once the agent graph exists.")
def test_agent_graph_end_to_end_live() -> None:  # pragma: no cover
    raise NotImplementedError

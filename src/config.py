"""Typed application configuration (pydantic-settings).

Single source of truth for every tunable and credential. Reads from environment
variables and an optional ``.env`` file. On Streamlit Community Cloud, the app
bridges ``st.secrets`` into ``os.environ`` at startup (see ``app/streamlit_app.py``),
so the same field names work in both places.

Resolved Bedrock model IDs (verified live in us-east-1 at M0, 2026-06-02) are baked
in as defaults. Both Claude 4.x models are **inference-profile only** on Bedrock — the
``us.`` cross-region inference-profile IDs are required; the bare ``anthropic.*`` IDs
return a validation error.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from env / .env / Streamlit secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- AWS / Bedrock ---------------------------------------------------------
    aws_region: str = "us-east-1"
    # Local dev uses a named profile; Streamlit Cloud uses static keys (least-priv IAM user).
    aws_profile: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None

    # --- Resolved Bedrock model IDs (M0-verified, us-east-1) --------------------
    # Reasoning tier (planner + synthesizer): Sonnet 4.6.
    model_planner: str = "us.anthropic.claude-sonnet-4-6"
    model_synthesizer: str = "us.anthropic.claude-sonnet-4-6"
    # Orchestration tier (executor query-gen + per-claim verifier): Haiku 4.5.
    model_executor: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    model_verifier: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # Contextual-Retrieval blurb generator runs on the cheap tier too.
    model_contextualizer: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Embeddings: Titan V2, 1024D, L2-normalized.
    model_embedding: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 1024

    # Reranker: Cohere Rerank 3.5 via the Bedrock Rerank API (available in us-east-1).
    model_reranker: str = "cohere.rerank-v3-5:0"

    # --- Supabase Postgres (pgvector) ------------------------------------------
    # Full Postgres connection string, e.g. the Supabase pooler URL:
    #   postgresql://postgres.<ref>:<pw>@aws-0-us-east-1.pooler.supabase.com:6543/postgres
    supabase_db_url: str | None = None

    # --- LangSmith (env-gated; no-op when key unset) ---------------------------
    langsmith_api_key: str | None = None
    langsmith_project: str = "gas-energy-copilot-v2"
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Retrieval pipeline (architectural defaults from the spec; not tuned) ---
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    fused_top_k: int = 50
    mmr_top_k: int = 20
    mmr_lambda: float = 0.6
    rerank_top_k: int = 10
    parent_sections_target: int = 7  # unique parent sections returned per sub-question

    # --- Agent planner ---------------------------------------------------------
    max_sub_questions: int = 5

    # --- Logging ---------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = Field(default=True, description="JSON logs in prod; pretty when False.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()


# Module-level singleton for ergonomic imports: ``from src.config import settings``.
settings = get_settings()

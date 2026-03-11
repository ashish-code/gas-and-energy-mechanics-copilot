"""
Module: config.py

PURPOSE:
    Central configuration management for the Gas & Energy Mechanics Copilot.
    All tunable parameters live here — the application never reads environment
    variables or TOML files directly; it always goes through this module.

ARCHITECTURE POSITION:
    Every layer (API, services, agents, tools) imports `app_config()` to get
    its settings. This is the single source of truth for runtime behaviour.

    CONFIG_DIR env var → settings.toml → typed_settings → ApplicationConfig
                                                               │
                    ┌──────────────┬──────────┬───────────────┼───────────────┬──────────┐
                    ▼              ▼          ▼               ▼               ▼          ▼
              AgentSettings   ApiSettings  RAGSettings  MemorySettings  LangfuseSettings EvalSettings

LIBRARY CHOICE — typed_settings + attrs:
    Why typed_settings:
      - Type-safe TOML loading with validation at startup (fail fast, not at runtime).
      - The @ts.settings decorator automatically maps TOML keys to Python attributes.
      - Supports nested settings via class composition (each sub-class is a TOML table).
    Why attrs:
      - Lighter than dataclasses for nested config; @define generates __init__,
        __repr__, __eq__ automatically.
      - Works seamlessly with typed_settings as the underlying data model.
    Alternatives considered:
      - Pydantic BaseSettings: excellent but heavier dependency; better for runtime
        validation of user input, not static config files.
      - dynaconf: powerful layered config but complex API for a TOML-only project.
      - python-decouple: simpler but no nested structure or type coercion.

KEY CONCEPTS:
    - All settings are immutable after load (attrs frozen is not set here to allow
      test overrides, but the singleton pattern means they are effectively immutable
      in production).
    - The `app_config()` singleton is constructed once on first call and cached.
      This is safe for a single-process server like uvicorn.
"""

import os
from pathlib import Path

from attrs import define
import typed_settings as ts


# ---------------------------------------------------------------------------
# Sub-settings: each maps to a [app.<section>] block in settings.toml
# ---------------------------------------------------------------------------


@define
@ts.settings
class ApiSettings:
    """
    HTTP API configuration — docs visibility, URL prefix, versioning.

    serve_docs_enabled: When True, Swagger UI (/docs) and ReDoc (/redoc) are mounted.
        Disable in production to avoid leaking API schema publicly.
    prefix: URL prefix for all API routes (e.g., "/api" → /api/v1/chat).
    """

    serve_docs_enabled: bool
    root_message: str
    prefix: str
    api_version: str


@define
@ts.settings
class CORSSettings:
    """
    Cross-Origin Resource Sharing policy.

    In development, localhost origins are allowed.
    In production, restrict `origins` to the actual frontend hostname.

    Why CORS matters for LLM APIs: browsers will reject JavaScript fetch() calls
    to your API unless the API explicitly allows the frontend's origin.
    """

    origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


@define
@ts.settings
class LoggingSettings:
    """
    Structured logging configuration.

    log_json: When True, emits newline-delimited JSON (ideal for CloudWatch/Datadog).
        When False, emits human-readable colored output (ideal for local dev).
    log_access_excluded_path_prefixes: Paths to exclude from access logs
        (e.g., ["/health"] to suppress noisy health-check polling).
    log_access_exclude_success_only: If True, only suppress successful responses
        on excluded paths — errors (4xx/5xx) on those paths are still logged.
    """

    log_level: str = "INFO"
    log_json: bool = False
    log_access_excluded_path_prefixes: list[str] = []
    log_access_exclude_success_only: bool = True


@define
@ts.settings
class AgentSettings:
    """
    LLM agent configuration — which Bedrock model to use and the base system prompt.

    bedrock_model_id: The Bedrock model identifier passed to boto3. This is separate
        from the MODEL env var used by CrewAI/LiteLLM; this field is for the
        bare Bedrock embedding calls in the RAG tool.
    system_prompt: Baseline instruction injected into every conversation. Agents
        may override this with task-specific backstory/goal from agents.yaml.
    """

    name: str = ""
    description: str = ""
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    system_prompt: str = ""


@define
@ts.settings
class RAGSettings:
    """
    Retrieval-Augmented Generation (RAG) pipeline configuration.

    index_dir: Path to the pre-built FAISS index directory (relative to CWD).
        Contains: index.faiss, chunks.parquet, meta.json
    top_k: Number of passages to retrieve per query. Higher = more context but
        more tokens consumed by the LLM. 5 is a good default for 400-word chunks.
    similarity_threshold: Minimum cosine similarity score [0.0, 1.0] to include
        a retrieved passage. 0.0 = return all results regardless of relevance.
        Raise to 0.3-0.5 if you see too many irrelevant chunks in answers.
    embedding_model: Amazon Bedrock Titan Embeddings V2 model ID. Must match the
        model used during index build (build_index.py) — mixing models corrupts results.
    """

    enabled: bool = True
    index_dir: str = "data/rag_index"
    top_k: int = 5
    embedding_region: str = "us-east-1"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    similarity_threshold: float = 0.0


@define
@ts.settings
class MemorySettings:
    """
    Multi-turn conversation memory backed by Amazon DynamoDB.

    PURPOSE:
        Without memory, every query is stateless — the agent has no context from
        prior turns in the same chat session. Memory enables follow-up questions
        like "Can you clarify that last point?" or "Now apply that to Part 195."

    DynamoDB SCHEMA:
        Table name: <table_name>
        Primary key: session_id (String, Hash key) + turn_id (String, Range key)
        Attributes:
          - role: "user" | "assistant"
          - content: the message text
          - timestamp: ISO8601 string
          - metadata: JSON map (agent name, retrieval scores, etc.)
          - ttl: Unix epoch seconds (DynamoDB auto-expires items after this time)

        Why composite key (session_id + turn_id):
          - Enables efficient Query of all turns for a session in one API call.
          - turn_id is zero-padded ("0001") so lexicographic sort = chronological order.

    WHY DynamoDB over alternatives:
        - vs Redis: Redis is faster (~1ms vs ~5ms) but requires ElastiCache VPC setup.
          DynamoDB is serverless — zero infrastructure to manage.
        - vs PostgreSQL/RDS: RDS requires provisioning, VPC peering, connection pooling.
          Chat logs don't need relational queries; key-value is sufficient.
        - vs S3: S3 is append-only log storage. DynamoDB supports efficient Query by
          session_id, which is critical for loading conversation history per request.

    max_turns: Maximum number of prior turns to load as context. Loading the entire
        session history could exceed the LLM's context window. 20 turns ≈ 10 exchanges.
    ttl_days: After this many days of inactivity, DynamoDB auto-deletes the turns.
        This is a cost-control mechanism — old chat logs don't accumulate forever.
    """

    enabled: bool = True
    table_name: str = "gas-energy-copilot-conversations"
    region: str = "us-east-1"
    max_turns: int = 20
    ttl_days: int = 30


@define
@ts.settings
class LangfuseSettings:
    """
    Langfuse LLM observability configuration.

    PURPOSE:
        Langfuse gives us full visibility into every LLM call, agent step, and
        tool use. Each user query becomes a "trace" with nested "spans" for
        routing, retrieval, synthesis, and judging. Quality scores from TruLens/
        DeepEval are attached to traces so you can correlate low scores with
        specific inputs.

    HOW IT INTEGRATES:
        Langfuse integrates with LiteLLM (which CrewAI uses internally) via
        environment variables. Setting LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
        causes LiteLLM to automatically send all LLM calls to Langfuse — no
        code changes to CrewAI required.

    WHY Langfuse over alternatives:
        - vs Arize Phoenix: Phoenix excels at embedding drift and ML monitoring.
          Langfuse excels at conversation-level tracing and RAG span hierarchy.
          We use Langfuse for traces + Phoenix could be added for embeddings later.
        - vs LangSmith: LangSmith requires LangChain; Langfuse is framework-agnostic.
        - vs OpenTelemetry: OTel is lower-level infrastructure. Langfuse adds
          LLM-specific semantics: prompt/completion tracking, token cost estimation,
          quality scores, and session grouping.
        - vs Weights & Biases (Weave): W&B Weave is strong for ML experiments
          but heavier than needed for production API tracing.

    enabled: Set to True in production with valid keys. False = no-op (no overhead).
    public_key: From Langfuse project settings. Safe to embed in config (not secret).
    secret_key: Keep in environment variable LANGFUSE_SECRET_KEY, not in TOML.
    host: Default is Langfuse Cloud. Change to your self-hosted URL if applicable.
    """

    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    project_name: str = "gas-energy-copilot"
    flush_at: int = 15   # number of events to buffer before flushing to Langfuse
    flush_interval: float = 0.5  # seconds between flushes


@define
@ts.settings
class EvalSettings:
    """
    Evaluation framework configuration for TruLens and DeepEval.

    PURPOSE:
        Evaluation is how we know if the system is getting better or worse over
        time. Two complementary tools are used:

        1. TruLens (live monitoring):
           Wraps the pipeline to score every real user interaction using the
           RAG Triad (context relevance, groundedness, answer relevance).
           Scores accumulate in a local SQLite DB and are viewable in a dashboard.

        2. DeepEval (CI/CD gate):
           Run as a pytest suite against a synthetic QA dataset. Blocks merges
           if metric thresholds are not met. Think of it as unit tests for LLM quality.

    THRESHOLDS:
        These values represent acceptable quality floors. Adjust after baselining
        your system — start lower and tighten as the system matures.

        faithfulness ≥ 0.7: At most 30% of answer claims can be unsupported by context.
        relevancy ≥ 0.7:    At least 70% of the answer must address the question.
        hallucination ≤ 0.3: At most 30% of answer sentences can contradict context.
        precision ≥ 0.6:    At least 60% of retrieved chunks must be relevant.
        recall ≥ 0.6:       Retrieval must find ≥60% of relevant chunks.

    trulens_db_url: SQLite in dev (file on disk), PostgreSQL URI in prod.
        Format: "sqlite:///path/to/db.sqlite" or "postgresql://user:pw@host/db"
    """

    enabled: bool = False
    trulens_db_url: str = "sqlite:///evaluation/trulens.db"

    # DeepEval metric thresholds (used in tests/test_eval_metrics.py)
    deepeval_threshold_faithfulness: float = 0.7
    deepeval_threshold_relevancy: float = 0.7
    deepeval_threshold_hallucination: float = 0.3   # UPPER bound — lower is better
    deepeval_threshold_precision: float = 0.6
    deepeval_threshold_recall: float = 0.6

    # Number of QA pairs to sample per CI evaluation run (full dataset is 300+)
    eval_sample_size: int = 30


# ---------------------------------------------------------------------------
# Top-level config: composes all sub-settings
# ---------------------------------------------------------------------------


@define
class ApplicationConfig:
    """
    Combined application configuration — the single object that the entire app uses.

    Each sub-setting maps to a [app.<section>] block in settings.toml.
    Adding a new settings class here requires:
      1. Defining the class above with @define @ts.settings
      2. Adding the field here in ApplicationConfig
      3. Adding the [app.<new_section>] block to settings.toml

    The `app_config()` function below returns the singleton instance of this class.
    """

    app_name: str
    service_name: str
    port: int
    host: str
    environment: str
    debug: bool

    # Nested sub-settings (each is a @ts.settings class)
    agent: AgentSettings
    api: ApiSettings
    cors: CORSSettings
    logging: LoggingSettings
    rag: RAGSettings
    memory: MemorySettings
    langfuse: LangfuseSettings
    eval: EvalSettings


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

# Module-level cache for the config singleton.
# Using a plain global (not threading.local) is safe because:
# - uvicorn uses a single main thread for startup
# - all async handlers share the same event loop and see the same global
__APP_CONFIG_INSTANCE: ApplicationConfig | None = None


def app_config() -> ApplicationConfig:
    """
    Return the singleton ApplicationConfig instance.

    Reads CONFIG_DIR from the environment on first call, loads the TOML file,
    validates all fields, and caches the result. Subsequent calls return the
    cached instance without any I/O.

    Raises:
        RuntimeError: If CONFIG_DIR is not set.
        FileNotFoundError: If settings.toml does not exist in CONFIG_DIR.
        typed_settings.ConfigurationError: If TOML values fail type validation.

    Example:
        config = app_config()
        print(config.rag.top_k)       # 5
        print(config.memory.enabled)  # True
    """
    global __APP_CONFIG_INSTANCE

    config_dir_env = os.environ.get("CONFIG_DIR")
    if not config_dir_env:
        raise RuntimeError(
            "CONFIG_DIR environment variable is not set. "
            "Point it to a directory containing settings.toml. "
            "Example: export CONFIG_DIR=./config"
        )

    config_dir = Path(config_dir_env).expanduser().resolve(strict=True)
    # The "!" prefix tells typed_settings that this file is required (raises if missing)
    config_files = [f"!{config_dir}/settings.toml"]

    if __APP_CONFIG_INSTANCE is None:
        __APP_CONFIG_INSTANCE = ts.load(ApplicationConfig, "app", config_files=config_files)

    return __APP_CONFIG_INSTANCE

import os
from pathlib import Path

from attrs import define
import typed_settings as ts


@define
@ts.settings
class ApiSettings:
    """API-specific sub-settings."""

    serve_docs_enabled: bool
    root_message: str
    prefix: str
    api_version: str


@define
@ts.settings
class CORSSettings:
    """CORS sub-settings"""

    origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


@define
@ts.settings
class LoggingSettings:
    """Logging-specific sub-settings"""

    log_level: str = "INFO"
    strands_log_level: str = "WARNING"
    # Whether or not logs should be JSON-formatted strings
    log_json: bool = False

    # List of path prefixes to exclude from access logs (e.g., ["/health", "/metrics"])
    log_access_excluded_path_prefixes: list[str] = []

    # If True, only exclude successful requests (2xx, 3xx status codes)
    # If False, exclude ALL requests matching the path prefixes regardless of status
    log_access_exclude_success_only: bool = True


@define
@ts.settings
class AgentSettings:
    name: str = ""
    description: str = ""
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    system_prompt: str = ""


@define
@ts.settings
class RAGSettings:
    """RAG-specific settings for retrieval-augmented generation."""

    enabled: bool = True
    index_dir: str = "data/rag_index"
    top_k: int = 5
    embedding_region: str = "us-east-1"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    similarity_threshold: float = 0.0  # Minimum similarity score to include results


@define
class ApplicationConfig:
    """Combined application configuration."""

    app_name: str
    service_name: str
    port: int
    host: str
    environment: str
    debug: bool

    # nested sub-settings
    agent: AgentSettings
    api: ApiSettings
    cors: CORSSettings
    logging: LoggingSettings
    rag: RAGSettings


# Single, global instance
__APP_CONFIG_INSTANCE = None


def app_config() -> ApplicationConfig:
    """Return a single, global instance of ApplicationConfig."""
    global __APP_CONFIG_INSTANCE

    # Read config directory from environment variable
    config_dir_env = os.environ.get("CONFIG_DIR")
    if not config_dir_env:
        raise RuntimeError("CONFIG_DIR environment variable is not set")

    config_dir = Path(config_dir_env).expanduser().resolve(strict=True)
    config_files = [f"!{config_dir}/settings.toml"]  # settings.toml is required

    if __APP_CONFIG_INSTANCE is None:
        __APP_CONFIG_INSTANCE = ts.load(ApplicationConfig, "app", config_files=config_files)

    return __APP_CONFIG_INSTANCE

"""Shared boto3 client factories for Bedrock.

One place to honor the credential precedence used everywhere: an explicit AWS profile
(local dev) > static keys (Streamlit Cloud) > the default boto3 chain. Clients are cached
so we reuse connections across the embedder, reranker, and LLM wrappers.
"""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.config import Config

from src.config import settings
from src.ratelimit import throttle


def _install_throttle(client: "boto3.client") -> "boto3.client":
    """Pace every HTTP send on this client through the global limiter.

    Registered at the botocore layer so it covers ALL Bedrock traffic uniformly — our own
    invoke_model/converse AND LangChain's ChatBedrockConverse and RAGAS's judge calls — without
    each call site having to remember to throttle.
    """
    client.meta.events.register("before-send.bedrock-runtime", lambda **_: throttle())
    client.meta.events.register("before-send.bedrock-agent-runtime", lambda **_: throttle())
    return client

_BOTO_CONFIG = Config(
    region_name=settings.aws_region,
    retries={"max_attempts": 5, "mode": "adaptive"},
    read_timeout=120,
    connect_timeout=10,
)


@lru_cache(maxsize=1)
def get_session() -> boto3.Session:
    if settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        return boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_session_token=settings.aws_session_token,
            region_name=settings.aws_region,
        )
    return boto3.Session(region_name=settings.aws_region)


@lru_cache(maxsize=1)
def bedrock_runtime() -> boto3.client:
    """For invoke_model (Titan embeddings) and Converse (LLMs)."""
    return _install_throttle(get_session().client("bedrock-runtime", config=_BOTO_CONFIG))


@lru_cache(maxsize=1)
def bedrock_agent_runtime() -> boto3.client:
    """For the Rerank API (Cohere Rerank 3.5)."""
    return _install_throttle(get_session().client("bedrock-agent-runtime", config=_BOTO_CONFIG))

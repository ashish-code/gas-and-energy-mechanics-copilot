"""Bedrock LLM helpers via the Converse API.

Two entry points:
  * `converse()` — raw boto3 Converse for utility calls (contextual blurbs, HyDE, multi-query)
    where we want minimal overhead and our own retry.
  * `chat_model()` — a LangChain `ChatBedrockConverse` for the agent graph (LangSmith-traced,
    supports `.with_structured_output(PydanticModel)`).

Claude 4.x on Bedrock is inference-profile only, so model ids are the `us.*` profile ids
from `src.config` — both paths accept them directly.
"""

from __future__ import annotations

from functools import lru_cache

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.bedrock_clients import bedrock_runtime
from src.config import settings


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def converse(
    model_id: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Single-turn Converse call returning the assistant text."""
    kwargs: dict = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = bedrock_runtime().converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=8)
def chat_model(model_id: str, temperature: float = 0.0, max_tokens: int = 4096):  # type: ignore[no-untyped-def]
    """Cached LangChain ChatBedrockConverse for a given model id (used by the agent graph)."""
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=model_id,
        region_name=settings.aws_region,
        temperature=temperature,
        max_tokens=max_tokens,
        client=bedrock_runtime(),
    )

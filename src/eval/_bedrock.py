"""RAGAS <-> Bedrock wiring: wrap our Bedrock LLM + Titan embeddings for RAGAS."""

from __future__ import annotations

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from src.bedrock_clients import bedrock_runtime
from src.config import settings


def ragas_llm() -> LangchainLLMWrapper:
    """Sonnet (reasoning tier) as the RAGAS judge/generator LLM."""
    model = ChatBedrockConverse(
        model=settings.model_synthesizer,
        region_name=settings.aws_region,
        temperature=0.0,
        # RAGAS faithfulness emits one NLI verdict per extracted statement; detailed answers
        # produce long JSON that truncates (-> parse error -> nan) at 2048. 8192 gives headroom.
        max_tokens=8192,
        client=bedrock_runtime(),
    )
    return LangchainLLMWrapper(model)


def ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Titan V2 as the RAGAS embedding model (keeps everything on Bedrock)."""
    emb = BedrockEmbeddings(
        model_id=settings.model_embedding,
        client=bedrock_runtime(),
        normalize=True,
    )
    return LangchainEmbeddingsWrapper(emb)

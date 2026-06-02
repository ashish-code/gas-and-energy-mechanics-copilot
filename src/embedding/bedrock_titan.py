"""Amazon Titan Embed V2 via Bedrock — 1024D, L2-normalized.

Titan v2 supports server-side L2 normalization (`normalize: true`); we also normalize
defensively so cosine == dot product downstream (pgvector cosine ops, MMR, semantic dips).
`invoke_model` embeds one text per call, so batch embedding uses a bounded thread pool.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.bedrock_clients import bedrock_runtime
from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)

# Titan v2 caps input around 8k tokens; truncate defensively by characters.
_MAX_CHARS = 40_000


class TitanEmbedder:
    """Thin, retrying wrapper around Titan Embed V2."""

    def __init__(self, model_id: str | None = None, dim: int | None = None) -> None:
        self.model_id = model_id or settings.model_embedding
        self.dim = dim or settings.embedding_dim
        self._client = bedrock_runtime()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embed_text(self, text: str) -> list[float]:
        """Embed one text -> L2-normalized 1024D vector."""
        body = json.dumps(
            {"inputText": text[:_MAX_CHARS], "dimensions": self.dim, "normalize": True}
        )
        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        vec = np.asarray(json.loads(resp["body"].read())["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str], *, max_workers: int = 2) -> list[list[float]]:
        """Embed many texts concurrently, preserving order."""
        if not texts:
            return []
        results: list[list[float]] = [[] for _ in texts]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.embed_text, t): i for i, t in enumerate(texts)}
            done = 0
            for fut, i in futures.items():
                results[i] = fut.result()
                done += 1
                if done % 250 == 0:
                    log.info("titan.embed_progress", done=done, total=len(texts))
        return results

    def embed_array(self, texts: list[str], *, max_workers: int = 2) -> np.ndarray:
        """embed_batch as a float32 ndarray (for MMR / semantic-dip math)."""
        return np.asarray(self.embed_batch(texts, max_workers=max_workers), dtype=np.float32)

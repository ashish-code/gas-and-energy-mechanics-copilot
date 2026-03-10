"""
RAG Search Tool for Pipeline Safety Regulations.

Reuses the FAISS + Amazon Bedrock Titan Embeddings V2 retrieval stack
built for this repo's Strands-based copilot.  By default the tool reads
the index from ``../data/rag_index/`` (one level above ``crew/``), so it
works out of the box when run from the ``crew/`` directory.

Set ``RAG_INDEX_DIR`` to any absolute or relative path to override.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Type

import boto3
import faiss
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class RAGSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language question or keyword search to look up in the "
            "PHMSA pipeline safety regulation documentation (49 CFR Parts 192, 193, 195)."
        ),
    )


class RAGSearchTool(BaseTool):
    """
    Search the PHMSA pipeline safety regulation index.

    Uses the FAISS IndexFlatIP (cosine similarity) built from 8,524 chunks
    of 49 CFR Parts 192 / 193 / 195 and a Wikipedia engineering supplement.
    Queries are embedded with Amazon Bedrock Titan Embeddings V2 (1024-D).
    """

    name: str = "search_pipeline_safety_docs"
    description: str = (
        "Search indexed PHMSA pipeline safety regulations (49 CFR Parts 192, 193, 195) "
        "and engineering reference material. Use for regulatory requirements, "
        "design standards, safety thresholds, and compliance obligations. "
        "Input: a natural-language question or keyword phrase. "
        "Output: the most relevant regulatory passages with source citations."
    )
    args_schema: Type[BaseModel] = RAGSearchInput

    # Lazy-loaded — excluded from Pydantic model fields
    _index: Any = None
    _chunks: Any = None
    _metadata: dict | None = None
    _bedrock_client: Any = None
    _index_dim: int = 1024

    # ── internal helpers ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._index is not None:
            return

        index_dir = Path(os.environ.get("RAG_INDEX_DIR", "../data/rag_index"))
        if not index_dir.exists():
            raise RuntimeError(
                f"RAG index not found at: {index_dir.resolve()}\n"
                "Set RAG_INDEX_DIR or run from the crew/ directory so the "
                "default path ../data/rag_index resolves correctly."
            )

        self._index = faiss.read_index(str(index_dir / "index.faiss"))
        self._chunks = pd.read_parquet(index_dir / "chunks.parquet")
        with open(index_dir / "meta.json") as f:
            self._metadata = json.load(f)
        self._index_dim = self._metadata.get("dim", 1024)

        region = os.environ.get("BEDROCK_EMBEDDING_REGION", "us-east-1")
        self._bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    def _embed(self, query: str) -> np.ndarray:
        model = os.environ.get("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        body = json.dumps({"inputText": query[:8000]})
        response = self._bedrock_client.invoke_model(
            modelId=model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        vec = np.array(
            json.loads(response["body"].read())["embedding"], dtype=np.float32
        )
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    # ── BaseTool interface ───────────────────────────────────────────────────

    def _run(self, query: str) -> str:  # type: ignore[override]
        try:
            self._load()
        except RuntimeError as exc:
            return f"[RAG unavailable] {exc}\nAnswering from general knowledge only."

        top_k = int(os.environ.get("RAG_TOP_K", "5"))
        query_vec = self._embed(query)
        distances, indices = self._index.search(query_vec.reshape(1, -1), top_k)

        results: list[dict] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            chunk = self._chunks.iloc[idx]
            parts = []
            if "filename" in chunk and pd.notna(chunk.get("filename")):
                parts.append(str(chunk["filename"]))
            if "page" in chunk and pd.notna(chunk.get("page")):
                parts.append(f"page {int(chunk['page'])}")
            source = " – ".join(parts) if parts else "unknown"
            results.append(
                {"rank": rank, "score": float(score), "text": chunk["text"], "source": source}
            )

        if not results:
            return "No relevant documentation found."

        lines = ["# Retrieved Regulatory Documentation\n"]
        for r in results:
            lines.append(f"## [{r['rank']}] {r['source']}  (score: {r['score']:.3f})")
            lines.append(r["text"])
            lines.append("\n---\n")
        return "\n".join(lines)

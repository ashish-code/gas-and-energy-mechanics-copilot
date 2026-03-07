"""RAG Service for retrieval-augmented generation using FAISS."""

import json
from pathlib import Path

import boto3
import faiss
import numpy as np
import pandas as pd
import structlog.stdlib

from gas_energy_copilot.ai_copilot.core.config import RAGSettings

log = structlog.stdlib.get_logger()


class RAGService:
    """Service for loading FAISS index and performing document retrieval."""

    def __init__(self, config: RAGSettings) -> None:
        self.config = config
        self._index: faiss.Index | None = None
        self._chunks: pd.DataFrame | None = None
        self._metadata: dict | None = None
        self._bedrock_client = None
        self._index_embedding_dim: int | None = None

    def initialize(self) -> None:
        """Load FAISS index, chunks, and metadata from disk."""
        if not self.config.enabled:
            log.info("RAG is disabled in configuration")
            return

        index_dir = Path(self.config.index_dir)
        if not index_dir.exists():
            raise RuntimeError(f"RAG index directory does not exist: {index_dir}")

        log.info(f"Loading RAG index from {index_dir}")

        self._index = faiss.read_index(str(index_dir / "index.faiss"))
        log.info(f"Loaded FAISS index with {self._index.ntotal} vectors")

        self._chunks = pd.read_parquet(index_dir / "chunks.parquet")
        log.info(f"Loaded {len(self._chunks)} document chunks")

        with open(index_dir / "meta.json", "r") as f:
            self._metadata = json.load(f)
        log.info(f"Loaded metadata: {self._metadata}")

        self._index_embedding_dim = self._metadata.get("dim", 1024)

        log.info("Initialising AWS Bedrock embedding client")
        self._bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=self.config.embedding_region,
        )

        log.info("RAG service initialisation complete")

    @property
    def is_ready(self) -> bool:
        """Return True when the index and embedding client are loaded."""
        return self._index is not None and self._chunks is not None

    def embed_query(self, query: str) -> np.ndarray:
        """Embed *query* using Bedrock Titan Embeddings V2."""
        if self._bedrock_client is None:
            raise RuntimeError("RAG service not initialised. Call initialize() first.")

        body = json.dumps({"inputText": query[:8000]})
        response = self._bedrock_client.invoke_model(
            modelId=self.config.embedding_model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        embedding = np.array(
            json.loads(response["body"].read())["embedding"],
            dtype=np.float32,
        )

        if len(embedding) != self._index_embedding_dim:
            log.warning(
                "Embedding dimension mismatch — index may have been built with a different model",
                query_dim=len(embedding),
                index_dim=self._index_embedding_dim,
                embedding_model=self.config.embedding_model,
            )

        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Return the top-k most relevant document chunks for *query*.

        Args:
            query:  The user's question.
            top_k:  Number of results (defaults to ``config.top_k``).

        Returns:
            List of dicts with keys: rank, score, text, source, metadata.
        """
        if not self.is_ready:
            raise RuntimeError("RAG service not initialised. Call initialize() first.")

        k = top_k if top_k is not None else self.config.top_k
        query_embedding = self.embed_query(query)
        distances, indices = self._index.search(query_embedding.reshape(1, -1), k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if distance < self.config.similarity_threshold:
                continue

            chunk = self._chunks.iloc[idx]

            source_parts = []
            if "filename" in chunk and pd.notna(chunk.get("filename")):
                source_parts.append(str(chunk["filename"]))
            if "page" in chunk and pd.notna(chunk.get("page")):
                source_parts.append(f"page {chunk['page']}")
            source = " - ".join(source_parts) if source_parts else "unknown"

            results.append({
                "rank": i + 1,
                "score": float(distance),
                "text": chunk["text"],
                "source": source,
                "metadata": {
                    "path": chunk.get("path"),
                    "filename": chunk.get("filename"),
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("chunk_id"),
                },
            })

        log.info(f"Retrieved {len(results)} chunks for query (top_k={k})")
        return results

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not retrieved_chunks:
            return "No relevant documentation found."

        parts = ["# Retrieved Documentation\n"]
        for chunk in retrieved_chunks:
            parts.append(f"## Source: {chunk['source']}")
            parts.append(f"Relevance Score: {chunk['score']:.3f}\n")
            parts.append(chunk["text"])
            parts.append("\n---\n")

        return "\n".join(parts)

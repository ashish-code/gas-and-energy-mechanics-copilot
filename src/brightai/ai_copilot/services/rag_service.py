"""RAG Service for retrieval-augmented generation using FAISS."""

import json
from pathlib import Path

import boto3
import faiss
import numpy as np
import pandas as pd
import structlog.stdlib

from brightai.ai_copilot.core.config import RAGSettings

log = structlog.stdlib.get_logger()


class RAGService:
    """Service for loading FAISS index and performing document retrieval."""

    def __init__(self, config: RAGSettings) -> None:
        self.config = config
        self._index: faiss.Index | None = None
        self._chunks: pd.DataFrame | None = None
        self._metadata: dict | None = None
        self._embedding_client = None
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

        # Load FAISS index
        index_path = index_dir / "index.faiss"
        self._index = faiss.read_index(str(index_path))
        log.info(f"Loaded FAISS index with {self._index.ntotal} vectors")

        # Load chunks (document text)
        chunks_path = index_dir / "chunks.parquet"
        self._chunks = pd.read_parquet(chunks_path)
        log.info(f"Loaded {len(self._chunks)} document chunks")

        # Load metadata
        meta_path = index_dir / "meta.json"
        with open(meta_path, "r") as f:
            self._metadata = json.load(f)
        log.info(f"Loaded metadata: {self._metadata}")

        # Store original index embedding dimension
        self._index_embedding_dim = self._metadata.get("dim", 1536)

        # Initialize embedding client based on provider
        if self.config.embedding_provider == "openai":
            log.info("Initializing OpenAI embedding client")
            from openai import OpenAI

            self._embedding_client = OpenAI()
        elif self.config.embedding_provider == "bedrock":
            log.info("Initializing AWS Bedrock embedding client")
            self._embedding_client = boto3.client("bedrock-runtime", region_name="us-west-2")
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding_provider}")

        log.info("RAG service initialization completed")

    @property
    def is_ready(self) -> bool:
        """Check if RAG service is initialized and ready."""
        return self._index is not None and self._chunks is not None

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query using configured provider."""
        if self._embedding_client is None:
            raise RuntimeError("Embedding client not initialized")

        if self.config.embedding_provider == "openai":
            return self._embed_openai(query)
        elif self.config.embedding_provider == "bedrock":
            return self._embed_bedrock(query)
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding_provider}")

    def _embed_openai(self, query: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        response = self._embedding_client.embeddings.create(
            model=self.config.embedding_model,
            input=query,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        # Normalize for cosine similarity (index uses IndexFlatIP with normalized vectors)
        faiss.normalize_L2(embedding.reshape(1, -1))

        return embedding

    def _embed_bedrock(self, query: str) -> np.ndarray:
        """Generate embedding using AWS Bedrock."""
        import json as json_lib

        # Prepare request based on Bedrock Titan Embeddings format
        body = json_lib.dumps({"inputText": query})

        response = self._embedding_client.invoke_model(
            modelId=self.config.embedding_model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json_lib.loads(response["body"].read())
        embedding = np.array(response_body["embedding"], dtype=np.float32)

        # Check dimension mismatch
        bedrock_dim = len(embedding)
        if bedrock_dim != self._index_embedding_dim:
            log.warning(
                f"Dimension mismatch: Bedrock={bedrock_dim}, Index={self._index_embedding_dim}. "
                f"Results may be suboptimal. Consider re-embedding your index with Bedrock."
            )

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding.reshape(1, -1))

        return embedding

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: The user's question or query
            top_k: Number of results to return (defaults to config.top_k)

        Returns:
            List of dictionaries containing retrieved chunks with metadata
        """
        if not self.is_ready:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")

        k = top_k if top_k is not None else self.config.top_k

        # Generate query embedding
        query_embedding = self.embed_query(query)

        # Search FAISS index
        distances, indices = self._index.search(query_embedding.reshape(1, -1), k)

        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Filter by similarity threshold
            if distance < self.config.similarity_threshold:
                continue

            chunk_data = self._chunks.iloc[idx]

            # Build source string from available fields
            source_parts = []
            if "filename" in chunk_data and pd.notna(chunk_data.get("filename")):
                source_parts.append(str(chunk_data["filename"]))
            if "page" in chunk_data and pd.notna(chunk_data.get("page")):
                source_parts.append(f"page {chunk_data['page']}")

            source = " - ".join(source_parts) if source_parts else "unknown"

            results.append(
                {
                    "rank": i + 1,
                    "score": float(distance),
                    "text": chunk_data["text"],
                    "source": source,
                    "metadata": {
                        "path": chunk_data.get("path"),
                        "filename": chunk_data.get("filename"),
                        "page": chunk_data.get("page"),
                        "chunk_id": chunk_data.get("chunk_id"),
                    },
                }
            )

        log.info(f"Retrieved {len(results)} chunks for query (top_k={k})")
        return results

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries

        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant documentation found."

        context_parts = ["# Retrieved Documentation\n"]
        for chunk in retrieved_chunks:
            context_parts.append(f"## Source: {chunk['source']}")
            context_parts.append(f"Relevance Score: {chunk['score']:.3f}\n")
            context_parts.append(chunk["text"])
            context_parts.append("\n---\n")

        return "\n".join(context_parts)

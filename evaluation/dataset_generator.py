"""
Module: evaluation/dataset_generator.py

PURPOSE:
    Generates a synthetic Question-Answer evaluation dataset from the existing
    FAISS index chunks using RAGAS TestsetGenerator. This dataset is used by:
      - DeepEval tests (tests/test_eval_metrics.py) as ground-truth QA pairs.
      - TruLens batch evaluation runs.
      - Benchmarking retrieval quality changes over time.

    Without this dataset, evaluation is manual — someone would need to write
    hundreds of test questions and expected answers by hand. RAGAS automates
    this by extracting questions directly from your existing corpus text.

WHY RAGAS FOR GENERATION (not DeepEval, not manual):
    - RAGAS understands document structure and generates questions that require
      multi-hop reasoning across sections (e.g., combining §192.3 definitions
      with §192.505 requirements) — the hardest test cases to write manually.
    - RAGAS provides three question types:
        * simple (40%): Direct lookup — "What is the maximum operating pressure
          for polyethylene pipe under §192.121?"
        * multi_context (40%): Cross-section — "What combined requirements apply
          to a new Class 3 natural gas pipeline?"
        * reasoning (20%): Interpretation — "If a pipeline's MAOP is reduced
          under §192.619, what documentation must an operator maintain?"
    - RAGAS filters out low-quality questions (too short, unanswerable, duplicates).
    - 300 pairs is statistically sufficient: DeepEval recommends ≥100 for metric
      variance; RAGAS recommends ≥200 for reliability across question types.

RAGAS LIBRARY OVERVIEW:
    RAGAS (Retrieval-Augmented Generation Assessment) is built on LangChain.
    Key components used here:
      - TestsetGenerator: Takes documents, generates diverse QA pairs.
      - Document: RAGAS wrapper around text chunks (page_content + metadata).
      - EvaluationDataset: Collection of QA pairs with metadata.

    RAGAS uses an LLM (we use Bedrock) to generate questions and answers. This
    means generation requires AWS credentials and takes ~5-15 minutes for 300 pairs.

    Alternative: LlamaIndex QueryEngine + DatasetGenerator (similar approach but
    LlamaIndex-specific; RAGAS is framework-agnostic and has better documentation
    for regulatory/legal domains).

USAGE:
    # Generate 300 QA pairs and save to evaluation/data/synthetic_qa.json
    uv run python scripts/generate_eval_dataset.py --n 300

    # Or use the Python API directly:
    from evaluation.dataset_generator import RegulatoryQADatasetGenerator
    gen = RegulatoryQADatasetGenerator()
    dataset = gen.generate(n_questions=100)
    gen.save(dataset, Path("evaluation/data/synthetic_qa.json"))
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Default output location for the generated dataset
DEFAULT_OUTPUT_PATH = Path("evaluation/data/synthetic_qa.json")

# Source of chunks: the pre-built FAISS index parquet file
DEFAULT_CHUNKS_PATH = Path("data/rag_index/chunks.parquet")

# RAGAS question type distribution
# These ratios are tuned for regulatory content where:
# - simple questions test basic retrieval (direct §-lookups)
# - multi_context tests cross-section reasoning (compliance scenarios)
# - reasoning tests interpretation (ambiguous regulatory language)
QUESTION_TYPE_DISTRIBUTION = {
    "simple": 0.4,
    "multi_context": 0.4,
    "reasoning": 0.2,
}


class RegulatoryQADatasetGenerator:
    """
    Generates a synthetic QA evaluation dataset from the FAISS index corpus.

    This class orchestrates the RAGAS TestsetGenerator over the existing 8,524
    regulatory chunks. It handles:
      1. Loading chunks from chunks.parquet (existing index artifact).
      2. Sampling a representative subset for generation (using all 8,524 chunks
         would be too slow and expensive; we sample ~500 for diversity).
      3. Converting chunks to RAGAS Document format.
      4. Running RAGAS TestsetGenerator with Bedrock as the LLM.
      5. Saving the resulting QA pairs to JSON for use by DeepEval tests.

    SAMPLING STRATEGY:
        We sample proportionally from each CFR part and Wikipedia to ensure
        the dataset covers the full regulatory corpus, not just the most common
        sections. Without sampling, RAGAS would over-generate on the longest
        sections (which have the most chunks) and under-generate on rare sections.

    LLM BACKEND:
        RAGAS uses LangChain's LLM interface. We use ChatBedrockConverse (LangChain's
        Bedrock wrapper) for consistency with the existing Bedrock setup.
        Alternative: ChatOpenAI — requires OpenAI API key, not our stack.
    """

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        llm_model_id: str | None = None,
        embedding_model_id: str | None = None,
        aws_region: str = "us-east-1",
    ) -> None:
        """
        Initialise the dataset generator.

        Args:
            chunks_path:      Path to chunks.parquet from the FAISS index build.
            llm_model_id:     Bedrock model ID for question/answer generation.
                              Defaults to the same model used for the main pipeline.
            embedding_model_id: Bedrock embedding model ID for semantic deduplication.
            aws_region:       AWS region for Bedrock API calls.
        """
        self.chunks_path = chunks_path
        self.aws_region = aws_region

        # Use the same model as the main agent pipeline for consistency
        self.llm_model_id = llm_model_id or os.environ.get(
            "MODEL", "bedrock/openai.gpt-oss-120b-1:0"
        ).replace("bedrock/", "")

        self.embedding_model_id = embedding_model_id or "amazon.titan-embed-text-v2:0"

    def _load_chunks(self, max_chunks: int = 500) -> list[dict[str, Any]]:
        """
        Load and sample chunks from the FAISS index parquet file.

        We sample max_chunks chunks with proportional stratification by source
        (CFR part and Wikipedia) to ensure broad coverage of the regulatory corpus.

        Args:
            max_chunks: Maximum number of chunks to use for generation.
                        Higher = more diverse dataset but slower and more expensive.
                        500 is a good balance (generates ~300 QA pairs in ~10 min).

        Returns:
            List of dicts with keys: text, source, section, title, path.

        Raises:
            FileNotFoundError: If chunks.parquet does not exist (index not built yet).
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}\n"
                "Run `uv run python scripts/build_index.py` first to build the RAG index."
            )

        df = pd.read_parquet(self.chunks_path)
        log.info(f"Loaded {len(df)} chunks from {self.chunks_path}")

        # Stratified sampling: proportional to each source's chunk count
        if len(df) > max_chunks:
            sampled = df.groupby("source", group_keys=False).apply(
                lambda group: group.sample(
                    n=max(1, int(max_chunks * len(group) / len(df))),
                    random_state=42,  # reproducible sampling
                )
            )
            df = sampled.sample(n=min(max_chunks, len(sampled)), random_state=42)
            log.info(f"Sampled {len(df)} chunks (stratified by source)")

        return df.to_dict(orient="records")

    def _chunks_to_ragas_documents(self, chunks: list[dict[str, Any]]) -> list:
        """
        Convert raw chunk dicts to RAGAS Document objects.

        RAGAS Document wraps text with metadata. The metadata is stored alongside
        each QA pair in the output dataset, making it traceable back to its source.

        Args:
            chunks: List of chunk dicts from load_chunks().

        Returns:
            List of ragas.testset.graph.Node objects (RAGAS internal format).

        Note:
            RAGAS ≥0.2 uses a different Document class than earlier versions.
            We use langchain_core.documents.Document which RAGAS accepts directly.
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "langchain-core is required for RAGAS. Run: uv add langchain-core ragas"
            )

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk.get("source", ""),
                    "section": chunk.get("section", ""),
                    "title": chunk.get("title", ""),
                    "path": chunk.get("path", ""),
                    "filename": chunk.get("filename", ""),
                },
            )
            documents.append(doc)

        log.info(f"Converted {len(documents)} chunks to RAGAS Document format")
        return documents

    def _build_ragas_llm_and_embeddings(self):
        """
        Build RAGAS-compatible LLM and embeddings wrappers using Bedrock.

        RAGAS requires LangChain-compatible LLM and embeddings objects.
        We use:
          - ChatBedrockConverse: LangChain's Bedrock chat model wrapper.
          - BedrockEmbeddings: LangChain's Bedrock embeddings wrapper.

        These are used by RAGAS internally to:
          1. Generate questions from document passages (LLM).
          2. Generate reference answers for generated questions (LLM).
          3. Deduplicate semantically similar questions (embeddings).

        Returns:
            Tuple of (llm, embeddings) LangChain objects.
        """
        try:
            from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-aws is required for Bedrock integration. "
                "Run: uv add langchain-aws"
            )

        llm = ChatBedrockConverse(
            model_id=self.llm_model_id,
            region_name=self.aws_region,
            temperature=0.3,   # some creativity for question diversity, but controlled
            max_tokens=2048,   # enough for detailed answers
        )

        embeddings = BedrockEmbeddings(
            model_id=self.embedding_model_id,
            region_name=self.aws_region,
        )

        return llm, embeddings

    def generate(self, n_questions: int = 300, max_source_chunks: int = 500) -> list[dict]:
        """
        Generate a synthetic QA evaluation dataset.

        Orchestrates the full generation pipeline:
        1. Load and sample chunks from the FAISS index.
        2. Convert to RAGAS Document format.
        3. Build LangChain LLM and embeddings for Bedrock.
        4. Run RAGAS TestsetGenerator.
        5. Return the dataset as a list of dicts.

        Args:
            n_questions:       Target number of QA pairs to generate.
                               RAGAS may produce slightly fewer after deduplication.
            max_source_chunks: Maximum chunks to use as source material.

        Returns:
            List of dicts, each with keys:
              - question: The generated question.
              - ground_truth: The reference answer (generated by RAGAS using the LLM).
              - contexts: List of relevant passage texts (from the source documents).
              - source: Source document label (e.g., "49 CFR Part 192").

        Raises:
            FileNotFoundError: If chunks.parquet is missing.
            ImportError: If ragas, langchain-core, or langchain-aws are not installed.
        """
        try:
            from ragas.testset import TestsetGenerator
            from ragas.testset.evolutions import simple, multi_context, reasoning
        except ImportError:
            raise ImportError(
                "ragas is required for dataset generation. "
                "Run: uv add ragas langchain-core langchain-aws"
            )

        log.info(f"Starting RAGAS dataset generation: {n_questions} questions")

        chunks = self._load_chunks(max_chunks=max_source_chunks)
        documents = self._chunks_to_ragas_documents(chunks)
        generator_llm, embeddings = self._build_ragas_llm_and_embeddings()

        # TestsetGenerator parameters:
        # - generator_llm: generates the questions
        # - critic_llm: evaluates and filters low-quality questions
        #   (we use the same model for both to avoid extra credentials)
        # - embeddings: deduplicates semantically similar questions
        generator = TestsetGenerator.from_langchain(
            generator_llm=generator_llm,
            critic_llm=generator_llm,   # same model as generator
            embeddings=embeddings,
        )

        # Generate the testset with the configured question type distribution
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            test_size=n_questions,
            distributions={
                simple: QUESTION_TYPE_DISTRIBUTION["simple"],
                multi_context: QUESTION_TYPE_DISTRIBUTION["multi_context"],
                reasoning: QUESTION_TYPE_DISTRIBUTION["reasoning"],
            },
            with_debugging_logs=False,  # suppress verbose RAGAS internal logs
        )

        # Convert RAGAS TestsetSample objects to plain dicts
        dataset = []
        for sample in testset.samples:
            dataset.append({
                "question": sample.user_input,
                "ground_truth": sample.reference,
                "contexts": sample.reference_contexts or [],
                "evolution_type": sample.synthesizer_name or "simple",
                "metadata": {
                    "generated_by": "ragas",
                    "n_source_documents": len(documents),
                },
            })

        log.info(f"Generated {len(dataset)} QA pairs ({n_questions} requested)")
        return dataset

    def save(self, dataset: list[dict], output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
        """
        Save the generated dataset to a JSON file.

        Uses JSON (not JSONL or Parquet) because:
          - DeepEval's EvaluationDataset.from_json() expects a JSON array.
          - JSON is human-readable for manual inspection and hand-curation.
          - The dataset is small enough (~300 items) that JSON is efficient.

        Args:
            dataset:     List of QA pair dicts from generate().
            output_path: Path to save the JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        log.info(f"Saved {len(dataset)} QA pairs to {output_path}")
        print(f"✓ Dataset saved: {output_path} ({len(dataset)} QA pairs)")

    def load(self, dataset_path: Path = DEFAULT_OUTPUT_PATH) -> list[dict]:
        """
        Load a previously generated dataset from JSON.

        Args:
            dataset_path: Path to the JSON dataset file.

        Returns:
            List of QA pair dicts.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                "Run: uv run python scripts/generate_eval_dataset.py --n 300"
            )

        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        log.info(f"Loaded {len(dataset)} QA pairs from {dataset_path}")
        return dataset

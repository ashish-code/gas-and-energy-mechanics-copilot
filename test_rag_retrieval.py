#!/usr/bin/env python3
"""Simple test script to verify RAG retrieval works without running the full app."""

import os
import sys
from pathlib import Path

# Test OpenAI API key first
print("=" * 60)
print("Testing OpenAI API Key...")
print("=" * 60)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not set!")
    print("\nPlease run:")
    print('export OPENAI_API_KEY="your-key-here"')
    sys.exit(1)

print(f"✅ OPENAI_API_KEY is set (ends with: ...{api_key[-8:]})")

# Test OpenAI connection
print("\nTesting OpenAI API connection...")
try:
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test query",
    )
    print(f"✅ OpenAI API is working! Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"❌ OpenAI API test failed: {e}")
    print("\nThe API key might be expired. Please request a new one from your admin.")
    sys.exit(1)

# Test FAISS index loading
print("\n" + "=" * 60)
print("Testing FAISS Index Loading...")
print("=" * 60)

index_dir = Path("data/rag_index")
if not index_dir.exists():
    print(f"❌ Index directory not found: {index_dir}")
    sys.exit(1)

print(f"✅ Index directory exists: {index_dir}")

try:
    import faiss
    import pandas as pd
    import json

    # Load FAISS index
    index_path = index_dir / "index.faiss"
    index = faiss.read_index(str(index_path))
    print(f"✅ FAISS index loaded: {index.ntotal} vectors")

    # Load chunks
    chunks_path = index_dir / "chunks.parquet"
    chunks = pd.read_parquet(chunks_path)
    print(f"✅ Chunks loaded: {len(chunks)} documents")
    print(f"   Columns: {list(chunks.columns)}")

    # Load metadata
    meta_path = index_dir / "meta.json"
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    print(f"✅ Metadata loaded:")
    print(f"   Embedding model: {metadata['embed_model']}")
    print(f"   Dimensions: {metadata['dim']}")
    print(f"   Num vectors: {metadata['num_vecs']}")

except Exception as e:
    print(f"❌ Failed to load index: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test retrieval
print("\n" + "=" * 60)
print("Testing RAG Retrieval...")
print("=" * 60)

test_query = "What are the main troubleshooting steps?"
print(f"Query: '{test_query}'")

try:
    import numpy as np

    # Generate query embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=test_query,
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    print(f"✅ Query embedding generated (dim: {len(query_embedding)})")

    # Search FAISS index
    top_k = 3
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    print(f"✅ Search completed, top {top_k} results:")

    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        chunk_data = chunks.iloc[idx]
        print(f"\n  Rank {i+1}:")
        print(f"  Score: {distance:.4f}")
        print(f"  Source: {chunk_data.get('source', 'unknown')}")
        print(f"  Text preview: {chunk_data['text'][:150]}...")

except Exception as e:
    print(f"❌ Retrieval test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All RAG components are working!")
print("=" * 60)
print("\nYou can now run the full application:")
print("  1. Set up AWS credentials (for AWS Bedrock access)")
print("  2. Run: just dev")
print("  3. Or run: just chat (for Streamlit UI)")

#!/usr/bin/env python3
"""Smoke-test RAG retrieval using Bedrock Titan Embeddings — no server required."""

import json
import os
import sys
from pathlib import Path

AWS_PROFILE = os.environ.get("AWS_PROFILE", "vscode-user")
AWS_REGION  = os.environ.get("AWS_REGION",  "us-east-1")
INDEX_DIR   = Path("data/rag_index")

# ── 1. Verify FAISS index ────────────────────────────────────────────────────
print("=" * 60)
print("Testing FAISS index ...")
print("=" * 60)

if not INDEX_DIR.exists():
    print(f"❌ Index directory not found: {INDEX_DIR}")
    print("   Run: just build-index")
    sys.exit(1)

try:
    import faiss
    import pandas as pd
    import numpy as np

    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    chunks = pd.read_parquet(INDEX_DIR / "chunks.parquet")
    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)

    print(f"✅ FAISS index loaded: {index.ntotal} vectors (dim={meta['dim']})")
    print(f"✅ Chunks loaded:      {len(chunks)} documents")
    print(f"   Embedding model:    {meta['embedding_model']}")
except Exception as e:
    print(f"❌ Failed to load index: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 2. Bedrock embedding ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("Testing Bedrock Titan Embeddings V2 ...")
print("=" * 60)

try:
    import boto3

    session  = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    bedrock  = session.client("bedrock-runtime")
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": "test query"}),
        contentType="application/json",
        accept="application/json",
    )
    embedding = np.array(json.loads(response["body"].read())["embedding"], dtype=np.float32)
    print(f"✅ Bedrock embedding OK: dim={len(embedding)}")
except Exception as e:
    print(f"❌ Bedrock embedding failed: {e}")
    print("   Ensure AWS credentials are configured and Bedrock access is enabled.")
    sys.exit(1)

# ── 3. Retrieval ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Testing RAG retrieval ...")
print("=" * 60)

query = "What are the main safety procedures for compressor stations?"
print(f"Query: '{query}'")

try:
    faiss.normalize_L2(embedding.reshape(1, -1))
    distances, indices = index.search(embedding.reshape(1, -1), 3)
    print(f"✅ Search complete — top 3 results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = chunks.iloc[idx]
        print(f"\n  Rank {i + 1}  score={dist:.4f}")
        print(f"  Source: {chunk.get('filename', 'unknown')}  page {chunk.get('page', '?')}")
        print(f"  Text:   {chunk['text'][:120]}...")
except Exception as e:
    print(f"❌ Retrieval failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✅ All RAG components working.")
print("   Start the server with: just dev")
print("=" * 60)

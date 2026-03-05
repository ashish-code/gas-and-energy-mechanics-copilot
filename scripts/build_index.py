"""
Build FAISS RAG index from public gas & energy engineering documents.

Downloads PDFs from public government and regulatory sources, chunks them,
embeds with Amazon Bedrock Titan Embeddings V2, and saves a FAISS index.

Usage:
    AWS_PROFILE=vscode-user python scripts/build_index.py

Output:
    data/rag_index/index.faiss
    data/rag_index/chunks.parquet
    data/rag_index/meta.json
"""

import json
import os
import re
import urllib.request
from pathlib import Path

import boto3
import faiss
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024          # Titan Embeddings V2 default output dim
CHUNK_SIZE = 500              # words per chunk
CHUNK_OVERLAP = 50            # word overlap between chunks
OUTPUT_DIR = Path("data/rag_index")

# Wikipedia articles covering gas & energy engineering topics
WIKIPEDIA_TOPICS = [
    "Natural gas",
    "Pipeline transport",
    "Natural gas processing",
    "Gas compressor",
    "Compressor station",
    "Liquefied natural gas",
    "Natural gas storage",
    "Gas turbine",
    "Gas meter",
    "Pressure regulator",
    "Natural gas vehicle",
    "Biogas",
    "Methane",
    "Gas leak detection",
    "Pipeline safety",
    "Gas chromatography",
    "Heat exchanger",
    "Pressure vessel",
    "Corrosion inhibitor",
    "Valve",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_wikipedia_article(topic: str) -> dict | None:
    """Fetch plain text of a Wikipedia article via the public REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.request.quote(topic)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gas-energy-copilot-indexer/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return {"title": data.get("title", topic), "text": data.get("extract", "")}
    except Exception as e:
        print(f"  FAILED ({e})")
        return None


def fetch_wikipedia_full(topic: str) -> dict | None:
    """Fetch full plain-text content of a Wikipedia article."""
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&titles={urllib.request.quote(topic)}&prop=extracts&explaintext=true&exsectionformat=plain"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gas-energy-copilot-indexer/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"title": page.get("title", topic), "text": text}
    except Exception as e:
        print(f"  FAILED ({e})")
        return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def embed_texts(bedrock: "boto3.client", texts: list[str]) -> np.ndarray:
    """Embed a list of texts using Bedrock Titan Embeddings V2."""
    embeddings = []
    for i, text in enumerate(texts):
        if i % 20 == 0:
            print(f"    Embedding {i + 1}/{len(texts)} ...")
        body = json.dumps({"inputText": text[:8000]})  # Titan max ~8K chars
        response = bedrock.invoke_model(
            modelId=EMBEDDING_MODEL,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Gas & Energy Mechanics Copilot — Index Builder ===\n")

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    # 1. Fetch Wikipedia articles
    print("Step 1: Fetching Wikipedia articles on gas & energy topics ...")
    articles = []
    for topic in WIKIPEDIA_TOPICS:
        print(f"  Fetching: {topic} ...", end=" ")
        article = fetch_wikipedia_full(topic)
        if article and len(article["text"]) > 200:
            articles.append(article)
            print(f"OK ({len(article['text'])} chars)")
        else:
            print("skipped (too short)")

    if not articles:
        raise RuntimeError("No articles fetched. Check network access.")

    print(f"\n  {len(articles)} article(s) fetched\n")

    # 2. Chunk
    print("Step 2: Chunking text ...")
    records = []
    chunk_id = 0
    for article in articles:
        for i, chunk in enumerate(chunk_text(article["text"])):
            if len(chunk.split()) < 20:
                continue
            records.append({
                "chunk_id": chunk_id,
                "filename": f"{article['title'].replace(' ', '_')}.txt",
                "path": f"wikipedia/{article['title']}",
                "page": i + 1,
                "text": chunk,
            })
            chunk_id += 1

    print(f"  Total chunks: {len(records)}\n")

    if not records:
        raise RuntimeError("No chunks produced.")

    chunks_df = pd.DataFrame(records)

    # 3. Embed
    print("Step 3: Embedding chunks with Bedrock Titan Embeddings V2 ...")
    texts = chunks_df["text"].tolist()
    embeddings = embed_texts(bedrock, texts)

    # Normalise for cosine similarity (IndexFlatIP)
    faiss.normalize_L2(embeddings)
    print(f"  Embedded {len(embeddings)} chunks, dim={embeddings.shape[1]}\n")

    # 4. Build FAISS index
    print("Step 4: Building FAISS index ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors\n")

    # 5. Save
    print("Step 5: Saving index, chunks, and metadata ...")
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    chunks_df.to_parquet(OUTPUT_DIR / "chunks.parquet", index=False)

    meta = {
        "dim": int(dim),
        "n_chunks": len(records),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_provider": "bedrock",
        "documents": [a["title"] for a in articles],
    }
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"""
=== Done ===
Index:  {OUTPUT_DIR}/index.faiss  ({index.ntotal} vectors, dim={dim})
Chunks: {OUTPUT_DIR}/chunks.parquet  ({len(records)} rows)
Meta:   {OUTPUT_DIR}/meta.json
    """)


if __name__ == "__main__":
    main()

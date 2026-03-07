"""
Build FAISS RAG index from public gas & energy regulatory documents.

Sources:
  - eCFR Title 49, Parts 192 (Natural Gas Pipelines), 193 (LNG Facilities),
    195 (Hazardous Liquid Pipelines) — via the eCFR versioner API
  - Wikipedia supplement: gas & energy engineering background articles

Embeds with Amazon Bedrock Titan Embeddings V2 (1024D) and saves a FAISS
IndexFlatIP (cosine similarity) index.

Usage:
    AWS_PROFILE=vscode-user uv run python scripts/build_index.py

Output:
    data/rag_index/index.faiss
    data/rag_index/chunks.parquet
    data/rag_index/meta.json
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
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
CHUNK_SIZE = 400              # words per chunk
CHUNK_OVERLAP = 60            # word overlap between chunks
OUTPUT_DIR = Path("data/rag_index")

# eCFR regulatory documents — Title 49, Subtitle B, Chapter I, Subchapter D
ECFR_DATE = "2025-01-01"
ECFR_BASE = "https://www.ecfr.gov/api/versioner/v1/full"
ECFR_PARTS = [
    {"part": "192", "label": "49 CFR Part 192 — Natural Gas Pipeline Safety"},
    {"part": "193", "label": "49 CFR Part 193 — LNG Facilities"},
    {"part": "195", "label": "49 CFR Part 195 — Hazardous Liquid Pipelines"},
]

# Wikipedia supplement articles
WIKIPEDIA_TOPICS = [
    "Natural gas",
    "Pipeline transport",
    "Natural gas processing",
    "Gas compressor",
    "Compressor station",
    "Liquefied natural gas",
    "Natural gas storage",
    "Gas turbine",
    "Pressure regulator",
    "Pipeline safety",
    "Heat exchanger",
    "Pressure vessel",
    "Corrosion inhibitor",
    "Valve",
    "Pipeline integrity management",
]

# Tags to skip when extracting text from eCFR XML
ECFR_SKIP_TAGS = {"CITA", "AUTH", "SOURCE", "EDNOTE", "FTNT", "FP-2", "NOTE"}

# ---------------------------------------------------------------------------
# eCFR helpers
# ---------------------------------------------------------------------------


def _ecfr_url(part: str) -> str:
    return (
        f"{ECFR_BASE}/{ECFR_DATE}/title-49.xml"
        f"?subtitle=B&chapter=I&subchapter=D&part={part}"
    )


def fetch_ecfr_part(part_info: dict) -> list[dict]:
    """
    Download a 49 CFR part from the eCFR versioner API and extract sections.

    Returns a list of dicts with keys: source, part, section_num, title, text
    """
    part = part_info["part"]
    label = part_info["label"]
    url = _ecfr_url(part)
    print(f"  Fetching {label} ...")
    print(f"    URL: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "gas-energy-copilot-indexer/1.0", "Accept": "application/xml"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_xml = resp.read()
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code} — skipping part {part}")
        return []
    except Exception as e:
        print(f"    ERROR: {e} — skipping part {part}")
        return []

    root = ET.fromstring(raw_xml)
    sections = []

    # eCFR XML structure:
    #   <DIV5 TYPE="PART"> → <DIV6 TYPE="SUBPART"> → <DIV8 TYPE="SECTION" N="192.1">
    #     <HEAD>§ 192.1 Title text</HEAD>
    #     <P>paragraph...</P>  ...
    for section_el in root.iter("DIV8"):
        if section_el.get("TYPE") != "SECTION":
            continue
        section_num = section_el.get("N", "")
        head_el = section_el.find("HEAD")
        title = head_el.text.strip() if head_el is not None and head_el.text else f"§ {section_num}"

        # Collect paragraph text, skip metadata/citation tags
        parts_text = []
        for child in section_el:
            if child.tag in ECFR_SKIP_TAGS:
                continue
            if child.tag == "HEAD":
                continue
            text = ET.tostring(child, method="text", encoding="unicode")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                parts_text.append(text)

        body = " ".join(parts_text).strip()
        if len(body.split()) < 10:
            continue  # skip empty/stub sections

        sections.append({
            "source": label,
            "part": f"49_CFR_{part}",
            "section_num": section_num,
            "title": title,
            "text": f"{title}\n\n{body}",
        })

    print(f"    Extracted {len(sections)} sections")
    return sections


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------


def fetch_wikipedia_full(topic: str) -> dict | None:
    """Fetch full plain-text content of a Wikipedia article."""
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&titles={urllib.request.quote(topic)}&prop=extracts&explaintext=true&exsectionformat=plain"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "gas-energy-copilot-indexer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"title": page.get("title", topic), "text": text}
    except Exception as e:
        print(f"    FAILED ({e})")
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.split()) >= 20]


# ---------------------------------------------------------------------------
# Bedrock embedding
# ---------------------------------------------------------------------------


def embed_texts(bedrock_client, texts: list[str]) -> np.ndarray:
    """Embed a list of texts using Bedrock Titan Embeddings V2."""
    embeddings = []
    for i, text in enumerate(texts):
        if i % 25 == 0:
            print(f"    Embedding {i + 1}/{len(texts)} ...")
        body = json.dumps({"inputText": text[:8000]})
        for attempt in range(3):
            try:
                response = bedrock_client.invoke_model(
                    modelId=EMBEDDING_MODEL,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                embeddings.append(result["embedding"])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"    Retry {attempt + 1} after error: {e}")
                time.sleep(2)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Gas & Energy Mechanics Copilot — Index Builder ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    records = []
    chunk_id = 0
    document_labels = []

    # ── Step 1: eCFR regulatory documents ───────────────────────────────────
    print("Step 1: Fetching eCFR regulatory documents ...")
    for part_info in ECFR_PARTS:
        sections = fetch_ecfr_part(part_info)
        if not sections:
            print(f"  WARNING: No sections extracted for {part_info['label']}")
            continue
        document_labels.append(part_info["label"])
        for sec in sections:
            for i, chunk in enumerate(chunk_text(sec["text"])):
                records.append({
                    "chunk_id": chunk_id,
                    "filename": f"{sec['part']}.txt",
                    "path": f"ecfr/{sec['part']}/§{sec['section_num']}",
                    "page": i + 1,
                    "source": sec["source"],
                    "section": sec["section_num"],
                    "title": sec["title"],
                    "text": chunk,
                })
                chunk_id += 1
        time.sleep(1)  # be polite to the eCFR API

    ecfr_count = len(records)
    print(f"\n  {ecfr_count} chunks from {len(ECFR_PARTS)} CFR parts\n")

    # ── Step 2: Wikipedia supplement ────────────────────────────────────────
    print("Step 2: Fetching Wikipedia supplement articles ...")
    for topic in WIKIPEDIA_TOPICS:
        print(f"  Fetching: {topic} ...", end=" ")
        article = fetch_wikipedia_full(topic)
        if not article or len(article["text"]) < 200:
            print("skipped (too short or failed)")
            continue
        print(f"OK ({len(article['text'])} chars)")
        document_labels.append(article["title"])
        for i, chunk in enumerate(chunk_text(article["text"])):
            records.append({
                "chunk_id": chunk_id,
                "filename": f"{article['title'].replace(' ', '_')}.txt",
                "path": f"wikipedia/{article['title']}",
                "page": i + 1,
                "source": "Wikipedia",
                "section": "",
                "title": article["title"],
                "text": chunk,
            })
            chunk_id += 1

    wiki_count = len(records) - ecfr_count
    print(f"\n  {wiki_count} chunks from {len(WIKIPEDIA_TOPICS)} Wikipedia articles")
    print(f"  Total chunks: {len(records)}\n")

    if not records:
        raise RuntimeError("No chunks produced. Check network access and API availability.")

    chunks_df = pd.DataFrame(records)

    # ── Step 3: Embed ────────────────────────────────────────────────────────
    print("Step 3: Embedding chunks with Bedrock Titan Embeddings V2 ...")
    texts = chunks_df["text"].tolist()
    embeddings = embed_texts(bedrock, texts)
    faiss.normalize_L2(embeddings)
    print(f"  Embedded {len(embeddings)} chunks, dim={embeddings.shape[1]}\n")

    # ── Step 4: Build FAISS index ────────────────────────────────────────────
    print("Step 4: Building FAISS IndexFlatIP ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors\n")

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    print("Step 5: Saving index, chunks, and metadata ...")
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    chunks_df.to_parquet(OUTPUT_DIR / "chunks.parquet", index=False)

    meta = {
        "dim": int(dim),
        "n_chunks": len(records),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_provider": "bedrock",
        "ecfr_date": ECFR_DATE,
        "documents": document_labels,
    }
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"""
=== Done ===
  Index:  {OUTPUT_DIR}/index.faiss  ({index.ntotal} vectors, dim={dim})
  Chunks: {OUTPUT_DIR}/chunks.parquet  ({len(records)} rows)
  Meta:   {OUTPUT_DIR}/meta.json
  Sources:
    - eCFR regulatory: {ecfr_count} chunks
    - Wikipedia:       {wiki_count} chunks
""")


if __name__ == "__main__":
    main()

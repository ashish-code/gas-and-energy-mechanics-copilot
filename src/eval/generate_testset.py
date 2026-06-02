"""Generate a RAGAS test set from the corpus (TestsetGenerator + Sonnet + Titan).

Builds LangChain documents from the parent sections in Supabase, then has RAGAS synthesize
(question, reference, reference_contexts) tuples and persists them to
data/golden_set/ragas_set.json.

NOTE: at this account's ~1 req/s Bedrock throughput, generation is slow — `testset_size`
defaults conservatively and is overridable. The contract targets 40; pass --size 40 to match
(expect a long run). No tuning of the generator — defaults only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document as LCDocument

from src.eval._bedrock import ragas_embeddings, ragas_llm
from src.logging_setup import configure_logging, get_logger
from src.store import connect

log = get_logger(__name__)

OUT_PATH = Path("data/golden_set/ragas_set.json")


def _corpus_documents() -> list[LCDocument]:
    """One LangChain Document per parent section (small-to-big parents)."""
    from src.store import fetch_all_chunks

    with connect() as conn:
        rows = fetch_all_chunks(conn)
    parents: dict[str, dict] = {}
    for r in rows:
        pid = r.get("parent_id") or r["id"]
        parents.setdefault(pid, {"texts": [], "meta": r})
        parents[pid]["texts"].append((r.get("chunk_index", 0), r["text"]))
    docs = []
    for pid, p in parents.items():
        text = "\n\n".join(t for _, t in sorted(p["texts"]))
        meta = p["meta"]["metadata"] or {}
        docs.append(LCDocument(page_content=text, metadata={"parent_id": pid, "source": p["meta"]["source"], **meta}))
    return docs


def generate(size: int = 20) -> list[dict]:
    from ragas.testset import TestsetGenerator

    docs = _corpus_documents()
    log.info("testset.docs", n=len(docs))
    generator = TestsetGenerator(llm=ragas_llm(), embedding_model=ragas_embeddings())
    testset = generator.generate_with_langchain_docs(docs, testset_size=size)

    df = testset.to_pandas()
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "question": row.get("user_input"),
                "reference": row.get("reference"),
                "reference_contexts": list(row.get("reference_contexts") or []),
            }
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    log.info("testset.saved", path=str(OUT_PATH), n=len(records))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20)
    args = ap.parse_args()
    configure_logging()
    generate(args.size)


if __name__ == "__main__":
    main()

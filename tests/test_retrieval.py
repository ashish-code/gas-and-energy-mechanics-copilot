"""Offline unit tests for the retrieval primitives (no Bedrock / DB).

Live retrieval quality is exercised by the e2e smoke + manual validation; these lock down
the deterministic pieces: BM25 tokenization, RRF fusion, MMR diversification, citation
parsing, and section-natural packing.
"""

from __future__ import annotations

from src.chunking.base import approx_tokens, pack_units
from src.chunking.phmsa_layout import parse_cited_regs
from src.retrieval.base import Candidate
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.mmr import mmr_select
from src.retrieval.sparse import tokenize


def _cand(cid: str, lane: str = "dense") -> Candidate:
    return Candidate(
        chunk_id=cid, doc_id="d", source="ecfr", text=cid, contextualized_text=cid,
        section_path=[], parent_id=cid, chunk_index=0, metadata={}, lanes=[lane],
    )


def test_tokenize_preserves_citation_tokens() -> None:
    toks = tokenize("MAOP under 49 CFR 192.619 and §195.452(j)(3).")
    assert "192.619" in toks  # the dotted citation stays one token
    assert "maop" in toks
    assert "195.452" in toks


def test_rrf_fuse_rewards_agreement_across_lanes() -> None:
    # 'b' is rank-2 in both lanes; 'a' and 'c' are rank-1 in one lane each.
    lane1 = [_cand("a"), _cand("b"), _cand("x")]
    lane2 = [_cand("c", "bm25"), _cand("b", "bm25"), _cand("y", "bm25")]
    fused = rrf_fuse([lane1, lane2], k=60, top_n=10)
    ids = [c.chunk_id for c in fused]
    assert ids[0] == "b"  # appears in both lanes -> highest fused score
    assert set(ids[:3]) == {"a", "b", "c"}
    # lane membership is merged on the surviving candidate
    assert set(next(c for c in fused if c.chunk_id == "b").lanes) == {"dense", "bm25"}


def test_mmr_prefers_diverse_second_pick() -> None:
    query = [1.0, 0.0, 0.0]
    # a, b nearly identical & most relevant; c is relevant but orthogonal.
    vecs = {
        "a": [0.99, 0.10, 0.0],
        "b": [0.98, 0.12, 0.0],
        "c": [0.80, 0.0, 0.60],
    }
    cands = [_cand("a"), _cand("b"), _cand("c")]
    picked = mmr_select(query, cands, vecs, lambda_mult=0.5, k=2)
    ids = [c.chunk_id for c in picked]
    assert ids[0] == "a"          # most relevant first
    assert ids[1] == "c"          # diversity beats the near-duplicate 'b'


def test_parse_cited_regs_extracts_sections() -> None:
    regs = parse_cited_regs("Integrity Management [195.452(j)(3)] - 1 item(s)", "violation of 192.451")
    assert "195.452(j)(3)" in regs
    assert "192.451" in regs
    assert parse_cited_regs(None, "no citations here") == []


def test_pack_units_respects_target_and_merges_tiny_tail() -> None:
    units = ["word " * 200, "word " * 200, "tiny tail"]  # two ~big + one tiny
    blocks = pack_units(units, target_tokens=300, max_tokens=800, min_tokens=30)
    assert all(approx_tokens(b) <= 800 for b in blocks)
    # the tiny trailing unit is merged back, not emitted as its own sub-min chunk
    assert approx_tokens(blocks[-1]) >= 30

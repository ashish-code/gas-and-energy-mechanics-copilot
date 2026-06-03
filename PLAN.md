# PLAN.md — Gas & Energy Mechanics Copilot v2

**Milestone 0 deliverable.** Produced before any code. Awaiting user approval.

Author: senior engineer (Claude Code) · Date: 2026-06-02 · Branch target: `v2`

---

## 1. What this is

A multi-agent **plan → execute → verify** RAG system over US federal pipeline-safety
material (49 CFR regs, PHMSA enforcement actions, NTSB accident reports). Built as a
Thomson Reuters Labs interview showcase: **production discipline applied to
research-grade retrieval techniques.** Corpus is a deliberately small (~2–3K chunk)
public-domain stand-in; the *architecture* is the contribution.

The v2 is effectively a **rewrite**, not an extension, of v1 (see §7).

---

## 2. Architecture (in my words)

Three LangGraph nodes over a typed Pydantic state object:

```
User query
   │
   ▼
PLAN  (Sonnet 4.5) ── decompose into 1–5 sub-questions, OR refuse if out-of-scope
   │   → Plan{ sub_questions[] }            (refusal is a first-class, visible outcome)
   ▼
EXECUTE (Haiku 4.5, loops per sub-question)
   │   per sub-question, run the full retrieval pipeline:
   │     multi-query (3 variants) + HyDE (1 hypothetical reg para)
   │       → for each of the 5 query forms: dense (Titan→pgvector HNSW, k=30)
   │                                        + sparse (rank-bm25, k=30)
   │     → RRF fuse all lanes (k=60) → top-50
   │     → MMR diversify (λ=0.6) → top-20
   │     → Cohere Rerank 3.5 via Bedrock → top-10
   │     → parent-section expansion (small-to-big) → dedupe → ~5–7 parent sections
   │   → Evidence[] per sub-question
   ▼
VERIFY + SYNTHESIZE
   │   Sonnet 4.5 → VerifiedAnswer{ summary, claims[]{text,citation,chunk_id,quote}, unsupported[] }
   │   per claim, Haiku 4.5 verifies:
   │     (a) mechanical: citation/chunk_id exists in corpus
   │     (b) mechanical: quoted span actually appears in the source chunk
   │     (c) LLM-as-judge: does the source span entail the claim? (NLI)
   │   failed claims → unsupported_claims (surfaced, never silently dropped)
   ▼
Streamlit UI renders each tier; LangSmith traces every node/lane/LLM call.
```

**Why this shape is defensible in 5 min:** decomposition makes multi-hop questions
tractable and *visible*; hybrid+rerank is the current SOTA-default retrieval recipe;
per-claim verification with surfaced failures is the trust differentiator a legal/research
audience cares about; cascade routing (Sonnet for reasoning, Haiku for orchestration)
controls cost without hand-tuning.

---

## 3. Resolved Bedrock model IDs (verified live, `us-east-1`)

Verified `2026-06-02` with profile `vscode-user`, account `414994224379`, region `us-east-1`.

| Role | Model | **Invocation ID to use in code** | Notes |
|---|---|---|---|
| Planner | Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | **Inference profile required** (user chose 4.6 over 4.5) |
| Synthesizer | Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | same |
| Executor | Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | **Inference profile required** |
| Verifier | Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | same |
| Embeddings | Amazon Titan Embed V2 | `amazon.titan-embed-text-v2:0` | 1024D, `normalize:true`, on-demand |
| Reranker | Cohere Rerank 3.5 | `cohere.rerank-v3-5:0` | **available in us-east-1 ✅** — no fallback needed |

**Critical finding — must honor in code:** `get-foundation-model` reports that both Claude
Sonnet 4.5 and Haiku 4.5 support **only `INFERENCE_PROFILE`**, not `ON_DEMAND`. Invoking the
bare `anthropic.claude-…` IDs returns a validation error. Code must use the `us.` cross-region
inference-profile IDs above. (LangChain `ChatBedrockConverse` accepts these directly as
`model_id`; the contextual-blurb/HyDE/multi-query calls that use raw boto3 `converse` must too.)

**Reranker decision:** Cohere Rerank 3.5 (`cohere.rerank-v3-5:0`) **is present in
`us-east-1`**. We use it as primary; the `bge-reranker-base` fallback path is **not built**
in v2 (documented as the contingency only). Invoked via the Bedrock Rerank API
(`bedrock-agent-runtime` `rerank`).

**Decision (user, M0):** use Sonnet **4.6** (`us.anthropic.claude-sonnet-4-6`) for planner +
synthesizer, overriding the spec's 4.5. Both are inference-profile only; one-line config swap.

These IDs will be persisted as the defaults in `src/config.py`.

---

## 4. File-by-file work breakdown (≤18h target; 2h buffer)

| Milestone | Files | Est (h) |
|---|---|---|
| **M1 Infra** | branch `v2`; `pyproject.toml`+`uv.lock`; `src/config.py`; `src/logging_setup.py`; `infra/supabase_schema.sql`; `.env.example`; `.streamlit/secrets.toml.example`; smoke-test skeleton | **2.0** |
| **M2 Ingestion** | `ingestion/ecfr.py` (build fresh — see §7); `ingestion/phmsa.py`; `ingestion/ntsb.py`; `ingestion/pipeline.py`; `chunking/base.py`; `chunking/ecfr_xml.py`; `chunking/phmsa_layout.py`; `chunking/ntsb_semantic.py`; `chunking/contextual.py`; `embedding/bedrock_titan.py`; `scripts/build_index.py` | **5.0** |
| **M3 Retrieval** | `dense.py`, `sparse.py`, `multi_query.py`, `hyde.py`, `hybrid.py`, `mmr.py`, `reranker.py`, `parent_doc.py`; `tests/test_retrieval.py` | **3.0** |
| **M4 Agents** | `agents/schemas.py`, `planner.py`, `executor.py`, `verifier.py`, `graph.py`; `tests/test_agents.py` | **4.0** |
| **M5 Observability** | `observability/langsmith_setup.py` | **0.5** |
| **M6 Eval** | `eval/generate_testset.py`, `eval/hand_curated.py`, `eval/run_metrics.py`; `scripts/run_eval.py` | **1.5** |
| **M7 Streamlit** | `app/streamlit_app.py`, `app/example_queries.py`; deploy to Streamlit Cloud | **1.5** |
| **M8 README/polish** | `README.md`, mermaid diagram, secret-scan | **0.5** |
| | **Total** | **18.0** |

Buffer: 2h. **Highest-risk milestone: M2 (scraping + fresh eCFR build).** Per the contract,
if any one source exceeds a 2h block, it gets dropped and noted here + in README Future Work.

---

## 5. Carry-forward from v1 (what's actually reusable)

- **Titan V2 embedding call** (`invoke_model` with `{"inputText": ...}` → `embedding`) —
  reuse the request/parse shape in `embedding/bedrock_titan.py`, add `normalize:true`,
  retry/backoff, and batching.
- **`.gitignore`, `pyproject` tooling conventions** (ruff/mypy config, line-length 120).
- **Region default `us-east-1`** and boto3 client patterns.
- That's essentially it — see §7 for why the rest is a rewrite.

---

## 6. Unknowns & risks

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | **eCFR ingestion does not exist in v1** (see §7) — must be built fresh from the eCFR Versioner XML API. | Certain | Budgeted in M2. Versioner API is stable/public; XML structure (`<DIV>` hierarchy) is well-documented. **Resolved at M2:** `full/{date}/title-49.xml?part=N` → `DIV8 TYPE=SECTION` (N=, HEAD, P, hierarchy_metadata citation), `DIV6`=subpart. |
| R1b | **Actual eCFR corpus is ~600 chunks, not ~1500.** Parts 192+193+195 in full = 152K words ≈ 592 paragraph-packed chunks. The spec's 1500 estimate overshot the real corpus. | Certain | Accept ~600 eCFR; total corpus ~1600 (eCFR+PHMSA+NTSB). Within the "2–3K, variety>volume" philosophy. Padding to 1500 would mean sub-100-token chunks (over-splitting) — rejected as tuning-for-a-number. Documented in README. |
| R2 | PHMSA enforcement portal is server-rendered/JS or rate-limits scraping. | Medium | Time-box to 2h; if hostile, drop source and proceed with eCFR+NTSB. |
| R3 | NTSB report PDFs vary in layout; `pdfplumber` header detection misses sections. | Medium | Structural pass falls back to whole-doc semantic chunking if no headers found; time-box. |
| R4 | Claude 4.5 inference-profile-only invocation surprises a code path using a raw model ID. | Low (now known) | All model IDs centralized in `config.py` as the `us.` profile IDs. |
| R5 | RAGAS `TestsetGenerator` + faithfulness run latency/cost on Bedrock for 40+10 questions. | Medium | Run once, cache to `data/golden_set/`. No tuning; accept default-tier scores. |
| R6 | Streamlit Cloud 1GB RAM + cold start. | Low | Reranker is remote (Bedrock), not local; only rank-bm25 (~20MB) + boto3 in memory. `warm_up_demo.py` pre-warms. |
| R7 | Supabase free-tier pgvector HNSW build on ~2–3K rows. | Low | Trivial at this scale; `m=16, ef_construction=64` per spec. |
| R8 | Contextual-Retrieval blurb step = 1 Haiku call per chunk (~1.7K calls). | Medium (cost/time) | Backoff; persist `contextualized_text` to DB; run once. |
| R9 | **This Bedrock account (`vscode-user`, 414994224379) is rate-limited to ~1 req/s.** Titan ≈1.5s/call sequentially; concurrency >2 triggers `ThrottlingException` storms (measured at M2). | Certain | Concurrency pinned to **2** everywhere (embedder, contextualizer, query-embed). Implications: **full index build ≈ 60–90 min one-time background job** (embeddings persist, so paid once); **per-query latency ≈ 20–40 s** (5 query-form embeds + LLM calls), consistent with the spec's 30–60 s cold-start budget. **Mitigation option for the user:** request a Bedrock on-demand quota increase for Titan/Claude InvokeModel, or run the build from an account/region with higher TPS. Not a blocker — just slow. |

---

## 7. Discrepancies between the contract and v1 reality (please note)

The contract says **"eCFR … carry forward from v1"** and **"structlog JSON + correlation IDs
(carry forward from v1)."** Inspecting v1:

1. **There is no eCFR ingestion in v1.** `scripts/build_index.py` actually fetches **Wikipedia**
   articles (20 gas/energy topics) with fixed-size 500-word/50-overlap chunking into FAISS.
   The README/RAG_IMPLEMENTATION docs describe eCFR aspirationally. → eCFR ingestion is **built
   fresh** in M2 (budgeted). No working code to port.
2. **v1 logging is a local shim**, not structlog JSON + correlation IDs. `src/brightai/logging.py`
   is a stdlib `logging.basicConfig` stub. → `logging_setup.py` is **built fresh** (structlog JSON
   + `structlog.contextvars` correlation IDs), small (~40 LOC), still ≤budget.
3. v1 uses **Strands agents + FAISS + word-chunking**; v2 uses **LangGraph + pgvector + source-router
   chunking**. Net: v2 is a rewrite. The reusable surface is the Titan embedding call only (§5).

None of this changes scope or budget — M2 already budgets fresh eCFR work — but I'm flagging it so
"carry forward" isn't mistaken for "port existing code."

---

## 8. Decisions taken within spec (no approval needed, listed for transparency)

- Python 3.11+ (v1 pins 3.13; I'll set `requires-python = ">=3.11"` so Streamlit Cloud is happy).
- Drop the `brightai` package namespace; new code lives under `src/` flat per the required structure.
- Drop Strands, FAISS, OpenAI, `typed-settings` deps entirely; add LangGraph, langchain-aws,
  pydantic-settings, supabase, rank-bm25, pdfplumber, ragas, langsmith.
- Cohere Rerank invoked via Bedrock Rerank API; no local reranker shipped.

---

## 9. User decisions (resolved at M0 approval)

1. **Supabase project** — ✅ user will create it and provide `SUPABASE_URL` + service-role key at M1.
   I supply the schema SQL.
2. **AWS credentials** — ✅ use `AWS_PROFILE=vscode-user` (acct `414994224379`) **locally only**. For
   Streamlit Cloud, user will create a **new dedicated, least-privilege IAM user** (Bedrock invoke +
   rerank only); I'll provide the IAM policy JSON and step-by-step at M7. The `vscode-user` static
   keys do **not** go into the cloud secret store.
3. **LangSmith** — ✅ user will create a LangSmith project; I'll provide signup/key/project steps at M5.
   Tracing stays env-gated (no-op when `LANGSMITH_API_KEY` unset) so local runs work without it.
4. **Model pin** — ✅ **Sonnet 4.6** (`us.anthropic.claude-sonnet-4-6`) for planner + synthesizer.
5. **eCFR built fresh** — ✅ approved; no v1 code to port (§7).

---

## 9b. Validation log (live, post-build)

Validated end-to-end against a 143-chunk smoke index, then the full corpus:

- **Corpus (full build):** 2,082 chunks / 213 docs — 859 eCFR (Parts 192/193/195), 200 PHMSA
  enforcement actions, ~1,023 NTSB (10 reports). In the "2–3K, variety>volume" target.
- **Write path:** fetch → source-router chunk → contextualize (Haiku) → embed (Titan) → upsert
  (psycopg/pgvector). ✓
- **Retrieval (live):** 9 fusion lanes (orig+3 multi-query+HyDE × dense/sparse) → RRF top-50 →
  MMR top-20 → **Cohere Rerank** top-10 → parent expansion. Probable-cause query returned the
  correct NTSB sections at rerank score 0.95. ✓ (~15 s isolated)
- **Agent (live):** plan (2 sub-questions) → execute → synthesize (8 atomic claims) → verify
  (6 supported via citation+quote+entailment; **2 correctly surfaced as unsupported** for
  overstating the evidence). ✓ (~48 s)
- **Refusal (live):** out-of-corpus query routed plan→refuse with a clear reason, 0 claims. ✓
- **Bug fixed:** synthesizer initially returned 0 claims (summary only); strengthened the
  prompt to mandate atomic claim decomposition. ✓
- **Tracing:** LangSmith enabled (env-gated). **Tests:** 14 offline unit tests pass.

## 10. Exit criterion for M0

User approves this `PLAN.md`. Then: `git checkout -b v2` and begin M1. No code is written before approval.

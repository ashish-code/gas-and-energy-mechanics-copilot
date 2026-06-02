-- Gas & Energy Mechanics Copilot v2 — Supabase / Postgres schema
-- Run in the Supabase SQL editor (or psql) against a fresh project.
-- Idempotent: safe to re-run.

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- documents: one row per source document (a CFR section group, a PHMSA action,
-- an NTSB report).
-- ---------------------------------------------------------------------------
create table if not exists documents (
    id            text primary key,
    source        text not null check (source in ('ecfr', 'phmsa_enforcement', 'ntsb_accident')),
    title         text,
    url           text,
    effective_date date,
    metadata      jsonb default '{}'::jsonb,
    ingested_at   timestamptz default now()
);

-- ---------------------------------------------------------------------------
-- chunks: paragraph-sized retrieval units (small-to-big: we embed these small
-- chunks and expand to the parent section at generation time).
--
--   text                 raw chunk text
--   contextualized_text  text with the Anthropic Contextual-Retrieval blurb
--                        prepended; THIS is what was embedded
--   embedding            1024D Titan V2, L2-normalized
--   parent_id            groups chunks belonging to the same parent section, for
--                        small-to-big expansion + dedup
-- ---------------------------------------------------------------------------
create table if not exists chunks (
    id                  text primary key,
    doc_id              text not null references documents(id) on delete cascade,
    source              text not null,
    text                text not null,
    contextualized_text text not null,
    embedding           vector(1024) not null,
    section_path        text[],
    parent_id           text,
    page                int,
    char_span           int4range,
    chunk_index         int,
    metadata            jsonb default '{}'::jsonb
);

-- HNSW index for dense retrieval. cosine ops because Titan vectors are L2-normalized.
create index if not exists chunks_embedding_hnsw on chunks
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- Metadata / lookup indices.
create index if not exists chunks_doc_id   on chunks(doc_id);
create index if not exists chunks_source   on chunks(source);
create index if not exists chunks_parent   on chunks(parent_id);
create index if not exists chunks_metadata_gin on chunks using gin(metadata);

-- ---------------------------------------------------------------------------
-- match_chunks: dense cosine retrieval RPC. Returns top-k by cosine similarity
-- (1 - cosine_distance). Optional source filter for metadata-scoped retrieval.
-- Called from src/retrieval/dense.py.
-- ---------------------------------------------------------------------------
create or replace function match_chunks(
    query_embedding vector(1024),
    match_count     int default 30,
    filter_source   text default null
)
returns table (
    id                  text,
    doc_id              text,
    source              text,
    text                text,
    contextualized_text text,
    section_path        text[],
    parent_id           text,
    page                int,
    metadata            jsonb,
    similarity          float
)
language sql stable
as $$
    select
        c.id, c.doc_id, c.source, c.text, c.contextualized_text,
        c.section_path, c.parent_id, c.page, c.metadata,
        1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    where filter_source is null or c.source = filter_source
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

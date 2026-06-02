"""Contextual Retrieval (Anthropic, Sep 2024).

For each chunk, ask Haiku to write a 1-2 sentence blurb situating the chunk within its
document, then prepend that blurb to the chunk text *before embedding*. This is the
highest single-shot retrieval lift in the literature: it resolves pronouns, implicit
subjects, and bare citations that a standalone paragraph lacks.

We give the model the document title + section path as the "context", not the whole
document (the corpus documents are small and the section path is a precise locator). Runs
on the cheap tier (Haiku) with bounded concurrency; failures degrade gracefully to the raw
text (so a flaky call never drops a chunk).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.chunking.base import Chunk
from src.config import settings
from src.llm import converse
from src.logging_setup import get_logger

log = get_logger(__name__)

_SYSTEM = (
    "You situate a regulatory/enforcement/accident-report excerpt within its source so it "
    "retrieves well on its own. Output ONLY a single 1-2 sentence context blurb — no preamble, "
    "no quotes, no restating the excerpt."
)

_PROMPT = """Document: {title}
Section: {section}

Excerpt:
{text}

Write a 1-2 sentence blurb that states what this excerpt is (its document, section/citation,
and what it covers) so it can be understood and retrieved out of context."""


def _blurb(chunk: Chunk, title: str) -> str:
    section = " > ".join(chunk.section_path) if chunk.section_path else "(unknown section)"
    try:
        blurb = converse(
            settings.model_contextualizer,
            _PROMPT.format(title=title, section=section, text=chunk.text[:4000]),
            system=_SYSTEM,
            max_tokens=120,
            temperature=0.0,
        ).strip()
        return blurb
    except Exception as e:  # noqa: BLE001 — never drop a chunk over a flaky context call
        log.warning("contextual.blurb_failed", chunk_id=chunk.chunk_id, error=str(e))
        return ""


def contextualize(
    chunks: list[Chunk],
    doc_titles: dict[str, str],
    *,
    max_workers: int = 2,
) -> list[Chunk]:
    """Populate `contextualized_text` for every chunk (blurb + "\\n\\n" + raw text)."""
    if not chunks:
        return chunks

    def work(ch: Chunk) -> None:
        title = doc_titles.get(ch.doc_id, ch.source)
        blurb = _blurb(ch, title)
        ch.contextualized_text = f"{blurb}\n\n{ch.text}" if blurb else ch.text

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for _ in pool.map(work, chunks):
            done += 1
            if done % 250 == 0:
                log.info("contextual.progress", done=done, total=len(chunks))
    log.info("contextual.done", chunks=len(chunks))
    return chunks

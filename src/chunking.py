"""Two chunking strategies for Hebrew insurance policy Markdown.

Both strategies produce dicts with identical keys so downstream code
(indexing, evaluation) can treat them uniformly:

  chunk_id   – unique string: "{family_id}_{strategy}_{doc_name}_{idx}"
  text       – "passage: " + raw_text  (e5 prefix required for retrieval)
  source_doc – document stem (filename without .md)
  strategy   – "fixed" | "section_aware"
  family_id  – multi-tenancy key
  anchor     – first 80 chars of raw_text; stable across both strategies,
               used as the citation key in gold_set.jsonl
  section    – nearest ## heading (section_aware only; None for fixed)
"""

from __future__ import annotations

import re

from src.config import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, SECTION_MAX_TOKENS

_E5_PREFIX = "passage: "

# Hebrew / multilingual text: roughly 4 characters per token (subword).
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    raw_text: str,
    doc_name: str,
    family_id: str,
    strategy: str,
    idx: int,
    section: str | None = None,
) -> dict:
    """Assemble a single chunk dict from already-extracted raw text."""
    return {
        "chunk_id": f"{family_id}_{strategy}_{doc_name}_{idx}",
        "text": f"{_E5_PREFIX}{raw_text}",
        "source_doc": doc_name,
        "strategy": strategy,
        "family_id": family_id,
        "anchor": raw_text[:80],
        "section": section,
    }


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed-size
# ---------------------------------------------------------------------------


def chunk_fixed(
    text: str,
    doc_name: str,
    family_id: str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[dict]:
    """Split *text* into overlapping character-level windows.

    Args:
        text:       Raw Markdown text (no e5 prefix yet).
        doc_name:   Document stem used in chunk_id and source_doc.
        family_id:  Multi-tenancy key.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        List of chunk dicts (empty list if *text* is blank).
    """
    if not text:
        return []

    stride = max(1, chunk_size - overlap)
    chunks: list[dict] = []
    idx = 0
    start = 0

    while start < len(text):
        raw = text[start : start + chunk_size]
        chunks.append(_make_chunk(raw, doc_name, family_id, "fixed", idx))
        idx += 1
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — Section-aware
# ---------------------------------------------------------------------------


def chunk_section_aware(
    text: str,
    doc_name: str,
    family_id: str,
    max_tokens: int = SECTION_MAX_TOKENS,
) -> list[dict]:
    """Split *text* on Docling ``##`` section headings.

    Each section becomes one chunk unless its character count exceeds
    ``max_tokens * _CHARS_PER_TOKEN``, in which case the section is
    further split with :func:`chunk_fixed` (10 % overlap).

    Text that appears before the first ``##`` heading is preserved as its
    own chunk with ``section=None``.

    Args:
        text:       Raw Markdown text produced by Docling.
        doc_name:   Document stem.
        family_id:  Multi-tenancy key.
        max_tokens: Soft token cap per chunk (character approximation used).

    Returns:
        List of chunk dicts (empty list if *text* is blank).
    """
    if not text:
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN

    # Split on newlines immediately before a "## " heading.
    # Text before the first heading (if any) lands in parts[0].
    parts = re.split(r"\n(?=## )", text)

    chunks: list[dict] = []
    idx = 0

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        # Extract the nearest ## heading for metadata.
        heading_match = re.match(r"^(## [^\n]*)", stripped)
        section = heading_match.group(1) if heading_match else None

        if len(stripped) <= max_chars:
            chunks.append(
                _make_chunk(stripped, doc_name, family_id, "section_aware", idx, section)
            )
            idx += 1
        else:
            # Section too long — fall back to fixed-size sub-chunking.
            sub_overlap = max(1, max_chars // 10)
            sub_chunks = chunk_fixed(
                stripped,
                doc_name,
                family_id,
                chunk_size=max_chars,
                overlap=sub_overlap,
            )
            for sub in sub_chunks:
                sub["chunk_id"] = f"{family_id}_section_aware_{doc_name}_{idx}"
                sub["strategy"] = "section_aware"
                sub["section"] = section
                # Re-derive anchor from the (already-prefixed) text field.
                sub["anchor"] = sub["text"][len(_E5_PREFIX) :][:80]
                idx += 1
            chunks.extend(sub_chunks)

    return chunks

"""Tests for src.chunking — two chunking strategies (Step 2).

All test input is synthetic markdown; no real corpus files are read here.
"""

import pytest

from src.chunking import chunk_fixed, chunk_section_aware

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

E5_PREFIX = "passage: "

# A minimal markdown document with two sections and enough text to produce
# multiple fixed-size chunks when chunk_size is small.
_SHORT_DOC = (
    "## כיסוי גניבה\n"
    "הפוליסה מכסה גניבת רכב עד 50,000 ש\"ח בכפוף לתנאים.\n"
    "## כיסוי תאונה\n"
    "הפוליסה מכסה נזקי תאונה עד 100,000 ש\"ח.\n"
)

_LONG_SECTION_DOC = (
    "## כיסוי מקיף\n"
    + ("מידע על כיסוי נוסף. " * 200)  # ~4000 chars — exceeds any sane max_tokens
)

# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


def test_fixed_returns_list_of_dicts():
    chunks = chunk_fixed(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, dict) for c in chunks)


def test_fixed_adds_e5_passage_prefix():
    chunks = chunk_fixed(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    for chunk in chunks:
        assert chunk["text"].startswith(E5_PREFIX), (
            f"Missing 'passage: ' prefix in: {chunk['text'][:60]!r}"
        )


def test_fixed_metadata_fields():
    chunks = chunk_fixed(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    for chunk in chunks:
        assert chunk["source_doc"] == "test_policy"
        assert chunk["strategy"] == "fixed"
        assert chunk["family_id"] == "fam1"
        assert "chunk_id" in chunk
        assert "anchor" in chunk


def test_fixed_chunk_ids_are_unique():
    chunks = chunk_fixed(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_fixed_chunk_id_contains_doc_name():
    chunks = chunk_fixed(_SHORT_DOC, doc_name="car_policy", family_id="fam1")
    for chunk in chunks:
        assert "car_policy" in chunk["chunk_id"]


def test_fixed_short_text_produces_single_chunk():
    text = "פיסקה קצרה בלבד."
    chunks = chunk_fixed(text, doc_name="tiny", family_id="fam1")
    assert len(chunks) == 1


def test_fixed_empty_text_produces_empty_list():
    chunks = chunk_fixed("", doc_name="empty", family_id="fam1")
    assert chunks == []


def test_fixed_consecutive_chunks_overlap():
    # With tiny chunk_size=50 and overlap=20 the chunks must share text.
    long_text = "א" * 200
    chunks = chunk_fixed(
        long_text, doc_name="x", family_id="fam1", chunk_size=50, overlap=20
    )
    assert len(chunks) >= 2
    # Strip prefix before checking overlap
    raw = [c["text"][len(E5_PREFIX):] for c in chunks]
    # End of chunk N must equal start of chunk N+1 for the overlapping chars
    assert raw[0][-20:] == raw[1][:20]


def test_fixed_anchor_is_text_without_prefix():
    chunks = chunk_fixed(_SHORT_DOC, doc_name="test", family_id="fam1")
    for chunk in chunks:
        raw_text = chunk["text"][len(E5_PREFIX):]
        assert chunk["anchor"] == raw_text[:80]


# ---------------------------------------------------------------------------
# Section-aware chunker
# ---------------------------------------------------------------------------


def test_section_aware_returns_list_of_dicts():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, dict) for c in chunks)


def test_section_aware_splits_on_headings():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    # _SHORT_DOC has 2 sections → should produce at least 2 chunks
    assert len(chunks) >= 2


def test_section_aware_adds_e5_passage_prefix():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    for chunk in chunks:
        assert chunk["text"].startswith(E5_PREFIX)


def test_section_aware_metadata_fields():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    for chunk in chunks:
        assert chunk["source_doc"] == "test_policy"
        assert chunk["strategy"] == "section_aware"
        assert chunk["family_id"] == "fam1"
        assert "chunk_id" in chunk
        assert "anchor" in chunk
        assert "section" in chunk


def test_section_aware_section_field_contains_heading():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    headings = [c["section"] for c in chunks if c["section"]]
    assert any("כיסוי גניבה" in h for h in headings)
    assert any("כיסוי תאונה" in h for h in headings)


def test_section_aware_long_section_is_further_split():
    # _LONG_SECTION_DOC has one giant section; it must produce multiple chunks.
    chunks = chunk_section_aware(
        _LONG_SECTION_DOC, doc_name="big", family_id="fam1", max_tokens=100
    )
    assert len(chunks) >= 2


def test_section_aware_chunk_ids_are_unique():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_section_aware_anchor_is_text_without_prefix():
    chunks = chunk_section_aware(_SHORT_DOC, doc_name="test_policy", family_id="fam1")
    for chunk in chunks:
        raw_text = chunk["text"][len(E5_PREFIX):]
        assert chunk["anchor"] == raw_text[:80]


def test_section_aware_text_before_first_heading_is_kept():
    doc = "מבוא כללי ללא כותרת.\n## כיסוי\nתוכן הכיסוי.\n"
    chunks = chunk_section_aware(doc, doc_name="doc", family_id="fam1")
    all_text = " ".join(c["text"] for c in chunks)
    assert "מבוא כללי" in all_text

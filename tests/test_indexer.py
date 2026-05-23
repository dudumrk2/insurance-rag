"""Tests for src.indexer — ChromaDB collection management.

All tests use a temporary directory and an in-memory ChromaDB client so
they run fast without touching the real on-disk index and without needing
the embedding model (fake random embeddings are used).
"""

import numpy as np
import pytest

from src.indexer import _collection_name, build_collection, load_collection

# Embedding dimension for multilingual-e5-large.
_DIM = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_chunks(n: int, strategy: str = "fixed", family_id: str = "fam1") -> list[dict]:
    return [
        {
            "chunk_id": f"{family_id}_{strategy}_doc_{i}",
            "text": f"passage: תוכן chunk מספר {i}",
            "source_doc": "doc",
            "strategy": strategy,
            "family_id": family_id,
            "anchor": f"תוכן chunk מספר {i}"[:80],
            "section": f"## סעיף {i}" if i % 2 == 0 else None,
        }
        for i in range(n)
    ]


def _fake_embeddings(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, _DIM)).astype(np.float32)
    # L2-normalise to mimic real model output.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture()
def chroma_client(tmp_path):
    """Ephemeral in-memory ChromaDB client (no disk I/O)."""
    chromadb = pytest.importorskip("chromadb")
    return chromadb.EphemeralClient()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collection_name_format():
    assert _collection_name("fixed") == "insurance_fixed"
    assert _collection_name("section_aware") == "insurance_section_aware"


def test_build_collection_creates_collection(chroma_client):
    chunks = _fake_chunks(5)
    embs = _fake_embeddings(5)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    assert col is not None


def test_build_collection_count_matches_chunks(chroma_client):
    n = 10
    chunks = _fake_chunks(n)
    embs = _fake_embeddings(n)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    assert col.count() == n


def test_build_collection_metadata_stored(chroma_client):
    chunks = _fake_chunks(3)
    embs = _fake_embeddings(3)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    result = col.get(include=["metadatas"])
    meta = result["metadatas"][0]
    assert "source_doc" in meta
    assert "strategy" in meta
    assert "family_id" in meta
    assert "anchor" in meta
    assert "section" in meta


def test_build_collection_ids_match_chunk_ids(chroma_client):
    chunks = _fake_chunks(4)
    embs = _fake_embeddings(4)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    result = col.get()
    stored_ids = set(result["ids"])
    expected_ids = {c["chunk_id"] for c in chunks}
    assert stored_ids == expected_ids


def test_build_collection_replaces_existing(chroma_client):
    """Calling build_collection twice should not raise and final count is fresh."""
    chunks_v1 = _fake_chunks(5)
    embs_v1 = _fake_embeddings(5)
    build_collection("fixed", chunks_v1, embs_v1, client=chroma_client)

    chunks_v2 = _fake_chunks(3)
    embs_v2 = _fake_embeddings(3)
    build_collection("fixed", chunks_v2, embs_v2, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    assert col.count() == 3


def test_load_collection_returns_built_collection(chroma_client):
    chunks = _fake_chunks(6)
    embs = _fake_embeddings(6)
    build_collection("section_aware", chunks, embs, client=chroma_client)

    col = load_collection("section_aware", client=chroma_client)
    assert col.count() == 6


def test_load_collection_raises_for_missing(chroma_client):
    """Loading a collection that was never built should raise."""
    with pytest.raises(Exception):
        load_collection("nonexistent", client=chroma_client)


def test_section_none_stored_as_empty_string(chroma_client):
    """ChromaDB metadata values must be str/int/float; None → ''."""
    chunks = _fake_chunks(2)
    chunks[1]["section"] = None
    embs = _fake_embeddings(2)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    result = col.get(include=["metadatas"])
    sections = [m["section"] for m in result["metadatas"]]
    assert all(isinstance(s, str) for s in sections), (
        "section must be stored as str, not None"
    )

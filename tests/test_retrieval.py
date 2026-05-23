"""Tests for src.retrieval — semantic search over indexed chunks.

All tests use a temporary in-memory ChromaDB client with fake embeddings
so they run fast without the embedding model or disk I/O.

Tests verify the retrieve() function returns chunks ranked by similarity
to a query, filtered by family_id, and with proper metadata.
"""

import numpy as np
import pytest

from src.retrieval import retrieve


# Embedding dimension for multilingual-e5-large.
_DIM = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_chunks(
    n: int,
    strategy: str = "fixed",
    family_id: str = "fam1",
) -> list[dict]:
    """Generate synthetic test chunks."""
    return [
        {
            "chunk_id": f"{family_id}_{strategy}_doc_{i}",
            "text": f"passage: תוכן chunk מספר {i}",
            "source_doc": f"doc_{i % 3}.md",
            "strategy": strategy,
            "family_id": family_id,
            "anchor": f"תוכן chunk מספר {i}"[:80],
            "section": f"## סעיף {i}" if i % 2 == 0 else None,
        }
        for i in range(n)
    ]


def _fake_embeddings(n: int) -> np.ndarray:
    """Generate synthetic L2-normalized embeddings."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, _DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _fake_query_embedding() -> np.ndarray:
    """Generate a synthetic query embedding (known seed for reproducibility)."""
    rng = np.random.default_rng(123)
    vec = rng.standard_normal(_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


def _axis_vector(*components: tuple[int, float]) -> np.ndarray:
    """Build an L2-normalized vector with the given (index, value) components.

    Used to construct embeddings whose cosine similarity to a query is known
    exactly, so ordering assertions are unambiguous (unlike random vectors,
    which are near-orthogonal and clamp to score 0.0).
    """
    vec = np.zeros(_DIM, dtype=np.float32)
    for idx, val in components:
        vec[idx] = val
    return vec / np.linalg.norm(vec)


@pytest.fixture()
def chroma_client():
    """Ephemeral in-memory ChromaDB client."""
    chromadb = pytest.importorskip("chromadb")
    return chromadb.EphemeralClient()


@pytest.fixture()
def populated_collection(chroma_client):
    """A ChromaDB collection with 5 fake chunks, fam1."""
    from src.indexer import build_collection

    chunks = _fake_chunks(5, strategy="fixed", family_id="fam1")
    embs = _fake_embeddings(5)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    return col


@pytest.fixture()
def mock_embed_fn():
    """A mock embed function that returns a known query vector."""
    return lambda query: _fake_query_embedding()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_list(populated_collection, mock_embed_fn):
    """retrieve() returns a list."""
    result = retrieve(
        query="test query",
        strategy="fixed",
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    assert isinstance(result, list)


def test_retrieve_returns_dicts_with_required_keys(
    populated_collection,
    mock_embed_fn,
):
    """Each result dict has all required keys."""
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    assert len(result) > 0, "Expected at least one result"

    required_keys = {
        "chunk_id",
        "text",
        "source_doc",
        "score",
        "anchor",
        "section",
        "family_id",
    }
    for item in result:
        assert isinstance(item, dict)
        assert required_keys.issubset(item.keys()), (
            f"Missing keys. Got: {item.keys()}, expected: {required_keys}"
        )


def test_retrieve_results_sorted_by_score_descending(
    populated_collection,
    mock_embed_fn,
):
    """Results are sorted by score in descending order."""
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    assert len(result) > 1, "Expected multiple results to verify ordering"
    scores = [item["score"] for item in result]
    assert scores == sorted(scores, reverse=True), (
        f"Scores not sorted descending: {scores}"
    )


def test_retrieve_orders_most_similar_first(chroma_client):
    """The chunk most similar to the query is returned first (descending score).

    Uses controlled embeddings with known cosine similarities so the ordering
    is unambiguous:
      - chunk A: identical to query        -> similarity 1.0
      - chunk B: 45 deg from query         -> similarity ~0.707
      - chunk C: orthogonal to query       -> similarity 0.0
    Chunks are inserted in WORST-first order to ensure the test fails if the
    function relies on insertion order rather than similarity.
    """
    from src.indexer import build_collection

    query_vec = _axis_vector((0, 1.0))
    chunk_a = _axis_vector((0, 1.0))  # identical -> most similar
    chunk_b = _axis_vector((0, 1.0), (1, 1.0))  # 45 deg
    chunk_c = _axis_vector((1, 1.0))  # orthogonal -> least similar

    def _chunk(name: str) -> dict:
        return {
            "chunk_id": f"fam1_fixed_{name}",
            "text": f"passage: chunk {name}",
            "source_doc": "doc.md",
            "strategy": "fixed",
            "family_id": "fam1",
            "anchor": name,
            "section": None,
        }

    # Insert worst-first (C, B, A) so insertion order != similarity order.
    chunks = [_chunk("C"), _chunk("B"), _chunk("A")]
    embs = np.vstack([chunk_c, chunk_b, chunk_a]).astype(np.float32)
    build_collection("fixed", chunks, embs, client=chroma_client)
    col = chroma_client.get_collection("insurance_fixed")

    result = retrieve(
        query="x",
        strategy="fixed",
        family_id="fam1",
        top_k=3,
        collection=col,
        embed_fn=lambda _query: query_vec,
    )

    ids = [r["chunk_id"] for r in result]
    assert ids == ["fam1_fixed_A", "fam1_fixed_B", "fam1_fixed_C"], (
        f"Expected most-similar-first ordering, got {ids}"
    )
    # Strictly descending scores for these distinct similarities.
    assert result[0]["score"] > result[1]["score"] > result[2]["score"], (
        f"Scores not strictly descending: {[r['score'] for r in result]}"
    )


def test_retrieve_respects_top_k(populated_collection, mock_embed_fn):
    """retrieve() returns at most top_k results."""
    top_k = 2
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        top_k=top_k,
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    assert len(result) <= top_k, (
        f"Got {len(result)} results, expected at most {top_k}"
    )


def test_retrieve_returns_fewer_than_top_k_if_collection_smaller(
    populated_collection,
    mock_embed_fn,
):
    """If collection has fewer chunks than top_k, return all chunks."""
    top_k = 100
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        top_k=top_k,
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    # populated_collection has exactly 5 chunks of fam1
    assert len(result) == 5, (
        f"Expected 5 results (size of collection), got {len(result)}"
    )


def test_retrieve_filters_by_family_id(chroma_client, mock_embed_fn):
    """retrieve() filters by family_id — fam1 chunks don't return fam2 chunks."""
    from src.indexer import build_collection

    # Build a collection with chunks from two families
    chunks_fam1 = _fake_chunks(3, strategy="fixed", family_id="fam1")
    chunks_fam2 = _fake_chunks(3, strategy="fixed", family_id="fam2")
    all_chunks = chunks_fam1 + chunks_fam2
    all_embs = _fake_embeddings(6)
    build_collection("fixed", all_chunks, all_embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")

    # Query for fam1 only
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        top_k=10,
        collection=col,
        embed_fn=mock_embed_fn,
    )

    # All results should be from fam1
    assert all(item["family_id"] == "fam1" for item in result), (
        f"Retrieved chunks from other families: {[item['family_id'] for item in result]}"
    )
    # Should get exactly 3 results (the fam1 chunks)
    assert len(result) == 3, f"Expected 3 results, got {len(result)}"


def test_retrieve_score_between_0_and_1(populated_collection, mock_embed_fn):
    """All scores are between 0 and 1 (cosine similarity range)."""
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    assert len(result) > 0, "Expected at least one result"

    for item in result:
        score = item["score"]
        assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"
        assert 0 <= score <= 1, (
            f"Score {score} out of valid range [0, 1]"
        )


def test_retrieve_empty_collection_returns_empty_list(
    chroma_client,
    mock_embed_fn,
):
    """Retrieving from an empty collection returns an empty list."""
    from src.indexer import build_collection

    chunks = []
    embs = _fake_embeddings(0)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    result = retrieve(
        query="test query",
        strategy="fixed",
        collection=col,
        embed_fn=mock_embed_fn,
    )
    assert result == [], f"Expected empty list, got {result}"


def test_retrieve_text_contains_passage_prefix(
    populated_collection,
    mock_embed_fn,
):
    """The 'text' field in results contains the stored 'passage: ' prefix."""
    result = retrieve(
        query="test query",
        strategy="fixed",
        family_id="fam1",
        collection=populated_collection,
        embed_fn=mock_embed_fn,
    )
    assert len(result) > 0, "Expected at least one result"

    # All texts should start with "passage: " because that's how they're stored
    for item in result:
        assert item["text"].startswith("passage: "), (
            f"Expected text to start with 'passage: ', got '{item['text'][:30]}...'"
        )


def test_retrieve_section_empty_string_when_none(
    chroma_client,
    mock_embed_fn,
):
    """When section is None in metadata, it's returned as empty string."""
    from src.indexer import build_collection

    chunks = _fake_chunks(2, strategy="fixed", family_id="fam1")
    chunks[0]["section"] = "## סעיף א'"
    chunks[1]["section"] = None
    embs = _fake_embeddings(2)
    build_collection("fixed", chunks, embs, client=chroma_client)

    col = chroma_client.get_collection("insurance_fixed")
    result = retrieve(
        query="test query",
        strategy="fixed",
        collection=col,
        embed_fn=mock_embed_fn,
    )

    # Both should have section as string (never None)
    for item in result:
        assert isinstance(item["section"], str), (
            f"Section must be string, got {type(item['section'])}"
        )

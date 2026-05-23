"""Tests for src.embedder.

Fast tests (always run): pure prefix logic + empty-list handling.
Slow tests (marked): load the real multilingual-e5-large model.

Run slow tests with:  pytest -m slow
Skip slow tests with: pytest -m "not slow"   (default CI behaviour)
"""

import numpy as np
import pytest

from src.embedder import _with_query_prefix, embed_texts


# ---------------------------------------------------------------------------
# Fast: pure logic — no model required
# ---------------------------------------------------------------------------


def test_query_prefix_added_when_missing():
    assert _with_query_prefix("כיסוי גניבה?") == "query: כיסוי גניבה?"


def test_query_prefix_not_doubled_when_present():
    already = "query: כיסוי גניבה?"
    assert _with_query_prefix(already) == already


def test_embed_texts_empty_list_returns_empty_array():
    result = embed_texts([])
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Slow: require real model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_embed_texts_shape():
    """embedding dimension for multilingual-e5-large is 1024."""
    texts = ["passage: שלום עולם", "passage: כיסוי ביטוח"]
    result = embed_texts(texts)
    assert result.shape == (2, 1024)


@pytest.mark.slow
def test_embed_texts_l2_normalized():
    """sentence-transformers with normalize_embeddings=True → unit vectors."""
    texts = ["passage: בדיקת נורמליזציה"]
    result = embed_texts(texts)
    norm = float(np.linalg.norm(result[0]))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.slow
def test_embed_query_returns_1d_unit_vector():
    from src.embedder import embed_query

    vec = embed_query("מה הפרנשייז על רכב?")
    assert vec.ndim == 1
    assert vec.shape[0] == 1024
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


@pytest.mark.slow
def test_embed_query_adds_prefix_transparently():
    """embed_query('text') and embed_texts(['query: text']) must be identical."""
    from src.embedder import embed_query

    query = "מה הפרנשייז?"
    via_query = embed_query(query)
    via_texts = embed_texts([f"query: {query}"])[0]
    assert np.allclose(via_query, via_texts, atol=1e-5)

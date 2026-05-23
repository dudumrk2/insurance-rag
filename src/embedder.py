"""Thin wrapper around sentence-transformers for multilingual-e5-large.

The model is loaded lazily on first use (singleton) so importing this module
has zero cost when building chunks or running tests that don't need embeddings.

e5 prefix convention (CRITICAL for retrieval quality):
  - Passages / chunks  → "passage: " + text
  - Queries at runtime → "query: " + text
  Chunks already carry the "passage: " prefix from src.chunking.
  embed_query() adds "query: " automatically if missing.
"""

from __future__ import annotations

import numpy as np

from src.config import EMBEDDING_MODEL

_model = None  # lazy singleton


def _get_model():
    """Load (or return cached) SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Public helpers (also used in tests)
# ---------------------------------------------------------------------------


def _with_query_prefix(text: str) -> str:
    """Add ``'query: '`` prefix if not already present."""
    if text.startswith("query: "):
        return text
    return f"query: {text}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of texts and return a float32 numpy array.

    Texts are expected to already carry the appropriate e5 prefix
    (``'passage: '`` for chunks, ``'query: '`` for queries).

    Args:
        texts:         List of strings to embed.
        batch_size:    Encoding batch size (trade-off: memory vs. speed).
        show_progress: Show tqdm progress bar (useful for large corpora).

    Returns:
        ``np.ndarray`` of shape ``(len(texts), 1024)``, L2-normalised.
        Returns an empty ``(0, 1024)`` array when *texts* is empty.
    """
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    model = _get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Adds the ``'query: '`` prefix required by multilingual-e5-large if the
    caller omits it.

    Returns:
        1-D ``np.ndarray`` of shape ``(1024,)``, L2-normalised.
    """
    return embed_texts([_with_query_prefix(query)])[0]

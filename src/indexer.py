"""ChromaDB collection management for the insurance RAG index.

Two collections are maintained, one per chunking strategy:
  - ``insurance_fixed``          (from chunk_fixed)
  - ``insurance_section_aware``  (from chunk_section_aware)

Each collection stores:
  - document text (with ``'passage: '`` prefix — for reference only)
  - pre-computed embedding vector
  - metadata: source_doc, strategy, family_id, anchor, section

Filtering by ``family_id`` is handled at query time via ChromaDB's ``where``
clause, enabling multi-tenancy without separate collections per family.

Usage::

    from src.indexer import build_collection, load_collection
    build_collection("fixed", chunks, embeddings)
    col = load_collection("fixed")
    results = col.query(query_embeddings=[vec], n_results=5,
                        where={"family_id": "demo_family_001"})
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import INDICES_DIR

_COLLECTION_PREFIX = "insurance"
_UPSERT_BATCH = 100  # ChromaDB recommends ≤ 5 461 but 100 is safe for RAM


def _collection_name(strategy: str) -> str:
    """Return the ChromaDB collection name for a given strategy."""
    return f"{_COLLECTION_PREFIX}_{strategy}"


def build_collection(
    strategy: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    persist_dir: Path = INDICES_DIR,
    client=None,  # chromadb.Client — injected for tests
) -> None:
    """Create (or replace) a ChromaDB collection for *strategy*.

    Args:
        strategy:    ``"fixed"`` or ``"section_aware"``.
        chunks:      List of chunk dicts produced by ``src.chunking``.
        embeddings:  Float32 array of shape ``(len(chunks), 1024)``.
        persist_dir: Directory for the PersistentClient (ignored when *client*
                     is provided, e.g. in tests).
        client:      Optional pre-built ChromaDB client (used in tests to
                     inject an EphemeralClient without disk I/O).
    """
    if client is None:
        import chromadb  # noqa: PLC0415

        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))

    name = _collection_name(strategy)

    # Drop existing collection so a rebuild is always clean.
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 — collection may not exist yet
        pass

    collection = client.create_collection(
        name,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches to keep memory usage bounded.
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        batch_embs = embeddings[start : start + _UPSERT_BATCH]

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=batch_embs.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source_doc": c["source_doc"],
                    "strategy": c["strategy"],
                    "family_id": c["family_id"],
                    "anchor": c["anchor"],
                    # ChromaDB metadata values must be str/int/float — not None.
                    "section": c["section"] or "",
                }
                for c in batch
            ],
        )


def load_collection(
    strategy: str,
    persist_dir: Path = INDICES_DIR,
    client=None,  # chromadb.Client — injected for tests
):
    """Load an existing ChromaDB collection for *strategy*.

    Raises:
        Exception: if the collection has not been built yet.
    """
    if client is None:
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(persist_dir))

    return client.get_collection(_collection_name(strategy))

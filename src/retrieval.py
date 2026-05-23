"""Semantic search over indexed insurance chunks.

retrieve() queries a ChromaDB collection by semantic similarity, filtered by
family_id, and returns the top-k most relevant chunks with metadata.

Usage::

    from src.retrieval import retrieve

    results = retrieve(
        query="What is covered under liability?",
        strategy="section_aware",
        family_id="demo_family_001",
        top_k=5,
    )
    for result in results:
        print(f"{result['score']:.2f} | {result['text'][:50]}...")
"""

from __future__ import annotations

from src.config import DEFAULT_FAMILY_ID, DEFAULT_TOP_K
from src.embedder import embed_query


def retrieve(
    query: str,
    strategy: str = "section_aware",
    family_id: str = DEFAULT_FAMILY_ID,
    top_k: int = DEFAULT_TOP_K,
    collection=None,  # chromadb.Collection — injected for tests
    embed_fn=None,  # function to embed query — injected for tests
) -> list[dict]:
    """Retrieve top-k chunks most similar to the query.

    Args:
        query:       The search query (Hebrew or English).
        strategy:    Chunking strategy: ``"fixed"`` or ``"section_aware"``.
        family_id:   Multi-tenant family ID to filter results.
        top_k:       Maximum number of results to return.
        collection:  ChromaDB collection (if None, loaded from disk using strategy).
        embed_fn:    Function to embed the query (if None, uses ``src.embedder.embed_query``).

    Returns:
        List of result dicts, each with:
          - chunk_id: str
          - text: str (the full "passage: ..." text as stored)
          - source_doc: str
          - score: float (cosine similarity, 0-1, higher = better)
          - anchor: str
          - section: str (empty string if None)
          - family_id: str

        Sorted by score descending. At most top_k results.
        Returns an empty list if no matches are found.

    Raises:
        Exception: if the collection has not been built yet (when collection=None).
    """
    # Load collection if not injected
    if collection is None:
        from src.indexer import load_collection

        collection = load_collection(strategy)

    # Embed the query
    if embed_fn is None:
        embed_fn = embed_query
    query_embedding = embed_fn(query)

    # Query ChromaDB with family_id filter
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"family_id": family_id},
        include=["documents", "metadatas", "distances"],
    )

    # Convert ChromaDB result to our format
    # ChromaDB returns: {ids, documents, metadatas, distances, ...}
    # distances is a list of lists (one inner list per query, we have 1 query)
    # For cosine distance with L2-normalized vectors: similarity = 1 - distance
    if not result["ids"] or not result["ids"][0]:
        return []

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    results = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        # Convert distance to similarity (cosine: sim = 1 - dist)
        # For L2-normalized vectors, sim can be in [-1, 1]; clamp to [0, 1] for practical use
        similarity = max(0.0, min(1.0, 1.0 - distance))

        results.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source_doc": metadata["source_doc"],
                "score": similarity,
                "anchor": metadata["anchor"],
                "section": metadata["section"],  # already a string (empty if was None)
                "family_id": metadata["family_id"],
            }
        )

    # Results from ChromaDB are already sorted by distance (ascending),
    # so we need to reverse to get descending similarity order
    results.reverse()

    return results

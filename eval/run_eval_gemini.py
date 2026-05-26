"""Phase 0 ablation — add gemini-embedding-001 as a 5th retrieval config.

Local validation only. NOT part of the frozen academic deployment/demo.

What it does:
  * Rebuilds the EXACT same 447 section_aware chunks from data/redacted/*.md
    (chunk_section_aware — the proven strategy).
  * Swaps ONLY the embedder: e5  ->  gemini-embedding-001 via the Google API,
    output_dimensionality=768, task_type RETRIEVAL_DOCUMENT for chunks and
    RETRIEVAL_QUERY for questions (NO e5 "passage:"/"query:" prefixes).
  * Builds an ephemeral ChromaDB collection (reusing src.indexer.build_collection).
  * Runs the SAME eval/run_eval._eval_run over the 50 gold questions.
  * Prints Hit@1/3/5 + MRR next to the e5 baseline (Hit@5 0.720 / MRR 0.534).

The GEMINI_API_KEY is loaded by insurance-rag's own src.config (python-dotenv)
on import — this script never reads or prints the key.

Usage:
    .venv\\Scripts\\python.exe eval\\run_eval_gemini.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

# Importing src.config (transitively, via src.chunking) loads .env automatically.
from src.config import DEFAULT_FAMILY_ID, REDACTED_DIR  # noqa: E402
from src.chunking import chunk_section_aware  # noqa: E402
from src.indexer import build_collection  # noqa: E402
from src.retrieval import retrieve  # noqa: E402
from run_eval import _eval_run  # noqa: E402  (reuse the tested eval harness)

import os  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

# --- Config ---------------------------------------------------------------
EMBED_MODEL = "gemini-embedding-001"
OUTPUT_DIM = 768
TOP_K = 5
GOLD_PATH = ROOT / "eval" / "gold_set.jsonl"

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

# e5 baseline (section_aware) for comparison
E5_BASELINE = {"hit@5": 0.720, "mrr": 0.534}

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# --- Embedding helpers ----------------------------------------------------
def _strip(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def _normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _embed_batch(texts: list[str], task_type: str, max_retries: int = 6) -> list[list[float]]:
    """Embed a batch with gemini-embedding-001, retrying on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = _client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=OUTPUT_DIM,
                ),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            err = str(e)
            transient = any(s in err for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "DEADLINE"))
            if attempt == max_retries or not transient:
                raise
            wait = 3 + attempt * 2
            print(f"  ↻ embed retry {attempt}/{max_retries} after transient error (waiting {wait}s): {err[:120]}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def embed_documents(raw_texts: list[str], batch: int = 32) -> np.ndarray:
    vectors: list[np.ndarray] = []
    total = len(raw_texts)
    for i in range(0, total, batch):
        chunk = raw_texts[i:i + batch]
        values = _embed_batch(chunk, "RETRIEVAL_DOCUMENT")
        vectors.extend(_normalize(v) for v in values)
        print(f"  embedded {min(i + batch, total)}/{total} chunks")
        time.sleep(0.5)
    return np.vstack(vectors).astype(np.float32)


def gemini_embed_query(query: str) -> np.ndarray:
    raw = _strip(query, _QUERY_PREFIX)
    values = _embed_batch([raw], "RETRIEVAL_QUERY")[0]
    return _normalize(values)


# --- Main -----------------------------------------------------------------
def main() -> None:
    # 1. Rebuild the exact 447 section_aware chunks.
    md_files = sorted(REDACTED_DIR.glob("*.md"))
    chunks: list[dict] = []
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        chunks.extend(chunk_section_aware(text, md_path.stem, DEFAULT_FAMILY_ID))
    print(f"Rebuilt {len(chunks)} section_aware chunks from {len(md_files)} docs.")

    # 2. Embed chunks with gemini-embedding-001 (strip e5 'passage: ' prefix).
    raw_texts = [_strip(c["text"], _PASSAGE_PREFIX) for c in chunks]
    print(f"Embedding {len(raw_texts)} chunks via {EMBED_MODEL} (dim={OUTPUT_DIM})...")
    embeddings = embed_documents(raw_texts)

    # 3. Build an ephemeral ChromaDB collection (cosine).
    import chromadb  # noqa: PLC0415

    strategy = "section_aware_gemini"
    for c in chunks:
        c["strategy"] = strategy  # makes the collection name unique
    chroma_client = chromadb.EphemeralClient()
    build_collection(strategy, chunks, embeddings, client=chroma_client)
    collection = chroma_client.get_collection(f"insurance_{strategy}")
    print(f"Built ephemeral collection 'insurance_{strategy}' with {collection.count()} vectors.")

    # 4. Load gold set + run the same eval, injecting the gemini query embedder.
    import json  # noqa: PLC0415

    with GOLD_PATH.open(encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(gold)} gold questions.\n")

    def gemini_retrieve(**kwargs):
        return retrieve(embed_fn=gemini_embed_query, **kwargs)

    metrics = _eval_run(
        gold,
        strategy,
        DEFAULT_FAMILY_ID,
        TOP_K,
        collection=collection,
        retrieve_fn=gemini_retrieve,
    )

    # 5. Report.
    print("\n" + "=" * 64)
    print("PHASE 0 ABLATION — gemini-embedding-001 (768d) vs e5 baseline")
    print("=" * 64)
    print(f"{'Config':<28}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    print("-" * 64)
    print(f"{'section_aware (e5, base)':<28}{'—':>8}{'—':>8}"
          f"{E5_BASELINE['hit@5']:>8.3f}{E5_BASELINE['mrr']:>8.3f}")
    print(f"{'section_aware (gemini-768)':<28}"
          f"{metrics['hit@1']:>8.3f}{metrics['hit@3']:>8.3f}"
          f"{metrics['hit@5']:>8.3f}{metrics['mrr']:>8.3f}")
    print("-" * 64)
    d_hit5 = metrics["hit@5"] - E5_BASELINE["hit@5"]
    d_mrr = metrics["mrr"] - E5_BASELINE["mrr"]
    print(f"{'delta vs e5':<28}{'':>8}{'':>8}{d_hit5:>+8.3f}{d_mrr:>+8.3f}")
    print("=" * 64)
    print(f"Questions: {metrics['n_questions']}  |  Hits@5: {metrics['n_hits_at_5']}")


if __name__ == "__main__":
    main()

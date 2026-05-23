"""CLI: embed all chunks and build ChromaDB indices (Step 3).

Reads ``data/processed/chunks_fixed.jsonl`` and
``data/processed/chunks_section_aware.jsonl``, embeds every chunk with
``intfloat/multilingual-e5-large``, and persists two ChromaDB collections
under ``indices/``:

    indices/  →  ChromaDB PersistentClient root
      insurance_fixed/
      insurance_section_aware/

Usage::

    python build_index.py [--strategy {fixed,section_aware,both}]
                          [--batch-size N]

Options:
    --strategy    Which index(es) to build (default: both).
    --batch-size  Embedding batch size (default: 32).

The script is idempotent: re-running it rebuilds the collection from scratch.
On a CPU-only machine the first run downloads ~1.2 GB for the e5-large model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import INDICES_DIR, PROCESSED_DIR
from src.embedder import embed_texts
from src.indexer import build_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("build_index")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from chunks.")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "section_aware", "both"],
        default="both",
        help="Which strategy's index to build (default: both).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32).",
    )
    return parser.parse_args()


def _load_chunks(strategy: str) -> list[dict]:
    path = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
    if not path.exists():
        log.error("Missing chunks file: %s — run scripts/chunk.py first.", path)
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build(strategy: str, batch_size: int) -> None:
    log.info("─── Strategy: %s ───", strategy)

    chunks = _load_chunks(strategy)
    log.info("Loaded %d chunks.", len(chunks))

    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    log.info("Embedding %d texts (batch_size=%d) …", len(texts), batch_size)
    embeddings: np.ndarray = embed_texts(texts, batch_size=batch_size, show_progress=True)
    elapsed = time.perf_counter() - t0
    log.info("Embedding done in %.1f s  (%.0f chunks/s).", elapsed, len(texts) / elapsed)

    log.info("Writing ChromaDB collection …")
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    build_collection(strategy, chunks, embeddings)
    log.info("Collection 'insurance_%s' built with %d vectors.", strategy, len(chunks))


def main() -> None:
    args = _parse_args()

    strategies = (
        ["fixed", "section_aware"] if args.strategy == "both" else [args.strategy]
    )

    for strategy in strategies:
        _build(strategy, args.batch_size)

    log.info("")
    log.info("=== Index build complete ===")
    log.info("Persistent store: %s", INDICES_DIR)


if __name__ == "__main__":
    main()

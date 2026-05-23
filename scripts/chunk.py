"""CLI: chunk all redacted Markdown files using both strategies.

Reads every ``data/redacted/*.md`` file, applies ``chunk_fixed`` and
``chunk_section_aware``, and writes two JSONL files:

    data/processed/chunks_fixed.jsonl
    data/processed/chunks_section_aware.jsonl

Each line is a JSON object with the fields documented in ``src/chunking.py``.

Usage::

    python scripts/chunk.py [--family-id FAMILY_ID]

Options:
    --family-id   Override the family_id metadata on every chunk
                  (default: ``config.DEFAULT_FAMILY_ID``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sure the project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_fixed, chunk_section_aware
from src.config import DEFAULT_FAMILY_ID, PROCESSED_DIR, REDACTED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("chunk")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk redacted Markdown files.")
    parser.add_argument(
        "--family-id",
        default=DEFAULT_FAMILY_ID,
        help=f"family_id metadata injected into every chunk (default: {DEFAULT_FAMILY_ID})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    family_id: str = args.family_id

    md_files = sorted(REDACTED_DIR.glob("*.md"))
    if not md_files:
        log.error("No .md files found in %s — run scripts/redact.py first.", REDACTED_DIR)
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    strategies = {
        "fixed": chunk_fixed,
        "section_aware": chunk_section_aware,
    }

    totals: dict[str, int] = {name: 0 for name in strategies}

    # Open both output files; write one JSON line per chunk.
    handles = {
        name: (PROCESSED_DIR / f"chunks_{name}.jsonl").open("w", encoding="utf-8")
        for name in strategies
    }

    try:
        for md_path in md_files:
            doc_name = md_path.stem
            text = md_path.read_text(encoding="utf-8")

            for strategy_name, fn in strategies.items():
                chunks = fn(text, doc_name=doc_name, family_id=family_id)
                for chunk in chunks:
                    handles[strategy_name].write(json.dumps(chunk, ensure_ascii=False) + "\n")
                totals[strategy_name] += len(chunks)
                log.info(
                    "%-14s → %s: %d chunk(s)",
                    strategy_name,
                    doc_name,
                    len(chunks),
                )
    finally:
        for fh in handles.values():
            fh.close()

    log.info("")
    log.info("=== Chunking summary ===")
    for name, count in totals.items():
        out_path = PROCESSED_DIR / f"chunks_{name}.jsonl"
        log.info("  %-20s %4d chunks  → %s", name, count, out_path)


if __name__ == "__main__":
    main()

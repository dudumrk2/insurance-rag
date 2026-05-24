"""Ablation study — compare retrieval strategies over the gold set.

Four configurations are evaluated:

  A  section_aware  — natural Markdown sections (≤700 tokens)
  B  fixed_500      — sliding window 500 tok / overlap 50  (existing index)
  C  fixed_300      — sliding window 300 tok / overlap 50  (built in-memory)
  D  fixed_700      — sliding window 700 tok / overlap 50  (built in-memory)

Metrics reported per configuration (over 50 gold questions):

  Hit@k  — fraction of questions where the gold passage appears in top-k results
  MRR    — Mean Reciprocal Rank

Usage::

    python eval/run_eval.py [--gold eval/gold_set.jsonl]
                            [--top-k 5]
                            [--out eval/ablation_results.md]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_FAMILY_ID, PROCESSED_DIR, REDACTED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("run_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOP_K_DEFAULT = 5
_GOLD_DEFAULT = Path("eval/gold_set.jsonl")
_OUT_DEFAULT = Path("eval/ablation_results.md")


# ---------------------------------------------------------------------------
# Metric helpers  (tested in tests/test_eval.py)
# ---------------------------------------------------------------------------


def _gold_rank(results: list[dict], gold_anchor: str) -> int | None:
    """Return the 1-based rank of the first result that contains *gold_anchor*.

    Matching is a substring check on the raw chunk text (after stripping the
    ``"passage: "`` prefix).  This handles both strategies:
    - section_aware: the gold anchor is exactly the first 80 chars of the chunk
    - fixed: the same passage may be in a differently-bounded chunk, so we look
      for the anchor string anywhere inside the retrieved text.

    Returns ``None`` when no result contains the gold anchor.
    """
    needle = gold_anchor.strip()
    for rank, result in enumerate(results, start=1):
        raw = result["text"][len("passage: "):]
        if needle in raw:
            return rank
    return None


def _hit_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of questions with a gold result in the top-*k* positions."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def _mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank over the question set."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------


def _eval_run(
    gold: list[dict],
    strategy: str,
    family_id: str,
    top_k: int,
    collection=None,          # ChromaDB collection; if None loaded from disk
    retrieve_fn=None,         # injected in tests
) -> dict:
    """Evaluate one retrieval configuration against the gold set.

    Returns a dict with keys: hit@1, hit@3, hit@5, mrr, n_questions.
    """
    if retrieve_fn is None:
        from src.retrieval import retrieve as retrieve_fn  # noqa: PLC0415

    ranks: list[int | None] = []
    for item in gold:
        results = retrieve_fn(
            query=item["question"],
            strategy=strategy,
            family_id=family_id,
            top_k=top_k,
            collection=collection,
        )
        ranks.append(_gold_rank(results, item["anchor"]))

    return {
        "hit@1": _hit_at_k(ranks, k=1),
        "hit@3": _hit_at_k(ranks, k=3),
        "hit@5": _hit_at_k(ranks, k=5),
        "mrr": _mrr(ranks),
        "n_questions": len(ranks),
        "n_hits_at_5": sum(1 for r in ranks if r is not None and r <= 5),
        "ranks": ranks,
    }


# ---------------------------------------------------------------------------
# Build ephemeral index (for fixed_300 / fixed_700 variants)
# ---------------------------------------------------------------------------


def _build_ephemeral_collection(chunk_size: int, chunk_overlap: int, family_id: str):
    """Chunk all redacted docs at *chunk_size*, embed, return an in-memory collection."""
    import chromadb  # noqa: PLC0415

    from src.chunking import chunk_fixed  # noqa: PLC0415
    from src.embedder import embed_texts  # noqa: PLC0415
    from src.indexer import build_collection  # noqa: PLC0415

    log.info("Building ephemeral fixed_%d index...", chunk_size)

    md_files = sorted(REDACTED_DIR.glob("*.md"))
    all_chunks: list[dict] = []
    for md_path in md_files:
        doc_name = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_fixed(text, doc_name, family_id, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    log.info("  Chunked → %d chunks (size=%d, overlap=%d)", len(all_chunks), chunk_size, chunk_overlap)

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, show_progress=True)
    log.info("  Embedded %d chunks.", len(all_chunks))

    strategy = f"fixed_{chunk_size}"
    # Override chunk strategy field so the collection name is unique
    for c in all_chunks:
        c["strategy"] = strategy

    client = chromadb.EphemeralClient()
    build_collection(strategy, all_chunks, embeddings, client=client)
    return client.get_collection(f"insurance_{strategy}"), len(all_chunks)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _render_table(runs: list[dict]) -> str:
    """Render a Markdown table from a list of run result dicts."""
    header = "| Configuration   | Chunks | Hit@1 | Hit@3 | Hit@5 |  MRR  |"
    sep    = "|-----------------|--------|-------|-------|-------|-------|"
    rows = []
    for r in runs:
        rows.append(
            f"| {r['label']:<15s} | {r['n_chunks']:>6} "
            f"| {_fmt(r['hit@1'])} | {_fmt(r['hit@3'])} | {_fmt(r['hit@5'])} "
            f"| {_fmt(r['mrr'])} |"
        )
    return "\n".join([header, sep] + rows)


def _render_report(runs: list[dict], top_k: int, n_questions: int) -> str:
    """Render the full Markdown ablation report."""
    table = _render_table(runs)

    # Find best run(s)
    best_mrr = max(r["mrr"] for r in runs)
    best_label = next(r["label"].strip() for r in runs if r["mrr"] == best_mrr)

    lines = [
        "# Ablation Study — Retrieval Strategy Comparison",
        "",
        f"**Gold set:** {n_questions} questions  |  **Top-k:** {top_k}",
        "",
        "## Results",
        "",
        table,
        "",
        "## Notes",
        "",
        f"- **Best MRR:** `{best_label}` ({_fmt(best_mrr)})",
        "- Hit@k = fraction of questions where the gold passage appears in top-k retrieved chunks.",
        "- MRR = Mean Reciprocal Rank (higher = gold chunk appears closer to position 1).",
        "- `section_aware` and `fixed_500` use the pre-built persistent ChromaDB indices.",
        "- `fixed_300` and `fixed_700` are built in-memory from the redacted Markdown files.",
        "- Gold anchors are first-80-char keys from section-aware chunks; substring matching",
        "  is used for fixed strategies (the same passage may sit inside a differently-bounded chunk).",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study over gold_set.jsonl.")
    parser.add_argument("--gold",  type=Path, default=_GOLD_DEFAULT)
    parser.add_argument("--top-k", type=int,  default=_TOP_K_DEFAULT)
    parser.add_argument("--out",   type=Path, default=_OUT_DEFAULT)
    parser.add_argument(
        "--skip-ephemeral",
        action="store_true",
        help="Skip fixed_300 and fixed_700 runs (faster, for quick sanity checks).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    top_k: int = args.top_k
    family_id = DEFAULT_FAMILY_ID

    # --- Load gold set ---
    if not args.gold.exists():
        log.error("Gold set not found: %s", args.gold)
        sys.exit(1)
    with args.gold.open(encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d gold questions from %s", len(gold), args.gold)

    # --- Load pre-built persistent collections ---
    from src.indexer import load_collection  # noqa: PLC0415

    log.info("Loading section_aware collection...")
    col_section = load_collection("section_aware")
    n_section = col_section.count()

    log.info("Loading fixed collection (fixed_500)...")
    col_fixed500 = load_collection("fixed")
    n_fixed500 = col_fixed500.count()

    # --- Run A: section_aware ---
    log.info("─── Run A: section_aware ───")
    metrics_a = _eval_run(gold, "section_aware", family_id, top_k, collection=col_section)
    log.info("  Hit@5=%.3f  MRR=%.3f", metrics_a["hit@5"], metrics_a["mrr"])

    # --- Run B: fixed_500 ---
    log.info("─── Run B: fixed_500 ───")
    metrics_b = _eval_run(gold, "fixed", family_id, top_k, collection=col_fixed500)
    log.info("  Hit@5=%.3f  MRR=%.3f", metrics_b["hit@5"], metrics_b["mrr"])

    runs = [
        {"label": "section_aware  ", **metrics_a, "n_chunks": n_section},
        {"label": "fixed_500      ", **metrics_b, "n_chunks": n_fixed500},
    ]

    # --- Runs C & D: ephemeral fixed_300 / fixed_700 ---
    if not args.skip_ephemeral:
        for size in (300, 700):
            log.info("─── Run: fixed_%d ───", size)
            col, n_chunks = _build_ephemeral_collection(size, 50, family_id)
            metrics = _eval_run(gold, f"fixed_{size}", family_id, top_k, collection=col)
            log.info("  Hit@5=%.3f  MRR=%.3f", metrics["hit@5"], metrics["mrr"])
            runs.append({"label": f"fixed_{size}      "[:15] + " ", **metrics, "n_chunks": n_chunks})

    # --- Output ---
    report = _render_report(runs, top_k, len(gold))
    print(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    log.info("Report written to %s", args.out)


if __name__ == "__main__":
    main()

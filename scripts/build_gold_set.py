"""Build eval/gold_set.jsonl — 50 Hebrew QA pairs over the real corpus.

Two-phase semi-automatic pipeline:

  Phase 1 — Generation
    For each redacted Markdown document, send the full text to Gemini and ask
    it to generate N question-answer pairs *with an exact supporting quote*.
    The quote is a verbatim substring of the document that proves the answer.

  Phase 2 — Anchor matching
    For each generated QA pair, find the section-aware chunk whose text
    contains the quote (or the longest common substring when no exact match
    exists).  Use that chunk's ``anchor`` field as the gold citation key —
    this anchor survives both chunking strategies unchanged.

Output format (one JSON object per line)::

    {
      "id":       "q001",
      "question": "מה הפרנשייז על נזק מלא לרכב?",
      "answer":   "הפרנשייז הוא 3,000 ש\"ח",
      "anchor":   "<first-80-chars-of-matching-chunk>",
      "source_doc": "car_policy"
    }

Usage::

    python scripts/build_gold_set.py [--out eval/gold_set.jsonl]

Requires ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` env var (or ADC).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, REDACTED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("build_gold_set")

# ---------------------------------------------------------------------------
# Config — questions per document
# ---------------------------------------------------------------------------

_QUOTAS: dict[str, int] = {
    "car_policy":    15,
    "car_policy1":   20,
    "health_policy": 25,
    "home_policy":   15,
}

_GENERATION_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Phase 1 — Gemini generation
# ---------------------------------------------------------------------------

_SYSTEM = (
    "אתה עוזר שמייצר שאלות הערכה לפוליסות ביטוח. "
    "ענה בעברית בלבד. החזר JSON בלבד, ללא הסברים נוספים."
)

_USER_TEMPLATE = """\
להלן טקסט מפוליסת ביטוח. צור בדיוק {n} שאלות-תשובות בעברית.

חוקים חשובים:
1. כל שאלה חייבת להיות ניתנת למענה ישיר מהטקסט בלבד.
2. התשובות צריכות להיות עובדתיות וקצרות (עד 2 משפטים).
3. הוסף שדה "quote" עם ציטוט מילולי מדויק מהטקסט (30-120 תווים) שמוכיח את התשובה.
4. כסה נושאים שונים: גבולות כיסוי, השתתפות עצמית, אי-כיסויים, תנאים, תקופות המתנה.
5. אל תחזור על אותו נושא פעמיים.

החזר **מערך JSON בלבד** (ללא markdown, ללא ```json):
[
  {{"question": "...", "answer": "...", "quote": "...ציטוט מדויק מהטקסט..."}}
]

הטקסט:
---
{text}
---"""


def _generate_pairs(doc_name: str, text: str, n: int) -> list[dict]:
    """Call Gemini to generate n QA pairs for one document."""
    from google import genai  # noqa: PLC0415

    client = genai.Client()

    # Truncate very long docs to first ~40 000 chars (context window headroom).
    MAX_CHARS = 40_000
    excerpt = text[:MAX_CHARS]
    if len(text) > MAX_CHARS:
        log.warning("%s truncated to %d chars for generation.", doc_name, MAX_CHARS)

    prompt = _USER_TEMPLATE.format(n=n, text=excerpt)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=_GENERATION_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            # Strip markdown code fences if Gemini adds them despite instructions.
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError(f"expected a JSON list, got {type(pairs).__name__}")
            # Keep only well-formed dict entries; a stray string/number here would
            # crash the downstream `pair.get(...)` loop in main().
            pairs = [p for p in pairs if isinstance(p, dict)]
            log.info("%s: generated %d pairs (requested %d).", doc_name, len(pairs), n)
            return pairs[:n]
        except Exception as exc:  # noqa: BLE001
            log.warning("Attempt %d failed for %s: %s", attempt + 1, doc_name, exc)
            time.sleep(2 ** attempt)

    log.error("All attempts failed for %s — skipping.", doc_name)
    return []


# ---------------------------------------------------------------------------
# Phase 2 — Anchor matching
# ---------------------------------------------------------------------------


def _load_section_chunks() -> list[dict]:
    path = PROCESSED_DIR / "chunks_section_aware.jsonl"
    if not path.exists():
        log.error("chunks_section_aware.jsonl not found — run scripts/chunk.py first.")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Minimum distinctive length for an anchor. A very short anchor (e.g. a stray
# table cell like 'ד 25') is a substring of many chunks across documents, so in
# eval it would match the wrong chunk and falsely inflate Hit@k / MRR.
_MIN_ANCHOR_LEN = 10


def _is_usable_anchor(chunk: dict) -> bool:
    """True if the chunk's anchor is long enough to be a distinctive citation key."""
    return len(chunk["anchor"].strip()) >= _MIN_ANCHOR_LEN


def _find_anchor(quote: str, doc_chunks: list[dict]) -> str | None:
    """Return the anchor of the chunk that best contains *quote*.

    Tries exact substring match first, then falls back to the chunk with the
    longest overlapping token sequence. Only chunks with a distinctive anchor
    (see ``_MIN_ANCHOR_LEN``) are considered, so the gold key never collapses to
    a non-unique fragment.
    """
    quote_clean = quote.strip()
    usable = [c for c in doc_chunks if _is_usable_anchor(c)]

    # Pass 1: exact substring.
    for chunk in usable:
        raw = chunk["text"][len("passage: "):]
        if quote_clean in raw:
            return chunk["anchor"]

    # Pass 2: longest word overlap (handles minor Docling whitespace diffs).
    quote_tokens = set(quote_clean.split())
    best_anchor = None
    best_score = 0
    for chunk in usable:
        raw = chunk["text"][len("passage: "):]
        chunk_tokens = set(raw.split())
        score = len(quote_tokens & chunk_tokens)
        if score > best_score:
            best_score = score
            best_anchor = chunk["anchor"]

    return best_anchor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold_set.jsonl via Gemini.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/gold_set.jsonl"),
        help="Output path (default: eval/gold_set.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = _load_section_chunks()
    # Pre-group chunks by source_doc for fast lookup.
    chunks_by_doc: dict[str, list[dict]] = {}
    for c in all_chunks:
        chunks_by_doc.setdefault(c["source_doc"], []).append(c)

    md_files = sorted(REDACTED_DIR.glob("*.md"))
    if not md_files:
        log.error("No redacted .md files found — run scripts/redact.py first.")
        sys.exit(1)

    all_pairs: list[dict] = []
    q_index = 1

    for md_path in md_files:
        doc_name = md_path.stem
        n = _QUOTAS.get(doc_name, 10)
        text = md_path.read_text(encoding="utf-8")
        doc_chunks = chunks_by_doc.get(doc_name, [])

        log.info("─── %s (target: %d questions) ───", doc_name, n)
        pairs = _generate_pairs(doc_name, text, n)

        matched = 0
        skipped = 0
        for pair in pairs:
            question = pair.get("question", "").strip()
            answer = pair.get("answer", "").strip()
            quote = pair.get("quote", "").strip()

            if not question or not answer:
                skipped += 1
                continue

            anchor = _find_anchor(quote, doc_chunks) if quote else None
            if anchor is None:
                # Never write an empty/non-distinctive anchor: in eval a short or
                # empty anchor matches the wrong chunk(s) and falsely inflates
                # Hit@k / MRR. Fall back to the first chunk with a usable anchor,
                # else skip the pair entirely.
                fallback = next((c["anchor"] for c in doc_chunks if _is_usable_anchor(c)), None)
                if fallback is None:
                    log.warning("  No usable anchor for: %s — skipping.", question[:60])
                    skipped += 1
                    continue
                log.warning("  No anchor matched for: %s — using first-usable-chunk fallback.", question[:60])
                anchor = fallback

            all_pairs.append({
                "id": f"q{q_index:03d}",
                "question": question,
                "answer": answer,
                "anchor": anchor,
                "source_doc": doc_name,
            })
            q_index += 1
            matched += 1

        log.info("  → %d written, %d skipped.", matched, skipped)

    with out_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    log.info("")
    log.info("=== Gold set complete ===")
    log.info("  Total pairs: %d", len(all_pairs))
    log.info("  Written to:  %s", out_path)


if __name__ == "__main__":
    main()

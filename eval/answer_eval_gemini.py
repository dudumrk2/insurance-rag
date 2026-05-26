"""Answer-quality evaluation with gemini-embedding-001 retrieval.

Runs the same 10 questions as section 5.5 of the report, but swaps **only the
retrieval embeddings** from `multilingual-e5-large` to `gemini-embedding-001`.
Generation is unchanged (Gemini 2.5 Flash, temperature=0.2).

The output is a Markdown file with Q / Expected / Got / Sources for each
question — meant to be reviewed manually and classified
(Correct / Partially correct / Incorrect / Hallucinated).

What it does:
  1. Rebuild the same 447 section_aware chunks from data/redacted/*.md
  2. Embed all chunks with gemini-embedding-001 (RETRIEVAL_DOCUMENT, 768d)
  3. Build an ephemeral ChromaDB collection (cosine)
  4. For each of the first 10 gold questions:
     - retrieve top-5 via gemini-embedded ChromaDB
     - generate answer with Gemini 2.5 Flash (same prompt as src.generation)
  5. Write side-by-side comparison to eval/answer_eval_gemini_results.md

The GEMINI_API_KEY is loaded automatically by src.config (python-dotenv).
The script never reads or prints the key.

Usage:
    .venv\\Scripts\\python.exe eval\\answer_eval_gemini.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 stdout on Windows so Hebrew progress prints don't crash cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importing src.config (transitively, via src.chunking) loads .env automatically.
from src.config import DEFAULT_FAMILY_ID, REDACTED_DIR  # noqa: E402
from src.chunking import chunk_section_aware  # noqa: E402
from src.generation import answer  # noqa: E402
from src.indexer import build_collection  # noqa: E402
from src.retrieval import retrieve  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

# --- Config ---------------------------------------------------------------
EMBED_MODEL = "gemini-embedding-001"
OUTPUT_DIM = 768
TOP_K = 5
N_QUESTIONS = 10
GOLD_PATH = ROOT / "eval" / "gold_set.jsonl"
OUT_PATH = ROOT / "eval" / "answer_eval_gemini_results.md"

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# --- Embedding helpers (mirrors run_eval_gemini.py) -----------------------
def _strip(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def _normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _embed_batch(texts: list[str], task_type: str, max_retries: int = 6) -> list[list[float]]:
    """Embed a batch, retrying on transient errors (429, 503, 500, DEADLINE)."""
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
    # 1. Rebuild the same 447 section_aware chunks.
    md_files = sorted(REDACTED_DIR.glob("*.md"))
    chunks: list[dict] = []
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        chunks.extend(chunk_section_aware(text, md_path.stem, DEFAULT_FAMILY_ID))
    print(f"Rebuilt {len(chunks)} section_aware chunks from {len(md_files)} docs.")

    # 2. Embed with gemini-embedding-001.
    raw_texts = [_strip(c["text"], _PASSAGE_PREFIX) for c in chunks]
    print(f"Embedding {len(raw_texts)} chunks via {EMBED_MODEL} (dim={OUTPUT_DIM})...")
    embeddings = embed_documents(raw_texts)

    # 3. Build ephemeral ChromaDB collection (cosine).
    import chromadb  # noqa: PLC0415

    strategy = "section_aware_gemini"
    for c in chunks:
        c["strategy"] = strategy
    chroma_client = chromadb.EphemeralClient()
    build_collection(strategy, chunks, embeddings, client=chroma_client)
    collection = chroma_client.get_collection(f"insurance_{strategy}")
    print(f"Built ephemeral collection 'insurance_{strategy}' with {collection.count()} vectors.\n")

    # 4. Load gold set, take first N questions.
    with GOLD_PATH.open(encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    questions = gold[:N_QUESTIONS]
    print(f"Running answer evaluation on {len(questions)} questions...\n")

    # 5. Custom retrieve fn that uses the gemini-embedded collection.
    def gemini_retrieve(query: str, strategy: str, family_id: str, top_k: int):
        return retrieve(
            query=query,
            strategy=strategy,
            family_id=family_id,
            top_k=top_k,
            collection=collection,
            embed_fn=gemini_embed_query,
        )

    # 6. Run answer() for each question.
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{N_QUESTIONS}] {q['id']} ({q['category']}): {q['question'][:60]}...")
        res = answer(
            question=q["question"],
            family_id=DEFAULT_FAMILY_ID,
            strategy=strategy,
            top_k=TOP_K,
            _retrieve_fn=gemini_retrieve,
        )
        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["answer"],
            "got": res["answer"],
            "expected_anchor": q["anchor"],
            "sources": res["sources"],
        })
        # Rate-limit a bit; Gemini Flash is generous but we have 10 calls + 10 embeds.
        time.sleep(1.0)

    # 7. Write side-by-side comparison.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Answer evaluation — gemini-embedding-001 retrieval\n\n")
        f.write("Same 10 questions as report section 5.5, retrieval swapped to ")
        f.write(f"`{EMBED_MODEL}` ({OUTPUT_DIM}d). Generation unchanged ")
        f.write("(Gemini 2.5 Flash, temperature=0.2).\n\n")
        f.write("---\n\n")
        for r in results:
            f.write(f"## Q{r['id']} — {r['category']}\n\n")
            f.write(f"**שאלה:** {r['question']}\n\n")
            f.write(f"**תשובה צפויה (gold):** {r['expected']}\n\n")
            f.write(f"**תשובה שהתקבלה (gemini retrieval + Flash):**\n\n")
            f.write(f"> {r['got']}\n\n")
            f.write(f"**Anchor צפוי:** `{r['expected_anchor'][:80]}...`\n\n")
            f.write("**Sources שהושבו:**\n\n")
            for s in r["sources"]:
                f.write(f"- `{s[:80]}...`\n")
            f.write("\n**Classification:** _(fill manually: Correct / Partially / Incorrect / Hallucinated)_\n\n")
            f.write("---\n\n")

    print(f"\n✓ Results saved to {OUT_PATH.relative_to(ROOT)}")
    print(f"  Review manually and classify each answer.")


if __name__ == "__main__":
    main()

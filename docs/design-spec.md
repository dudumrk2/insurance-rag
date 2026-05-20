# Insurance Policy RAG — Design Spec

**Date:** 2026-05-20
**Status:** Approved (pending written-spec review)
**Branch:** `feat/insurance-rag`
**Author:** brainstorming session (Dudu + Claude)

---

## 1. Purpose & Context

Build a complete, controllable Retrieval-Augmented Generation (RAG) pipeline over a
corpus of Israeli (Hebrew) insurance policies. The project serves two goals:

1. **Academic mid-term assignment** — a self-contained project that meets a fixed rubric
   (data prep, loading, chunking, embedding, indexing, retrieval, generation, citations,
   gold set, evaluation, ablation, 4-page report). Must expose a fixed interface:
   `answer(question: str) -> dict`.
2. **Reusable feature for `ai-wealth-monitor`** — the same engine plugs into the existing
   chat (`/api/chat/ask`) so the demo user can upload policies and query them.

The work is sequenced **assignment-first**, then integration. Phase 1 (steps 1–7) is the
submission; Phase 2 (step 8) is the chat integration.

### Why this corpus is suitable for RAG
Israeli insurance policies contain dense, contract-specific knowledge (coverage limits,
deductibles, exclusions, waiting periods) that a baseline LLM does **not** know and would
hallucinate or be vague about without retrieval. This satisfies the assignment's "not a
generic Wikipedia-style corpus" requirement.

---

## 2. Scope

### In scope (Phase 1 — submission)
- PDF → Markdown conversion (Docling)
- PII redaction (regex + known family strings), text-mode
- Two chunking strategies (fixed-size, section-aware)
- Embeddings (`intfloat/multilingual-e5-large`)
- Vector store (ChromaDB, persistent)
- Retrieval + generation (Gemini 2.5 Flash) with citations
- Gold set of 50 Hebrew questions, anchor-based citations
- Evaluation (Hit@5, MRR, manual review of ≥10) + ablation table
- 4-page report

### In scope (Phase 2 — integration)
- `query_insurance_policies` tool registered in `dashboard_chat.py`
- `demo_seeder` indexes redacted demo policies under `family_id=DEMO_UID`
- Demo bypass skips mock for insurance questions only
- Installable as editable package (`pip install -e ./insurance-rag`)

### Out of scope (noted as "future work" in report)
- Cross-encoder reranking
- Hybrid (BM25 + dense) retrieval — optional stretch ablation only
- Upload→index live hook in `InsuranceFlow` (marked TODO)

---

## 3. Architecture

Three phases, separate entry points:

```
Phase 1 (Ingest, manual)   scripts/redact.py     PDF(raw) → Docling MD → redact → data/redacted/*.md
Phase 2 (Index, reproducible)  build_index.py     redacted MD → chunk(×2) → embed → ChromaDB(×2 collections)
Phase 3 (Ask, online)      src/rag_system.py     question → retrieve top-k → Gemini → {answer, sources, retrieved_chunks}
```

Two consumers of `answer()`:
- **Eval/CLI:** `eval/run_eval.py` imports `answer()`.
- **Integration:** `backend/routers/dashboard_chat.py` registers `answer()` as a Gemini tool.

---

## 4. Repository layout

```
insurance-rag/
├── data/
│   ├── raw/                          # original PDFs — gitignored (contain PII)
│   ├── redacted/                     # *.md, PII removed — committed
│   ├── processed/
│   │   ├── chunks_fixed_size.jsonl
│   │   └── chunks_section_aware.jsonl
│   ├── redaction_log.json            # what was removed + where (no PII values)
│   └── MANIFEST.md
├── src/
│   ├── __init__.py
│   ├── config.py                     # paths, model names, chunk sizes
│   ├── pdf_to_md.py                  # Docling → markdown
│   ├── redaction.py                  # regex + known-strings → clean MD
│   ├── chunking.py                   # FixedSizeChunker, SectionAwareChunker
│   ├── embeddings.py                 # sentence-transformers e5 wrapper
│   ├── vector_store.py               # ChromaDB wrapper (add/query/reset)
│   ├── retrieval.py                  # retrieve(query, k, strategy, family_id)
│   ├── generation.py                 # call Gemini, parse citations
│   ├── rag_system.py                 # answer() — public interface
│   └── utils.py                      # token counting, file IO, logging
├── scripts/
│   └── redact.py                     # CLI: data/raw/*.pdf → data/redacted/*.md
├── eval/
│   ├── gold_set.jsonl                # 50 Hebrew questions, anchor-based
│   ├── generate_gold_candidates.py   # Claude generates candidates for review
│   ├── run_eval.py                   # runs answer() over gold set, prints metrics
│   └── results/
│       ├── eval_fixed_size_500.json
│       ├── eval_section_aware.json
│       └── ablation_table.md
├── tests/
│   ├── conftest.py                   # fixtures (tiny MD corpus, 2 families)
│   ├── test_chunking.py
│   ├── test_redaction.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   └── test_e2e.py                   # build_index → answer, tenancy isolation
├── build_index.py                    # reproducible index build entry point
├── pyproject.toml                    # editable install
├── requirements.txt
├── report.md                         # → report.pdf
├── README.md                         # exact run instructions
└── .gitignore                        # data/raw/, indices/, __pycache__, *.pyc
```

Index storage (gitignored, rebuilt by `build_index.py`):
```
insurance-rag/indices/
├── chroma_fixed_size/
└── chroma_section_aware/
```

---

## 5. Components & interfaces

| Module | Public function | In | Out |
|---|---|---|---|
| `pdf_to_md` | `convert(pdf_path)` | path | str (markdown) |
| `redaction` | `redact(md_text, known_strings)` | str + list | (str_redacted, log_dict) |
| `chunking` | `FixedSizeChunker(size, overlap).split(doc)` | doc | list[Chunk] |
| `chunking` | `SectionAwareChunker(max_size).split(doc)` | doc | list[Chunk] |
| `embeddings` | `Embedder().encode(texts, is_query=False)` | list[str] | np.ndarray |
| `vector_store` | `VectorStore(strategy).add / query / reset` | — | — |
| `retrieval` | `retrieve(query, k, strategy, family_id)` | str+int+str+str | list[dict] |
| `generation` | `generate(question, chunks)` | str + list | dict(text, used_chunks) |
| `rag_system` | `answer(question, family_id, strategy="section_aware")` | str+str+str | dict (per spec) |

### Standard `Chunk` shape (everywhere)
```python
{
    "chunk_id": "health_policy__sa__sec_03",   # {doc_id}__{strategy_abbrev}__{seq:03d}
    "doc_id": "health_policy",
    "text": "...",
    "metadata": {
        "source": "health_policy.pdf",
        "page_start": 12,
        "page_end": 13,
        "section": "ניתוחים מיוחדים",
        "family_id": "demo_family_001",
        "strategy": "section_aware",
        "policy_type": "בריאות"
    }
}
```
Strategy abbreviations: `sa` (section_aware), `fs500` / `fs300` / `fs700` (fixed_size by size).

### `answer()` return contract (assignment-mandated)
```python
{
    "answer": str,
    "sources": list[str],          # chunk_ids cited
    "retrieved_chunks": list[dict] # each: chunk_id, text, score, metadata
}
```

---

## 6. Data flow details

### Phase 1 — Ingestion (manual, one-time)
1. `Docling.convert(pdf)` → structured markdown with `##` headings.
2. `redaction.redact(md, known_strings)`:
   - **Regex pass:** Israeli ID `\b\d{9}\b`, phone `\b0(5\d|[2-4]|7\d|8|9)\-?\d{7}\b`,
     email, license plate `\b\d{2,3}-\d{3}-\d{2,3}\b`, IBAN `\bIL\d{2}(?:\s?\d{4}){5}\b`.
   - **Known-strings pass:** names/IDs/email from family profile
     (mirrors `flow_utils.prepare_pdf_for_vision` logic, on text not images).
3. Output `data/redacted/*.md` + `data/redaction_log.json`.
   - Log records pattern type + surrounding context ONLY — never the raw PII value.
   - **Human review of the log is required before submission.**

### Phase 2 — Indexing (`build_index.py`, deterministic)
For each `data/redacted/*.md`:
- `FixedSizeChunker(500, 50).split()` → `chunks_fixed_size.jsonl` → embed → `VectorStore("fixed_size_500")`
- `SectionAwareChunker(700).split()` → `chunks_section_aware.jsonl` → embed → `VectorStore("section_aware")`

Reproducibility:
- Chunkers are deterministic (no randomness).
- `torch.manual_seed(42)`, inference mode, no dropout.
- ChromaDB `PersistentClient` with fixed path.
- `build_index.py --reset` deletes `indices/` and rebuilds identically.
- `data/processed/*.jsonl` is the source of truth; deleting `indices/` is always safe.

### Phase 3 — Question answering (online)
1. `retrieve(q, k=5, strategy, family_id)`:
   - `Embedder.encode([q], is_query=True)` → applies `"query: "` prefix.
   - `VectorStore.query(emb, k, where={"family_id": family_id})`.
   - Returns top-k chunks with scores.
2. `generate(q, chunks)`:
   - Builds prompt with context blocks tagged `[chunk_id: ...]`.
   - Gemini 2.5 Flash, temperature 0.2.
   - System rule: answer only from context; if absent say
     "המידע לא נמצא בפוליסות שלי."; cite chunk_ids used.
3. Parse `[chunk_id: ...]` markers → `sources`. Return full contract dict.

### Prompt structure
```
אתה עוזר לענות על שאלות מתוך פוליסות ביטוח.
ענה אך ורק מתוך הקונטקסט. אם התשובה לא נמצאת בקונטקסט, אמור:
"המידע לא נמצא בפוליסות שלי."
ציטוטים: בסוף התשובה ציין באילו chunks השתמשת בפורמט [chunk_id].
(אם התשובה היא 'לא מכוסה', עדיין ציין את ה-chunk שאיפשר את ההסקה.)

שאלה:
{question}

קונטקסט:
[chunk_id: car_policy__sa__sec_05, source: car_policy.pdf, pages: 12-13]
...

תשובה:
```

---

## 7. Embedding model notes (critical)

`intfloat/multilingual-e5-large` is asymmetric and trained with prefixes:
- Documents/chunks → prefix `"passage: "`
- Search queries → prefix `"query: "`

`Embedder.encode(texts, is_query)` applies the correct prefix. Omitting/swapping prefixes
silently degrades retrieval. Embedding dim = 1024, L2-normalized.

---

## 8. Chunking strategies & ablation

**Strategy 1 — Fixed-size:** 500 tokens, overlap 50, e5 tokenizer. Naive baseline; may cut
mid-sentence.

**Strategy 2 — Section-aware:** splits on Docling `##` headings (הגדרות / כיסויים / חריגים /
תגמולי ביטוח / ביטולים). Sections over 700 tokens recursively sub-split by paragraph then
sentence. Empty/heading-only sections merge with the section below.

**Ablation table (4 rows):**
| Experiment | Hit@5 | Answer accuracy | Notes |
|---|---|---|---|
| fixed_size 500 | — | — | baseline |
| section_aware | — | — | structure-aware (headline comparison) |
| fixed_size 300 | — | — | smaller, sharper for short factual |
| fixed_size 700 | — | — | more context, more noise |

Optional stretch: dense vs hybrid (BM25 + dense via RRF) as a 5th row.

---

## 9. Gold set

- 50 Hebrew questions, 10 per category: factual, numerical, temporal, negation, comparison.
- Generation: **Claude** produces candidates (deliberately a different LLM than the
  Gemini answerer, to avoid circular evaluation); human reviews/edits/adds the hard
  negation & comparison cases.
- **Anchor-based citations** (not chunk_ids, which differ per strategy):
```json
{
  "question": "מהי תקרת הכיסוי לרובוטיקה כירורגית?",
  "reference_answer": "...",
  "must_cite": { "source": "health_policy.pdf", "pages": [12, 13], "section_anchor": "ניתוחים מיוחדים" },
  "category": "numerical"
}
```
- Hit@k = at least one of the top-k retrieved chunks overlaps the anchor (page range or
  section). Survives both chunking strategies → fair comparison.

---

## 10. Multi-tenancy

- Every chunk carries `family_id` in metadata.
- `VectorStore.query()` requires `family_id` as a mandatory parameter; calling without it
  raises (assert/ValueError). A loud failure is preferred over silent cross-family leakage.
- Assignment corpus runs under fixed `family_id="demo_family_001"`; integration uses
  `config.DEMO_UID`.

---

## 11. Integration with chat (Phase 2)

`backend/routers/dashboard_chat.py`:
- Register a new Gemini tool:
```python
def query_insurance_policies(question: str) -> str:
    """ענה על שאלה מתוך פוליסות הביטוח של המשפחה (חיפוש סמנטי)."""
    from insurance_rag.src.rag_system import answer
    result = answer(question, family_id=uid, strategy="section_aware")
    return f"{result['answer']}\n\nמקורות: {', '.join(result['sources'])}"
```
- Keep existing `read_full_policy` tool (full-text path for deep contractual questions).
  Gemini picks the right tool per query.
- **Demo bypass change** — currently the demo user gets a static mock for every question
  (`dashboard_chat.py` DEMO_BYPASS). Change so it skips the mock **only for insurance**:
```python
is_insurance = request.context_filter == "ביטוח" or "ביטוח" in request.question
if uid == config.DEMO_UID and not is_insurance:
    return {"response": ...}   # existing mock for pension/stocks/default
# insurance demo questions fall through to real Gemini + RAG (family_id=DEMO_UID)
```
- `demo_seeder`: after seeding Firestore, index the redacted demo policies into Chroma
  under `family_id=DEMO_UID`.
- Packaging: install `insurance-rag` as editable package so `backend` can import it.

Three preconditions for the demo to work:
1. `GEMINI_API_KEY` present (else the real path falls back to a mock).
2. Demo policies indexed under `DEMO_UID` (seeder).
3. `is_insurance` correctly detects insurance questions.

---

## 12. Error handling & edge cases

| Scenario | Handling |
|---|---|
| `family_id` has no indexed policies | return "המידע לא נמצא...", empty sources/chunks |
| Docling fails on a PDF | `redact.py` exits code 2 for that file, continues to next |
| Regex missed some PII | logged; **manual log review before submission** is the safety net |
| Single heading section > 700 tokens | recursive sub-split (paragraph → sentence) |
| Empty/heading-only section | merge with section below |
| Gemini 429 / rate limit | exponential backoff 1s/2s/4s; after 3 fails → graceful error dict |
| Gemini answer without `[chunk_id]` | sources=[] but retrieved_chunks still populated |
| Empty/whitespace query | early ValueError in answer() |
| Non-Hebrew query (Arabic/Russian) | e5 + Gemini are multilingual; behavior not guaranteed (documented as limitation) |
| Two policies same name | doc_id gets short hash suffix: `health_policy_a3f2` |
| Corrupt Chroma collection | `build_index.py --reset` rebuilds from `data/processed/` |

---

## 13. Testing strategy

**Unit (pytest):**
- `chunking`: word-boundary splits, overlap correctness, `##` detection, recursive
  sub-split, determinism (same input → same output).
- `redaction`: catches Israeli ID/phone/email, removes known string, log contains no PII.
- `embeddings`: shape (1024), L2-normalized, batching.
- `vector_store`: add/query/reset, `where` filter, persistent reload.
- `retrieval`: `family_id` mandatory (raises without it), k respected, scores descending.

**Integration (`tests/test_e2e.py`):**
- Fixture: 2 small MD files, 2 families. `build_index()` → `answer()` for family A.
- Assert no family-B chunk appears in results (tenancy boundary).

**Determinism:** run `build_index.py` twice, assert `chunks_*.jsonl` hashes match.

**Eval as the big test:** `eval/run_eval.py` produces Hit@5, MRR, and manual-review
classification of ≥10 answers (Correct / Partial / Incorrect / Hallucinated).

Not tested: LLM stability (Gemini not mocked — empirical via eval); Docling perf (one-time).

---

## 14. Build order

```
Step 1: Scaffold + redaction      → data/redacted/*.md + log for review
Step 2: Chunking (2 strategies)   → chunks_*.jsonl + unit tests
Step 3: Embeddings + ChromaDB     → build_index.py runs, indices/ built
Step 4: Retrieval + generation    → answer() works from CLI on one question
Step 5: Gold set (50 Qs)          → gold_set.jsonl after your review
Step 6: Eval + ablation           → metrics + ablation table
Step 7: Report (4 pages)          → report.pdf
─────────────────────────────────  (Steps 1–7 = submission)
Step 8: Chat integration          → tool + demo_seeder + demo bypass (Phase 2)
```

---

## 15. Stack summary

- Python (matches existing backend)
- `docling` — PDF → Markdown
- `sentence-transformers` + `intfloat/multilingual-e5-large` — embeddings
- `chromadb` — persistent vector store
- `google-genai` (Gemini 2.5 Flash) — generation (already in backend)
- `anthropic` (Claude) — gold-set candidate generation only
- `pytest` — tests
- PyMuPDF (transitively via Docling) — note AGPL-3.0 license constraint

---

## 16. MANIFEST.md fields (to fill during Step 1)

```
Corpus name: Family Insurance Policies (Hebrew)
Domain: Insurance / legal-financial
Source of documents: User's own insurance policies (PII removed)
Number of documents: 3+ (car, health, home)
Approximate pages / tokens: ~30+ pages (verify after Docling)
File types: PDF → Markdown
License / permission: Personal documents, used with permission, PII redacted
Why suitable for RAG: contract-specific knowledge a baseline LLM lacks
What questions: coverage limits, deductibles, exclusions, waiting periods, renewal dates
```

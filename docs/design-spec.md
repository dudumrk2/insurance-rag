# Insurance Policy RAG — Design Spec

**Date:** 2026-05-20
**Status:** Historical design spec. The implementation is complete; this file records the original plan and has been annotated where the final code differs. Use `README.md` and `docs/report.md` as the source of truth for current behavior.
**Branch:** `master` (standalone repo; Phase 2 integration happens on `feat/insurance-rag` in `ai-wealth-monitor`)
**Author:** brainstorming session (Dudu + Claude)

---

## 1. Purpose & Context

Build a complete, controllable Retrieval-Augmented Generation (RAG) pipeline over a
corpus of Israeli (Hebrew) insurance policies. The project serves two goals:

1. **Academic mid-term assignment** — a self-contained project that meets a fixed rubric
   (data prep, loading, chunking, embedding, indexing, retrieval, generation, citations,
   gold set, evaluation, ablation, 4-page report). Must expose a fixed interface:
   `answer(question: str) -> dict`. Our implementation satisfies this: `family_id` and
   `strategy` have defaults, so `answer("שאלה?")` works as-is for the assignment.
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
- Retrieval + generation (Gemini 2.5 Flash) with retrieved source anchors; strict model-selected citations are future work
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
- Upload→index live hook in `InsuranceFlow` (future integration work)

---

## 3. Architecture

Three pipeline stages, separate entry points:

```
Stage 1 (Ingest, manual)       scripts/redact.py   PDF(raw) → Docling MD → redact → data/redacted/*.md
Stage 2 (Index, reproducible)  build_index.py      redacted MD → chunk(×2) → embed → ChromaDB(×2 collections)
Stage 3 (Ask, online)          src/generation.py   question → retrieve top-k → Gemini → {answer, sources, strategy, question}
```

Current consumers of `answer()`:
- **CLI / demo server:** direct imports from `src.generation`.
- **Flask demo:** `server.py` exposes `/ask` and calls `answer()`.
- **Eval:** `eval/run_eval.py` evaluates retrieval directly rather than scoring generated answers.

---

## 4. Repository layout

```
insurance-rag/
├── data/
│   ├── raw/                          # original PDFs — gitignored (contain PII)
│   ├── redacted/                     # *.md, PII removed — committed
│   ├── processed/
│   │   ├── chunks_fixed.jsonl
│   │   └── chunks_section_aware.jsonl
│   ├── redaction_log.json            # what was removed + where (no PII values)
│   └── MANIFEST.md
├── src/
│   ├── __init__.py
│   ├── config.py                     # paths, model names, chunk sizes
│   ├── pdf_to_md.py                  # Docling → markdown
│   ├── redaction.py                  # regex + known-strings → clean MD
│   ├── chunking.py                   # chunk_fixed(), chunk_section_aware()
│   ├── embedder.py                   # sentence-transformers e5 wrapper
│   ├── indexer.py                    # ChromaDB collection builder
│   ├── retrieval.py                  # retrieve(query, k, strategy, family_id)
│   ├── generation.py                 # answer() public interface + Gemini call
│   └── utils.py                      # logging and small helpers
├── scripts/
│   ├── redact.py                     # CLI: data/raw/*.pdf → data/redacted/*.md
│   ├── chunk.py                      # CLI: data/redacted/*.md → data/processed/chunks_*.jsonl
│   └── build_gold_set.py             # Gemini generates candidates for review
├── eval/
│   ├── gold_set.jsonl                # 50 Hebrew questions, anchor-based
│   ├── run_eval.py                   # retrieval Hit@k/MRR over gold set
│   ├── ablation_results.md
│   ├── embedding_ablation_results.md
│   └── answer_eval_gemini_results.md
├── tests/
│   ├── conftest.py                   # fixtures (tiny MD corpus, 2 families)
│   ├── test_chunking.py
│   ├── test_redaction.py
│   ├── test_embedder.py
│   ├── test_indexer.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_eval.py
│   └── test_server.py
├── build_index.py                    # reproducible index build entry point
├── pyproject.toml                    # editable install
├── requirements.txt
├── docs/report.md                    # final report
├── README.md                         # exact run instructions
└── .gitignore                        # data/raw/, indices/, __pycache__, *.pyc
```

Index storage (gitignored, rebuilt by `build_index.py`):
```
insurance-rag/indices/
└── chroma.sqlite3                    # Chroma PersistentClient root

Collections:
- insurance_fixed
- insurance_section_aware
```

---

## 5. Components & interfaces

| Module | Public function | In | Out |
|---|---|---|---|
| `pdf_to_md` | `convert(pdf_path)` | path | str (markdown) |
| `redaction` | `redact(md_text, known_strings)` | str + list | (str_redacted, log_dict) |
| `chunking` | `chunk_fixed(text, doc_name, family_id, chunk_size, overlap)` | text + metadata | list[dict] |
| `chunking` | `chunk_section_aware(text, doc_name, family_id, max_tokens)` | text + metadata | list[dict] |
| `embedder` | `embed_texts(texts)` / `embed_query(text)` | list[str] / str | np.ndarray |
| `indexer` | `build_collection(strategy, chunks, embeddings)` | chunks + embeddings | Chroma collection |
| `retrieval` | `retrieve(query, k, strategy, family_id)` | str+int+str+str | list[dict] |
| `generation` | `answer(question, family_id="demo_family_001", strategy="section_aware")` | str+str+str | dict |

### Standard `Chunk` shape (everywhere)
```python
{
    "chunk_id": "demo_family_001_section_aware_health_policy_3",
    "text": "passage: ...",
    "source_doc": "health_policy",
    "strategy": "section_aware",
    "family_id": "demo_family_001",
    "anchor": "...",                  # first 80 raw-text chars
    "section": "## ניתוחים מיוחדים"   # None for fixed chunks
}
```
Strategy names in chunk metadata: `fixed` and `section_aware`.

### `answer()` return contract (implemented)
```python
{
    "answer": str,
    "sources": list[str],          # anchors from all retrieved chunks, not model-selected citations
    "strategy": str,
    "question": str
}
```

Returning `retrieved_chunks` with per-chunk metadata is a documented future improvement.

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
- `chunk_fixed(..., chunk_size=500, overlap=50)` → `chunks_fixed.jsonl` → embed → Chroma collection `insurance_fixed`
- `chunk_section_aware(..., max_tokens=700)` → `chunks_section_aware.jsonl` → embed → Chroma collection `insurance_section_aware`

Reproducibility:
- Chunkers are deterministic (no randomness).
- ChromaDB `PersistentClient` uses the fixed `indices/` path.
- `build_collection()` deletes and recreates the target collection on each run.
- `data/processed/*.jsonl` is the source of truth; deleting `indices/` is always safe.

### Phase 3 — Question answering (online)
1. `retrieve(q, k=5, strategy, family_id)`:
   - `embed_query(q)` → applies `"query: "` prefix.
   - ChromaDB `collection.query(..., where={"family_id": family_id})`.
   - Returns top-k chunks with scores.
2. `generate(q, chunks)`:
   - Builds a Hebrew prompt with raw concatenated context blocks.
   - Gemini 2.5 Flash, temperature 0.2.
   - System rule: answer in Hebrew based on the supplied context.
3. Return answer plus `sources = [chunk["anchor"] for chunk in chunks]`.

### Prompt structure
```
System: אתה עוזר המתמחה בפוליסות ביטוח. ענה בעברית בלבד על בסיס ההקשר שסופק.

User: הקשר:
{plain concatenated retrieved chunks}

שאלה: {question}
```

---

## 7. Embedding model notes (critical)

`intfloat/multilingual-e5-large` is asymmetric and trained with prefixes:
- Documents/chunks → prefix `"passage: "`
- Search queries → prefix `"query: "`

`embed_query(text)` applies the query prefix; chunks already carry the passage prefix
from `src.chunking`. Omitting/swapping prefixes silently degrades retrieval.
Embedding dim = 1024, L2-normalized.

---

## 8. Chunking strategies & ablation

**Strategy 1 — Fixed-size:** 500 characters, overlap 50 characters. Naive baseline; may cut
mid-sentence. The original plan used token windows, but the implemented code uses character windows.

**Strategy 2 — Section-aware:** splits on Docling `##` headings (הגדרות / כיסויים / חריגים /
תגמולי ביטוח / ביטולים). Sections over 2,800 characters (approximately 700 Hebrew tokens) fall back to fixed sub-chunking.

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

- 50 Hebrew questions selected from 75 generated candidates.
- Generation: **Gemini 2.5 Flash** produces candidates; human review selects the final set.
- Evaluation scoring is mechanical substring matching over real chunks, but the question distribution can still be model-biased because Gemini generated the candidates.
- **Anchor-based citations** (not chunk_ids, which differ per strategy):
```json
{
  "question": "מהי תקרת הכיסוי לרובוטיקה כירורגית?",
  "reference_answer": "...",
  "must_cite": { "source": "health_policy.pdf", "pages": [12, 13], "section_anchor": "ניתוחים מיוחדים" },
  "category": "numerical"
}
```
- Hit@k = at least one of the top-k retrieved chunk texts contains the gold anchor.
  This survives both chunking strategies and supports fair comparison.

---

## 10. Multi-tenancy

- Every chunk carries `family_id` in metadata.
- `retrieve()` accepts `family_id` and always sends it as a ChromaDB metadata filter.
  The default assignment/demo value is `demo_family_001`.
- Assignment corpus runs under fixed `family_id="demo_family_001"`; integration uses
  `config.DEMO_UID`.

---

## 11. Integration with chat (Phase 2)

`backend/routers/dashboard_chat.py`:
- Register a new Gemini tool:
```python
def query_insurance_policies(question: str) -> str:
    """ענה על שאלה מתוך פוליסות הביטוח של המשפחה (חיפוש סמנטי)."""
    from src.generation import answer
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
| `family_id` has no indexed policies | current implementation passes empty context to generation; explicit refusal is future work |
| Docling fails on a PDF | `redact.py` exits code 2 for that file, continues to next |
| Regex missed some PII | logged; **manual log review before submission** is the safety net |
| Single heading section > ~2,800 chars | fixed-size fallback sub-chunking |
| Empty section | skipped; heading-only nonblank sections are kept as chunks |
| Gemini 429 / rate limit | not retried in `src/generation.py`; retries exist in gold-set generation only |
| Gemini answer without citations | sources still contain all retrieved anchors |
| Empty/whitespace query | no dedicated early rejection in `answer()`; caller validation is future work |
| Non-Hebrew query (Arabic/Russian) | e5 + Gemini are multilingual; behavior not guaranteed (documented as limitation) |
| Two policies same stem | not specially handled; filenames must be unique in `data/redacted/` |
| Corrupt Chroma collection | delete `indices/` or rerun `build_index.py`; each collection is recreated |

---

## 13. Testing strategy

**Unit (pytest):**
- `chunking`: character-window splits, overlap correctness, `##` detection,
  fixed-size fallback for oversized sections, determinism (same input → same output).
- `redaction`: catches Israeli ID/phone/email, removes known string, log contains no PII.
- `embedder`: shape (1024), L2-normalized, batching.
- `indexer`: build/load Chroma collections, `where` filter metadata, persistent reload.
- `retrieval`: `family_id` filter applied, k respected, Chroma order preserved.

**Integration/server:**
- `tests/test_generation.py` mocks retrieval/generation to verify the public contract.
- `tests/test_server.py` mocks `answer()` and verifies `/ask` behavior and CORS.

**Determinism:** run `build_index.py` twice, assert `chunks_*.jsonl` hashes match.

**Eval as the big test:** `eval/run_eval.py` produces Hit@k and MRR. Manual answer
classification is recorded separately in the report and Gemini answer-eval artifacts.

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
- `google-genai` (Gemini 2.5 Flash) — generation and gold-set candidate generation
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

# Insurance Policy RAG

A Retrieval-Augmented Generation (RAG) pipeline over a corpus of Hebrew insurance
policies. Built for a course mid-term assignment, designed to also plug into the
`ai-wealth-monitor` application later.

> **Status:** Complete — ingestion, chunking, embeddings, retrieval, generation, evaluation, and Flask demo are all implemented.

## What this is

Ask natural-language questions about insurance policies (coverage limits, deductibles,
exclusions, waiting periods, renewal dates) and get answers grounded in the policy text,
with retrieved source anchors for auditability.

Public interface:

```python
answer(question: str) -> dict
# → {"answer": str, "sources": list[str], "strategy": str, "question": str}
```

> **Known gap:** `sources` contains all retrieved chunk anchors, not model-selected citations.
> Returning `retrieved_chunks` with per-chunk metadata is a documented future improvement.

## Documentation

Start here — the design and the reasoning behind it:

- [`docs/report.md`](docs/report.md) — **current implementation truth**: final pipeline,
  evaluation results, known gaps, and reproducible run steps.
- [`docs/design-spec.md`](docs/design-spec.md) — **historical design spec**: original
  architecture plan; now annotated where the final implementation differs.
- [`docs/DESIGN_RATIONALE.md`](docs/DESIGN_RATIONALE.md) — **the why**: the decision
  journey, alternatives we weighed, final implementation notes, and a primer on the RAG concepts used
  (e5 prefixes, dense vs BM25 vs hybrid, bi-encoder vs cross-encoder reranking).
- [`docs/mid_term_assignment.pdf`](docs/mid_term_assignment.pdf) — the original
  assignment brief.

## Stack

- `docling` — PDF → Markdown
- `sentence-transformers` + `intfloat/multilingual-e5-large` — embeddings
- `chromadb` — persistent vector store
- `google-genai` (Gemini 2.5 Flash) — answer generation and gold-set candidate generation
- `pytest` — tests

## Running

Each build step needs only its own dependencies (the full stack pulls PyTorch
~2GB). Install incrementally:

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows; use bin/activate on *nix

# Step 1 — redaction (PDF → redacted Markdown)
pip install -e ".[pdf]"
cp data/known_pii.example.json data/known_pii.json   # then fill in real names/IDs (gitignored)
python scripts/redact.py                              # data/raw/*.pdf → data/redacted/*.md + log
#   → review data/redaction_log.json before committing

# Step 2 — build vector indices
pip install -e ".[embeddings,vectorstore,generation]"
python build_index.py

# Step 3 — run retrieval evaluation (ablation study)
python eval/run_eval.py

# Step 4 — start the Flask demo server
pip install -e ".[server]"
python server.py
```

Run the fast test suite with `pip install -e ".[dev]" && pytest -m "not slow"`.
For the full embedding-backed suite, install `.[all]` and run `pytest`.

## Privacy

The corpus is built from real personal insurance policies. **Raw PDFs
(`data/raw/`) are gitignored and never committed.** Only PII-redacted Markdown
(`data/redacted/`) enters version control. See the redaction section of the design spec.

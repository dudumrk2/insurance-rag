# Insurance Policy RAG

A Retrieval-Augmented Generation (RAG) pipeline over a corpus of Hebrew insurance
policies. Built for a course mid-term assignment, designed to also plug into the
`ai-wealth-monitor` application later.

> **Status:** Design complete, implementation not started yet.

## What this is

Ask natural-language questions about insurance policies (coverage limits, deductibles,
exclusions, waiting periods, renewal dates) and get answers grounded in the policy text,
with citations to the exact source chunks.

Public interface (assignment-mandated):

```python
answer(question: str) -> dict
# → {"answer": str, "sources": list[str], "retrieved_chunks": list[dict]}
```

## Documentation

Start here — the design and the reasoning behind it:

- [`docs/design-spec.md`](docs/design-spec.md) — **the what**: full technical design
  (architecture, components, data flow, interfaces, testing).
- [`docs/DESIGN_RATIONALE.md`](docs/DESIGN_RATIONALE.md) — **the why**: the decision
  journey, alternatives we weighed, and a primer on the RAG concepts used
  (e5 prefixes, dense vs BM25 vs hybrid, bi-encoder vs cross-encoder reranking).
- [`docs/mid_term_assignment.pdf`](docs/mid_term_assignment.pdf) — the original
  assignment brief.

## Stack

- `docling` — PDF → Markdown
- `sentence-transformers` + `intfloat/multilingual-e5-large` — embeddings
- `chromadb` — persistent vector store
- `google-genai` (Gemini 2.5 Flash) — answer generation
- `anthropic` (Claude) — gold-set question generation only
- `pytest` — tests

## Running (once implemented)

```bash
pip install -r requirements.txt
python src/redact.py          # data/raw/*.pdf → data/redacted/*.md (manual, one-time)
python build_index.py         # build the vector indices
python eval/run_eval.py       # run evaluation over the gold set
```

## Privacy

The corpus is built from real personal insurance policies. **Raw PDFs
(`data/raw/`) are gitignored and never committed.** Only PII-redacted Markdown
(`data/redacted/`) enters version control. See the redaction section of the design spec.

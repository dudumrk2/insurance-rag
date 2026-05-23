# Corpus Manifest

> Fill the corpus-specific values (documents, pages, tokens) once the real PDFs
> are added to `data/raw/` and converted via `scripts/redact.py`.

| Field | Value |
|---|---|
| Corpus name | Family Insurance Policies (Hebrew) |
| Domain | Insurance / legal-financial |
| Source of documents | User's own insurance policies (PII removed) |
| Number of documents | 4 (car policy, car coverages/costs, health policy, home policy) |
| Approximate pages / tokens | ~671 KB redacted Markdown · ~4,291 lines · ~670,000 chars |
| File types | PDF → Markdown |
| License / permission | Personal documents, used with permission, PII redacted |
| Why suitable for RAG | Contract-specific knowledge a baseline LLM lacks (coverage limits, deductibles, exclusions, waiting periods) |
| What questions | Coverage limits, deductibles, exclusions, waiting periods, renewal dates |

## Privacy

Raw PDFs (`data/raw/`) contain PII and are **gitignored — never committed**. Only
PII-redacted Markdown (`data/redacted/`) and the redaction log (`data/redaction_log.json`,
which records counts and post-redaction context only, never raw PII) enter version control.

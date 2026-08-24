# Insurance RAG: Hebrew Policy Retrieval System

A full Retrieval-Augmented Generation (RAG) pipeline designed for Hebrew insurance policies. The system parses complex PDF documents (such as car and health insurance policies), cleanses PII (Personally Identifiable Information), chunks the text using structure-aware strategies, retrieves accurate segments via dense embeddings, and generates precise, grounded answers. 

![Project Overview](docs/roadmap.html) *(See the [Live Roadmap & Architecture](https://dudumrk2.github.io/insurance-rag/roadmap.html))*

> [!NOTE]  
> A detailed technical report (in Hebrew) covering the methodology, ablation studies, and evaluation is available in [`docs/report.md`](docs/report.md). The original Hebrew README is available in [`README.he.md`](README.he.md).

## Core Architecture

```text
PDF
 └─► Docling ──► Markdown ──► Redaction ──► data/redacted/*.md
                                                    │
                              ┌─────────────────────┤
                              ▼                     ▼
                         chunk_fixed         chunk_section_aware
                         (500/300/700 chars)  (≤2,800 chars/section)
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                              multilingual-e5-large
                              (1024-dim, "passage: " prefix)
                                         │
                                         ▼
                                  ChromaDB
                              (cosine similarity, family_id filter)
                                         │
                                         ▼
                              retrieve(query, top_k=5)
                                         │
                                         ▼
                              Gemini 2.5 Flash
                              (Hebrew system prompt, T=0.2)
                                         │
                                         ▼
                              answer + retrieved anchors
```

## Results & Ablation Study

An ablation study was performed against a carefully curated 50-question "Gold Set". It demonstrated that **structure-aware chunking (`section_aware`) significantly outperforms fixed-size windows**. By dividing text according to natural markdown headers, the system preserves the semantic integrity of legal clauses and limits.

### Chunking Strategy Comparison

| Configuration | Segments | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| **`section_aware`** | 447 | **0.380** | **0.660** | **0.740** | **0.529** |
| `fixed_500` | 944 | 0.060 | 0.220 | 0.300 | 0.139 |
| `fixed_300` | 1,700 | 0.160 | 0.220 | 0.280 | 0.196 |
| `fixed_700` | 656 | 0.080 | 0.180 | 0.260 | 0.138 |

### Embedding Model Comparison
Tested on the `section_aware` strategy. Note: The `gemini-embedding-001` variant leverages asymmetric task types instead of the E5 prefixing logic.

| Model | Dimensions | Hit@5 | MRR | Correctness (10 sampled Qs) |
|---|---|---|---|---|
| **`gemini-embedding-001`** | 768 | **0.800** | **0.615** | **70%** |
| `multilingual-e5-large` | 1024 | 0.720 | 0.534 | 60% |

## Getting Started

### Prerequisites
- Python >=3.10
- A Gemini API Key (set via `.env`)

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dudumrk2/insurance-rag.git
   cd insurance-rag
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[all]"
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Run the CLI to test the pipeline:**
   ```python
   from src.generation import answer
   result = answer('מה הפרנשייז על נזק מלא לרכב?', strategy='section_aware')
   print(result['answer'])
   ```

To re-run the entire pipeline (PDF parsing -> Redaction -> Indexing -> Evaluation), follow the detailed command list in the [Report Appendix A](docs/report.md#א-הרצת-ה-pipeline-מקצה-לקצה).

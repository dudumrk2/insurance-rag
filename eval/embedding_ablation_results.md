# Phase 0 — Embedding Model Ablation Results

> **Script:** `eval/run_eval_gemini.py`
> **Date:** 2026-05-25
> **Status:** completed — gemini-embedding-001 selected for ai-wealth-monitor integration

## Setup

| Parameter | Value |
|---|---|
| Chunks | 447 section_aware (identical to main ablation) |
| Gold questions | 50 (eval/gold_set.jsonl) |
| top_k | 5 |
| Embedding model — new | gemini-embedding-001 via Google API |
| output_dimensionality | 768 |
| task_type — indexing | RETRIEVAL_DOCUMENT |
| task_type — query | RETRIEVAL_QUERY |
| Embedding model — baseline | intfloat/multilingual-e5-large (1024d, local) |
| Index type | ChromaDB EphemeralClient (cosine) |
| Only change vs main ablation | embedding model (everything else identical) |

## Results

| Model | Dim | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| **gemini-embedding-001** ⭐ | 768 | **0.500** | **0.740** | **0.800** | **0.615** |
| multilingual-e5-large (baseline) | 1,024 | — | — | 0.720 | 0.534 |
| **∆ (gemini − e5)** | −256 | — | — | **+0.080 (+11%)** | **+0.081 (+15%)** |

*The e5 baseline (0.720/0.534) is from an independent run of run_eval.py.
It differs slightly from section 5.2 of the report (0.740/0.529) due to HNSW approximation variance between runs.*

## Raw output (captured from terminal)

```
PHASE 0 ABLATION — gemini-embedding-001 (768d) vs e5 baseline
================================================================
Config                      Hit@1   Hit@3   Hit@5     MRR
----------------------------------------------------------------
section_aware (e5, base)        —       —   0.720   0.534
section_aware (gemini-768)  0.500   0.740   0.800   0.615
----------------------------------------------------------------
delta vs e5                             +0.080  +0.081
================================================================
Questions: 50  |  Hits@5: 40
```

## Decision

**gemini-embedding-001 selected** for ai-wealth-monitor integration.

Rationale:
- +11% Hit@5 / +15% MRR on Hebrew over e5-large
- No local model (~3GB torch + sentence-transformers not needed)
- Smaller output dimension (768 vs 1024) — lower Firestore storage
- Already uses RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY task_type (same asymmetric principle as e5 prefixes)
- google-genai SDK already in ai-wealth-monitor backend requirements

# Ablation Study — Retrieval Strategy Comparison

**Gold set:** 50 questions  |  **Top-k:** 5

## Results

| Configuration   | Chunks | Hit@1 | Hit@3 | Hit@5 |  MRR  |
|-----------------|--------|-------|-------|-------|-------|
| section_aware   |    447 | 0.400 | 0.660 | 0.720 | 0.534 |
| fixed_500       |    944 | 0.040 | 0.200 | 0.280 | 0.119 |
| fixed_300        |   1700 | 0.180 | 0.240 | 0.300 | 0.216 |
| fixed_700        |    656 | 0.080 | 0.220 | 0.280 | 0.150 |

## Notes

- **Best MRR:** `section_aware` (0.534)
- Hit@k = fraction of questions where the gold passage appears in top-k retrieved chunks.
- MRR = Mean Reciprocal Rank (higher = gold chunk appears closer to position 1).
- `section_aware` and `fixed_500` use the pre-built persistent ChromaDB indices.
- `fixed_300` and `fixed_700` are built in-memory from the redacted Markdown files.
- Gold anchors are first-80-char keys from section-aware chunks; substring matching
  is used for fixed strategies (the same passage may sit inside a differently-bounded chunk).

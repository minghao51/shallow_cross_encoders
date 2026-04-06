# SPLADE Reranker

## Overview

A sparse encoder reranker using SPLADE-style sparse embeddings. SPLADE produces sparse vectors where non-zero dimensions correspond to informative terms, combining the interpretability of lexical matching with the semantic understanding of neural encoders.

**Implementation**: `src/reranker/strategies/splade.py`

**Encoder**: `sentence_transformers.SparseEncoder` (pretrained model).

---

## Mathematical Formulation

### Sparse Embedding

Each document D is encoded into a sparse vector where each non-zero dimension corresponds to a vocabulary term with an importance weight:

```
E(D) = {t₁: w₁, t₂: w₂, ..., tₖ: wₖ}  where wᵢ > 0
```

Unlike dense embeddings that compress all information into a fixed vector, sparse embeddings retain explicit term-level importance scores learned from the pretrained model.

### Term Pruning

Only the top-k terms by weight are retained:

```
E'(D) = top-k terms from E(D) by weight (descending)
```

### MaxSim Scoring

For a query Q with sparse embedding E(Q) and document D with sparse embedding E'(D):

```
score(Q, D) = Σ_{t ∈ Q ∩ D} min(w_Q(t), w_D(t))
```

Where:
- `w_Q(t)` is the importance weight of term t in the query
- `w_D(t)` is the importance weight of term t in the document
- The sum is over terms that appear in both query and document

**Intuition**: For each shared term, take the minimum importance weight between query and document. This ensures both sides consider the term significant.

---

## DAG Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FIT PHASE                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│        SparseEncoder (pretrained)           │
│  encode(docs) → sparse vectors              │
│  {term_id: weight, ...}                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Term Pruning (top-k)                 │
│  Sort by weight descending, keep top 128    │
│  Convert term_ids to strings                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          Sparse Index (stored)              │
│  [{term₁: w₁, term₂: w₂, ...}, ...]         │
└─────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│                   SCORE PHASE                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐
│   Query Q   │     │Sparse Index │
│             │     │ (pre-built) │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   │
┌─────────────────────────────────────────────┐
│  SparseEncoder.encode([Q]) → {t: w, ...}    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         MaxSim (per query-doc pair)         │
│  score = Σ min(w_Q(t), w_D(t))              │
│  for t in Q ∩ D                             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────┐
│  RankedDocs │  (doc, score, rank, metadata)
└─────────────┘
```

---

## Approach & Methodology

### Fit Phase

1. **Load pretrained SparseEncoder** (lazy, on first use)
2. **Encode all documents** in batches (batch_size=32)
3. **Prune to top-k terms** per document (default: 128)
4. **Store sparse index**: List of {term: weight} dicts

### Score Phase

1. **Encode query** as sparse vector
2. **For each document** in the index:
   - Find shared terms between query and document
   - For each shared term: `score += min(w_Q(t), w_D(t))`
3. **Rank**: Sort documents by score (descending)

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pretrained SparseEncoder** | No training needed, leverages learned term importance |
| **Top-k pruning** | Controls memory and compute, removes noise terms |
| **Min-based scoring** | Both query and document must consider term important |
| **String term keys** | Interpretable — you can see which terms contribute |
| **Auto-fit on rerank** | Convenience: fits if not already done |

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `model_name` | config default | Pretrained sparse encoder model |
| `top_k_terms` | 128 | Maximum terms retained per document |

### Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Encoding (fit) | O(n·d_sparse) | n = docs, d_sparse = sparse dim |
| Pruning | O(n·k·log(k)) | k = top_k_terms |
| Scoring | O(n·k) | k = avg shared terms |

### Performance

| Metric | Value |
|--------|-------|
| Latency (20 docs) | TBD |
| Latency (200 docs) | TBD |
| NDCG@10 | TBD |
| MRR | TBD |
| P@1 | TBD |

### When to Use

- **Interpretable matching**: You can see exactly which terms contribute to scores
- **Lexical + semantic blend**: SPLADE naturally combines both signals
- **No training data needed**: Uses pretrained model, no fit labels required
- **Multilingual support**: Some models support multiple languages

### Limitations

- **Requires sentence-transformers**: Additional dependency with neural model download
- **GPU benefits encoding**: While scoring is CPU-friendly, encoding benefits from GPU
- **Vocabulary-dependent**: Terms outside the model's vocabulary get no weight
- **Single-pass encoding**: No query-document interaction during encoding

# BM25 Engine

## Overview

BM25 (Best Matching 25) is a probabilistic lexical retrieval algorithm that scores documents based on query term frequency, inverse document frequency, and document length normalization. It serves as the foundational baseline in the reranking pipeline.

**Implementation**: `src/reranker/lexical.py`

**Backend**: Wraps `rank_bm25` library with a pure-Python fallback when unavailable.

---

## Mathematical Formulation

### BM25 Scoring Function

For a query Q with terms q₁, q₂, ..., qₙ and document D:

```
score(D, Q) = Σᵢ IDF(qᵢ) · (TF(qᵢ, D) · (k₁ + 1)) / (TF(qᵢ, D) + k₁ · (1 - b + b · |D|/avgdl))
```

Where:
- **TF(qᵢ, D)**: Term frequency of qᵢ in document D
- **IDF(qᵢ)**: Inverse document frequency = log((N - df(qᵢ) + 0.5) / (df(qᵢ) + 0.5) + 1)
- **|D|**: Document length (number of tokens)
- **avgdl**: Average document length across the corpus
- **N**: Total number of documents in the corpus

### Hyperparameters

| Parameter | Value | Role |
|-----------|-------|------|
| k₁ | 1.5 | Term frequency saturation — controls how quickly TF saturates |
| b | 0.75 | Length normalization — 0 means no normalization, 1 means full normalization |

### IDF Smoothing

The implementation uses a smoothed IDF variant:

```
IDF(qᵢ) = log((N - df(qᵢ) + 0.5) / (df(qᵢ) + 0.5) + 1)
```

The `+ 1` at the end ensures IDF is always positive, even for very common terms.

---

## DAG Components

```
┌─────────────┐
│   Query Q   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Tokenize   │  query.lower().split()
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│              BM25 Scoring                   │
│  ┌───────────┐  ┌───────────┐  ┌──────────┐│
│  │ TF Lookup │  │ IDF Calc  │  │ Length   ││
│  │ per term  │  │ per term  │  │ Norm     ││
│  └─────┬─────┘  └─────┬─────┘  └────┬─────┘│
│        └────────┬──────┘       └─────┘     │
│                 ▼                          │
│         Σ (IDF × TF × Norm)                │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│          Normalization (optional)           │
│  scores = scores / max(scores)              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────┐
│  RankedDocs │  (doc, score, rank, metadata)
└─────────────┘
```

---

## Approach & Methodology

### Fit Phase

1. **Tokenize corpus**: Split each document into lowercase tokens
2. **Compute statistics**:
   - Document frequency (df) for each unique term
   - Average document length (avgdl)
3. **Initialize backend**:
   - Try `rank_bm25.BM25Okapi` (optimized C extension)
   - Fall back to pure-Python implementation

### Score Phase

1. **Tokenize query**: Split into lowercase tokens
2. **For each document**, compute BM25 score by summing over query terms:
   - Look up term frequency in document
   - Compute IDF from pre-computed document frequencies
   - Apply length normalization using document length vs average
3. **Clamp negative scores** to zero
4. **Normalize** scores to [0, 1] range by dividing by maximum

### Backend Selection

| Backend | When Used | Characteristics |
|---------|-----------|-----------------|
| `rank_bm25` | Library available | Faster, optimized implementation |
| Pure Python | Library unavailable | Slower but self-contained, identical results |

### Design Decisions

- **Case-insensitive**: All text is lowercased before processing
- **Simple tokenization**: Whitespace splitting (no stemming, no stopword removal)
- **Score normalization**: Optional max-normalization for consistent score ranges
- **Auto-fit on rerank**: If not fitted, automatically fits on the provided documents

### Limitations

- **Lexical only**: No semantic understanding; "car" and "automobile" are unrelated
- **No phrase matching**: Terms are treated independently
- **No field weighting**: All text treated as a single field
- **Vocabulary mismatch**: Fails when query and document use different terms for same concept

### Performance

| Metric | Value |
|--------|-------|
| Latency (20 docs) | ~0.13ms |
| Latency (200 docs) | ~1.02ms |
| NDCG@10 (expanded v2) | 0.6647 ± 0.23 |
| MRR | 0.9242 |
| P@1 | 0.9167 |

### When to Use

- **Fast baseline**: Quick sanity check before deploying semantic models
- **Large corpora**: Scales well with document count (7.8x latency growth for 10x corpus)
- **Exact term matching**: When query terms must appear in results
- **Resource-constrained**: No embedding model required, minimal memory footprint

# Binary Quantized Reranker

## Overview

A two-stage semantic reranker that combines ultra-fast binary quantization with learned bilinear interaction. Stage 1 uses Hamming distance on binarized embeddings for rapid candidate filtering. Stage 2 applies a learned diagonal weight matrix to refine scores for top candidates.

**Implementation**: `src/reranker/strategies/binary_reranker.py`

**Training**: Logistic Regression on elementwise query-document products.

---

## Mathematical Formulation

### Stage 1: Binary Quantization + Hamming Distance

**Quantization**: Each dense embedding vector is thresholded at zero:

```
b(v) = (v > 0) ∈ {0, 1}^d
```

**Hamming Distance**: Count of differing bits between query and document binary vectors:

```
H(q, d) = Σᵢ |b(q)ᵢ - b(d)ᵢ|
```

**Normalized Hamming Score**: Converted to [0, 1] similarity:

```
S_hamming(q, d) = 1 - H(q, d) / max(H_max, 1)
```

Where `H_max` is the maximum Hamming distance across all candidates.

### Stage 2: Bilinear Interaction

**Elementwise Product Features**: For training, the model learns from:

```
x = q ⊙ d  (elementwise/Hadamard product)
```

**Bilinear Score**: A learned diagonal weight matrix W scales the elementwise product:

```
S_bilinear(q, d) = q^T W d = Σᵢ wᵢ · qᵢ · dᵢ
```

Where `W = diag(w₁, w₂, ..., w_d)` is a diagonal matrix learned via Logistic Regression.

**Weight Extraction**: The logistic regression coefficients become the diagonal weights:

```
w = |coef(LogisticRegression(q ⊙ d, labels))|
```

### Combined Scoring

For each document:
1. Compute `S_hamming` for all documents
2. Select top-k candidates by Hamming score (`hamming_top_k`, default 500)
3. Replace scores for top-k bilinear candidates (`bilinear_top_k`, default 50) with `S_bilinear`

```
S_final(q, dᵢ) = { S_bilinear(q, dᵢ)  if dᵢ ∈ top-k bilinear candidates
                 { S_hamming(q, dᵢ)   otherwise
```

---

## DAG Components

```
┌─────────────┐     ┌─────────────┐
│   Query Q   │     │  Documents  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────────┐
│           Embedder (shared)                 │
│  e(Q) → dense vector, e(D) → dense vectors  │
└──────────┬──────────────────────┬────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐  ┌──────────────────────┐
│   Quantize       │  │   Quantize           │
│   b(e(Q))        │  │   b(e(D))            │
│   (sign > 0)     │  │   (sign > 0)         │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────┐
│        Hamming Distance Matrix              │
│  H[i] = count_nonzero(b(q) != b(dᵢ))        │
│  S_hamming[i] = 1 - H[i] / max(H)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Top-k Selection                      │
│  Select top hamming_top_k (default 500)     │
│  For top bilinear_top_k (default 50):       │
│    S_final[i] = q^T W dᵢ                    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────┐
│  RankedDocs │  (doc, score, rank, metadata)
└─────────────┘
```

### Training DAG

```
┌──────────┐  ┌──────────┐  ┌────────┐
│ Queries  │  │  Docs    │  │ Labels │
└────┬─────┘  └────┬─────┘  └───┬────┘
     │             │            │
     ▼             ▼            │
┌─────────────────────────┐     │
│    Embedder             │     │
│  e(Q), e(D) → vectors   │     │
└──────────┬──────────────┘     │
           │                    │
           ▼                    │
┌─────────────────────────┐     │
│  Elementwise Product    │     │
│  x = e(Q) ⊙ e(D)        │     │
└──────────┬──────────────┘     │
           │                    │
           ▼                    │
┌─────────────────────────────────────────────┐
│     Logistic Regression                     │
│  P(y=1|x) = σ(w^T x + b)                   │
│  W = diag(|w|)                              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────┐
│  Bilinear Weights W     │
│  (diagonal matrix)      │
└─────────────────────────┘
```

---

## Approach & Methodology

### Fit Phase

1. **Encode documents**: Generate dense embeddings for all unique documents
2. **Quantize**: Convert embeddings to binary vectors via sign thresholding
3. **Train bilinear model** (if labels have variance):
   - Encode query-document pairs
   - Compute elementwise products as features
   - Fit Logistic Regression with C=1.0, max_iter=500
   - Extract absolute coefficients as diagonal weights
4. **Fallback**: If labels are constant, use DummyClassifier and uniform weights

### Score Phase

1. **Encode query**: Generate dense embedding for the query
2. **Quantize**: Convert query embedding to binary vector
3. **Hamming scoring**: Compute Hamming distances to all document binary vectors
4. **Normalize**: Convert distances to [0, 1] similarity scores
5. **Bilinear refinement**: For top-k candidates, replace Hamming scores with bilinear scores
6. **Rank**: Sort documents by final scores (descending)

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Sign quantization** | Simple, fast, preserves direction information |
| **Diagonal W matrix** | O(d) computation vs O(d²) for full matrix |
| **Two-stage cascade** | Fast filtering + precise refinement |
| **Absolute weights** | Ensures all dimensions contribute positively |
| **Hamming as fallback** | Bilinear model may not always be available |

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `hamming_top_k` | 500 | Number of candidates considered for bilinear refinement |
| `bilinear_top_k` | 50 | Number of top candidates receiving bilinear scores |
| `random_state` | 42 | Reproducibility for Logistic Regression |

### Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Quantization | O(d) per vector | Simple thresholding |
| Hamming distance | O(d) per document | Bitwise XOR + population count |
| Bilinear score | O(d) per candidate | Elementwise multiply + sum |
| Total scoring | O(n·d + k·d) | n=docs, k=bilinear_top_k |

### Performance

| Metric | Value |
|--------|-------|
| Latency (20 docs) | ~0.69ms |
| Latency (200 docs) | ~4.02ms |
| NDCG@10 (expanded v2) | **0.8475 ± 0.18** (best overall) |
| MRR | 0.9271 |
| P@1 | 0.9167 |
| Scaling (20→200 docs) | 5.8x |

### When to Use

- **Best accuracy/latency trade-off**: Highest NDCG@10 among all strategies
- **Large corpora**: Hamming distance scales linearly with document count
- **Memory-constrained**: Binary vectors use 1/8 the memory of float32
- **Production deployments**: Fast enough for real-time, accurate enough for quality

### Ablation Findings

- **Bilinear refinement adds no value** on current synthetic data (hamming-only achieves same NDCG@10)
- **Aggressive pruning** (100/10 vs 500/50) has minimal impact (-0.0025 NDCG@10)
- **Recommendation**: Use hamming-only mode for 5-10% latency savings

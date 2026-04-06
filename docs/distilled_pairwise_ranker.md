# Distilled Pairwise Ranker

## Overview

A pairwise preference model that ranks documents by conducting pairwise comparisons between all document pairs. Trained on LLM-generated preference labels, it uses Logistic Regression to predict which of two documents is more relevant to a query. For small candidate sets, it runs a full tournament; for larger sets, it uses a merge-rank algorithm.

**Implementation**: `src/reranker/strategies/distilled.py`

**Training data**: LLM-generated pairwise preferences (doc_a vs doc_b for a query).

---

## Mathematical Formulation

### Pairwise Feature Vector

For a query q and two documents d_a, d_b, a 7-dimensional feature vector is constructed:

```
x(q, d_a, d_b) = [
    q · d_a,                    # sim(q, d_a): query-doc_a similarity
    q · d_b,                    # sim(q, d_b): query-doc_b similarity
    q · d_a - q · d_b,          # sim_diff: relative similarity advantage
    ‖d_a - d_b‖,                # doc_distance: how different the docs are
    |d_a|,                      # len_a: document a length (tokens)
    |d_b|,                      # len_b: document b length (tokens)
    |d_a| - |d_b|,              # len_diff: relative length advantage
]
```

Where:
- `q · d` is the dot product of dense embeddings
- `‖d_a - d_b‖` is the Euclidean distance between document embeddings
- `|d|` is the token count of the document

### Pairwise Preference Model

Logistic Regression predicts the probability that document A is preferred over document B:

```
P(d_a ≻ d_b | q) = σ(w^T x(q, d_a, d_b) + b)
```

Where σ is the sigmoid function, w ∈ ℝ⁷ are learned weights, and b is the bias term.

### Full Tournament Scoring (n ≤ 50 documents)

For small candidate sets, every pair is compared:

```
For each pair (i, j) where i < j:
    p_ij = P(d_i ≻ d_j | q)
    score(d_i) += p_ij
    score(d_j) += 1 - p_ij

Final score(d_i) = Σ_{j≠i} P(d_i ≻ d_j | q)
```

This is equivalent to a round-robin tournament where each document earns points proportional to its predicted win probability against every other document.

**Complexity**: O(n²) pairwise comparisons for n documents.

### Merge-Rank Scoring (n > 50 documents)

For larger candidate sets, a divide-and-conquer merge-rank algorithm is used:

```
merge_rank(docs):
    if len(docs) <= 1: return docs

    mid = len(docs) // 2
    left = merge_rank(docs[:mid])
    right = merge_rank(docs[mid:])

    merged = []
    while left and right:
        p = P(left[0] ≻ right[0] | q)
        if p >= 0.5:
            merged.append(left[0])
            left[0].score += p
            right[0].score += (1 - p)
            left.pop(0)
        else:
            merged.append(right[0])
            right[0].score += (1 - p)
            left[0].score += p
            right.pop(0)

    merged.extend(left)
    merged.extend(right)
    return merged
```

**Final score**: `score(d_i) = merge_score(d_i) + (total - position(d_i))`

**Complexity**: O(n log n) pairwise comparisons.

---

## DAG Components

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
└─────────────────────────────────────────────────────────────┘

┌──────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ Queries  │  │ Doc A  │  │ Doc B  │  │ Labels │
└────┬─────┘  └───┬────┘  └───┬────┘  └───┬────┘
     │            │           │           │
     ▼            ▼           ▼           │
┌──────────────────────────────────────┐   │
│           Embedder                   │   │
│  e(q), e(d_a), e(d_b) → vectors      │   │
└──────────────┬───────────────────────┘   │
               │                           │
               ▼                           │
┌──────────────────────────────────────────┐│
│     Pairwise Feature Construction        ││
│  x = [q·a, q·b, q·a-q·b, ‖a-b‖,        ││
│       len_a, len_b, len_a-len_b]         ││
└──────────────┬───────────────────────────┘│
               │                           │
               ▼                           │
┌──────────────────────────────────────────────────────────┐
│  Logistic Regression                                     │
│  P(y=1|x) = σ(w^T x + b)                                │
│  Loss: -Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]               │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────┐
│  Trained LR Model       │
│  (weights w, bias b)    │
└─────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│                   RERANKING PHASE                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐
│   Query Q   │     │  Documents  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────────┐
│           Embedder                          │
│  e(Q), e(D₁), ..., e(Dₙ) → vectors          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Tournament / Merge-Rank             │
│  if n <= 50: Full tournament O(n²)          │
│  if n > 50:  Merge-rank O(n log n)          │
│                                              │
│  For each pair (i, j):                      │
│    p = P(d_i ≻ d_j | Q) from LR model      │
│    Accumulate scores                         │
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

1. **For each training preference** (query, doc_a, doc_b, label):
   - Encode query, doc_a, and doc_b
   - Compute 7 pairwise features
2. **Stack features** into matrix X ∈ ℝ^(n×7)
3. **Train Logistic Regression** on (X, y):
   - C=1.0 (regularization strength)
   - max_iter=500
   - Binary labels: 1 if doc_a preferred, 0 if doc_b preferred
4. **Fallback**: If labels are constant, use DummyClassifier

### Reranking Phase

1. **Encode query and all documents**
2. **Choose strategy based on document count**:
   - **n ≤ 50**: Full tournament — compare all n(n-1)/2 pairs
   - **n > 50**: Merge-rank — divide-and-conquer with O(n log n) comparisons
3. **Accumulate scores** from pairwise comparisons
4. **Rank**: Sort documents by accumulated score (descending)

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **7 features** | Minimal set capturing similarity, distance, and length |
| **Tournament for small n** | Exact pairwise aggregation, no approximation |
| **Merge-rank for large n** | Scalable alternative to O(n²) tournament |
| **Logistic Regression** | Simple, fast, interpretable, well-calibrated probabilities |
| **LLM-distilled training** | Leverages LLM judgment without LLM inference cost at runtime |

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `logistic_c` | 1.0 | Inverse regularization strength |
| `logistic_max_iter` | 500 | Maximum optimization iterations |
| `full_tournament_max_docs` | 50 | Threshold for tournament vs merge-rank |
| `random_state` | 42 | Reproducibility |

### Complexity

| Operation | Tournament | Merge-Rank |
|-----------|------------|------------|
| Comparisons | O(n²) | O(n log n) |
| Embedding | O(n·d) | O(n·d) |
| Feature construction | O(n²·d) | O(n log n · d) |
| Total | O(n²·d) | O(n log n · d) |

### Performance

| Metric | Value |
|--------|-------|
| Latency | ~0.17ms |
| Accuracy (expanded v2) | 86.0% ± 34.7% |
| Training data | 1,764 preference pairs |

### When to Use

- **Pairwise comparisons**: When you need to know which of two documents is better
- **Ultra-fast inference**: Logistic Regression is extremely fast at prediction time
- **LLM distillation**: Captures LLM judgment quality without LLM cost
- **Small candidate sets**: Tournament scoring is exact for ≤50 documents

### Limitations

- **Not a direct ranker**: Produces pairwise preferences, not absolute scores
- **High variance**: ±34.7% std dev indicates performance varies significantly by query
- **Transitivity not guaranteed**: Pairwise preferences may contain cycles (A≻B, B≻C, C≻A)
- **Tournament cost**: O(n²) comparisons become expensive for large candidate sets

# MultiReranker

## Overview

An ensemble strategy that combines multiple rerankers using Reciprocal Rank Fusion (RRF). Each reranker scores documents independently, and their rankings are fused into a unified result. This approach requires no training and handles different score distributions gracefully.

**Implementation**: `src/reranker/strategies/multi.py`

**Fusion method**: Reciprocal Rank Fusion (RRF) with configurable weights.

---

## Mathematical Formulation

### Reciprocal Rank Fusion (RRF)

Given N rerankers, each producing a ranked list of documents, the RRF score for document d is:

```
RRF(d) = Σᵢ₌₁ᴺ wᵢ / (k + rankᵢ(d) + 1)
```

Where:
- `rankᵢ(d)` is the position of document d in reranker i's ranking (0-indexed)
- `wᵢ` is the weight for reranker i (default: 1.0 for all)
- `k` is a constant that controls how much low-ranked documents are penalized (default: 60)

### From Scores to RRF

Each reranker produces scores for the same set of documents. These scores are converted to rankings:

```
For each reranker i:
    rankᵢ = argsort(-scoresᵢ)  # descending order

Then apply RRF formula above.
```

### Weighted Fusion

When weights differ from uniform:

```
RRF_weighted(d) = Σᵢ₌₁ᴺ wᵢ / (k + rankᵢ(d) + 1)
```

Higher-weighted rerankers contribute more to the final score.

### Single Reranker Passthrough

When only one reranker is provided, scores are passed through directly (no fusion needed):

```
score(d) = ranker.rerank(query, docs)[d].score
```

---

## DAG Components

```
┌─────────────────────────────────────────────────────────────┐
│                   RERANK PHASE                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐
│   Query Q   │     │  Documents  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────────┐
│         Parallel Reranker Execution         │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Reranker │  │ Reranker │  │ Reranker │  │
│  │    1     │  │    2     │  │    N     │  │
│  │ scores₁  │  │ scores₂  │  │ scoresₙ  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │         │
│       ▼             ▼             ▼         │
│  rank₁(d)      rank₂(d)      rankₙ(d)      │
└─────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Reciprocal Rank Fusion              │
│  RRF(d) = Σ wᵢ / (k + rankᵢ(d) + 1)        │
│                                              │
│  For each document d:                        │
│    score = 0                                 │
│    for each reranker i:                      │
│      score += wᵢ / (k + rankᵢ(d) + 1)       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Sort by RRF score (desc)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────┐
│  RankedDocs │  (doc, score, rank, metadata)
└─────────────┘
```

---

## Approach & Methodology

### Rerank Phase

1. **Run each reranker independently** on (query, docs)
2. **Extract scores** from each reranker's results
3. **Apply weights** if configured (multiply scores by wᵢ)
4. **Convert to rankings**: Sort documents by each reranker's scores
5. **Apply RRF fusion**: For each document, sum 1/(k + rank + 1) across all rerankers
6. **Final ranking**: Sort documents by fused RRF scores (descending)

### Reranker Protocol

Any object implementing the rerank contract can be used:

```python
class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...
```

### Configuration

```python
@dataclass
class MultiRerankerConfig:
    rrf_k: int = 60              # RRF constant (higher = more weight to top ranks)
    weights: list[float] | None  # Per-reranker weights (default: uniform)
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **RRF over score averaging** | Handles different score distributions and scales |
| **k=60 default** | Established value from literature (Cormack et al.) |
| **Optional weights** | Allows tuning contribution of each reranker |
| **No training required** | Pure fusion, no labels or fitting needed |
| **Single reranker passthrough** | Avoid unnecessary overhead when only one reranker |

### Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Per-reranker scoring | Σ O(costᵢ) | Sum of individual reranker costs |
| Ranking conversion | O(N·n·log(n)) | N rerankers, n documents |
| RRF fusion | O(N·n) | Linear in rerankers and documents |
| Final sorting | O(n·log(n)) | Sort by fused scores |

### Performance

| Metric | Value |
|--------|-------|
| Latency | Σ(latency of component rerankers) |
| NDCG@10 | TBD |
| MRR | TBD |
| P@1 | TBD |

### When to Use

- **Ensemble approach**: Combine strengths of multiple rerankers
- **No training data**: RRF requires no labels or fitting
- **Robustness**: Individual reranker weaknesses are compensated
- **Heterogeneous signals**: Combines lexical, semantic, and learned rerankers
- **Production reliability**: If one reranker fails, others still contribute

### Limitations

- **No persistence**: Does not support save/load (save individual rerankers instead)
- **Sequential execution**: Rerankers run one after another (not parallelized)
- **Cumulative latency**: Total latency is the sum of all component rerankers
- **No interaction**: Rerankers don't share information during scoring

### Example Combinations

| Combination | Rationale |
|-------------|-----------|
| BM25 + Hybrid | Lexical baseline + learned model |
| Binary + Late Interaction | Fast binary filter + token-level precision |
| Hybrid + Distilled | GBDT ensemble + pairwise preferences |
| All available | Maximum robustness ensemble |

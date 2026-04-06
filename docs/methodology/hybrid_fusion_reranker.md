# Hybrid Fusion Reranker

## Overview

A GBDT-based reranker that fuses lexical, semantic, and heuristic signals into a unified relevance score. It combines a learned classifier (XGBoost or sklearn GradientBoosting) with a weighted heuristic blend, averaging both for the final score.

**Implementation**: `src/reranker/strategies/hybrid.py`

**Classifier**: XGBoost (preferred) or sklearn GradientBoostingClassifier (fallback).

---

## Mathematical Formulation

### Feature Vector

For each query-document pair (q, d), a feature vector x ∈ ℝ^f is constructed:

```
x = [sem_score, bm25_score, vec_norm_diff, token_overlap_ratio,
     query_coverage_ratio, shared_token_char_sum, exact_phrase_match,
     query_len, doc_len, ...adapter_features]
```

### Feature Definitions

| Feature | Formula | Description |
|---------|---------|-------------|
| `sem_score` | q · d | Dot product of dense embeddings |
| `bm25_score` | BM25(q, d) | Normalized BM25 score |
| `vec_norm_diff` | ‖q - d‖ | Euclidean distance between embeddings |
| `token_overlap_ratio` | |Q ∩ D| / |Q ∪ D| | Jaccard similarity of token sets |
| `query_coverage_ratio` | |Q ∩ D| / |Q| | Fraction of query terms found in document |
| `shared_token_char_sum` | Σ_{t ∈ Q∩D} |t| | Total character length of shared tokens |
| `exact_phrase_match` | 𝟙[q ⊆ d] | 1 if query string is substring of document, 0 otherwise |
| `query_len` | |tokens(q)| | Number of tokens in query |
| `doc_len` | |tokens(d)| | Number of tokens in document |

### Heuristic Blended Score

A weighted linear combination of selected features:

```
S_blend(q, d) = w₁·sem_score + w₂·bm25_score + w₃·token_overlap
              + w₄·query_coverage + w₅·(shared_char / query_tokens)
              + w₆·exact_phrase + w₇·keyword_hit_rate
```

**Default weights** (from config):

| Weight | Feature | Value |
|--------|---------|-------|
| w₁ | sem_score | 0.25 |
| w₂ | bm25_score | 0.20 |
| w₃ | token_overlap | 0.15 |
| w₄ | query_coverage | 0.20 |
| w₅ | shared_char | 0.10 |
| w₆ | exact_phrase | 0.10 |
| w₇ | keyword_hit | 0.05 |

### Model Score

The GBDT classifier outputs a probability:

```
S_model(q, d) = P(y=1 | x) = σ(F_GBDT(x))
```

Where F_GBDT is the ensemble of decision trees and σ is the sigmoid function.

### Final Score

The final score is the average of the model and heuristic scores:

```
S_final(q, d) = (S_model(q, d) + S_blend(q, d)) / 2
```

### XGBoost Configuration

| Hyperparameter | Value | Role |
|----------------|-------|------|
| n_estimators | 120 | Number of boosting rounds |
| max_depth | 4 | Maximum tree depth |
| learning_rate | 0.08 | Step size shrinkage |
| subsample | 0.9 | Row subsampling ratio |
| colsample_bytree | 0.9 | Column subsampling ratio |
| eval_metric | logloss | Binary classification loss |

---

## DAG Components

```
┌─────────────┐     ┌─────────────┐
│   Query Q   │     │  Documents  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────────┐
│          Feature Construction               │
│  ┌─────────┐  ┌─────────┐  ┌────────────┐  │
│  │ Embedder│  │ BM25    │  │ Tokenizer  │  │
│  │ q, dᵢ   │  │ Engine  │  │ Q, D sets  │  │
│  └────┬────┘  └────┬────┘  └─────┬──────┘  │
│       │            │             │          │
│       ▼            ▼             ▼          │
│  ┌──────────────────────────────────────┐   │
│  │ Feature Vector xᵢ ∈ ℝ^f              │   │
│  │ [sem, bm25, norm_diff, overlap, ...] │   │
│  └──────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  GBDT Classifier │  │  Heuristic Blend     │
│  S_model = P(y=1)│  │  S_blend = Σ wⱼ·fⱼ  │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────┐
│           Final Score                       │
│  S_final = (S_model + S_blend) / 2          │
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
┌─────────────────────────────────────────────┐
│  Feature Construction (per pair)            │
│  xᵢ = build_features(qᵢ, dᵢ)                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  XGBoost / GradientBoostingClassifier       │
│  min logloss: Σ -[yᵢ log(pᵢ) + (1-yᵢ)log(1-pᵢ)] │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────┐
│  Trained GBDT Model     │
└─────────────────────────┘
```

---

## Approach & Methodology

### Fit Phase

1. **For each training pair** (query, doc, label):
   - Encode query and document embeddings
   - Compute BM25 scores
   - Extract lexical features (token overlap, coverage, etc.)
   - Run adapter features if configured
2. **Stack features** into matrix X ∈ ℝ^(n×f)
3. **Train GBDT classifier** on (X, y):
   - Try XGBoost first; fall back to sklearn GradientBoosting
   - Uses logloss for binary classification
4. **Store feature names** for consistent inference

### Score Phase

1. **Build feature matrix** for all query-document pairs
2. **Compute heuristic blend**: Weighted sum of key features
3. **Get model predictions**: P(y=1|x) from GBDT
4. **Average**: Final score = (model + blend) / 2
5. **Rank**: Sort documents by final scores (descending)

### HeuristicAdapter Protocol

Custom adapters can inject additional features:

```python
class HeuristicAdapter(Protocol):
    def compute(self, query: str, doc: str) -> dict[str, float]: ...
```

**Built-in adapter**: `KeywordMatchAdapter` — computes term hit rate (fraction of query terms found in document).

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Model + blend average** | Combines learned patterns with interpretable heuristics |
| **XGBoost first** | Faster training, better performance than sklearn GBDT |
| **9 base features** | Covers semantic, lexical, and structural signals |
| **Adapter extensibility** | Domain-specific features without modifying core logic |
| **Auto BM25 fit** | Fits BM25 on candidate docs if not provided |

### Performance

| Metric | Value |
|--------|-------|
| Latency (20 docs) | ~1.62ms |
| Latency (200 docs) | ~6.02ms |
| NDCG@10 (expanded v2) | 0.8007 ± 0.13 |
| MRR | 1.0000 |
| P@1 | 1.0000 |
| Scaling (20→200 docs) | 3.7x (best scaling) |

### Ablation Findings

- **Adapters contribute nothing** on current data — 9 built-in features are sufficient
- **XGBoost vs sklearn GBDT**: XGBoost preferred when available
- **Recommendation**: No adapters needed for general use

### When to Use

- **Production accuracy-critical**: Perfect MRR and P@1
- **Best scaling behavior**: 3.7x latency growth for 10x corpus
- **Interpretable scoring**: Heuristic blend provides transparent feature contributions
- **Extensible**: Add domain-specific adapters for custom signals

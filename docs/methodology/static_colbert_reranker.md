# Static ColBERT Reranker

## Overview

A late interaction reranker inspired by the ColBERT architecture. Instead of collapsing documents into single vectors (mean pooling), it stores per-token embeddings and computes fine-grained token-level alignment at query time using the MaxSim operator.

**Implementation**: `src/reranker/strategies/late_interaction.py`

**Key innovation**: Captures term-level alignment that mean-pooling loses, while remaining CPU-efficient through static pre-computed token vectors.

---

## Mathematical Formulation

### Token-Level Embeddings

Each document D is tokenized into tokens t₁, t₂, ..., tₘ, and each token is embedded independently:

```
E(D) = [e(t₁), e(t₂), ..., e(tₘ)]  where e(tᵢ) ∈ ℝ^d
```

Unlike standard sentence embeddings that produce a single vector, this produces a matrix of shape (m, d).

### MaxSim Scoring

For a query Q with token embeddings E(Q) = [e(q₁), ..., e(qₙ)] and document token embeddings E(D):

```
score(Q, D) = Σᵢ₌₁ⁿ maxⱼ₌₁ᵐ cosine(e(qᵢ), e(dⱼ))
```

Where cosine similarity is:

```
cosine(u, v) = (u · v) / (‖u‖ · ‖v‖)
```

**Intuition**: Each query token finds its best-matching document token. The score is the sum of these best matches. This captures multi-term alignment — different query terms can match different parts of the document.

### Salience Weighting (Optional)

When enabled, document token vectors are weighted by TF-IDF salience before MaxSim:

```
TF(t, D) = count(t, D)
IDF(t, D) = log(1 + |D| / (TF(t, D) + 1))
salience(t) = TF(t, D) × IDF(t, D)

E'(D) = [salience(t₁)·e(t₁), ..., salience(tₘ)·e(tₘ)]
```

### Token Pruning

Documents with more than `top_k_tokens` tokens are pruned to the most salient tokens:

```
If salience enabled:  keep top-k tokens by salience score
If salience disabled: keep first k tokens
```

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
│           Tokenize (per doc)                │
│  tokens = doc.lower().split()               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Encode Tokens (per token)            │
│  vectors[i] = embedder.encode(tokens[i])    │
│  Shape: (num_tokens, embed_dim)             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Optional: Compute Salience           │
│  tf = term frequency per token              │
│  idf = log(1 + |tokens| / (tf + 1))        │
│  salience = tf × idf                        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Token Pruning (if needed)            │
│  Keep top-k by salience or first-k          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          TokenIndex (stored)                │
│  {tokens, vectors, salience?}               │
└─────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│                   SCORE PHASE                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐
│   Query Q   │     │ TokenIndex  │
│             │     │  (pre-built)│
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   │
┌──────────────┐           │
│  Tokenize Q  │           │
│  Encode Q    │           │
│  E(Q) ∈ ℝ^(n×d)          │
└──────┬───────┘           │
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────────┐
│         MaxSim (per query-doc pair)         │
│  ┌─────────────────────────────────────┐    │
│  │ Normalize E(Q) and E(D) vectors     │    │
│  │ sim_matrix = E(Q) @ E(D)^T          │    │
│  │ max_sims = max(sim_matrix, axis=1)  │    │
│  │ score = sum(max_sims)               │    │
│  └─────────────────────────────────────┘    │
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

1. **For each document**:
   - Tokenize by whitespace (lowercase)
   - Encode each token independently via embedder
   - Optionally compute TF-IDF salience weights
   - Prune to top-k tokens if exceeding limit
   - Store as `TokenIndex(tokens, vectors, salience?)`
2. **Build index**: List of TokenIndex entries, one per document

### Score Phase

1. **Tokenize and encode query**: Get per-token embeddings E(Q)
2. **For each document** in the index:
   - Retrieve pre-computed token embeddings E(D)
   - Apply salience weighting if enabled: E'(D) = E(D) × salience
   - Normalize both E(Q) and E(D) to unit vectors
   - Compute similarity matrix: S = E(Q) @ E(D)^T
   - MaxSim: For each query token, take max similarity across document tokens
   - Score = sum of max similarities
3. **Rank**: Sort documents by MaxSim score (descending)

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Whitespace tokenization** | Simple, fast, no external dependencies |
| **Per-token embedding** | Captures term-level semantics lost in mean pooling |
| **Static vectors** | Pre-computed at fit time, no re-encoding at query time |
| **TF-IDF salience** | Downweights common tokens, upweights distinctive ones |
| **Token pruning** | Controls memory and compute for long documents |

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `top_k_tokens` | 128 | Maximum tokens to retain per document |
| `use_salience` | True | Whether to apply TF-IDF weighting |

### Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Token encoding (fit) | O(m·d) per doc | m = token count |
| Salience computation | O(m log m) | Sorting for top-k |
| MaxSim (score) | O(n·m·d) per pair | Matrix multiply |
| Pruned MaxSim | O(n·k·d) | k = top_k_tokens |

### Performance

| Metric | Value |
|--------|-------|
| Latency (20 docs) | ~0.54ms |
| Latency (200 docs) | ~3.90ms |
| NDCG@10 (expanded v2) | 0.8007 ± 0.13 |
| MRR | 1.0000 |
| P@1 | 1.0000 |
| Scaling (20→200 docs) | 7.2x |

### Ablation Findings

- **Salience weighting has no impact** on current synthetic data
- **Token count reduction** (128 → 64) has no impact on accuracy
- **Recommendation**: Use 64 tokens without salience for efficiency

### When to Use

- **Token-level precision needed**: When term-by-term alignment matters
- **CPU-efficient late interaction**: No GPU required, static vectors
- **Medium corpora**: Scales well up to ~200 documents
- **Production**: Perfect MRR and P@1, competitive NDCG@10

### Comparison to Original ColBERT

| Aspect | Original ColBERT | Static ColBERT |
|--------|-----------------|----------------|
| Tokenizer | BERT WordPiece | Whitespace split |
| Embeddings | Contextual (BERT) | Static (model2vec) |
| Training | End-to-end | Pre-trained embedder |
| MaxSim | GPU-optimized | CPU matrix multiply |
| Use case | Full search engine | Reranking stage |

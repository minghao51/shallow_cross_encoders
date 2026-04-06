# Pipeline Reranker

## Overview

A multi-stage cascading pipeline that feeds documents through a sequence of reranking stages, each filtering and re-ranking the candidate set before passing top-k to the next stage. This enables combining fast, coarse filters with expensive, precise rerankers efficiently.

**Implementation**: `src/reranker/strategies/pipeline.py`

**Design principle**: Cheap stages first, expensive stages last — maximize quality while minimizing compute.

---

## Mathematical Formulation

### Cascade Architecture

Given an initial candidate set D₀ with |D₀| = n documents, and K stages with top-k parameters k₁, k₂, ..., k_K:

```
D₁ = top_k₁(Reranker₁(Q, D₀))
D₂ = top_k₂(Reranker₂(Q, D₁))
...
D_K = top_k_K(Reranker_K(Q, D_{K-1}))
```

Where:
- `Reranker_i(Q, D)` produces a ranked list of documents in D for query Q
- `top_k(L)` returns the first k documents from ranked list L
- Each stage i receives |D_{i-1}| documents and outputs |D_i| ≤ k_i documents

### Final Scoring

The final ranking preserves the scores from the last stage:

```
score_final(d) = score_K(d)  for d ∈ D_K
rank_final(d) = position of d in D_K (1-indexed)
```

### Latency Model

Total pipeline latency is the sum of stage latencies:

```
T_total = Σᵢ T_i(|D_{i-1}|)
```

Where T_i(m) is the latency of stage i on m documents. Since each stage reduces the candidate set, later (more expensive) stages run on fewer documents:

```
T_total = T₁(n) + T₂(k₁) + T₃(k₂) + ... + T_K(k_{K-1})
```

This is significantly cheaper than running the most expensive stage on all n documents.

---

## DAG Components

```
┌─────────────┐
│   Query Q   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│         Initial Candidates D₀               │
│         |D₀| = n documents                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: BM25 (fast lexical filter)                        │
│  Input: n docs → Output: top-k₁ docs                        │
│  Records: input_count, output_count, latency_ms, top_score  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ D₁ (|D₁| = k₁)
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: BinaryQuantizedReranker (fast binary semantic)    │
│  Input: k₁ docs → Output: top-k₂ docs                       │
│  Records: input_count, output_count, latency_ms, top_score  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ D₂ (|D₂| = k₂)
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: HybridFusionReranker (GBDT reranker)              │
│  Input: k₂ docs → Output: top-k₃ docs                       │
│  Records: input_count, output_count, latency_ms, top_score  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ D₃ (|D₃| = k₃)
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: StaticColBERTReranker (late interaction)          │
│  Input: k₃ docs → Output: top-k₄ docs                       │
│  Records: input_count, output_count, latency_ms, top_score  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ D₄ (|D₄| = k₄)
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: DistilledPairwiseRanker (pairwise tournament)     │
│  Input: k₄ docs → Output: final ranking                     │
│  Records: input_count, output_count, latency_ms, top_score  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         PipelineResult                      │
│  - final_ranking: list[RankedDoc]           │
│  - stage_results: [stage_metadata, ...]     │
│  - total_latency_ms: float                  │
└─────────────────────────────────────────────┘
```

---

## Approach & Methodology

### Configuration

The pipeline is configured as a list of stages, each with:
- **name**: Identifier for the stage
- **reranker**: The reranking strategy instance
- **top_k**: Number of documents to pass to the next stage

```python
pipeline = PipelineReranker()
pipeline.add_stage("bm25", bm25_reranker, top_k=500)
pipeline.add_stage("binary", binary_reranker, top_k=200)
pipeline.add_stage("hybrid", hybrid_reranker, top_k=50)
pipeline.add_stage("colbert", colbert_reranker, top_k=20)
pipeline.add_stage("pairwise", pairwise_ranker, top_k=10)
```

### Execution

1. **Start with all candidate documents**
2. **For each stage** in order:
   - Run the stage's reranker on current candidates
   - Record stage metadata (input count, output count, latency, top score)
   - Select top-k documents for next stage
   - If no documents remain, stop early
3. **Build final ranking** with scores from the last stage
4. **Return** PipelineResult with ranking and per-stage metadata

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Cascading top-k** | Each stage reduces candidate set, saving compute for expensive stages |
| **Ordered by cost** | Cheapest stages first, most expensive last |
| **Stage metadata** | Full observability into each stage's contribution |
| **Early termination** | Stops if a stage produces no candidates |
| **Passthrough fallback** | If no stages configured, returns documents as-is |

### Recommended Pipeline Configuration

| Stage | Reranker | top_k | Rationale |
|-------|----------|-------|-----------|
| 1 | BM25 | 500 | Fast lexical filter, eliminates obviously irrelevant docs |
| 2 | BinaryQuantized | 200 | Fast binary semantic, removes lexical-only matches |
| 3 | HybridFusion | 50 | GBDT with rich features, strong quality filter |
| 4 | StaticColBERT | 20 | Token-level precision, fine-grained alignment |
| 5 | DistilledPairwise | 10 | Pairwise tournament, final quality pass |

### Performance

| Metric | Value |
|--------|-------|
| Total latency (expanded v2) | ~2.21ms |
| NDCG@10 | 0.5281 ± 0.13 |
| MRR | 0.7778 |
| P@1 | 0.6667 |

### Why Pipeline Underperforms Individual Stages

The pipeline's NDCG@10 (0.528) is lower than individual stages because:
1. **Error accumulation**: Early stages may filter out relevant documents
2. **Score incompatibility**: Different stages use different scoring scales
3. **Aggressive filtering**: top-k thresholds may be too restrictive
4. **Final stage bottleneck**: The last stage's quality caps the overall result

### When to Use

- **Large candidate pools**: When starting with 1000+ documents
- **Latency budgets**: When you need to balance quality with response time
- **Progressive refinement**: When each stage adds complementary signals
- **Debugging**: Per-stage metadata helps identify bottlenecks

### Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| Cascading | Reduces compute for expensive stages | Early stages may filter relevant docs |
| Multi-signal | Combines lexical, semantic, pairwise signals | More complex to tune and maintain |
| Observability | Per-stage latency and quality metrics | More components to monitor |
| Flexibility | Stages can be swapped independently | Each stage must be fitted separately |

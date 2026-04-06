# Unified Benchmark Results

Generated: 2026-04-04 15:16:05
Seed: 42
Embedder: minishlab/potion-base-32M
Quick mode: False
Test pairs: 10
Test preferences: 7
Test contradictions: 7

## Baseline Results

| Strategy | Experiment | NDCG@10 | MRR | P@1 | Latency (ms) |
|----------|------------|---------|-----|-----|--------------|
| bm25 | bm25_baseline | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.09 |
| hybrid | hybrid_baseline | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.56 |
| distilled | distilled_baseline | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.27 |
| late_interaction | colbert_baseline | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.43 |
| binary_reranker | binary_baseline | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.44 |
| pipeline | pipeline_baseline | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2.48 |
| distilled | distilled_baseline | acc=1.0000 ± 0.0000 | - | - | 0.27 |
| consistency | consistency_baseline | recall=1.0000, fpr=0.0000 | - | - | 0.32 |
| splade | splade_skipped | SKIPPED (requires sentence-transformers) |

## MultiReranker (RRF Fusion)

| Experiment | Rerankers | NDCG@10 | MRR | P@1 | Latency (ms) |
|------------|-----------|---------|-----|-----|--------------|
| multi_bm25_binary | bm25, binary | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.56 |
| multi_hybrid_bm25 | hybrid, bm25 | 0.8870 ± 0.0786 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.73 |

## Ablation Studies

### hybrid

| Experiment | NDCG@10 | Δ vs Baseline |
|------------|---------|---------------|
| hybrid_ablation_no_adapters | 0.8870 | +0.0000 |

### late_interaction

| Experiment | NDCG@10 | Δ vs Baseline |
|------------|---------|---------------|
| colbert_ablation_no_salience | 0.8870 | +0.0000 |
| colbert_ablation_64_tokens | 0.8870 | +0.0000 |

### binary_reranker

| Experiment | NDCG@10 | Δ vs Baseline |
|------------|---------|---------------|
| binary_ablation_hamming_only | 0.8870 | +0.0000 |
| binary_ablation_aggressive | 0.8870 | +0.0000 |

### consistency

| Experiment | Recall | FPR | Δ Recall vs Baseline |
|------------|--------|-----|---------------------|
| consistency_ablation_relaxed | 1.0000 | 0.0000 | +0.0000 |
| consistency_ablation_strict | 1.0000 | 0.0000 | +0.0000 |

## Scaling Results (Latency in ms)

| Corpus Size | BM25 (ms) | Hybrid (ms) | ColBERT (ms) | Binary (ms) |
|-------------|-----------|-------------|--------------|-------------|
| 20 | 0.08 | 1.45 | 0.32 | 0.39 |
| 50 | 0.07 | 1.48 | 0.36 | 0.33 |
| 100 | 0.07 | 1.46 | 0.32 | 0.38 |
| 200 | 0.08 | 1.47 | 0.30 | 0.41 |

## Embedder Model Comparison

| Model | Dim | Strategy | NDCG@10 | MRR | P@1 | Latency (ms) |
|-------|-----|----------|---------|-----|-----|--------------|
| potion-base-8M | 64 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.17 |
| potion-base-8M | 128 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.17 |
| potion-base-8M | 256 | hybrid | 1.0000 | 1.0000 | 1.0000 | 0.64 |
| potion-base-8M | 256 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.16 |
| potion-base-8M | 512 | hybrid | 1.0000 | 1.0000 | 1.0000 | 0.63 |
| potion-base-8M | 512 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.19 |
| potion-base-32M | 64 | hybrid | 1.0000 | 1.0000 | 1.0000 | 0.63 |
| potion-base-32M | 64 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.24 |
| potion-base-32M | 128 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.19 |
| potion-base-32M | 256 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.18 |
| potion-base-32M | 512 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 0.20 |
| potion-base-8M | 64 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.61 |
| potion-base-8M | 64 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.29 |
| potion-base-8M | 128 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.59 |
| potion-base-8M | 128 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.23 |
| potion-base-8M | 256 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.28 |
| potion-base-8M | 512 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.26 |
| potion-base-32M | 64 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.23 |
| potion-base-32M | 128 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.68 |
| potion-base-32M | 128 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.24 |
| potion-base-32M | 256 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.57 |
| potion-base-32M | 256 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.29 |
| potion-base-32M | 512 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.59 |
| potion-base-32M | 512 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.27 |
| potion-multilingual-128M | 64 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.70 |
| potion-multilingual-128M | 64 | binary_reranker | 0.9262 | 0.9000 | 0.8000 | 0.38 |
| potion-multilingual-128M | 64 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.33 |
| potion-multilingual-128M | 128 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.69 |
| potion-multilingual-128M | 128 | binary_reranker | 0.9262 | 0.9000 | 0.8000 | 0.29 |
| potion-multilingual-128M | 128 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.31 |
| potion-multilingual-128M | 256 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.72 |
| potion-multilingual-128M | 256 | binary_reranker | 0.9262 | 0.9000 | 0.8000 | 0.19 |
| potion-multilingual-128M | 256 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.36 |
| potion-multilingual-128M | 512 | hybrid | 0.9262 | 0.9000 | 0.8000 | 0.80 |
| potion-multilingual-128M | 512 | binary_reranker | 0.9262 | 0.9000 | 0.8000 | 0.21 |
| potion-multilingual-128M | 512 | late_interaction | 0.9262 | 0.9000 | 0.8000 | 0.29 |

**Best NDCG@10**: minishlab/potion-base-8M (dim=64, strategy=binary_reranker) = 1.0000

## Key Findings

- **Best NDCG@10**: bm25_baseline (0.8870)
- **Fastest**: bm25_baseline (0.09ms)

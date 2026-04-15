# Technical Roadmap: Lightweight Reranking & Consistency System

> **Objective:** Build a modular, CPU-native reranking and consistency-checking pipeline powered by `model2vec` static embeddings, distilled LLM judgment, and hybrid lexical-semantic features — designed to be embedded as a drop-in component in an existing system.
>
> **Architecture Principle:** Every strategy resolves to a single, clean interface:
> ```python
> reranker.rerank(query: str, docs: list[str]) -> list[RankedDoc]
> ```
> No GPU required. No inference server. Pure Python, importable modules.

---

## Stack

| Component | Technology | Role |
|---|---|---|
| Vector Engine | `model2vec` (`potion-base-8M` or `Potion-32M`) | Generates dense static representations locally |
| Lexical Engine | `rank_bm25` | Provides exact-match and acronym signal |
| Reranking Logic | `xgboost` / `scikit-learn` | Shallow cross-encoder that weighs fused features |
| LLM Teacher | OpenRouter API | Generates synthetic labels — used in training only, never inference |
| Structured Extraction | `pydantic` v2 | Schema-enforced claim extraction for consistency checks |
| Environment | Python 3.11+ with `uv` | Ultra-fast dependency resolution and isolated execution |
| Data Format | JSONL / Parquet | Reproducible synthetic datasets |

---

## Phase 0 — Synthetic Data Engine

> **Goal:** Build the LLM teacher pipeline that generates training data for all three downstream strategies. This is the keystone phase — nothing else trains without it.

### 0.1 Project Scaffold

```
reranker/
├── pyproject.toml           # managed by uv
├── data/
│   ├── raw/                 # LLM-generated JSONL outputs
│   └── processed/           # Feature matrices, ready for training
├── src/
│   └── reranker/
│       ├── __init__.py
│       ├── protocols.py     # RankedDoc, HeuristicAdapter, BaseReranker ABCs
│       ├── embedder.py      # model2vec wrapper
│       ├── lexical.py       # BM25 wrapper
│       ├── data/
│       │   └── synth.py     # Synthetic generation pipeline
│   ├── strategies/
│   │   ├── hybrid.py        # Phase 2: Hybrid Fusion Reranker
│   │   ├── distilled.py     # Phase 3: Distilled Pairwise Ranker
│   │   └── consistency.py   # Phase 4: Consistency Engine
│   └── eval/
│       └── metrics.py       # NDCG, MRR, latency, cost tracking
├── notebooks/
│   └── 00_data_exploration.ipynb
└── scripts/
    ├── generate_pairs.py
    ├── generate_preferences.py
    └── generate_contradictions.py
```

**Setup:**
```bash
uv init reranker
cd reranker
uv add model2vec rank-bm25 xgboost scikit-learn pydantic openai httpx
uv add --dev pytest pytest-benchmark ruff
```

### 0.2 Dataset A — Relevance-Labeled Pairs (for Strategy 2)

Generate query → document pairs with a graded relevance score (0–3).

```python
# scripts/generate_pairs.py
PROMPT = """
You are a relevance judge. Given a query and a document, rate relevance 0–3.
0 = Irrelevant | 1 = Tangential | 2 = Relevant | 3 = Highly Relevant

Return ONLY valid JSON: {"score": <int>, "rationale": "<one sentence>"}

Query: {query}
Document: {doc}
"""
```

**Targets:**
- ~2,000 query-document pairs
- Balanced label distribution (avoid collapse to majority class)
- Store as `data/raw/pairs.jsonl` — fields: `query`, `doc`, `score`, `rationale`

### 0.3 Dataset B — Pairwise Preferences (for Strategy 1)

Generate (query, doc_a, doc_b) triples with binary preference labels.

```python
# scripts/generate_preferences.py
PROMPT = """
Given a query, decide which document better answers it.
Return ONLY valid JSON: {"preferred": "A" or "B", "confidence": 0.0–1.0}

Query: {query}
Document A: {doc_a}
Document B: {doc_b}
"""
```

**Targets:**
- ~1,500 preference triples (derived from Dataset A pairs — use high-confidence pairs)
- Store as `data/raw/preferences.jsonl` — fields: `query`, `doc_a`, `doc_b`, `preferred`, `confidence`
- Log total API cost per run for ROI baselining

### 0.4 Dataset C — Structured Claims with Contradictions (for Strategy 3)

Generate documents containing deliberate factual contradictions on shared entities.

```python
# scripts/generate_contradictions.py
PROMPT = """
Generate two short document excerpts about the same subject.
Document B should contain a factual contradiction of a specific numerical or categorical
claim made in Document A. The rest of the content should be semantically similar.

Return ONLY valid JSON matching this schema:
{
  "subject": "<entity name>",
  "doc_a": "<text>",
  "doc_b": "<text with contradiction>",
  "contradicted_field": "<field name>",
  "value_a": "<value in doc_a>",
  "value_b": "<value in doc_b>"
}
"""
```

**Targets:**
- ~500 contradiction pairs
- ~200 non-contradicting control pairs (same entity, consistent values)
- Store as `data/raw/contradictions.jsonl`

### Phase 0 Exit Criteria

- [ ] All three JSONL datasets generated and versioned
- [ ] Label distribution visualised and validated (no severe class imbalance)
- [ ] Total API cost logged — this is the cost baseline Strategy 1 aims to beat
- [ ] `data/raw/` committed with a reproducibility seed

---

## Phase 1 — Core Infrastructure

> **Goal:** Wire up the embedding pipeline, BM25 engine, and feature construction utilities. Define the shared interfaces every strategy implements. Hit the latency target on bare metal.

### 1.1 Core Protocols (`src/reranker/protocols.py`)

```python
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

@dataclass
class RankedDoc:
    doc: str
    score: float
    rank: int
    metadata: dict = None

@runtime_checkable
class HeuristicAdapter(Protocol):
    """Domain-specific feature injector. Implement this to extend any reranker."""
    def compute(self, query: str, doc: str) -> dict[str, float]: ...

@runtime_checkable  
class BaseReranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...
```

> **Design note:** `HeuristicAdapter` is the plugin point for domain-specific signals. The pipeline consumer registers one or more adapters — the reranker calls `.compute()` on each and appends the returned floats to the feature vector. The core library is never modified.

### 1.2 Embedder Wrapper (`src/reranker/embedder.py`)

```python
from model2vec import StaticModel

class Embedder:
    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        self.model = StaticModel.from_pretrained(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize=True)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # cosine, since normalized
```

**Model selection note:**
- `potion-base-8M` — default, lowest latency, good for high-throughput
- `potion-base-32M` — higher accuracy ceiling, still CPU-native; switch if eval shows accuracy gap

### 1.3 Lexical Wrapper (`src/reranker/lexical.py`)

```python
from rank_bm25 import BM25Okapi

class BM25Engine:
    def fit(self, corpus: list[str]) -> None:
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def score(self, query: str, normalize: bool = True) -> np.ndarray:
        scores = self.bm25.get_scores(query.lower().split())
        if normalize and scores.max() > 0:
            scores = scores / scores.max()
        return scores
```

### 1.4 Latency Benchmark

Run before any model training. This is your performance floor contract.

```python
# scripts/benchmark_core.py
import time, numpy as np
from reranker.embedder import Embedder

embedder = Embedder()
texts = ["This is a sample document."] * 100

start = time.perf_counter()
embedder.encode(texts)
elapsed = (time.perf_counter() - start) / len(texts) * 1000

print(f"Avg embedding latency: {elapsed:.2f}ms per doc")
# Target: < 2ms per doc on CPU
```

### Phase 1 Exit Criteria

- [ ] `Embedder`, `BM25Engine` importable and tested
- [ ] `BaseReranker` and `HeuristicAdapter` protocols defined
- [ ] Avg embedding latency < 2ms/doc on CPU (no GPU)

---

## Phase 2 — Hybrid Fusion Reranker (Strategy 2)

> **Goal:** Build the foundational reranker that fuses semantic, lexical, and pluggable domain signals into a single XGBoost model. This is the scaffold all other strategies extend.

### 2.1 Feature Construction

For a (query `Q`, document `D`) pair, construct:

| Feature | Formula | Signal |
|---|---|---|
| `sem_score` | `Q_vec · D_vec` | Semantic similarity |
| `bm25_score` | `BM25(Q, D)` | Exact lexical match |
| `vec_norm_diff` | `‖Q_vec - D_vec‖₂` | Embedding distance |
| `query_len` | `len(Q.split())` | Query complexity proxy |
| `doc_len` | `len(D.split())` | Document length bias |
| `heuristic_*` | `adapter.compute(Q, D)` | Pluggable domain signals |

```python
# src/reranker/strategies/hybrid.py
class HybridFusionReranker:
    def __init__(self, adapters: list[HeuristicAdapter] = None):
        self.embedder = Embedder()
        self.adapters = adapters or []
        self.model = None  # fitted XGBClassifier / XGBRanker

    def _build_features(self, query: str, docs: list[str]) -> np.ndarray:
        q_vec = self.embedder.encode([query])[0]
        d_vecs = self.embedder.encode(docs)
        # ... construct feature matrix
        # append adapter outputs per doc
        for adapter in self.adapters:
            for i, doc in enumerate(docs):
                hfeats = list(adapter.compute(query, doc).values())
                # append to row i

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4)
        self.model.fit(X, y)

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        X = self._build_features(query, docs)
        scores = self.model.predict_proba(X)[:, 1]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [RankedDoc(doc=d, score=s, rank=i+1) for i, (d, s) in enumerate(ranked)]
```

### 2.2 Training Pipeline

```python
# scripts/train_hybrid.py
# 1. Load data/raw/pairs.jsonl
# 2. Embed all queries and documents (batch for efficiency)
# 3. Construct feature matrix X, label vector y (binarise scores >= 2 = relevant)
# 4. Train/val/test split: 70/15/15
# 5. Fit HybridFusionReranker
# 6. Evaluate: NDCG@10, MRR, vs BM25-only baseline
# 7. Serialize model to data/models/hybrid_reranker.json
```

### 2.3 Adapter Pattern — Example Stub

```python
class KeywordMatchAdapter:
    """Example: boost score if query terms appear verbatim in doc."""
    def compute(self, query: str, doc: str) -> dict[str, float]:
        terms = query.lower().split()
        hit_rate = sum(1 for t in terms if t in doc.lower()) / max(len(terms), 1)
        return {"keyword_hit_rate": hit_rate}
```

Pipeline consumers implement `HeuristicAdapter` and pass their list to the constructor — the core library is never modified.

### Phase 2 Exit Criteria

- [ ] XGBoost reranker trained on synthetic pairs
- [ ] **NDCG@10 ≥ BM25-only + 10 points** on held-out test set
- [ ] `HeuristicAdapter` protocol validated with at least one stub adapter
- [ ] `rerank()` end-to-end latency < 5ms for a 20-document candidate list

---

## Phase 3 — Distilled Pairwise Ranker (Strategy 1)

> **Goal:** Distill LLM pairwise judgment into a sub-millisecond local classifier. Eliminate API calls from the inference path entirely.

### 3.1 Pairwise Feature Construction

For a (query `Q`, doc_a `A`, doc_b `B`) triple:

| Feature | Formula |
|---|---|
| `qa_sim` | `Q_vec · A_vec` |
| `qb_sim` | `Q_vec · B_vec` |
| `sim_diff` | `qa_sim - qb_sim` |
| `ab_dist` | `‖A_vec - B_vec‖₂` |
| `a_len`, `b_len` | document lengths |
| `len_diff` | `a_len - b_len` |

```python
# src/reranker/strategies/distilled.py
class DistilledPairwiseRanker:
    """
    Predicts which of two documents better answers a query.
    Trained on LLM preference labels; runs entirely on CPU at inference.
    """
    def __init__(self):
        self.embedder = Embedder()
        self.model = LogisticRegression(C=1.0, max_iter=500)

    def _build_pairwise_features(self, query, doc_a, doc_b) -> np.ndarray:
        q, a, b = self.embedder.encode([query, doc_a, doc_b])
        return np.array([
            np.dot(q, a),
            np.dot(q, b),
            np.dot(q, a) - np.dot(q, b),
            np.linalg.norm(a - b),
            len(doc_a.split()),
            len(doc_b.split()),
            len(doc_a.split()) - len(doc_b.split()),
        ])

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        # Tournament-style: score each doc via aggregated pairwise wins
        scores = np.zeros(len(docs))
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                feats = self._build_pairwise_features(query, docs[i], docs[j])
                pred = self.model.predict_proba([feats])[0, 1]
                scores[i] += pred
                scores[j] += (1 - pred)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [RankedDoc(doc=d, score=s, rank=i+1) for i, (d, s) in enumerate(ranked)]
```

> **Note on tournament scaling:** For large candidate sets (>50 docs), switch from full round-robin to a merge-sort–based comparison strategy to stay O(n log n).

### 3.2 ROI Measurement Script

```python
# scripts/measure_roi.py
# Compare three modes on identical preference test set:
# Mode A: Full LLM judge (OpenRouter API call per pair) — log cost + latency
# Mode B: Distilled model (local CPU) — log cost ($0) + latency
# Mode C: Semantic similarity only (no model) — log cost ($0) + latency
# Output: accuracy vs Mode A, latency ratio, projected monthly cost at N queries/day
```

### Phase 3 Exit Criteria

- [ ] Distilled classifier accuracy within **5% of LLM judge** on held-out preference set
- [ ] Single pairwise inference < **1ms** on CPU
- [ ] ROI script showing projected cost reduction (target: >80% vs pure API approach)
- [ ] Model serialised as `data/models/pairwise_ranker.pkl`

---

## Phase 4 — Consistency Engine (Strategy 3)

> **Goal:** Detect factual contradictions across document fragments using Pydantic-structured extraction + CPU-native distance matrices. No inference server required.

### 4.1 Generic Claim Schema

```python
# src/reranker/strategies/consistency.py
from pydantic import BaseModel, Field
from typing import Any

class Claim(BaseModel):
    entity: str = Field(description="The subject entity this claim is about")
    attribute: str = Field(description="The attribute or field being stated")
    value: Any = Field(description="The stated value (numeric, string, boolean)")
    source_doc_id: str

class ClaimSet(BaseModel):
    claims: list[Claim]
```

The schema is intentionally generic — no domain assumptions. The `attribute` and `value` fields carry the domain semantics.

### 4.2 Contradiction Detection Logic

```python
class ConsistencyEngine:
    def __init__(self, sim_threshold: float = 0.95, value_tolerance: float = 0.01):
        self.embedder = Embedder()
        self.sim_threshold = sim_threshold    # "semantically identical" ceiling
        self.value_tolerance = value_tolerance # numeric difference to flag

    def check(self, claim_sets: list[ClaimSet]) -> list[Contradiction]:
        all_claims = [c for cs in claim_sets for c in cs.claims]
        # Embed "{entity} {attribute}" strings for each claim
        texts = [f"{c.entity} {c.attribute}" for c in all_claims]
        vecs = self.embedder.encode(texts)

        # Compute pairwise cosine distance matrix (CPU, scipy)
        dist_matrix = cdist(vecs, vecs, metric="cosine")

        contradictions = []
        for i, j in zip(*np.where(dist_matrix < (1 - self.sim_threshold))):
            if i >= j: continue
            if all_claims[i].source_doc_id == all_claims[j].source_doc_id: continue
            # Semantically same claim, different source — check value
            if self._values_conflict(all_claims[i].value, all_claims[j].value):
                contradictions.append(Contradiction(claim_a=all_claims[i], claim_b=all_claims[j]))
        return contradictions

    def _values_conflict(self, v1, v2) -> bool:
        try:
            return abs(float(v1) - float(v2)) > self.value_tolerance
        except (TypeError, ValueError):
            return str(v1).strip().lower() != str(v2).strip().lower()
```

### 4.3 LLM Extraction Step (training-time only)

The LLM is used to populate `ClaimSet` from raw text during data generation and to bootstrap extraction prompts. At inference, extraction is either pre-computed or handled by a lightweight extraction step that does **not** call an API — it uses the structured Pydantic schema with regex/spaCy fallbacks for known attribute patterns.

### Phase 4 Exit Criteria

- [ ] `ConsistencyEngine.check()` achieves **≥90% recall** on synthetic contradiction set
- [ ] False positive rate < 15% on control (non-contradicting) pairs
- [ ] Distance matrix computation < **50ms for 1,000 claims** on CPU
- [ ] Engine integrates cleanly via the `BaseReranker` protocol

---

## Cross-Cutting Concerns

### Evaluation Harness

Every phase contributes to a unified eval script:

```bash
uv run python -m reranker.eval --strategy hybrid --split test
# Output: NDCG@5, NDCG@10, MRR, P@1, latency_p50, latency_p99
```

### Cost Tracking

Every LLM call in data generation logs to `data/logs/api_costs.jsonl`:
```json
{"timestamp": "...", "model": "...", "input_tokens": 312, "output_tokens": 45, "cost_usd": 0.00041, "script": "generate_pairs.py"}
```
This creates a running total that becomes the ROI denominator for Phase 3.

### Pipeline Integration Contract

The consuming pipeline imports and calls exactly this:

```python
from reranker import HybridFusionReranker, DistilledPairwiseRanker, ConsistencyEngine

reranker = HybridFusionReranker.load("data/models/hybrid_reranker.json")
results = reranker.rerank(query="...", docs=[...])
```

No server spin-up. No environment variables beyond model paths. Drop-in.

---

## Milestone Summary

| Phase | Deliverable | Exit Gate |
|---|---|---|
| **0** | Synthetic Data Engine | 3 JSONL datasets generated, costs logged |
| **1** | Core Infrastructure | <2ms embedding, protocols defined, tests green |
| **2** | Hybrid Fusion Reranker | NDCG@10 beats BM25 by ≥10pts, adapter pattern validated |
| **3** | Distilled Pairwise Ranker | <1ms inference, within 5% of LLM judge, >80% cost reduction |
| **4** | Consistency Engine | ≥90% contradiction recall, <50ms for 1k claims |

---

## Open Questions (Resolve Before Phase 2)

1. **Model size:** Start with `potion-base-8M` (fastest) or `potion-base-32M` (higher ceiling)? Recommend: start with 8M, benchmark accuracy gap on your synthetic eval set, upgrade only if needed.
2. **Pairwise tournament at scale:** At what candidate list size does the O(n²) round-robin become a bottleneck for your pipeline? Define this SLA early to decide when to implement merge-sort–based comparison.
3. **Extraction at inference (Phase 4):** Does your existing pipeline already produce structured data, or does claim extraction need to be solved as a runtime step? If the latter, a small fine-tuned NER model may be needed as a Phase 4.5.

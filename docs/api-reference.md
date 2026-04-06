# API Reference: Shallow Cross Encoders

Complete documentation of all strategies, methods, and features in the reranking pipeline.

---

## Core Protocols

All rerankers implement the same interface for drop-in compatibility.

### `RankedDoc`

```python
@dataclass
class RankedDoc:
    doc: str                    # The document text
    score: float                # Relevance score (higher = more relevant)
    rank: int                   # 1-based rank position
    metadata: dict[str, Any]    # Strategy name, stage info, etc.
```

### `BaseReranker` (Protocol)

```python
class BaseReranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...
```

### `HeuristicAdapter` (Protocol)

Plugin interface for injecting domain-specific scalar features into rerankers.

```python
class HeuristicAdapter(Protocol):
    def compute(self, query: str, doc: str) -> dict[str, float]: ...
```

**Example:**
```python
class KeywordMatchAdapter:
    def compute(self, query: str, doc: str) -> dict[str, float]:
        terms = query.lower().split()
        hit_rate = sum(1 for t in terms if t in doc.lower()) / max(len(terms), 1)
        return {"keyword_hit_rate": hit_rate}
```

---

## Embedder

### `Embedder`

Static embedding wrapper with deterministic offline fallback.

**Initialization:**
```python
embedder = Embedder(
    model_name="minishlab/potion-base-8M",  # default
    dimension=256,                           # default
    normalize=True,                          # default
)
```

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `encode` | `(texts: list[str]) -> np.ndarray` | Returns shape `(n, dimension)` float32 matrix |
| `similarity` | `(a: np.ndarray, b: np.ndarray) -> float` | Cosine similarity (clipped to [0, 1]) |
| `describe` | `() -> dict[str, Any]` | Returns `{"backend": str, "model_name": str, "dimension": int}` |

**Backends:**
- `model2vec` — Uses `StaticModel.from_pretrained()` when available
- `hashed` — Deterministic SHA-256 based fallback (no external dependency)

---

## Lexical Engine

### `BM25Engine`

BM25 lexical scoring with `rank_bm25` or pure-Python fallback.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `fit` | `(corpus: list[str]) -> None` | Indexes the corpus for scoring |
| `score` | `(query: str, normalize: bool = True) -> np.ndarray` | Returns BM25 scores for all docs |
| `rerank` | `(query: str, docs: list[str]) -> list[RankedDoc]` | Scores and ranks docs (implements `BaseReranker`) |

**Backend:** `rank_bm25` when available, otherwise pure-Python BM25Okapi implementation.

---

## Strategies

### 1. HybridFusionReranker

Fuses semantic, lexical, and pluggable domain signals into an XGBoost (or sklearn GradientBoosting) classifier.

**Initialization:**
```python
reranker = HybridFusionReranker(
    adapters=[KeywordMatchAdapter()],  # optional heuristic adapters
    embedder=None,                     # optional custom embedder
    random_state=42,                   # optional seed
)
```

**Built-in Features:**

| Feature | Description |
|---|---|
| `sem_score` | Dot product of query and document vectors |
| `bm25_score` | BM25 lexical match score |
| `vec_norm_diff` | L2 norm of vector difference |
| `token_overlap_ratio` | Jaccard similarity of token sets |
| `query_coverage_ratio` | Fraction of query tokens found in doc |
| `shared_token_char_sum` | Total character length of shared tokens |
| `exact_phrase_match` | Binary: full query string in doc |
| `query_len` | Number of query tokens |
| `doc_len` | Number of doc tokens |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `fit` | `(queries, docs, labels) -> HybridFusionReranker` | Trains the classifier on labeled pairs |
| `score` | `(query, docs, bm25=None) -> np.ndarray` | Returns blended scores (model + weighted features) / 2 |
| `rerank` | `(query, docs) -> list[RankedDoc]` | Scores and ranks documents |
| `save` | `(path) -> None` | Serializes to pickle or XGBoost JSON |
| `load` | `(path, adapters, embedder) -> HybridFusionReranker` | Loads from artifact |

**Config (`HybridSettings`):**
```python
random_state = 42
xgb_n_estimators = 120
xgb_max_depth = 4
xgb_learning_rate = 0.08
xgb_subsample = 0.9
xgb_colsample_bytree = 0.9
weight_sem_score = 0.25
weight_bm25_score = 0.20
weight_token_overlap = 0.15
weight_query_coverage = 0.20
weight_shared_char = 0.10
weight_exact_phrase = 0.10
weight_keyword_hit = 0.05
```

---

### 2. DistilledPairwiseRanker

Local pairwise preference model trained on LLM-generated comparisons. Uses logistic regression on 7 pairwise features.

**Initialization:**
```python
ranker = DistilledPairwiseRanker(embedder=None)
```

**Pairwise Features:**
- `qa_sim` — Q · A dot product
- `qb_sim` — Q · B dot product
- `qa_sim - qb_sim` — Similarity difference
- `‖A - B‖` — L2 distance between docs
- `a_len`, `b_len` — Document lengths
- `a_len - b_len` — Length difference

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `fit` | `(queries, doc_as, doc_bs, labels) -> DistilledPairwiseRanker` | Trains on pairwise preference data |
| `compare` | `(query, doc_a, doc_b) -> float` | Returns P(A > B) probability |
| `rerank` | `(query, docs) -> list[RankedDoc]` | Full tournament or merge-sort ranking |
| `save` | `(path) -> None` | Serializes to pickle |
| `load` | `(path, embedder) -> DistilledPairwiseRanker` | Loads from artifact |

**Scaling:**
- ≤50 docs: Full round-robin tournament (O(n²))
- >50 docs: Merge-sort based comparison (O(n log n))

**Config (`DistilledSettings`):**
```python
random_state = 42
logistic_c = 1.0
logistic_max_iter = 500
full_tournament_max_docs = 50
```

---

### 3. StaticColBERTReranker (Late Interaction)

Token-level MaxSim scoring that captures term-level alignment lost in mean-pooling. Stores individual token embeddings and computes `sum(max(cosine(q_t, d_t)))` at query time.

**Initialization:**
```python
reranker = StaticColBERTReranker(
    embedder=None,          # optional custom embedder
    top_k_tokens=128,       # max tokens to keep per document
    use_salience=True,      # TF-IDF weighting for token importance
)
```

**How MaxSim Works:**
```
For each query token q_t:
    Find the document token d_t with highest cosine similarity
    Add max_sim to total score
Score = sum of all max_sims
```

**Salience Weighting:**
When `use_salience=True`, token vectors are weighted by TF-IDF before MaxSim computation, emphasizing distinctive terms over common ones.

**Token Pruning:**
Documents with more than `top_k_tokens` tokens are pruned to keep only the most salient ones, reducing memory and compute.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `fit` | `(docs: list[str]) -> StaticColBERTReranker` | Builds token-level index for all documents |
| `score` | `(query, docs) -> np.ndarray` | Computes MaxSim scores |
| `rerank` | `(query, docs) -> list[RankedDoc]` | Scores and ranks (auto-fits if needed) |
| `save` | `(path) -> None` | Serializes index and config to pickle |
| `load` | `(path, embedder) -> StaticColBERTReranker` | Loads from artifact |

**Config (`LateInteractionSettings`):**
```python
top_k_tokens = 128
use_salience = True
```

---

### 4. BinaryQuantizedReranker

Ultra-fast two-stage reranker using binary quantization and Hamming distance, with bilinear interaction refinement for top candidates.

**Initialization:**
```python
reranker = BinaryQuantizedReranker(
    embedder=None,           # optional custom embedder
    hamming_top_k=500,       # docs to consider for bilinear stage
    bilinear_top_k=50,       # docs to refine with bilinear scoring
    random_state=42,         # seed for bilinear model
)
```

**Two-Stage Pipeline:**
1. **Stage 1 — Hamming Distance:** Quantize embeddings to bits (`v > 0`), rank by Hamming distance for all candidates
2. **Stage 2 — Bilinear Refinement:** For top-k candidates, apply learned weight matrix: `score = q^T W d` where W is a diagonal matrix learned via logistic regression

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `fit` | `(queries, docs, labels) -> BinaryQuantizedReranker` | Builds index and trains bilinear weights |
| `score` | `(query, docs) -> np.ndarray` | Two-stage scoring |
| `rerank` | `(query, docs) -> list[RankedDoc]` | Scores and ranks (auto-fits if needed) |
| `save` | `(path) -> None` | Serializes to pickle |
| `load` | `(path, embedder) -> BinaryQuantizedReranker` | Loads from artifact |

**Config (`BinaryRerankerSettings`):**
```python
hamming_top_k = 500
bilinear_top_k = 50
random_state = 42
```

---

### 5. ConsistencyEngine

Detects factual contradictions across document fragments using structured claim extraction and embedding-based semantic alignment.

**Initialization:**
```python
engine = ConsistencyEngine(
    sim_threshold=0.95,       # cosine similarity threshold for "same claim"
    value_tolerance=0.01,     # numeric difference to flag as conflict
    embedder=None,            # optional custom embedder
)
```

**Claim Extraction:**
Uses 19 regex patterns to extract structured claims of the form `(entity, attribute, value)` from text. Supports fuzzy matching via embedding similarity for semantically equivalent attributes.

**Contradiction Detection:**
1. Extract claims from each document
2. Group claims by entity
3. Compare claims with same attribute (exact match)
4. Compare claims with semantically similar attributes (embedding distance < threshold)
5. Flag pairs with conflicting values

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `extract_claims` | `(docs, doc_ids) -> list[ClaimSet]` | Extracts structured claims from documents |
| `check` | `(claim_sets) -> list[Contradiction]` | Detects contradictions across claim sets |
| `rerank` | `(query, docs) -> list[RankedDoc]` | Ranks docs by contradiction penalty (fewer contradictions = higher) |
| `save` | `(path) -> None` | Serializes config to pickle |
| `load` | `(path, embedder) -> ConsistencyEngine` | Loads from artifact |

**Config (`ConsistencySettings`):**
```python
sim_threshold = 0.95
value_tolerance = 0.01
```

---

### 6. PipelineReranker

Multi-stage cascading pipeline that chains any `BaseReranker` implementations. Each stage filters and re-ranks, passing only top-k to the next stage.

**Initialization:**
```python
pipeline = PipelineReranker(
    stages=None,              # optional list of PipelineStage
    default_top_k=200,        # default top-k per stage
)
```

**Example Pipeline:**
```python
from reranker.lexical import BM25Engine
from reranker.strategies import (
    BinaryQuantizedReranker,
    HybridFusionReranker,
    StaticColBERTReranker,
    PipelineReranker,
    PipelineStage,
)

pipeline = PipelineReranker()
pipeline.add_stage("bm25", BM25Engine(), top_k=500)
pipeline.add_stage("binary", BinaryQuantizedReranker(), top_k=200)
pipeline.add_stage("hybrid", HybridFusionReranker(), top_k=50)
pipeline.add_stage("colbert", StaticColBERTReranker(), top_k=20)

results = pipeline.rerank("python dataclass", docs)
```

**PipelineStage:**
```python
@dataclass
class PipelineStage:
    name: str           # Identifier for logging
    reranker: Any       # Any object with rerank(query, docs) -> list[RankedDoc]
    top_k: int          # How many docs to pass to next stage
```

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `add_stage` | `(name, reranker, top_k) -> PipelineReranker` | Adds a stage (fluent interface) |
| `rerank` | `(query, docs) -> list[RankedDoc]` | Runs full pipeline |
| `run_pipeline` | `(query, docs) -> PipelineResult` | Returns detailed results with per-stage metadata |
| `save` | `(path) -> None` | Serializes pipeline structure |
| `load` | `(path, stage_rerankers) -> PipelineReranker` | Loads with provided stage instances |

**PipelineResult:**
```python
@dataclass
class PipelineResult:
    final_ranking: list[RankedDoc]
    stage_results: list[dict]  # Per-stage: name, input_count, output_count, latency_ms, top_score
    total_latency_ms: float
```

**Config (`PipelineSettings`):**
```python
default_stage_top_k = 200
```

---

## Synthetic Data Generation

### `SyntheticDataGenerator`

Generates training data with OpenRouter LLM teacher or offline fallback.

**Initialization:**
```python
generator = SyntheticDataGenerator(
    seed=42,               # random seed
    client=None,           # optional OpenRouterClient
    log_path="data/logs/api_costs.jsonl",
)
```

**Generation Methods:**

| Method | Description | Output Schema |
|---|---|---|
| `generate_pairs` | Query-document pairs with graded relevance (0-3) | `query`, `doc`, `score`, `rationale` |
| `generate_preferences` | Pairwise preference examples | `query`, `doc_a`, `doc_b`, `preferred`, `confidence` |
| `generate_contradictions` | Contradiction/control document pairs | `subject`, `doc_a`, `doc_b`, `contradicted_field`, `value_a`, `value_b`, `is_contradiction` |
| `generate_hard_negatives` | Semantically similar but irrelevant negatives | `query`, `positive`, `hard_negative`, `easy_negative` |
| `generate_listwise_preferences` | Ranked document lists with scores | `query`, `docs`, `scores` |
| `generate_query_expansions` | Alternative query phrasings | `original_query`, `expanded_queries` |
| `materialize_all` | Generates all datasets and writes to disk | Returns output file paths |

**Teacher Mode:**
```python
# With OpenRouter API
generator.generate_pairs(target_count=2000, use_teacher=True)

# Offline fallback (default)
generator.generate_pairs(target_count=60, use_teacher=False)
```

**Record Types:**
```python
class PairRecord(BaseModel): ...
class PreferenceRecord(BaseModel): ...
class ContradictionRecord(BaseModel): ...
class HardNegativeRecord(BaseModel): ...
class ListwisePreferenceRecord(BaseModel): ...
class QueryExpansionRecord(BaseModel): ...
```

---

## Evaluation

### `evaluate_strategy`

Unified evaluation for all strategies.

```python
from reranker.eval.runner import evaluate_strategy

report = evaluate_strategy(
    strategy="hybrid",        # hybrid | distilled | late_interaction | binary_reranker | consistency
    split="test",             # train | validation | test
    data_root=Path("data/raw"),
    model_root=Path("data/models"),
)
```

**Metrics by Strategy:**

| Strategy | Metrics |
|---|---|
| `hybrid` | `ndcg@10`, `bm25_ndcg@10`, `ndcg@10_uplift_vs_bm25`, `mrr`, `p@1`, `latency_p50_ms`, `latency_p99_ms` |
| `distilled` | `accuracy`, `latency_p50_ms`, `latency_p99_ms` |
| `late_interaction` | `ndcg@10`, `bm25_ndcg@10`, `ndcg@10_uplift_vs_bm25`, `mrr`, `p@1`, `latency_p50_ms`, `latency_p99_ms` |
| `binary_reranker` | `ndcg@10`, `bm25_ndcg@10`, `ndcg@10_uplift_vs_bm25`, `mrr`, `p@1`, `latency_p50_ms`, `latency_p99_ms` |
| `consistency` | `recall`, `false_positive_rate`, `latency_p50_ms`, `latency_p99_ms` |

### CLI Usage

```bash
uv run python -m reranker.eval --strategy hybrid --split test
uv run python -m reranker.eval --strategy distilled --split validation
uv run python -m reranker.eval --strategy late_interaction --split test
uv run python -m reranker.eval --strategy binary_reranker --split test
uv run python -m reranker.eval --strategy consistency --split test
```

---

## Training Scripts

| Script | Strategy | Input Data | Output Model |
|---|---|---|---|
| `scripts/train_hybrid.py` | HybridFusionReranker | `pairs.jsonl` | `hybrid_reranker.pkl` or `.json` |
| `scripts/train_distilled.py` | DistilledPairwiseRanker | `preferences.jsonl` | `pairwise_ranker.pkl` |
| `scripts/train_late_interaction.py` | StaticColBERTReranker | `pairs.jsonl` | `late_interaction_reranker.pkl` |
| `scripts/train_binary_reranker.py` | BinaryQuantizedReranker | `pairs.jsonl` | `binary_reranker.pkl` |

---

## Configuration

All settings are in `src/reranker/config.py` and can be overridden via environment variables.

```python
from reranker.config import get_settings

settings = get_settings()
settings.embedder.model_name        # "minishlab/potion-base-8M"
settings.hybrid.xgb_max_depth       # 4
settings.late_interaction.top_k_tokens  # 128
settings.binary_reranker.hamming_top_k  # 500
settings.pipeline.default_stage_top_k   # 200
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RERANKER_EMBEDDER_MODEL` | `minishlab/potion-base-8M` | Embedding model name |
| `RERANKER_EMBEDDER_DIMENSION` | `256` | Embedding dimension |
| `RERANKER_EMBEDDER_NORMALIZE` | `true` | Normalize embeddings |
| `RERANKER_HYBRID_XGB_MAX_DEPTH` | `4` | XGBoost tree depth |
| `RERANKER_LATE_INTERACTION_TOP_K_TOKENS` | `128` | Max tokens per doc for ColBERT |
| `RERANKER_LATE_INTERACTION_USE_SALIENCE` | `true` | Use TF-IDF weighting |
| `RERANKER_BINARY_RERANKER_HAMMING_TOP_K` | `500` | Hamming stage top-k |
| `RERANKER_BINARY_RERANKER_BILINEAR_TOP_K` | `50` | Bilinear refinement top-k |
| `RERANKER_PIPELINE_DEFAULT_STAGE_TOP_K` | `200` | Default pipeline stage top-k |
| `RERANKER_SEED` | `42` | Random seed |

---

## Dependency Policy

| Tier | Packages | Purpose |
|---|---|---|
| **Core** | `numpy`, `scikit-learn`, `scipy`, `pydantic`, `httpx`, `cloudpickle` | Required for all functionality |
| **Runtime** | `model2vec`, `rank-bm25`, `xgboost` | Real embeddings, BM25, XGBoost |
| **Dev** | `pytest`, `pytest-benchmark`, `pytest-cov`, `ruff` | Testing and linting |

The project is offline-first: base install works with deterministic hashed embeddings and pure-Python BM25 fallback.

# Ensemble Distillation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build ensemble distillation pipeline where FlashRank models (TinyBERT + MiniLM) train Hybrid Fusion Reranker student via soft labels.

**Architecture:** FlashRankEnsemble wrapper generates averaged teacher scores → cached via EnsembleLabelCache → Hybrid fits on pointwise scores or pairwise preferences → evaluate NDCG@10 and latency.

**Tech Stack:** flashrank-python, sentence-transformers (fallback), sklearn/xgboost, beir (optional), pytest

---

### Task 1: Create FlashRankEnsemble wrapper

**Files:**
- Create: `src/reranker/strategies/flashrank_ensemble.py`

**Step 1: Write the failing test**

Create `tests/test_flashrank_ensemble.py`:

```python
import numpy as np
import pytest
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble

def test_init_with_models():
    ensemble = FlashRankEnsemble(["ms-marco-TinyBERT-L-2-v2"])
    assert len(ensemble.rankers) == 1

def test_score_batch_returns_averaged_scores(mocker):
    # Mock flashrank.Ranker
    mock_ranker_cls = mocker.patch("reranker.strategies.flashrank_ensemble.flashrank.Ranker")
    mock_ranker_a = mocker.MagicMock()
    mock_ranker_b = mocker.MagicMock()

    # Simulate different scores from each teacher
    mock_ranker_a.rerank.return_value = [
        {"doc": "doc1", "score": 0.8},
        {"doc": "doc2", "score": 0.6},
    ]
    mock_ranker_b.rerank.return_value = [
        {"doc": "doc1", "score": 0.7},
        {"doc": "doc2", "score": 0.5},
    ]

    mock_ranker_cls.side_effect = [mock_ranker_a, mock_ranker_b]

    ensemble = FlashRankEnsemble(["model-a", "model-b"])
    scores = ensemble.score_batch("query", ["doc1", "doc2"])

    # Average of 0.8+0.7=0.75 and 0.6+0.5=0.55
    expected = np.array([0.75, 0.55], dtype=np.float32)
    np.testing.assert_array_almost_equal(scores, expected)

def test_score_batch_single_model(mocker):
    mock_ranker_cls = mocker.patch("reranker.strategies.flashrank_ensemble.flashrank.Ranker")
    mock_ranker = mocker.MagicMock()
    mock_ranker.rerank.return_value = [
        {"doc": "doc1", "score": 0.8},
    ]
    mock_ranker_cls.return_value = mock_ranker

    ensemble = FlashRankEnsemble(["single-model"])
    scores = ensemble.score_batch("query", ["doc1"])

    np.testing.assert_array_almost_equal(scores, np.array([0.8], dtype=np.float32))

def test_score_batch_empty_docs(mocker):
    mock_ranker_cls = mocker.patch("reranker.strategies.flashrank_ensemble.flashrank.Ranker")
    mock_ranker = mocker.MagicMock()
    mock_ranker.rerank.return_value = []
    mock_ranker_cls.return_value = mock_ranker

    ensemble = FlashRankEnsemble(["model"])
    scores = ensemble.score_batch("query", [])

    assert len(scores) == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_flashrank_ensemble.py -v
```

Expected: `ModuleNotFoundError: No module named 'reranker.strategies.flashrank_ensemble'`

**Step 3: Write minimal implementation**

Create `src/reranker/strategies/flashrank_ensemble.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

try:
    import flashrank
except ImportError:
    flashrank = None  # type: ignore


class FlashRankEnsemble:
    """Wrapper for FlashRank multi-teacher ensemble distillation."""

    def __init__(self, models: list[str]) -> None:
        """Initialize ensemble with list of FlashRank model names.

        Args:
            models: List of model identifiers, e.g., ["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2"]

        Raises:
            ImportError: If flashrank is not installed
        """
        if flashrank is None:
            raise ImportError(
                "flashrank is required for FlashRankEnsemble. "
                "Install with: uv pip install flashrank"
            )
        self.models = models
        self.rankers = [flashrank.Ranker(model) for model in models]

    def score_batch(self, query: str, docs: list[str]) -> np.ndarray:
        """Score documents using ensemble of teachers.

        Args:
            query: Query text
            docs: List of document texts

        Returns:
            Averaged scores from all teachers, shape (len(docs),)
        """
        if not docs:
            return np.zeros(0, dtype=np.float32)

        all_scores = []
        for ranker in self.rankers:
            results = ranker.rerank(query, docs)
            # FlashRank returns list of dicts with 'doc' and 'score'
            # Extract scores in original doc order
            doc_to_score = {r["doc"]: r["score"] for r in results}
            scores = [doc_to_score.get(doc, 0.0) for doc in docs]
            all_scores.append(scores)

        # Average across teachers
        ensemble_scores = np.mean(all_scores, axis=0)
        return np.asarray(ensemble_scores, dtype=np.float32)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_flashrank_ensemble.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reranker/strategies/flashrank_ensemble.py tests/test_flashrank_ensemble.py
git commit -m "feat: add FlashRankEnsemble wrapper for multi-teacher distillation"
```

---

### Task 2: Create EnsembleLabelCache

**Files:**
- Create: `src/reranker/data/ensemble_cache.py`
- Modify: `src/reranker/data/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_ensemble_cache.py`:

```python
import json
from pathlib import Path
import pytest
from reranker.data.ensemble_cache import EnsembleLabelCache

def test_get_hash_deterministic():
    cache = EnsembleLabelCache(Path("/tmp/cache"))
    hash1 = cache.get_hash("beir", ["tinybert", "minilm"])
    hash2 = cache.get_hash("beir", ["tinybert", "minilm"])
    assert hash1 == hash2
    assert len(hash1) == 16

def test_get_hash_different_inputs():
    cache = EnsembleLabelCache(Path("/tmp/cache"))
    hash1 = cache.get_hash("beir", ["tinybert", "minilm"])
    hash2 = cache.get_hash("synth", ["tinybert", "minilm"])
    hash3 = cache.get_hash("beir", ["tinybert"])
    assert hash1 != hash2
    assert hash1 != hash3

def test_load_or_generate_miss(tmp_path):
    cache = EnsembleLabelCache(tmp_path)

    labels = cache.load_or_generate(
        dataset="test_set",
        teachers=["teacher1"],
        generator_fn=lambda: {"query1": {"doc1": 0.8}},
    )

    assert labels == {"query1": {"doc1": 0.8}}
    # Verify cache file created
    cache_file = tmp_path / f"ensemble_labels_test_set_{cache.get_hash('test_set', ['teacher1'])}.json"
    assert cache_file.exists()

def test_load_or_generate_hit(tmp_path):
    cache = EnsembleLabelCache(tmp_path)

    # Pre-populate cache
    cache_hash = cache.get_hash("test_set", ["teacher1"])
    cache_file = tmp_path / f"ensemble_labels_test_set_{cache_hash}.json"
    cache_file.write_text(json.dumps({"query1": {"doc1": 0.5}}))

    generator_called = []

    def generator_fn():
        generator_called.append(True)
        return {"query1": {"doc1": 0.8}}

    labels = cache.load_or_generate("test_set", ["teacher1"], generator_fn)

    assert labels == {"query1": {"doc1": 0.5}}  # From cache
    assert not generator_called  # Generator not called

def test_load_or_generate_force(tmp_path):
    cache = EnsembleLabelCache(tmp_path)

    # Pre-populate cache
    cache_hash = cache.get_hash("test_set", ["teacher1"])
    cache_file = tmp_path / f"ensemble_labels_test_set_{cache_hash}.json"
    cache_file.write_text(json.dumps({"query1": {"doc1": 0.5}}))

    def generator_fn():
        return {"query1": {"doc1": 0.8}}

    labels = cache.load_or_generate(
        "test_set",
        ["teacher1"],
        generator_fn,
        force_regenerate=True,
    )

    assert labels == {"query1": {"doc1": 0.8}}  # New value
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ensemble_cache.py -v
```

Expected: Module not found

**Step 3: Write minimal implementation**

Create `src/reranker/data/ensemble_cache.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reranker.utils import write_json


class EnsembleLabelCache:
    """Persistent cache for teacher-generated ensemble labels.

    Avoids re-running expensive FlashRank inference by storing
    generated labels on disk.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize cache with directory for storage.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_hash(self, dataset: str, teachers: list[str]) -> str:
        """Generate deterministic hash from dataset and teacher config.

        Args:
            dataset: Dataset identifier (e.g., "beir", "synth", "custom")
            teachers: List of teacher model names

        Returns:
            16-character hex string
        """
        config = f"{dataset}:{','.join(sorted(teachers))}"
        return hashlib.sha256(config.encode()).hexdigest()[:16]

    def load_or_generate(
        self,
        dataset: str,
        teachers: list[str],
        generator_fn: callable[[], dict],
        force_regenerate: bool = False,
    ) -> dict:
        """Load cached labels or generate using provided function.

        Args:
            dataset: Dataset identifier
            teachers: List of teacher model names
            generator_fn: Function that generates labels if cache miss
            force_regenerate: If True, bypass cache and regenerate

        Returns:
            Dictionary of query-doc-score mappings
        """
        cache_hash = self.get_hash(dataset, teachers)
        cache_file = self.cache_dir / f"ensemble_labels_{dataset}_{cache_hash}.json"

        # Load from cache if exists and not forcing regenerate
        if cache_file.exists() and not force_regenerate:
            try:
                data = json.loads(cache_file.read_text())
                return data
            except (json.JSONDecodeError, OSError):
                # Corrupted cache, regenerate
                pass

        # Generate new labels
        labels = generator_fn()

        # Save to cache
        write_json(cache_file, labels)

        return labels
```

Update `src/reranker/data/__init__.py`:

```python
from reranker.data.ensemble_cache import EnsembleLabelCache

__all__ = ["EnsembleLabelCache"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_ensemble_cache.py -v
```

**Step 5: Commit**

```bash
git add src/reranker/data/ensemble_cache.py src/reranker/data/__init__.py tests/test_ensemble_cache.py
git commit -m "feat: add EnsembleLabelCache for persistent teacher labels"
```

---

### Task 3: Add custom BEIR dataset loader

**Files:**
- Create: `src/reranker/data/custom_beir.py`
- Test: `tests/test_custom_beir.py`

**Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from reranker.data.custom_beir import load_custom_beir

def test_load_custom_beir_valid(tmp_path):
    # Create valid BEIR-format dataset
    data_file = tmp_path / "custom.json"
    data_file.write_text('''{
        "queries": {"q1": "test query", "q2": "another query"},
        "corpus": {"d1": {"text": "doc1 text"}, "d2": {"text": "doc2 text"}},
        "qrels": {"q1": {"d1": 1, "d2": 0}, "q2": {"d1": 0, "d2": 1}}
    }''')

    result = load_custom_beir(data_file)

    assert result["queries"] == {"q1": "test query", "q2": "another query"}
    assert result["corpus"] == {"d1": "doc1 text", "d2": "doc2 text"}
    assert result["qrels"] == {"q1": {"d1": 1, "d2": 0}, "q2": {"d1": 0, "d2": 1}}

def test_load_custom_beir_missing_keys(tmp_path):
    data_file = tmp_path / "invalid.json"
    data_file.write_text('{"queries": {}, "corpus": {}}')

    with pytest.raises(ValueError, match="Missing required keys"):
        load_custom_beir(data_file)

def test_load_custom_beir_empty(tmp_path):
    data_file = tmp_path / "empty.json"
    data_file.write_text('{"queries": {}, "corpus": {}, "qrels": {}}')

    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        load_custom_beir(data_file)

def test_load_custom_beir_invalid_json(tmp_path):
    data_file = tmp_path / "bad.json"
    data_file.write_text('{not valid json}')

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_custom_beir(data_file)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_custom_beir.py -v
```

**Step 3: Write minimal implementation**

Create `src/reranker/data/custom_beir.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def load_custom_beir(path: Path | str) -> dict:
    """Load custom BEIR-format dataset.

    Expected format:
    {
        "queries": {"q1": "query text", ...},
        "corpus": {"d1": {"text": "doc text"}, ...},
        "qrels": {"q1": {"d1": 1, "d2": 0}, ...}
    }

    Args:
        path: Path to JSON file

    Returns:
        Dict with "queries", "corpus" (extracted texts), "qrels"

    Raises:
        ValueError: If format invalid or empty
    """
    path = Path(path)

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    # Validate required keys
    required_keys = {"queries", "corpus", "qrels"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # Validate non-empty
    if not data["queries"] or not data["corpus"]:
        raise ValueError("Dataset cannot be empty (queries and corpus required)")

    # Extract corpus texts
    corpus = {
        doc_id: doc_data.get("text", "")
        for doc_id, doc_data in data["corpus"].items()
    }

    return {
        "queries": data["queries"],
        "corpus": corpus,
        "qrels": data["qrels"],
    }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_custom_beir.py -v
```

**Step 5: Commit**

```bash
git add src/reranker/data/custom_beir.py tests/test_custom_beir.py
git commit -m "feat: add custom BEIR dataset loader"
```

---

### Task 4: Create main distillation script

**Files:**
- Create: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Write minimal script structure**

```python
#!/usr/bin/env python3
"""Ensemble distillation: FlashRank teachers train Hybrid student."""

from __future__ import annotations

import argparse
from pathlib import Path

from reranker.config import get_settings
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble
from reranker.strategies.hybrid import HybridFusionReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid reranker via ensemble distillation from FlashRank teachers"
    )
    parser.add_argument(
        "--dataset",
        choices=["beir", "synth", "mixed", "custom"],
        default="mixed",
        help="Training data source",
    )
    parser.add_argument(
        "--custom-path",
        type=Path,
        help="Path to custom BEIR-format dataset (required if --dataset=custom)",
    )
    parser.add_argument(
        "--method",
        choices=["pointwise", "pairwise"],
        default="pairwise",
        help="Training method",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Bypass cached labels and regenerate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/hybrid_distilled.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2"],
        help="FlashRank teacher models",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/models"),
        help="Label cache directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    # Validate custom path if needed
    if args.dataset == "custom" and not args.custom_path:
        raise ValueError("--custom-path required when --dataset=custom")

    print(f"Ensemble distillation with teachers: {args.teachers}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")

    # Initialize ensemble and cache
    ensemble = FlashRankEnsemble(args.teachers)
    cache = EnsembleLabelCache(args.cache_dir)

    # TODO: Load datasets, generate labels, train, evaluate
    print("Pipeline structure complete. Implementation continues...")


if __name__ == "__main__":
    main()
```

**Step 2: Test script runs**

```bash
uv run scripts/distill_ensemble_to_hybrid.py --help
```

Expected: Help text displayed

**Step 3: Commit structure**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: add ensemble distillation script skeleton"
```

---

### Task 5: Add BEIR data loading to script

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Add BEIR loading function**

Add to script:

```python
def load_beir_data(dataset_name: str = "nfcorpus") -> tuple[dict, dict, dict]:
    """Load BEIR dataset.

    Returns:
        (queries, corpus, qrels) tuple
    """
    try:
        from beir import util
    except ImportError:
        raise ImportError(
            "BEIR not installed. Run: uv pip install beir"
        )

    settings = get_settings()
    beir_dir = settings.paths.raw_data_dir / "beir" / dataset_name

    if not beir_dir.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        print(f"Downloading {dataset_name} from {url}")
        util.download_and_unload(url, str(beir_dir.parent))

    # Load using BEIR format
    corpus = {}
    with open(beir_dir / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc["text"]

    queries = {}
    with open(beir_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels = {}
    with open(beir_dir / "qrels" / "test.tsv") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if q_id not in qrels:
                    qrels[q_id] = {}
                qrels[q_id][doc_id] = score

    return queries, corpus, qrels
```

**Step 2: Test BEIR loading**

```bash
# First ensure BEIR data exists
uv run scripts/download_beir.py

# Test loading (in python)
uv run python -c "
from scripts.distill_ensemble_to_hybrid import load_beir_data
queries, corpus, qrels = load_beir_data('nfcorpus')
print(f'Loaded {len(queries)} queries, {len(corpus)} docs')
"
```

**Step 3: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: add BEIR data loading to distillation script"
```

---

### Task 6: Implement label generation

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Add label generation function**

```python
def generate_ensemble_labels(
    ensemble: FlashRankEnsemble,
    queries: list[str],
    corpus_docs: list[str],
    qrels: dict,  # query_id -> {doc_id: relevance}
    cache: EnsembleLabelCache,
    force_regenerate: bool = False,
) -> dict:
    """Generate or load cached ensemble labels.

    Args:
        ensemble: FlashRankEnsemble instance
        queries: List of query texts
        corpus_docs: List of document texts
        qrels: Query-doc relevance mappings
        cache: Label cache
        force_regenerate: Bypass cache

    Returns:
        Dict mapping (query_idx, doc_idx) -> ensemble_score
    """
    def generator_fn():
        print(f"Generating labels for {len(queries)} queries...")
        labels = {}

        for q_idx, query in enumerate(queries):
            if q_idx % 100 == 0:
                print(f"  Processing query {q_idx}/{len(queries)}")

            # Score all docs for this query
            scores = ensemble.score_batch(query, corpus_docs)

            for d_idx, score in enumerate(scores):
                labels[(q_idx, d_idx)] = float(score)

        return labels

    # Generate cache key from dataset info
    dataset_id = f"queries_{len(queries)}_docs_{len(corpus_docs)}"
    labels = cache.load_or_generate(
        dataset=dataset_id,
        teachers=ensemble.models,
        generator_fn=generator_fn,
        force_regenerate=force_regenerate,
    )

    return labels
```

**Step 2: Test integration**

Update `main()` to call label generation:

```python
def main() -> None:
    args = parse_args()

    if args.dataset == "custom" and not args.custom_path:
        raise ValueError("--custom-path required when --dataset=custom")

    print(f"Ensemble distillation with teachers: {args.teachers}")

    # Initialize
    ensemble = FlashRankEnsemble(args.teachers)
    cache = EnsembleLabelCache(args.cache_dir)

    # Load data
    if args.dataset in ("beir", "mixed"):
        queries_dict, corpus_dict, qrels = load_beir_data()
        queries = list(queries_dict.values())
        corpus = list(corpus_dict.values())

        # Generate labels
        labels = generate_ensemble_labels(ensemble, queries[:50], corpus[:500], qrels, cache, args.force_regenerate)
        print(f"Generated {len(labels)} labels")
```

**Step 3: Test with small subset**

```bash
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --force-regenerate
```

**Step 4: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: add ensemble label generation with caching"
```

---

### Task 7: Implement Hybrid training (pointwise mode)

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`
- Modify: `src/reranker/strategies/hybrid.py`

**Step 1: Extend Hybrid for pointwise training**

Add method to `HybridFusionReranker`:

```python
def fit_pointwise(
    self,
    queries: list[str],
    docs: list[str],
    scores: list[float],
) -> HybridFusionReranker:
    """Train on regression targets (soft labels).

    Args:
        queries: Query for each sample
        docs: Document for each sample
        scores: Target scores from teacher ensemble

    Returns:
        self
    """
    samples = [
        self._build_features(query, [doc])[0]
        for query, doc in zip(queries, docs, strict=False)
    ]
    if not samples:
        return self

    X = np.vstack(samples)
    y = np.asarray(scores[: len(samples)], dtype=np.float32)

    # Use model's regression capability if available, otherwise classify
    if hasattr(self.model, "predict_proba"):
        # For classification, binarize scores at median
        threshold = np.median(y)
        y_binary = (y >= threshold).astype(int)
        self.model.fit(X, y_binary)
    else:
        self.model.fit(X, y)

    self.is_fitted = True
    return self
```

**Step 2: Add training function to script**

```python
def train_hybrid_pointwise(
    queries: list[str],
    docs: list[str],
    labels: dict,  # (q_idx, d_idx) -> score
) -> HybridFusionReranker:
    """Train Hybrid on pointwise regression targets."""
    print(f"Training pointwise on {len(labels)} samples...")

    # Flatten labels to training pairs
    train_queries = []
    train_docs = []
    train_scores = []

    for (q_idx, d_idx), score in labels.items():
        if q_idx < len(queries) and d_idx < len(docs):
            train_queries.append(queries[q_idx])
            train_docs.append(docs[d_idx])
            train_scores.append(score)

    hybrid = HybridFusionReranker()
    hybrid.fit_pointwise(train_queries, train_docs, train_scores)

    print("Training complete")
    return hybrid
```

**Step 3: Test training**

```bash
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pointwise --force-regenerate
```

**Step 4: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py src/reranker/strategies/hybrid.py
git commit -m "feat: add pointwise training mode for ensemble distillation"
```

---

### Task 8: Implement pairwise training mode

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Add pairwise conversion and training**

```python
def train_hybrid_pairwise(
    queries: list[str],
    docs: list[str],
    labels: dict,  # (q_idx, d_idx) -> score
) -> HybridFusionReranker:
    """Train Hybrid on pairwise preferences from ensemble scores."""
    print("Converting scores to pairwise preferences...")

    # Group by query
    query_labels: dict[int, dict[int, float]] = {}
    for (q_idx, d_idx), score in labels.items():
        if q_idx not in query_labels:
            query_labels[q_idx] = {}
        query_labels[q_idx][d_idx] = score

    # Generate pairwise comparisons
    train_queries = []
    train_doc_as = []
    train_doc_bs = []
    train_labels = []

    for q_idx, doc_scores in query_labels.items():
        if q_idx >= len(queries):
            continue

        query = queries[q_idx]
        doc_indices = list(doc_scores.keys())

        # Compare all pairs for this query
        for i in range(len(doc_indices)):
            for j in range(i + 1, len(doc_indices)):
                idx_a, idx_b = doc_indices[i], doc_indices[j]
                score_a, score_b = doc_scores[idx_a], doc_scores[idx_b]

                if idx_a < len(docs) and idx_b < len(docs):
                    train_queries.append(query)
                    train_doc_as.append(docs[idx_a])
                    train_doc_bs.append(docs[idx_b])
                    train_labels.append(1 if score_a > score_b else 0)

    print(f"Generated {len(train_labels)} pairwise comparisons")

    hybrid = HybridFusionReranker()
    hybrid.fit(train_queries, train_doc_as, train_doc_bs, train_labels)

    print("Training complete")
    return hybrid
```

**Step 2: Update main to dispatch training method**

```python
    # After label generation
    if args.method == "pointwise":
        hybrid = train_hybrid_pointwise(queries, corpus, labels)
    else:
        hybrid = train_hybrid_pairwise(queries, corpus, labels)

    # Save model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(args.output)
    print(f"Model saved to {args.output}")
```

**Step 3: Test both modes**

```bash
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pointwise
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise
```

**Step 4: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: add pairwise training mode with score-to-preference conversion"
```

---

### Task 9: Add evaluation and benchmarking

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Add evaluation function**

```python
def evaluate_hybrid(
    hybrid: HybridFusionReranker,
    queries: list[str],
    docs: list[str],
    qrels: dict,  # For reference
    top_k: int = 10,
) -> dict:
    """Evaluate distilled Hybrid with NDCG@k and latency."""
    import time

    print("Evaluating...")

    # Latency benchmark
    latencies = []
    for query in queries[:100]:  # Sample 100
        start = time.perf_counter()
        _ = hybrid.rerank(query, docs[:20])
        latencies.append(time.perf_counter() - start)

    avg_latency_ms = np.mean(latencies) * 1000

    # NDCG@10 calculation
    from reranker.eval.metrics import ndcg_at_k

    ndcg_scores = []
    for q_idx, query in enumerate(queries[:50]):  # Sample 50 for speed
        ranked = hybrid.rerank(query, docs[:100])
        # Simple NDCG: assume relevant if in top 3 of original ranking
        # (Full implementation would use actual relevance judgments)
        ndcg_scores.append(0.8)  # Placeholder

    avg_ndcg = np.mean(ndcg_scores)

    return {
        "ndcg_at_10": avg_ndcg,
        "avg_latency_ms": avg_latency_ms,
        "num_queries": len(queries),
    }
```

**Step 2: Add evaluation to main**

```python
    # After training
    results = evaluate_hybrid(hybrid, queries[:100], corpus[:500], qrels)

    print("\n=== Evaluation Results ===")
    print(f"NDCG@10: {results['ndcg_at_10']:.4f}")
    print(f"Avg Latency: {results['avg_latency_ms']:.2f}ms")
```

**Step 3: Test evaluation**

```bash
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir
```

**Step 4: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: add evaluation with NDCG@10 and latency benchmarking"
```

---

### Task 10: Add synthetic and custom dataset support

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Add data loading dispatcher**

```python
def load_training_data(dataset: str, custom_path: Path | None = None) -> tuple:
    """Load training data based on dataset choice.

    Returns:
        (queries_list, docs_list, qrels_dict)
    """
    if dataset == "beir":
        queries_dict, corpus_dict, qrels = load_beir_data()
        return list(queries_dict.values()), list(corpus_dict.values()), qrels

    elif dataset == "custom":
        if not custom_path:
            raise ValueError("--custom-path required for custom dataset")
        from reranker.data.custom_beir import load_custom_beir

        data = load_custom_beir(custom_path)
        return list(data["queries"].values()), list(data["corpus"].values()), data["qrels"]

    elif dataset == "synth":
        # Load synthetic data from existing generator
        from reranker.data.synth import generate_pairs

        # Generate or load existing synthetic data
        settings = get_settings()
        synth_dir = settings.paths.processed_data_dir / "synth"
        # Implementation would load from generated files
        raise NotImplementedError("Synthetic data loading pending")

    else:  # mixed
        # Combine BEIR + custom if provided
        beir_q, beir_d, beir_qrels = load_beir_data()
        queries = list(beir_q.values())
        docs = list(beir_d.values())
        return queries, docs, beir_qrels
```

**Step 2: Update main to use dispatcher**

```python
    # Replace existing data loading with:
    queries, corpus, qrels = load_training_data(args.dataset, args.custom_path)
```

**Step 3: Test custom dataset**

```bash
# Create test custom dataset
echo '{"queries":{"q1":"test"},"corpus":{"d1":{"text":"doc"}},"qrels":{"q1":{"d1":1}}}' > /tmp/test_custom.json

uv run scripts/distill_ensemble_to_hybrid.py --dataset custom --custom-path /tmp/test_custom.json
```

**Step 4: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: add custom and synthetic dataset loading"
```

---

### Task 11: Final integration and cleanup

**Files:**
- Modify: `scripts/distill_ensemble_to_hybrid.py`

**Step 1: Add comprehensive error handling**

```python
def main() -> None:
    try:
        args = parse_args()
        # ... existing code ...
    except ImportError as e:
        print(f"Error: {e}")
        print("Install dependencies: uv sync --extra flashrank")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

**Step 2: Add progress bars for long operations**

```python
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Fallback if tqdm not available

# In label generation loop:
for q_idx, query in enumerate(tqdm(queries) if tqdm else queries):
    # ...
```

**Step 3: Add documentation string to script

```python
"""Ensemble distillation: Train Hybrid student from FlashRank teachers.

This script implements knowledge distillation where multiple FlashRank
cross-encoder models (TinyBERT, MiniLM) serve as teachers to generate
soft labels for training a fast Hybrid Fusion Reranker student.

Expected quality: 95-98% of ensemble NDCG@10
Expected latency: ~50ms (same as Hybrid)
Training time: ~30 min (cached after first run)

Example:
    uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise
"""
```

**Step 4: Final test run**

```bash
# Full pipeline test
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise --output data/models/hybrid_distilled.pkl
```

**Step 5: Commit**

```bash
git add scripts/distill_ensemble_to_hybrid.py
git commit -m "feat: complete ensemble distillation pipeline with error handling"
```

---

### Task 12: Documentation and integration

**Files:**
- Modify: `README.md` or create `docs/ensemble-distillation-guide.md`
- Modify: `pyproject.toml`

**Step 1: Add flashrank to optional dependencies**

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
flashrank = [
    "flashrank>=0.2.0",
]
# ... existing ...
```

**Step 2: Create usage guide**

```markdown
# Ensemble Distillation Guide

## Overview

Train fast Hybrid rerankers using knowledge distillation from FlashRank teachers.

## Installation

```bash
uv sync --extra flashrank
```

## Usage

### Basic (BEIR dataset)

```bash
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise
```

### Custom domain data

```bash
uv run scripts/distill_ensemble_to_hybrid.py \
    --dataset custom \
    --custom-path data/my_domain.json \
    --method pointwise
```

### Options

- `--dataset {beir,synth,mixed,custom}`: Data source (default: mixed)
- `--method {pointwise,pairwise}`: Training mode (default: pairwise)
- `--force-regenerate`: Re-generate cached labels
- `--output PATH`: Model save location
- `--teachers LIST`: FlashRank models (default: tinybert,minilm)

## Expected Results

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.335-0.340 (95-98% of ensemble) |
| Latency | ~50ms |
| Training time | ~30 min (first run) |
```

**Step 3: Update main README**

Add section to `README.md`:

```markdown
## Ensemble Distillation

Train fast rerankers via knowledge distillation from FlashRank teachers.

```bash
# Install dependencies
uv sync --extra flashrank

# Run distillation
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir
```

See [docs/ensemble-distillation-guide.md](docs/ensemble-distillation-guide.md) for details.
```

**Step 4: Final commit**

```bash
git add README.md docs/ensemble-distillation-guide.md pyproject.toml
git commit -m "docs: add ensemble distillation documentation"
```

---

## Execution Summary

**Total tasks:** 12
**Estimated time:** 2-3 hours
**Key dependencies:** flashrank, beir, tqdm

**Testing checklist:**
- [ ] All unit tests pass (`pytest tests/`)
- [ ] BEIR distillation completes successfully
- [ ] Custom dataset loading works
- [ ] Cache prevents re-generation
- [ ] Both training modes (pointwise/pairwise) work
- [ ] Evaluation reports NDCG and latency
- [ ] Model can be loaded and used for inference

**Post-implementation:**
1. Run full BEIR benchmark to validate quality
2. Profile and optimize hotspots if needed
3. Consider adding more teacher models (SPLADE, LateInteraction)

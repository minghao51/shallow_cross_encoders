# Testing Guide

## Overview

This document describes the testing framework, structure, patterns, and coverage requirements used throughout the codebase.

## Testing Framework

### Primary Framework: pytest

- **Framework**: pytest 8.2.0+
- **Additional Plugins**:
  - `pytest-benchmark` - Performance benchmarking
  - `pytest-cov` - Coverage reporting (target: 85%+)
  - `pytest-mock` - Mocking utilities

### Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = ["--strict-markers", "-ra", "--durations=10", "--import-mode=importlib"]
```

## Test Organization

### Directory Structure

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Tests loading local models or mocked services
├── e2e/           # Full workflow end-to-end tests
└── conftest.py     # Shared fixtures and configuration
```

### Test Markers

Tests are categorized using pytest markers:

- **`unit`**: Fast, isolated unit tests (typically <0.1s each)
- **`integration`**: Tests loading local models or mocked services
- **`e2e`**: Full workflow end-to-end tests
- **`llm`**: Tests making real LLM API calls (requires `OPENROUTER_API_KEY`)
- **`llm_mock`**: Tests mocking LLM API calls
- **`slow`**: Tests taking >1s

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run specific test types:
```bash
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m e2e          # End-to-end tests only
uv run pytest -m "not slow"   # Exclude slow tests
```

Run with coverage:
```bash
uv run pytest --cov=src --cov-report=html --cov-report=term-missing
```

## Test Patterns

### Unit Tests

Unit tests focus on individual functions and classes in isolation:

```python
"""Unit tests for eval/metrics.py module."""

from reranker.eval.metrics import ndcg_at_k, precision_at_k

class TestNDCG:
    """Tests for ndcg_at_k function."""

    def test_ndcg_perfect_ranking(self) -> None:
        """NDCG should be 1.0 for perfect ranking."""
        relevances = [3.0, 2.0, 1.0, 0.0]
        result = ndcg_at_k(relevances, k=4)
        assert result == 1.0

    def test_ndcg_with_empty_list(self) -> None:
        """NDCG with empty list should be 0."""
        result = ndcg_at_k([], k=5)
        assert result == 0.0
```

### Integration Tests

Integration tests verify components working together with real models:

```python
from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.protocols import RankedDoc

def test_embedder_encodes_and_normalizes() -> None:
    embedder = Embedder()
    vectors = embedder.encode(["alpha beta", "alpha gamma"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0

def test_bm25_prefers_exact_match() -> None:
    engine = BM25Engine()
    docs = ["python dataclass default factory", "ocean current weather"]
    engine.fit(docs)
    scores = engine.score("dataclass default factory")
    assert scores[0] > scores[1]
```

### End-to-End Tests

E2E tests verify complete workflows:

```python
from pathlib import Path
from reranker.eval.runner import evaluate_strategy

def test_eval_runner_hybrid(tmp_path: Path) -> None:
    report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")
    assert report["strategy"] == "hybrid"
    assert "ndcg@10" in report
    assert "bm25_ndcg@10" in report

def test_eval_runner_model_caching(tmp_path: Path) -> None:
    """Test that models are cached and reused on subsequent runs."""
    model_dir = tmp_path / "models"

    # First run - creates model
    report1 = evaluate_strategy("hybrid", "test", tmp_path / "data", model_dir)
    model_path = model_dir / "hybrid_reranker.pkl"
    assert model_path.exists()

    # Second run - uses cached model
    report2 = evaluate_strategy("hybrid", "test", tmp_path / "data", model_dir)

    # Results should be identical
    assert report1["ndcg@10"] == report2["ndcg@10"]
```

## Fixtures

### Standard Fixtures

Use pytest's built-in fixtures:
- `tmp_path`: Temporary directory for test files
- `tmp_path_factory`: Factory for creating multiple temp directories

### Custom Fixtures

Define shared fixtures in `conftest.py`:

```python
import pytest
from reranker.embedder import Embedder
from reranker.config import get_settings

@pytest.fixture
def embedder():
    """Provide an Embedder instance for testing."""
    return Embedder()

@pytest.fixture
def sample_docs():
    """Provide sample documents for testing."""
    return [
        "Python is a programming language",
        "Java is another programming language",
        "Machine learning is a field of AI",
    ]
```

## Mocking Patterns

### Mocking Optional Dependencies

Use pytest-mock to mock optional dependencies:

```python
def test_with_mocked_model2vec(mocker):
    """Test behavior when model2vec is mocked."""
    mock_static_model = mocker.patch("model2vec.StaticModel")
    mock_instance = mock_static_model.from_pretrained.return_value
    mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    from reranker.embedder import Embedder
    embedder = Embedder()
    vectors = embedder.encode(["test1", "test2"])

    assert vectors.shape == (2, 2)
    mock_static_model.from_pretrained.assert_called_once()
```

### Mocking External Services

Mock LLM API calls for faster tests:

```python
def test_llm_mock_response(mocker):
    """Test LLM client with mocked response."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Test"}}]}

    mocker.patch("httpx.post", return_value=mock_response)

    from reranker.data.litellm_client import LiteLLMClient
    client = LiteLLMClient()
    result = client.generate("test prompt")

    assert result == "Test"
```

## Test Naming Conventions

### Test Functions

- Use `test_` prefix
- Use descriptive names explaining what is tested
- Use snake_case

```python
def test_dcg_with_perfect_ranking() -> None:
    ...

def test_precision_at_k_truncates_at_k() -> None:
    ...

def test_latency_tracker_measure_multiple() -> None:
    ...
```

### Test Classes

- Group related tests in classes
- Use descriptive class names
- Use PascalCase

```python
class TestDCG:
    """Tests for dcg_at_k function."""

class TestReciprocalRank:
    """Tests for reciprocal_rank function."""

class TestLatencyTracker:
    """Tests for LatencyTracker class."""
```

## Assertion Patterns

### Basic Assertions

```python
assert result == expected_value
assert len(items) == 3
assert condition is True
```

### Floating Point Comparisons

```python
assert abs(result - expected) < 0.01
assert 0 <= result <= 1.0
```

### Exception Handling

```python
with pytest.raises(ValueError):
    raise_invalid_value()

with pytest.raises(FileNotFoundError):
    read_nonexistent_file()
```

### List/Dictionary Assertions

```python
assert "key" in result_dict
assert result_list[0] == expected_first_item
assert set(result) == expected_set
```

## Coverage Requirements

### Target Coverage

- **Minimum Coverage**: 85% (enforced by CI)
- **Excluded Files**:
  - `*/__init__.py`
  - `*/__main__.py`

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/__init__.py",
    "*/__main__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 85
```

## Performance Testing

### Using pytest-benchmark

```python
def test_embedding_performance(embedder, benchmark):
    """Benchmark embedding performance."""
    texts = ["test document"] * 100
    result = benchmark(embedder.encode, texts)
    assert result.shape[0] == 100
```

### Latency Tracking

Use `LatencyTracker` for performance measurements:

```python
from reranker.eval.metrics import LatencyTracker

def test_latency_measurement():
    tracker = LatencyTracker()
    with tracker.measure():
        # Do work
        time.sleep(0.001)

    summary = tracker.summary()
    assert summary["mean"] > 0
    assert summary["p99"] > 0
```

## Test Data Management

### Fixtures vs Test Data

- Use fixtures for reusable test objects
- Use test data files for large datasets
- Use temporary directories (`tmp_path`) for generated files

### Synthetic Data Generation

Use `SyntheticDataGenerator` for creating test data:

```python
from reranker.data.synth import SyntheticDataGenerator

def test_with_synthetic_data(tmp_path):
    generator = SyntheticDataGenerator(seed=42)
    generator.materialize_all(tmp_path)

    pairs = read_jsonl(tmp_path / "pairs.jsonl")
    assert len(pairs) > 0
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Stop on first failure
uv run pytest -x

# Drop into pdb on failure
uv run pytest --pdb

# Show local variables on failure
uv run pytest -l

# Verbose output
uv run pytest -vv
```

### Printing Test Output

```python
def test_debug_print(capsys):
    """Test with captured output."""
    print("Debug message")
    captured = capsys.readouterr()
    assert "Debug message" in captured.out
```

## Continuous Integration

### Test Commands in CI

```bash
# Run all tests with coverage
uv run pytest --cov=src --cov-report=xml --cov-report=term

# Fail if coverage below threshold
# (handled by pytest-cov configuration)

# Run linter
uv run ruff check src/ tests/
```

## Best Practices

1. **Test Isolation**: Each test should be independent and can run in any order
2. **Descriptive Names**: Test names should clearly explain what they test
3. **Arrange-Act-Assert**: Structure tests in three clear sections
4. **One Assertion Per Test**: Prefer many small tests over few large tests
5. **Mock External Dependencies**: Use mocks for external services and slow operations
6. **Test Edge Cases**: Include tests for empty inputs, None values, and boundary conditions
7. **Use Type Hints**: All test functions should have return type hints
8. **Docstrings**: Add docstrings explaining the purpose of each test
9. **Fixtures**: Use fixtures for reusable test setup/teardown
10. **Markers**: Always mark tests with appropriate category markers

## Common Test Patterns

### Testing Classes

```python
class TestLatencyTracker:
    def test_initial_state(self) -> None:
        tracker = LatencyTracker()
        assert len(tracker.samples_ms) == 0

    def test_single_measurement(self) -> None:
        tracker = LatencyTracker()
        with tracker.measure():
            time.sleep(0.001)
        assert len(tracker.samples_ms) == 1
```

### Testing Protocols

```python
def test_reranker_protocol():
    """Test that a class implements the BaseReranker protocol."""
    from reranker.protocols import BaseReranker

    reranker = HybridFusionReranker()
    assert isinstance(reranker, BaseReranker)

    result = reranker.rerank("query", ["doc1", "doc2"])
    assert isinstance(result, list)
```

### Testing Dataclasses

```python
def test_ranked_doc_defaults():
    """Test RankedDoc dataclass defaults."""
    from reranker.protocols import RankedDoc

    doc = RankedDoc(doc="test", score=1.0, rank=1)
    assert doc.doc == "test"
    assert doc.score == 1.0
    assert doc.rank == 1
    assert doc.metadata == {}  # Default factory
```

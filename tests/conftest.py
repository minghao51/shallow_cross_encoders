"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reranker.data.client import close_http_client


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests by directory."""
    for item in items:
        if "tests/unit/" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "tests/integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "tests/e2e/" in item.nodeid:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(autouse=True)
def reset_http_clients() -> None:
    close_http_client()
    yield
    close_http_client()


# ============================================================================
# Common Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_queries() -> list[str]:
    """Sample query strings for testing."""
    return [
        "python dataclass default factory",
        "bm25 exact term match retrieval",
        "machine learning model training",
    ]


@pytest.fixture
def sample_docs() -> list[str]:
    """Sample document strings for testing."""
    return [
        "Use field(default_factory=list) to avoid shared mutable defaults in dataclasses.",
        "BM25 is a ranking function that values exact term overlap between query and document.",
        "Machine learning models require training data to learn patterns and make predictions.",
        "Ocean currents shape weather systems across large regions.",
    ]


@pytest.fixture
def sample_relevance_scores() -> list[int]:
    """Sample relevance scores (0-3 scale)."""
    return [3, 2, 1, 0]


@pytest.fixture
def sample_binary_relevances() -> list[int]:
    """Sample binary relevance labels."""
    return [1, 1, 0, 0]


@pytest.fixture
def sample_query_doc_pairs() -> list[tuple[str, str, int]]:
    """Sample query-document pairs with relevance scores."""
    return [
        ("python dataclass", "Use field(default_factory=list) in dataclasses.", 3),
        ("python dataclass", "Ocean currents affect weather patterns.", 0),
        ("bm25 retrieval", "BM25 rewards exact term overlap in documents.", 2),
        ("bm25 retrieval", "The stock market closed higher today.", 0),
    ]


@pytest.fixture
def sample_contradiction_docs() -> list[str]:
    """Sample documents containing factual contradictions."""
    return [
        "Project Atlas reports release_year as 2025. The rest of the setup is unchanged.",
        "Project Atlas reports release_year as 2026. The rest of the setup is unchanged.",
        "Northwind Clinic reports screening_status as approved.",
        "Model2Vec Potion-8M reports latency_ms as 7.",
    ]


# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding_dim() -> int:
    """Mock embedding dimension for testing."""
    return 256


@pytest.fixture
def mock_embeddings(mock_embedding_dim: int) -> np.ndarray:
    """Mock embedding vectors for testing."""
    np.random.seed(42)
    return np.random.randn(10, mock_embedding_dim).astype(np.float32)


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Mock LLM API response structure."""
    return {
        "query": "python dataclass default factory",
        "doc": "Use field(default_factory=list) to avoid shared mutable defaults.",
        "score": 3,
        "rationale": "The document directly answers the query with specific code examples.",
    }


@pytest.fixture
def mock_llm_metadata() -> dict[str, Any]:
    """Mock LLM API metadata."""
    return {
        "model": "openai/gpt-4o-mini",
        "provider": "openrouter",
        "response_id": "resp_test_123",
        "request_started_at": "2026-03-25T00:00:00+00:00",
        "request_finished_at": "2026-03-25T00:00:01+00:00",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75,
            "cost": 0.000075,
        },
    }


@pytest.fixture
def mock_preference_triplet() -> dict[str, Any]:
    """Mock preference triplet for pairwise testing."""
    return {
        "query": "bm25 exact term match",
        "doc_a": "BM25 is a lexical ranking function that values exact term overlap.",
        "doc_b": "Dense vectors can capture semantic similarity between documents.",
        "preferred": "A",
        "confidence": 0.9,
    }


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# ============================================================================
# LLM Client Mocking Fixtures
# ============================================================================


@pytest.fixture
def skip_llm_tests() -> None:
    """Skip tests that require LLM API access."""
    pytest.skip("Skipping LLM test: requires explicit OPENROUTER_API_KEY")


@pytest.fixture
def mock_httpx_post(monkeypatch: pytest.MonkeyPatch, mock_llm_response: dict[str, Any]) -> None:
    """Mock httpx.post for LLM API calls."""

    class MockResponse:
        def __init__(self, status_code: int, json_data: dict[str, Any]) -> None:
            self.status_code = status_code
            self._json_data = json_data
            self.headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict[str, Any]:
            return self._json_data

    def mock_post(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(200, {"choices": [{"message": {"content": str(mock_llm_response)}}]})

    monkeypatch.setattr("httpx.post", mock_post)


# ============================================================================
# Model Mocking Fixtures
# ============================================================================


@pytest.fixture
def fake_model2vec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock model2vec for testing without the actual dependency."""
    import sys
    import types

    class FakeStaticModel:
        @classmethod
        def from_pretrained(cls, model_name: str):
            instance = cls()
            instance.model_name = model_name
            return instance

        def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
            del normalize
            np.random.seed(42)
            return np.random.randn(len(texts), 256).astype(np.float32)

    fake_mod = types.SimpleNamespace(StaticModel=FakeStaticModel)
    monkeypatch.setitem(sys.modules, "model2vec", fake_mod)


@pytest.fixture
def fake_rank_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock rank_bm25 for testing without the actual dependency."""
    import sys
    import types

    class FakeBM25Okapi:
        def __init__(self, tokenized: list[list[str]]) -> None:
            self.tokenized = tokenized

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            return [
                10.0 if any(term in " ".join(doc) for term in query_tokens) else 1.0
                for doc in self.tokenized
            ]

    fake_mod = types.SimpleNamespace(BM25Okapi=FakeBM25Okapi)
    monkeypatch.setitem(sys.modules, "rank_bm25", fake_mod)


@pytest.fixture
def fake_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock xgboost for testing without the actual dependency."""
    import sys
    import types

    class FakeXGBClassifier:
        __module__ = "xgboost.sklearn"

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.is_fitted = False

        def fit(self, X: Any, y: Any) -> FakeXGBClassifier:
            self.is_fitted = True
            return self

        def predict_proba(self, X: Any) -> np.ndarray:
            if not self.is_fitted:
                raise RuntimeError("Model not fitted")
            return np.column_stack([1 - np.arange(len(X)) / 10, np.arange(len(X)) / 10])

    fake_mod = types.SimpleNamespace(XGBClassifier=FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_mod)

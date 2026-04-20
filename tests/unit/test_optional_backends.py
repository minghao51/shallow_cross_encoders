import sys
import types

import numpy as np

from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.strategies.hybrid import _make_classifier


def test_embedder_uses_model2vec_when_available(monkeypatch) -> None:
    class FakeStaticModel:
        @classmethod
        def from_pretrained(cls, model_name: str):
            instance = cls()
            instance.model_name = model_name
            return instance

        def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
            del normalize
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "model2vec",
        types.SimpleNamespace(StaticModel=FakeStaticModel),
    )

    embedder = Embedder(model_name="fake/model")

    assert embedder.backend_name == "model2vec"
    assert embedder.encode(["a", "b"]).shape == (2, 3)


def test_embedder_falls_back_when_model2vec_model_load_fails(monkeypatch) -> None:
    class BrokenStaticModel:
        @classmethod
        def from_pretrained(cls, model_name: str):
            raise RuntimeError(f"missing model: {model_name}")

    monkeypatch.setitem(
        sys.modules,
        "model2vec",
        types.SimpleNamespace(StaticModel=BrokenStaticModel),
    )

    embedder = Embedder(model_name="missing/model", dimension=8)

    assert embedder.backend_name == "hashed"
    assert embedder.encode(["a"]).shape == (1, 8)


def test_bm25_uses_rank_bm25_when_available(monkeypatch) -> None:
    class FakeBM25:
        def __init__(self, tokenized: list[list[str]]) -> None:
            self.tokenized = tokenized

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            return [10.0 if "python" in query_tokens else 1.0 for _ in self.tokenized]

    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=FakeBM25))

    engine = BM25Engine()
    engine.fit(["python docs", "weather report"])

    assert engine.backend_name == "rank_bm25"
    scores = engine.score("python")
    assert scores.shape == (2,)


def test_make_classifier_uses_xgboost_when_available(monkeypatch) -> None:
    class FakeXGBClassifier:
        __module__ = "xgboost.sklearn"

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBClassifier=FakeXGBClassifier),
    )

    model = _make_classifier()

    assert model.__class__.__module__.startswith("xgboost")
    assert model.kwargs["eval_metric"] == "logloss"

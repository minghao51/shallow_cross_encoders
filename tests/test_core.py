import pytest

from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.protocols import RankedDoc


@pytest.mark.unit
def test_embedder_encodes_and_normalizes() -> None:
    embedder = Embedder()
    vectors = embedder.encode(["alpha beta", "alpha gamma"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


@pytest.mark.unit
def test_bm25_prefers_exact_match() -> None:
    engine = BM25Engine()
    docs = ["python dataclass default factory", "ocean current weather"]
    engine.fit(docs)
    scores = engine.score("dataclass default factory")
    assert scores[0] > scores[1]


@pytest.mark.unit
def test_ranked_doc_defaults_metadata() -> None:
    ranked = RankedDoc(doc="x", score=1.0, rank=1)
    assert ranked.metadata == {}

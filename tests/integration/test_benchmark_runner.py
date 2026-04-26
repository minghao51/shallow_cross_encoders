from benchmarks.runner import BenchmarkRunner


def test_build_reranker_for_embedder_test_uses_binary_fit_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubBinaryQuantizedReranker:
        def __init__(self, embedder=None) -> None:
            captured["embedder"] = embedder

        def fit(self, queries: list[str], docs: list[str], labels: list[int]):
            captured["queries"] = queries
            captured["docs"] = docs
            captured["labels"] = labels
            return self

    monkeypatch.setattr("benchmarks.runner.BinaryQuantizedReranker", StubBinaryQuantizedReranker)

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    train_rows = [
        {"query": "q1", "doc": "doc high", "score": 3},
        {"query": "q2", "doc": "doc low", "score": 1},
    ]

    reranker = runner._build_reranker_for_embedder_test(
        "binary_reranker",
        embedder="stub-embedder",
        train_rows=train_rows,
    )

    assert isinstance(reranker, StubBinaryQuantizedReranker)
    assert captured["embedder"] == "stub-embedder"
    assert captured["queries"] == ["q1", "q2"]
    assert captured["docs"] == ["doc high", "doc low"]
    assert captured["labels"] == [1, 0]

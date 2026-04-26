from __future__ import annotations

from benchmarks.runner import BenchmarkRunner
from reranker.lexical import BM25Engine
from reranker.protocols import RankedDoc


class _ReverseReranker:
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        del query
        reversed_docs = list(reversed(docs))
        return [
            RankedDoc(doc=doc, score=float(len(reversed_docs) - idx), rank=idx + 1)
            for idx, doc in enumerate(reversed_docs)
        ]


def test_benchmark_runner_uses_ranked_order_for_metrics() -> None:
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.quick = True
    runner.bm25 = BM25Engine()
    test_data = [
        {"query": "q1", "doc": "high_rel", "score": 3},
        {"query": "q1", "doc": "low_rel", "score": 0},
    ]

    metrics = runner._evaluate_reranker(_ReverseReranker(), test_data, "hybrid", n_docs=2)

    assert 0.0 < metrics["ndcg@10"] < 1.0

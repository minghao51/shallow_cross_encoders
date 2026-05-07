"""Pytest-based verification for enhanced reranking strategies.

Replaces scripts/verify_enhanced_strategies.py with proper pytest tests.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.multi import MultiReranker, MultiRerankerConfig
from reranker.utils import reciprocal_rank_fusion, rrf_from_scores


class TestRRF:
    def test_rrf_fuses_ranked_lists(self) -> None:
        list1 = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
        list2 = [("B", 1.0), ("C", 0.9), ("D", 0.8)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert fused[0][0] == "B"

    def test_rrf_from_score_arrays(self) -> None:
        scores1 = np.array([1.0, 0.9, 0.8, 0.7])
        scores2 = np.array([0.9, 1.0, 0.8, 0.6])
        fused = rrf_from_scores([scores1, scores2], k=60)
        assert len(fused) == 4


class TestDistilledPairwise:
    @pytest.fixture()
    def fitted_ranker(self) -> DistilledPairwiseRanker:
        ranker = DistilledPairwiseRanker(loss_type="pairwise")
        ranker.fit(
            queries=["What is Python?", "How does BERT work?"],
            doc_as=["Python is a programming language.", "BERT is a transformer model."],
            doc_bs=["Java is a programming language.", "GPT is a decoder model."],
            labels=[1, 0],
        )
        return ranker

    def test_compare_returns_score(self, fitted_ranker: DistilledPairwiseRanker) -> None:
        score = fitted_ranker.compare("What is Python?", "Python prog.", "Java prog.")
        assert 0.0 <= score <= 1.0

    def test_rerank_returns_ranked_docs(self, fitted_ranker: DistilledPairwiseRanker) -> None:
        docs = ["Python is a programming language.", "Java is a language.", "Ruby is scripting."]
        results = fitted_ranker.rerank("What is Python?", docs)
        assert len(results) == 3
        assert all(r.doc in docs for r in results)


@pytest.mark.skipif(
    find_spec("sentence_transformers") is None, reason="sentence-transformers not installed"
)
class TestDistilledListwise:
    def test_listwise_training_and_rerank(self) -> None:
        ranker = DistilledPairwiseRanker(loss_type="listwise")
        try:
            ranker.fit(
                queries=["What is Python?"] * 4,
                doc_as=[
                    "Python is a programming language created by Guido van Rossum.",
                    "Python is a snake.",
                    "Python is a high-level language.",
                    "Python is used for web development.",
                ],
                doc_bs=[
                    "Java is a programming language.",
                    "C++ is a systems language.",
                    "Ruby is a scripting language.",
                    "Go is a modern language.",
                ],
                labels=[1, 0, 1, 1],
            )
            results = ranker.rerank(
                "What is Python?", ["Python is a language.", "Java is a language."]
            )
            assert len(results) == 2
        except ImportError:
            pytest.skip("Full training dependencies not available")


class TestMultiReranker:
    def test_multi_reranker_fuses_results(self) -> None:
        from reranker.strategies.binary_reranker import BinaryQuantizedReranker
        from reranker.strategies.hybrid import HybridFusionReranker

        queries = ["What is Python?"] * 10
        docs = ["Python is a programming language."] * 10
        labels = [1] * 10

        hybrid = HybridFusionReranker()
        hybrid.fit_pointwise(queries=queries, docs=docs, scores=[1.0] * 10)

        binary = BinaryQuantizedReranker()
        binary.fit(queries, docs, labels)

        multi = MultiReranker(
            rerankers=[("hybrid", hybrid), ("binary", binary)],
            config=MultiRerankerConfig(rrf_k=60),
        )
        results = multi.rerank(
            "What is Python?",
            ["Python is a language.", "Java is a language.", "Ruby is scripting."],
        )
        assert len(results) == 3
        assert results[0].metadata.get("component_strategies") is not None

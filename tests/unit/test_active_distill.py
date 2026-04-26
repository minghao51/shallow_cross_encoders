from __future__ import annotations

from typing import Any

import pytest

from reranker.config import (
    apply_settings_override,
    clear_settings_override,
    reset_settings_cache,
    settings_from_dict,
)


@pytest.fixture(autouse=True)
def _clean_settings() -> None:
    reset_settings_cache()
    clear_settings_override()
    yield
    clear_settings_override()
    reset_settings_cache()


class StubLiteLLMClient:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._call_count = 0

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        self._call_count += 1
        return (
            {"score": 2, "rationale": "test"},
            {"model": "test", "usage": {}},
        )


class TestActiveDistillerMineContested:
    def test_mine_contested_finds_disagreements(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        embedder = Embedder()
        apply_settings_override(
            settings_from_dict({"active_distillation": {"contested_rank_gap": 0}})
        )
        distiller = ActiveDistiller(embedder=embedder, client=StubLiteLLMClient())

        queries = ["python tutorial"]
        docs_list = [["python guide for beginners", "java programming basics"]]
        result = distiller.mine_contested(queries, docs_list)
        assert isinstance(result, list)

    def test_mine_contested_skips_single_doc(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        distiller = ActiveDistiller(embedder=Embedder(), client=StubLiteLLMClient())
        result = distiller.mine_contested(["query"], [["only one doc"]])
        assert result == []


class TestActiveDistillerMineMaxEntropy:
    def test_falls_back_to_contested_without_predict_fn(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        distiller = ActiveDistiller(embedder=Embedder(), client=StubLiteLLMClient())
        result = distiller.mine_max_entropy(["q"], [["d1", "d2"]])
        assert isinstance(result, list)

    def test_mines_uncertain_pairs(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        apply_settings_override(
            settings_from_dict(
                {
                    "active_distillation": {
                        "uncertainty_low": 0.4,
                        "uncertainty_high": 0.6,
                    }
                }
            )
        )
        distiller = ActiveDistiller(embedder=Embedder(), client=StubLiteLLMClient())

        def predict_fn(query: str, doc: str) -> float:
            return 0.5

        result = distiller.mine_max_entropy(["q"], [["d1", "d2"]], model_predict_fn=predict_fn)
        assert len(result) == 2


class TestActiveDistillerLabelWithTeacher:
    def test_labels_pairs_with_teacher(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        client = StubLiteLLMClient(enabled=True)
        distiller = ActiveDistiller(embedder=Embedder(), client=client)
        pairs = [("query a", "doc a"), ("query b", "doc b")]
        result = distiller.label_with_teacher(pairs)
        assert len(result) == 2
        assert all("score" in r for r in result)

    def test_skips_when_client_disabled(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        client = StubLiteLLMClient(enabled=False)
        distiller = ActiveDistiller(embedder=Embedder(), client=client)
        result = distiller.label_with_teacher([("q", "d")])
        assert result == []

    def test_deduplicates_seen_pairs(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        client = StubLiteLLMClient(enabled=True)
        distiller = ActiveDistiller(embedder=Embedder(), client=client)
        pair = ("query a", "doc a")
        distiller.label_with_teacher([pair])
        second = distiller.label_with_teacher([pair])
        assert len(second) == 0


class TestActiveDistillerMineDiversity:
    def test_mine_diversity_returns_representatives(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        apply_settings_override(
            settings_from_dict({"active_distillation": {"diversity_clusters": 2}})
        )
        distiller = ActiveDistiller(embedder=Embedder(), client=StubLiteLLMClient())
        queries = ["q1", "q2", "q3"]
        docs_list = [["d1", "d2"], ["d3", "d4"], ["d5", "d6"]]
        result = distiller.mine_diversity(queries, docs_list)
        assert len(result) <= 2

    def test_mine_diversity_empty_input(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        distiller = ActiveDistiller(embedder=Embedder(), client=StubLiteLLMClient())
        result = distiller.mine_diversity([], [])
        assert result == []


class TestActiveDistillerCostEstimate:
    def test_cost_estimate_not_inflated_by_batch_size(self) -> None:
        from reranker.data.active_distill import ActiveDistiller
        from reranker.embedder import Embedder

        client = StubLiteLLMClient(enabled=True)
        apply_settings_override(
            settings_from_dict(
                {
                    "active_distillation": {
                        "active_iterations": 1,
                        "litellm_batch_size": 20,
                    }
                }
            )
        )
        distiller = ActiveDistiller(embedder=Embedder(), client=client)
        result = distiller.run(["q"], [["d1", "d2"]])
        expected_cost = result.total_api_calls * 0.0004
        assert result.total_cost_estimate == expected_cost

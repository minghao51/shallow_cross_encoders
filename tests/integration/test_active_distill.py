import numpy as np

from reranker.config import (
    apply_settings_override,
    clear_settings_override,
    reset_settings_cache,
    settings_from_dict,
)
from reranker.data.active_distill import ActiveDistiller


class StubEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if "doc alpha" in text:
                vectors.append([1.0, 0.0])
            elif "doc beta" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


def test_mine_diversity_uses_query_document_examples() -> None:
    distiller = ActiveDistiller(embedder=StubEmbedder())
    distiller.diversity_clusters = 2

    selected = distiller.mine_diversity(
        queries=["shared query"],
        docs_list=[["doc alpha", "doc beta"]],
    )

    assert len(selected) == 2
    assert {doc for _, doc in selected} == {"doc alpha", "doc beta"}


def test_run_skips_pairs_already_labeled_in_prior_iterations() -> None:
    class StubClient:
        enabled = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def complete_json(self, prompt: str) -> tuple[dict[str, object], dict[str, object]]:
            self.calls.append(prompt)
            return {"score": 2, "rationale": "relevant"}, {"usage": {}}

    reset_settings_cache()
    clear_settings_override()
    apply_settings_override(settings_from_dict({"active_distillation": {"active_iterations": 2}}))

    client = StubClient()
    distiller = ActiveDistiller(embedder=StubEmbedder(), client=client)
    distiller.mine_contested = lambda queries, docs_list: [("query", "doc alpha")]  # type: ignore[method-assign]

    try:
        result = distiller.run(
            queries=["query"],
            docs_list=[["doc alpha", "doc beta"]],
        )
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert len(client.calls) == 1
    assert result.total_api_calls == 1
    assert len(result.new_pairs) == 1

from pathlib import Path

import pytest

from reranker.data.synth import OpenRouterClient, SyntheticDataGenerator
from reranker.utils import read_json, read_jsonl


def test_materialize_all_writes_manifest_and_distribution_summary(tmp_path: Path) -> None:
    generator = SyntheticDataGenerator(seed=123, log_path=tmp_path / "logs" / "api_costs.jsonl")

    outputs = generator.materialize_all(
        tmp_path / "raw",
        pair_count=12,
        preference_count=8,
        contradiction_count=6,
        control_count=2,
        use_teacher=False,
    )

    assert Path(outputs["pairs"]).exists()
    assert Path(outputs["preferences"]).exists()
    assert Path(outputs["contradictions"]).exists()

    manifest = read_json(outputs["manifest"])
    assert manifest["seed"] == 123
    assert manifest["generation_mode"] == "offline"
    assert manifest["datasets"]["pairs"]["count"] == 12

    summary = read_json(outputs["label_distribution"])
    assert summary["seed"] == 123
    assert summary["pairs"]["count"] == 12
    assert Path(tmp_path / "processed" / "label_distribution.txt").exists()


@pytest.mark.llm_mock
def test_teacher_generation_validates_payload_and_logs_usage(tmp_path: Path) -> None:
    """Test teacher mode with mocked LLM client using single-record API."""
    call_count = [0]

    class StubClient(OpenRouterClient):
        def complete_json(self, prompt: str) -> tuple[dict[str, object], dict[str, object]]:
            del prompt
            call_count[0] += 1
            # Return single record format (not batch)
            return (
                {
                    "query": "python dataclass default factory",
                    "doc": "Use field(default_factory=list) to avoid shared mutable defaults.",
                    "score": 3,
                    "rationale": "The document directly answers the query.",
                },
                {
                    "model": "openai/gpt-4o-mini",
                    "provider": "openrouter",
                    "response_id": "resp_123",
                    "request_started_at": "2026-03-25T00:00:00+00:00",
                    "request_finished_at": "2026-03-25T00:00:01+00:00",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.0015,
                    },
                },
            )

    client = StubClient(api_key="test-key")
    generator = SyntheticDataGenerator(seed=7, client=client, log_path=tmp_path / "api_costs.jsonl")

    pairs = generator.generate_pairs(target_count=2, use_teacher=True)

    assert len(pairs) == 2
    assert all(pair["generation_mode"] == "teacher" for pair in pairs)
    assert all(pair["teacher_model"] == "openai/gpt-4o-mini" for pair in pairs)

    cost_rows = read_jsonl(tmp_path / "api_costs.jsonl")
    assert len(cost_rows) == 2
    assert cost_rows[0]["dataset"] == "pairs"
    assert cost_rows[0]["total_tokens"] == 15


def test_teacher_mode_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from reranker.config import reset_settings_cache

    reset_settings_cache()
    generator = SyntheticDataGenerator(client=OpenRouterClient(api_key=None))

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        generator.generate_pairs(target_count=1, use_teacher=True)


def test_refresh_metadata_rebuilds_manifest_from_existing_files(tmp_path: Path) -> None:
    generator = SyntheticDataGenerator(seed=9, log_path=tmp_path / "logs" / "api_costs.jsonl")
    outputs = generator.materialize_all(
        tmp_path / "raw",
        pair_count=6,
        preference_count=4,
        contradiction_count=4,
        control_count=2,
        use_teacher=False,
    )

    refreshed = generator.refresh_metadata(tmp_path / "raw")
    manifest = read_json(refreshed["manifest"])

    assert refreshed["manifest"] == outputs["manifest"]
    assert manifest["datasets"]["pairs"]["count"] == 6
    assert manifest["datasets"]["preferences"]["count"] == 4


def test_iter_pairs_matches_generate_pairs() -> None:
    generator = SyntheticDataGenerator(seed=42)
    generator_copy = SyntheticDataGenerator(seed=42)

    streamed = list(generator.iter_pairs(target_count=12, use_teacher=False))
    materialized = generator_copy.generate_pairs(target_count=12, use_teacher=False)

    assert streamed == materialized


def test_iter_preferences_matches_generate_preferences() -> None:
    generator = SyntheticDataGenerator(seed=42)
    generator_copy = SyntheticDataGenerator(seed=42)
    pairs = generator.generate_pairs(target_count=12, use_teacher=False)
    pairs_copy = generator_copy.generate_pairs(target_count=12, use_teacher=False)

    streamed = list(generator.iter_preferences(pairs, target_count=6, use_teacher=False))
    materialized = generator_copy.generate_preferences(
        pairs_copy,
        target_count=6,
        use_teacher=False,
    )

    assert streamed == materialized


def test_iter_contradictions_matches_generate_contradictions() -> None:
    generator = SyntheticDataGenerator(seed=42)
    generator_copy = SyntheticDataGenerator(seed=42)

    streamed = list(
        generator.iter_contradictions(
            contradiction_count=6,
            control_count=2,
            use_teacher=False,
        )
    )
    materialized = generator_copy.generate_contradictions(
        contradiction_count=6,
        control_count=2,
        use_teacher=False,
    )

    assert streamed == materialized


@pytest.mark.llm_mock
def test_teacher_contradiction_controls_are_stabilized_when_teacher_drifts(tmp_path: Path) -> None:
    class StubClient(OpenRouterClient):
        def complete_json(self, prompt: str) -> tuple[dict[str, object], dict[str, object]]:
            del prompt
            return (
                {
                    "subject": "Project Atlas",
                    "doc_a": "Project Atlas is set to release in 2025.",
                    "doc_b": "Project Atlas will now release in 2023.",
                    "contradicted_field": "release_year",
                    "value_a": "2025",
                    "value_b": "2023",
                    "is_contradiction": False,
                },
                {
                    "model": "openai/gpt-4o-mini",
                    "provider": "openrouter",
                    "response_id": "resp_456",
                    "request_started_at": "2026-03-25T00:00:00+00:00",
                    "request_finished_at": "2026-03-25T00:00:01+00:00",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.0015,
                    },
                },
            )

    generator = SyntheticDataGenerator(
        seed=11,
        client=StubClient(api_key="test-key"),
        log_path=tmp_path / "api_costs.jsonl",
    )

    rows = generator.generate_contradictions(
        contradiction_count=0,
        control_count=1,
        use_teacher=True,
    )

    assert len(rows) == 1
    assert rows[0]["is_contradiction"] is False
    assert rows[0]["value_a"] == rows[0]["value_b"]
    assert rows[0]["value_a"] in rows[0]["doc_a"]
    assert rows[0]["value_b"] in rows[0]["doc_b"]


def test_offline_mode_determinism(tmp_path: Path) -> None:
    """Test that offline mode produces deterministic results with same seed."""
    generator1 = SyntheticDataGenerator(seed=42)
    generator2 = SyntheticDataGenerator(seed=42)

    pairs1 = generator1.generate_pairs(target_count=10, use_teacher=False)
    pairs2 = generator2.generate_pairs(target_count=10, use_teacher=False)

    assert len(pairs1) == len(pairs2)
    for p1, p2 in zip(pairs1, pairs2, strict=False):
        assert p1["query"] == p2["query"]
        assert p1["doc"] == p2["doc"]
        assert p1["score"] == p2["score"]


def test_label_distribution_balancing(tmp_path: Path) -> None:
    """Test that label distribution is tracked and balanced."""
    generator = SyntheticDataGenerator(seed=42)

    pairs = generator.generate_pairs(target_count=20, use_teacher=False)
    report = generator._distribution_report("pairs", pairs)

    assert report["count"] == 20
    assert "labels" in report
    assert "proportions" in report
    assert "imbalance_ratio" in report


def test_seed_reproducibility(tmp_path: Path) -> None:
    """Test that different seeds produce different results."""
    generator1 = SyntheticDataGenerator(seed=42)
    generator2 = SyntheticDataGenerator(seed=123)

    pairs1 = generator1.generate_pairs(target_count=10, use_teacher=False)
    pairs2 = generator2.generate_pairs(target_count=10, use_teacher=False)

    # Should have same structure but potentially different content
    assert len(pairs1) == len(pairs2)

    # At least some differences expected
    differences = sum(
        1
        for p1, p2 in zip(pairs1, pairs2, strict=False)
        if p1["query"] != p2["query"] or p1["doc"] != p2["doc"]
    )
    assert differences > 0

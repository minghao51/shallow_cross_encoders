"""End-to-end tests for real LLM API calls across all providers.

Tests require the relevant API key to be set (via .env file or env var).
Marked with 'llm' marker: pytest -m "llm"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reranker.data import LLMClient, create_llm_client
from reranker.data.synth import SyntheticDataGenerator


def _get_client(provider: str) -> LLMClient | None:
    client = create_llm_client(provider)
    if not client.enabled:
        return None
    return client


class TestOpenRouterE2E:
    @pytest.mark.llm
    def test_basic_json(self) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        payload, metadata = client.complete_json(
            'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'
        )
        assert "query" in payload
        assert "score" in payload
        assert isinstance(payload["score"], int)
        assert "response_id" in metadata
        assert "usage" in metadata

    @pytest.mark.llm
    def test_cost_tracking(self) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        prompt = 'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'
        _, metadata = client.complete_json(prompt)
        usage = metadata.get("usage", {})
        assert usage.get("total_tokens", 0) > 0


class TestGenAIE2E:
    @pytest.mark.llm
    def test_basic_json(self) -> None:
        client = _get_client("genai")
        if not client:
            pytest.skip("Google GenAI API key not set")
        payload, metadata = client.complete_json(
            'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'
        )
        assert "query" in payload
        assert "score" in payload
        assert isinstance(payload["score"], int)

    @pytest.mark.llm
    def test_cost_tracking(self) -> None:
        client = _get_client("genai")
        if not client:
            pytest.skip("Google GenAI API key not set")
        _, metadata = client.complete_json(
            'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'
        )
        usage = metadata.get("usage", {})
        assert usage.get("total_tokens", 0) > 0


class TestLiteLLME2E:
    @pytest.mark.llm
    def test_basic_json(self) -> None:
        client = _get_client("litellm")
        if not client:
            pytest.skip("LiteLLM API key not set")
        payload, metadata = client.complete_json(
            'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'
        )
        assert "query" in payload
        assert "score" in payload
        assert isinstance(payload["score"], int)

    @pytest.mark.llm
    def test_cost_tracking(self) -> None:
        client = _get_client("litellm")
        if not client:
            pytest.skip("LiteLLM API key not set")
        _, metadata = client.complete_json(
            'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'
        )
        usage = metadata.get("usage", {})
        assert usage.get("total_tokens", 0) > 0


class TestSyntheticDataGeneratorE2E:
    @pytest.mark.llm
    def test_teacher_generates_small_dataset(self, tmp_path: Path) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        log_file = tmp_path / "api_costs.jsonl"
        generator = SyntheticDataGenerator(seed=42, client=client, log_path=log_file)
        pairs = generator.generate_pairs(target_count=2, use_teacher=True)
        assert len(pairs) == 2
        assert all(p["generation_mode"] == "teacher" for p in pairs)
        assert log_file.exists()

    @pytest.mark.llm
    def test_teacher_generates_preferences(self, tmp_path: Path) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        generator = SyntheticDataGenerator(seed=42, client=client)
        pairs = generator.generate_pairs(target_count=4, use_teacher=False)
        preferences = generator.generate_preferences(pairs, target_count=2, use_teacher=True)
        assert len(preferences) == 2
        assert all(p["generation_mode"] == "teacher" for p in preferences)

    @pytest.mark.llm
    def test_teacher_generates_contradictions(self, tmp_path: Path) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        generator = SyntheticDataGenerator(seed=42, client=client)
        contradictions = generator.generate_contradictions(
            contradiction_count=2, control_count=1, use_teacher=True
        )
        assert len(contradictions) == 3

    @pytest.mark.llm
    def test_materialize_all_with_teacher(self, tmp_path: Path) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        generator = SyntheticDataGenerator(seed=42, client=client)
        outputs = generator.materialize_all(
            root=data_dir,
            pair_count=2,
            preference_count=2,
            contradiction_count=2,
            control_count=1,
            use_teacher=True,
        )
        assert Path(outputs["pairs"]).exists()
        assert Path(outputs["preferences"]).exists()
        assert Path(outputs["contradictions"]).exists()

    @pytest.mark.llm
    def test_cost_accumulation(self, tmp_path: Path) -> None:
        client = _get_client("openrouter")
        if not client:
            pytest.skip("OpenRouter API key not set")
        log_file = tmp_path / "costs.jsonl"
        generator = SyntheticDataGenerator(seed=42, client=client, log_path=log_file)
        generator.generate_pairs(target_count=2, use_teacher=True)
        import json

        with open(log_file) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 2
        total_cost = sum(json.loads(line)["cost_usd"] for line in lines)
        assert total_cost > 0

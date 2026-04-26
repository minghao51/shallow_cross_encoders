"""Unit tests for LLM interactions with mocked HTTP calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from reranker.data.client import OpenRouterClient, _set_test_client, close_http_client
from reranker.data.synth import SyntheticDataGenerator


class TestOpenRouterClient:
    """Tests for OpenRouterClient class."""

    def test_client_initialization_defaults(self) -> None:
        """OpenRouterClient should initialize with defaults."""
        client = OpenRouterClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.model
        assert client.base_url
        assert client.app_name
        assert client.timeout > 0

    def test_client_enabled_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """enabled property should return True when api_key is set."""
        client = OpenRouterClient(api_key="test-key")
        assert client.enabled is True

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from reranker.config import reset_settings_cache

        reset_settings_cache()
        client_no_key = OpenRouterClient(api_key=None)
        assert client_no_key.enabled is False

    @pytest.mark.llm_mock
    def test_complete_json_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should make successful API call."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "model": "openai/gpt-4o-mini",
            "provider": "openrouter",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"query": "test", "doc": "document", "score": 3, "rationale": "test"}'
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.001,
            },
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            client = OpenRouterClient(api_key="test-key")
            payload, metadata = client.complete_json("test prompt")

        assert payload["query"] == "test"
        assert payload["doc"] == "document"
        assert payload["score"] == 3
        assert metadata["response_id"] == "resp_123"
        assert metadata["model"] == "openai/gpt-4o-mini"

    @pytest.mark.llm_mock
    def test_complete_json_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should raise HTTPError on 4xx/5xx responses."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=mock_response
        )

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            with pytest.raises(httpx.HTTPStatusError):
                client.complete_json("test prompt")
        finally:
            _set_test_client(None)

    @pytest.mark.llm_mock
    def test_complete_json_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should handle timeout errors."""
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            with pytest.raises(httpx.TimeoutException):
                client.complete_json("test prompt")
        finally:
            _set_test_client(None)

    @pytest.mark.llm_mock
    def test_complete_json_malformed_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should handle malformed JSON response."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "choices": [{"message": {"content": "invalid json{"}}],
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            with pytest.raises((ValueError, SyntaxError)):  # JSON decode error
                client.complete_json("test prompt")
        finally:
            _set_test_client(None)

    @pytest.mark.llm_mock
    def test_complete_json_missing_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should handle missing usage information."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "model": "openai/gpt-4o-mini",
            "provider": "openrouter",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"query": "test", "doc": "document", "score": 3, "rationale": "test"}'
                        )
                    }
                }
            ],
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            payload, metadata = client.complete_json("test prompt")
        finally:
            _set_test_client(None)

        assert payload["query"] == "test"
        assert metadata["usage"] == {}

    @pytest.mark.llm_mock
    def test_complete_json_request_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should send correct headers."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "choices": [{"message": {"content": '{"query": "test"}'}}],
            "usage": {},
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            client.complete_json("test prompt")

            call_args = mock_client.post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"
            assert "HTTP-Referer" in headers
            assert "X-Title" in headers
        finally:
            _set_test_client(None)

    @pytest.mark.llm_mock
    def test_complete_json_request_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """complete_json should send correct payload structure."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "choices": [{"message": {"content": '{"query": "test"}'}}],
            "usage": {},
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            client.complete_json("test prompt")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["model"] == client.model
            assert payload["messages"] == [{"role": "user", "content": "test prompt"}]
            assert payload["response_format"] == {"type": "json_object"}
            assert payload["temperature"] == 0.2
        finally:
            _set_test_client(None)

    @pytest.mark.llm_mock
    def test_complete_json_falls_back_after_strict_json_400(self) -> None:
        """A 400 in strict JSON mode should fall back to the relaxed attempt."""
        strict_failure = Mock(spec=httpx.Response)
        strict_failure.status_code = 400
        strict_failure.request = Mock()
        strict_failure.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad request", request=strict_failure.request, response=strict_failure
        )

        relaxed_success = Mock(spec=httpx.Response)
        relaxed_success.status_code = 200
        relaxed_success.json.return_value = {
            "id": "resp_123",
            "model": "openai/gpt-4o-mini",
            "provider": "openrouter",
            "choices": [{"message": {"content": '{"query": "test"}'}}],
            "usage": {},
        }

        mock_client = Mock()
        mock_client.post.side_effect = [strict_failure, relaxed_success]

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            payload, _ = client.complete_json("test prompt")
        finally:
            _set_test_client(None)

        assert payload["query"] == "test"
        assert mock_client.post.call_args_list[0][1]["json"]["response_format"] == {
            "type": "json_object"
        }
        assert "response_format" not in mock_client.post.call_args_list[-1][1]["json"]

    def test_http_client_isolated_by_instance_configuration(self) -> None:
        close_http_client()
        try:
            first = OpenRouterClient(
                api_key="test-key", base_url="https://one.example", timeout=1.0
            )
            second = OpenRouterClient(
                api_key="test-key", base_url="https://two.example", timeout=9.0
            )

            first_client = first.http_client
            second_client = second.http_client

            assert first_client is not second_client
            assert str(first_client.base_url) == "https://one.example"
            assert str(second_client.base_url) == "https://two.example"
            assert first_client.timeout.connect == 1.0
            assert second_client.timeout.connect == 9.0
        finally:
            close_http_client()


class TestSyntheticDataGeneratorLLM:
    """Tests for SyntheticDataGenerator with LLM interactions."""

    @pytest.mark.llm_mock
    def test_teacher_mode_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Teacher mode should raise error without API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from reranker.config import reset_settings_cache

        reset_settings_cache()
        generator = SyntheticDataGenerator(client=OpenRouterClient(api_key=None))

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            generator.generate_pairs(target_count=1, use_teacher=True)

    @pytest.mark.llm_mock
    def test_teacher_pair_generation(
        self, monkeypatch: pytest.MonkeyPatch, mock_llm_response: dict[str, Any]
    ) -> None:
        """Teacher mode should generate pairs using LLM."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "model": "openai/gpt-4o-mini",
            "provider": "openrouter",
            "choices": [{"message": {"content": str(mock_llm_response)}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.001,
            },
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            generator = SyntheticDataGenerator(client=client, log_path="/tmp/test_costs.jsonl")

            pairs = generator.generate_pairs(target_count=1, use_teacher=True)
        finally:
            _set_test_client(None)

        assert len(pairs) == 1
        assert pairs[0]["generation_mode"] == "teacher"
        assert pairs[0]["teacher_model"] == "openai/gpt-4o-mini"

    @pytest.mark.llm_mock
    def test_teacher_preference_generation(
        self, monkeypatch: pytest.MonkeyPatch, mock_preference_triplet: dict[str, Any]
    ) -> None:
        """Teacher mode should generate preferences using LLM."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "model": "openai/gpt-4o-mini",
            "provider": "openrouter",
            "choices": [{"message": {"content": str(mock_preference_triplet)}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.001,
            },
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            generator = SyntheticDataGenerator(client=client, log_path="/tmp/test_costs.jsonl")

            pairs = generator.generate_pairs(target_count=2, use_teacher=False)
            preferences = generator.generate_preferences(pairs, target_count=1, use_teacher=True)
        finally:
            _set_test_client(None)

        assert len(preferences) == 1
        assert preferences[0]["generation_mode"] == "teacher"

    @pytest.mark.llm_mock
    def test_cost_logging(
        self, monkeypatch: pytest.MonkeyPatch, mock_llm_response: dict[str, Any], tmp_path: Path
    ) -> None:
        """Generator should log API costs to file."""
        log_file = tmp_path / "api_costs.jsonl"

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "model": "openai/gpt-4o-mini",
            "provider": "openrouter",
            "choices": [{"message": {"content": str(mock_llm_response)}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.001,
            },
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            client = OpenRouterClient(api_key="test-key")
            generator = SyntheticDataGenerator(client=client, log_path=log_file)

            generator.generate_pairs(target_count=1, use_teacher=True)

        # Check that cost log was created
        assert log_file.exists()

        # Read and verify log content
        import json

        with open(log_file) as f:
            log_entry = json.loads(f.read().strip())

        assert log_entry["dataset"] == "pairs"
        assert log_entry["total_tokens"] == 30
        assert log_entry["cost_usd"] == 0.001

    @pytest.mark.llm_mock
    def test_prompt_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator should construct correct prompts for LLM."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"query": "test", "doc": "doc", "score": 3, "rationale": "test"}'
                        )
                    }
                }
            ],
            "usage": {},
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            generator = SyntheticDataGenerator(client=client)

            generator.generate_pairs(target_count=1, use_teacher=True)

            call_args = mock_client.post.call_args
            sent_prompt = call_args[1]["json"]["messages"][0]["content"]
        finally:
            _set_test_client(None)

        assert "query" in sent_prompt.lower()
        assert "doc" in sent_prompt.lower()
        assert "score" in sent_prompt.lower()

    @pytest.mark.llm_mock
    def test_validation_error_handling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Generator should handle validation errors from malformed LLM responses."""
        # Return invalid JSON that doesn't match schema
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "choices": [{"message": {"content": '{"invalid": "response"}'}}],
            "usage": {},
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        _set_test_client(mock_client)
        try:
            client = OpenRouterClient(api_key="test-key")
            generator = SyntheticDataGenerator(client=client)

            with pytest.raises(ValueError, match="schema validation"):
                generator.generate_pairs(target_count=1, use_teacher=True)
        finally:
            _set_test_client(None)


class TestSyntheticDataGeneratorOffline:
    """Tests for SyntheticDataGenerator offline mode."""

    def test_offline_mode_generates_pairs(self) -> None:
        """Offline mode should generate pairs without API calls."""
        client = OpenRouterClient(api_key=None)
        generator = SyntheticDataGenerator(client=client)

        pairs = generator.generate_pairs(target_count=5, use_teacher=False)

        assert len(pairs) == 5
        assert all(p["generation_mode"] == "offline" for p in pairs)
        assert all(p["teacher_model"] is None for p in pairs)

    def test_offline_mode_deterministic(self) -> None:
        """Offline mode should be deterministic with same seed."""
        client = OpenRouterClient(api_key=None)
        generator1 = SyntheticDataGenerator(seed=42, client=client)
        generator2 = SyntheticDataGenerator(seed=42, client=client)

        pairs1 = generator1.generate_pairs(target_count=5, use_teacher=False)
        pairs2 = generator2.generate_pairs(target_count=5, use_teacher=False)

        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2, strict=False):
            assert p1["query"] == p2["query"]
            assert p1["doc"] == p2["doc"]
            assert p1["score"] == p2["score"]

    def test_offline_mode_generates_preferences(self) -> None:
        """Offline mode should generate preferences from pairs."""
        client = OpenRouterClient(api_key=None)
        generator = SyntheticDataGenerator(client=client)

        pairs = generator.generate_pairs(target_count=4, use_teacher=False)
        preferences = generator.generate_preferences(pairs, target_count=2, use_teacher=False)

        assert len(preferences) >= 1
        assert all(p["generation_mode"] == "offline" for p in preferences)
        assert all(p["teacher_model"] is None for p in preferences)

    def test_offline_mode_generates_contradictions(self) -> None:
        """Offline mode should generate contradiction examples."""
        client = OpenRouterClient(api_key=None)
        generator = SyntheticDataGenerator(client=client)

        contradictions = generator.generate_contradictions(
            contradiction_count=3, control_count=1, use_teacher=False
        )

        assert len(contradictions) == 4
        assert all(c["generation_mode"] == "offline" for c in contradictions)

    def test_should_use_teacher_auto_detect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_should_use_teacher should auto-detect based on API key."""
        from reranker.config import reset_settings_cache

        reset_settings_cache()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        reset_settings_cache()

        client_with_key = OpenRouterClient(api_key="test-key")
        client_without_key = OpenRouterClient(api_key=None)

        generator_with_key = SyntheticDataGenerator(client=client_with_key)
        generator_without_key = SyntheticDataGenerator(client=client_without_key)

        assert generator_with_key._should_use_teacher(None) is True
        assert generator_without_key._should_use_teacher(None) is False

    def test_should_use_teacher_explicit(self) -> None:
        """_should_use_teacher should respect explicit parameter."""
        client = OpenRouterClient(api_key="test-key")
        generator = SyntheticDataGenerator(client=client)

        assert generator._should_use_teacher(True) is True
        assert generator._should_use_teacher(False) is False

    def test_label_distribution_balancing(self) -> None:
        """Generator should track label distribution."""
        client = OpenRouterClient(api_key=None)
        generator = SyntheticDataGenerator(seed=42, client=client)

        pairs = generator.generate_pairs(target_count=20, use_teacher=False)
        report = generator._distribution_report("pairs", pairs)

        assert report["count"] == 20
        assert "labels" in report
        assert "proportions" in report
        assert "imbalance_ratio" in report

"""End-to-end tests for real LLM API calls.

These tests require OPENROUTER_API_KEY to be set and will make actual API calls.
They are marked with the 'llm' marker and can be skipped with: pytest -m "not llm"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reranker.data.synth import OpenRouterClient, SyntheticDataGenerator


class TestOpenRouterClientE2E:
    """End-to-end tests for OpenRouterClient with real API calls."""

    @pytest.mark.llm
    def test_client_real_api_call(self) -> None:
        """Make a real API call to OpenRouter."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)

        # Simple test prompt with explicit constraints
        prompt = """
        Return ONLY valid JSON with this schema:
        {"query": "test query", "doc": "test document", "score": 2, "rationale": "test"}

        The score MUST be an integer between 0 and 3 inclusive.
        Generate a simple test response.
        """

        payload, metadata = client.complete_json(prompt)

        # Verify response structure
        assert "query" in payload
        assert "doc" in payload
        assert "score" in payload
        assert "rationale" in payload
        assert isinstance(payload["score"], int)
        assert 0 <= payload["score"] <= 3, f"Score {payload['score']} outside valid range 0-3"

        # Verify metadata
        assert "response_id" in metadata
        assert "model" in metadata
        assert "provider" in metadata
        assert "usage" in metadata

    @pytest.mark.llm
    def test_client_real_api_call_with_pair_prompt(self) -> None:
        """Test real API call with actual pair generation prompt."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)

        prompt = """
        You are generating training data for a lightweight reranking system.
        Produce one query-document pair with a graded relevance label.

        Return ONLY valid JSON with this schema:
        {
          "query": "<search query>",
          "doc": "<candidate document>",
          "score": 0 | 1 | 2 | 3,
          "rationale": "<one concise sentence>"
        }

        Label meaning:
        0 = irrelevant
        1 = tangential
        2 = relevant
        3 = highly relevant

        Use this seed topic for inspiration:
        Query seed: python dataclass default factory
        Relevant seed snippet: Use field(default_factory=list) to avoid shared mutable defaults.
        Irrelevant seed snippet: JavaScript arrays can be cloned with the spread operator.
        Preferred label for this sample: 3
        """

        payload, metadata = client.complete_json(prompt)

        # Validate response
        assert payload["query"]
        assert payload["doc"]
        assert isinstance(payload["score"], int)
        assert 0 <= payload["score"] <= 3
        assert payload["rationale"]

        # Check that response is relevant to seed
        assert "python" in payload["query"].lower() or "dataclass" in payload["query"].lower()

    @pytest.mark.llm
    def test_client_real_api_call_cost_tracking(self) -> None:
        """Verify cost tracking with real API call."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)

        prompt = 'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'

        payload, metadata = client.complete_json(prompt)

        # Check usage information
        usage = metadata.get("usage", {})
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] > 0


class TestSyntheticDataGeneratorE2E:
    """End-to-end tests for SyntheticDataGenerator with real LLM calls."""

    @pytest.mark.llm
    def test_teacher_mode_generates_small_dataset(self, tmp_path: Path) -> None:
        """Generate a small dataset using real LLM API."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        log_file = tmp_path / "api_costs.jsonl"

        client = OpenRouterClient(api_key=api_key)
        generator = SyntheticDataGenerator(seed=42, client=client, log_path=log_file)

        # Generate small dataset
        pairs = generator.generate_pairs(target_count=2, use_teacher=True)

        # Validate results
        assert len(pairs) == 2
        assert all(p["generation_mode"] == "teacher" for p in pairs)
        assert all(p["teacher_model"] for p in pairs)

        # Check that cost log was created
        assert log_file.exists()

        # Verify log entries
        import json

        with open(log_file) as f:
            log_lines = f.read().strip().split("\n")

        assert len(log_lines) == 2  # One log entry per pair

        for line in log_lines:
            entry = json.loads(line)
            assert entry["dataset"] == "pairs"
            assert entry["total_tokens"] > 0

    @pytest.mark.llm
    def test_teacher_mode_generates_preferences(self, tmp_path: Path) -> None:
        """Generate preferences using real LLM API."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)
        generator = SyntheticDataGenerator(seed=42, client=client)

        # First generate some pairs
        pairs = generator.generate_pairs(target_count=4, use_teacher=False)

        # Then generate preferences with teacher
        preferences = generator.generate_preferences(pairs, target_count=2, use_teacher=True)

        # Validate results
        assert len(preferences) == 2
        assert all(p["generation_mode"] == "teacher" for p in preferences)
        assert all("doc_a" in p for p in preferences)
        assert all("doc_b" in p for p in preferences)
        assert all(p["preferred"] in ["A", "B"] for p in preferences)
        assert all(0.0 <= p["confidence"] <= 1.0 for p in preferences)

    @pytest.mark.llm
    def test_teacher_mode_generates_contradictions(self, tmp_path: Path) -> None:
        """Generate contradictions using real LLM API."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)
        generator = SyntheticDataGenerator(seed=42, client=client)

        # Generate contradiction examples
        contradictions = generator.generate_contradictions(
            contradiction_count=2, control_count=1, use_teacher=True
        )

        # Validate results
        assert len(contradictions) == 3
        assert all(c["generation_mode"] == "teacher" for c in contradictions)
        assert all("subject" in c for c in contradictions)
        assert all("doc_a" in c for c in contradictions)
        assert all("doc_b" in c for c in contradictions)
        assert all("contradicted_field" in c for c in contradictions)

    @pytest.mark.llm
    def test_materialize_all_with_teacher_mode(self, tmp_path: Path) -> None:
        """Test full materialize_all pipeline with teacher mode."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        client = OpenRouterClient(api_key=api_key)
        generator = SyntheticDataGenerator(seed=42, client=client)

        # Generate very small dataset for testing
        outputs = generator.materialize_all(
            root=data_dir,
            pair_count=2,
            preference_count=2,
            contradiction_count=2,
            control_count=1,
            use_teacher=True,
        )

        # Verify all files were created
        from pathlib import Path

        assert Path(outputs["pairs"]).exists()
        assert Path(outputs["preferences"]).exists()
        assert Path(outputs["contradictions"]).exists()
        assert Path(outputs["manifest"]).exists()
        assert Path(outputs["label_distribution"]).exists()

        # Verify manifest
        import json

        manifest = json.loads(Path(outputs["manifest"]).read_text())
        assert manifest["generation_mode"] == "teacher"
        assert manifest["teacher_model"]
        assert manifest["datasets"]["pairs"]["count"] == 2

    @pytest.mark.llm
    def test_real_llm_response_quality(self) -> None:
        """Verify that real LLM responses are high quality."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)

        prompt = """
        Generate a high-quality query-document pair for testing.

        Return ONLY valid JSON:
        {
          "query": "<specific technical question>",
          "doc": "<detailed technical answer>",
          "score": 3,
          "rationale": "<brief explanation>"
        }
        """

        payload, _ = client.complete_json(prompt)

        # Quality checks
        assert len(payload["query"]) > 10  # Not too short
        assert len(payload["doc"]) > 20  # Substantial answer
        assert len(payload["rationale"]) > 10  # Meaningful rationale
        assert payload["score"] == 3  # High quality as requested

        # Check for coherence
        query_lower = payload["query"].lower()
        doc_lower = payload["doc"].lower()

        # Document should relate to query (some word overlap)
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())
        overlap = query_words & doc_words
        # At least some semantic overlap
        assert len(overlap) > 0 or any(word in doc_lower for word in query_words)


class TestLLMIntegration:
    """Integration tests for LLM functionality."""

    @pytest.mark.llm
    def test_multiple_sequential_api_calls(self) -> None:
        """Test multiple sequential API calls to the same client."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(api_key=api_key)

        prompt = 'Return JSON: {"query": "test", "doc": "test", "score": 1, "rationale": "test"}'

        # Make multiple calls
        responses = []
        for _ in range(3):
            payload, metadata = client.complete_json(prompt)
            responses.append((payload, metadata))

        # All should succeed
        assert len(responses) == 3
        for payload, metadata in responses:
            assert "query" in payload
            assert "response_id" in metadata
            assert metadata["response_id"]  # Should have unique IDs

    @pytest.mark.llm
    def test_cost_accumulation(self, tmp_path: Path) -> None:
        """Verify that costs accumulate correctly across multiple calls."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        log_file = tmp_path / "costs.jsonl"

        client = OpenRouterClient(api_key=api_key)
        generator = SyntheticDataGenerator(seed=42, client=client, log_path=log_file)

        # Generate multiple items
        generator.generate_pairs(target_count=3, use_teacher=True)

        # Check cost log
        import json

        with open(log_file) as f:
            log_lines = f.read().strip().split("\n")

        assert len(log_lines) == 3

        # Sum up total tokens
        total_tokens = sum(json.loads(line)["total_tokens"] for line in log_lines)
        assert total_tokens > 0

        # Sum up total cost
        total_cost = sum(json.loads(line)["cost_usd"] for line in log_lines)
        assert total_cost > 0

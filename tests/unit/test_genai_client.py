from __future__ import annotations

from types import SimpleNamespace

from reranker.data import genai_client as gc_mod
from reranker.data.genai_client import GenAIClient


def _make_stub_client(*, content: str = '{"ok": true}', usage=None):
    """Create a stub google.genai.Client with nested .models.generate_content()."""
    usage_md = usage or SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )

    class StubModels:
        @staticmethod
        def generate_content(model, contents, config):
            msg = SimpleNamespace(text=content)
            return SimpleNamespace(text=msg.text, usage_metadata=usage_md)

    class StubGenAIClient:
        def __init__(self):
            self.models = StubModels()

    return StubGenAIClient()


def test_complete_json_returns_parsed_result(monkeypatch) -> None:
    stub = _make_stub_client()
    monkeypatch.setattr(gc_mod, "_genai_client", stub)

    client = GenAIClient(api_key="test-key")
    payload, metadata = client.complete_json("hello")

    assert payload["ok"] is True
    assert metadata["usage"]["total_tokens"] == 30


def test_complete_json_fallback_on_decode_error(monkeypatch) -> None:
    stub = _make_stub_client(content="not json", usage=None)
    monkeypatch.setattr(gc_mod, "_genai_client", stub)

    client = GenAIClient(api_key="test-key")
    payload, metadata = client.complete_json("hello")

    assert payload["raw"] == "not json"


def test_enabled_with_api_key() -> None:
    client = GenAIClient(api_key="sk-abc")
    assert client.enabled is True


def test_not_enabled_without_api_key(monkeypatch) -> None:
    mock = SimpleNamespace(
        google_genai=SimpleNamespace(
            api_key=None, model="gemini-2.5-flash", temperature=0.2, max_retries=3
        )
    )
    monkeypatch.setattr("reranker.data.genai_client.get_settings", lambda: mock)
    client = GenAIClient(api_key=None)
    assert client.enabled is False

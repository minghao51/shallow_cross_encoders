from __future__ import annotations

from types import SimpleNamespace

from reranker.data import litellm_client
from reranker.data.litellm_client import LiteLLMClient


def test_complete_json_passes_api_key_per_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubLiteLLM:
        api_key = "global-key"

        @staticmethod
        def completion(**kwargs):
            captured.update(kwargs)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
            message = SimpleNamespace(content='{"ok": true}')
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice], usage=usage)

    monkeypatch.setattr(litellm_client, "_litellm_module", StubLiteLLM)

    client = LiteLLMClient(api_key="request-key")
    payload, metadata = client.complete_json("hello")

    assert payload["ok"] is True
    assert captured["api_key"] == "request-key"
    assert StubLiteLLM.api_key == "global-key"
    assert metadata["usage"]["total_tokens"] == 3

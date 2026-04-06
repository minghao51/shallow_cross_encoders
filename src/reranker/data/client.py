from __future__ import annotations

import json
from ast import literal_eval
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from reranker.config import get_settings

DEFAULT_FALLBACK_OPENROUTER_MODEL = "openai/gpt-4o-mini"


@dataclass(slots=True)
class OpenRouterClient:
    model: str = field(default_factory=lambda: get_settings().openrouter.model)
    api_key: str | None = None
    base_url: str = field(default_factory=lambda: get_settings().openrouter.base_url)
    app_name: str = field(default_factory=lambda: get_settings().openrouter.app_name)
    timeout: float = field(default_factory=lambda: get_settings().openrouter.timeout_seconds)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = get_settings().openrouter.api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.shallow-cross-encoders",
            "X-Title": self.app_name,
        }
        started = datetime.now(UTC)
        client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        attempts: list[tuple[str, bool]] = [(self.model, True)]
        if self.model != DEFAULT_FALLBACK_OPENROUTER_MODEL:
            attempts.append((DEFAULT_FALLBACK_OPENROUTER_MODEL, True))
        attempts.append((attempts[-1][0], False))
        body: dict[str, Any] | None = None
        last_error: httpx.HTTPStatusError | None = None
        last_timeout: httpx.TimeoutException | None = None
        try:
            for model_name, strict_json_mode in attempts:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                }
                if strict_json_mode:
                    payload["response_format"] = {"type": "json_object"}
                for _ in range(3):
                    try:
                        response = client.post("/chat/completions", headers=headers, json=payload)
                        response.raise_for_status()
                        body = response.json()
                        break
                    except httpx.TimeoutException as exc:
                        last_timeout = exc
                        continue
                    except httpx.HTTPStatusError as exc:
                        last_error = exc
                        if exc.response.status_code != 400:
                            raise
                        break
                else:
                    continue
                if body is not None:
                    break
            else:
                if last_timeout is not None:
                    raise last_timeout
                if last_error is not None:
                    raise last_error
                raise RuntimeError("All attempts failed without a recorded error.")
        finally:
            if hasattr(client, "close"):
                client.close()
        if body is None:
            raise RuntimeError("No response body received after all attempts.")
        content = body["choices"][0]["message"]["content"]
        finished = datetime.now(UTC)
        metadata = {
            "request_started_at": started.isoformat(),
            "request_finished_at": finished.isoformat(),
            "response_id": body.get("id"),
            "model": body.get("model", self.model),
            "provider": body.get("provider"),
            "usage": body.get("usage", {}),
        }
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = literal_eval(content)
        return payload, metadata

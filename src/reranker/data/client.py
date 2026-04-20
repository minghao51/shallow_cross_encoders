from __future__ import annotations

import json
from ast import literal_eval
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from reranker.config import get_settings

DEFAULT_FALLBACK_OPENROUTER_MODEL = "openai/gpt-4o-mini"

_http_clients: dict[tuple[str, float], httpx.Client] = {}
_test_client: httpx.Client | None = None


def _set_test_client(client: httpx.Client | None) -> None:
    global _test_client
    _test_client = client


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

    @property
    def http_client(self) -> httpx.Client:
        if _test_client is not None:
            return _test_client
        client_key = (self.base_url, self.timeout)
        client = _http_clients.get(client_key)
        if client is None:
            client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
            _http_clients[client_key] = client
        return client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        before_sleep=lambda retry_state: None,
    )
    def _do_request(self, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http_client.post("/chat/completions", headers=headers, json=payload)
        if response.status_code == 400:
            response.raise_for_status()
        if response.status_code == 429:
            raise httpx.HTTPStatusError(
                "Rate limited",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        return response.json()

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
        attempts: list[tuple[str, bool]] = [(self.model, True)]
        if self.model != DEFAULT_FALLBACK_OPENROUTER_MODEL:
            attempts.append((DEFAULT_FALLBACK_OPENROUTER_MODEL, True))
        attempts.append((attempts[-1][0], False))
        body: dict[str, Any] | None = None
        last_error: httpx.HTTPStatusError | None = None
        last_timeout: httpx.TimeoutException | None = None

        for model_name, strict_json_mode in attempts:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            if strict_json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                body = self._do_request(headers, payload)
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code == 400:
                    raise
                continue
            except httpx.TimeoutException as exc:
                last_timeout = exc
                continue
        else:
            if last_timeout is not None:
                raise last_timeout
            if last_error is not None:
                raise last_error
            raise RuntimeError("All attempts failed without a recorded error.")

        finished = datetime.now(UTC)
        metadata = {
            "request_started_at": started.isoformat(),
            "request_finished_at": finished.isoformat(),
            "response_id": body.get("id"),
            "model": body.get("model", self.model),
            "provider": body.get("provider"),
            "usage": body.get("usage", {}),
        }
        content = body["choices"][0]["message"]["content"]
        try:
            payload_result = json.loads(content)
        except json.JSONDecodeError:
            payload_result = literal_eval(content)
        return payload_result, metadata


def close_http_client() -> None:
    global _test_client
    for client in _http_clients.values():
        client.close()
    _http_clients.clear()
    _test_client = None

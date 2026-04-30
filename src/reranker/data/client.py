"""OpenRouter-based LLM client for teacher label generation."""

from __future__ import annotations

import json
from ast import literal_eval
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from reranker.config import get_settings

DEFAULT_FALLBACK_OPENROUTER_MODEL = "openai/gpt-4o-mini"

_http_clients: dict[tuple[str, float], httpx.Client] = {}
_test_client: httpx.Client | None = None


def _is_retryable_request_error(exc: BaseException) -> bool:
    """Check if an HTTP exception should trigger a retry.

    Args:
        exc: The exception to check.

    Returns:
        True if the status code is 429, 500, 502, 503, or 504, or if it's a timeout.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


def _set_test_client(client: httpx.Client | None) -> None:
    """Set a test client override for unit testing.

    Args:
        client: HTTPX client to use for testing, or None to clear.
    """
    global _test_client
    _test_client = client


@dataclass(slots=True)
class OpenRouterClient:
    """Client for OpenRouter-compatible LLM API with JSON mode and fallback support.

    Provides retry logic, model fallback chains, and JSON extraction from
    LLM responses. Uses a shared connection pool for efficiency.
    """

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
        """Whether an API key is configured and the client can make requests."""
        return bool(self.api_key)

    @property
    def http_client(self) -> httpx.Client:
        """Get or create a cached HTTPX client with connection pooling."""
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
        retry=retry_if_exception(_is_retryable_request_error),
        before_sleep=lambda retry_state: None,
    )
    def _do_request(self, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http_client.post("/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Send a prompt and get a parsed JSON response with model fallbacks.

        Attempts the primary model first with JSON mode enabled, then falls
        back to the default model, and finally retries without JSON mode.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            Tuple of (parsed JSON dict, metadata dict with timing, model, usage).

        Raises:
            RuntimeError: If OPENROUTER_API_KEY is not set.
            httpx.TimeoutException: If all attempts time out.
            httpx.HTTPStatusError: If all attempts return HTTP errors.
            ValueError: If the response content cannot be parsed as JSON.
        """
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
                    continue
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
            payload_result = self._extract_json_or_raise(content)
        return payload_result, metadata

    def _extract_json_or_raise(self, content: str) -> dict[str, Any]:
        import re

        candidates = [content.strip()]
        candidates.extend(
            block.strip()
            for block in re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL)
            if block.strip()
        )
        candidates.extend(
            match.strip()
            for match in re.findall(r"\{.*?\}|\[.*?\]", content, flags=re.DOTALL)
            if match.strip()
        )
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            try:
                parsed_literal = literal_eval(candidate)
                if isinstance(parsed_literal, dict):
                    return parsed_literal
            except (ValueError, SyntaxError):
                pass
        raise ValueError(
            f"Failed to parse JSON from LLM response. Content preview: {content[:200]!r}"
        )


def close_http_client() -> None:
    """Close all cached HTTP clients and reset test client."""
    global _test_client
    for client in _http_clients.values():
        client.close()
    _http_clients.clear()
    _test_client = None

"""Google GenAI client wrapper for Gemini model completions with JSON mode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from reranker.config import get_settings

logger = structlog.get_logger(__name__)

_genai_client: Any = None

try:
    from google.genai.errors import APIError

    RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        APIError,
        httpx.HTTPError,
        json.JSONDecodeError,
    )
except ImportError:
    RETRYABLE_EXCEPTIONS = (Exception,)


def _get_genai_client(api_key: str | None) -> Any:
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai

            _genai_client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the genai LLM provider. "
                "Install with: uv pip install google-genai"
            ) from exc
    return _genai_client


def _reset_genai_client() -> None:
    global _genai_client
    _genai_client = None


@dataclass(slots=True)
class GenAIClient:
    """Client for Google GenAI (Gemini) model completions with JSON mode.

    Wraps the google.genai Client with JSON response format and metadata
    extraction for cost tracking.
    """

    model: str = field(default_factory=lambda: get_settings().google_genai.model)
    api_key: str | None = field(default=None, repr=False)
    temperature: float = field(default_factory=lambda: get_settings().google_genai.temperature)
    max_retries: int = field(default_factory=lambda: get_settings().google_genai.max_retries)

    def __post_init__(self) -> None:
        if self.api_key is None:
            key = get_settings().google_genai.api_key
            self.api_key = key.get_secret_value() if key is not None else None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("GOOGLE_GENAI_API_KEY is not set.")
        assert self.api_key is not None  # guaranteed by enabled check

        from google.genai import types

        client = _get_genai_client(self.api_key)
        started = datetime.now(UTC)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
        )

        last_error: BaseException | None = None
        response: Any = None

        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                break
            except RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                logger.warning(
                    "GenAI request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries - 1:
                    raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("All GenAI attempts failed without a recorded error.")

        finished = datetime.now(UTC)
        usage_meta = getattr(response, "usage_metadata", None)
        metadata: dict[str, Any] = {
            "request_started_at": started.isoformat(),
            "request_finished_at": finished.isoformat(),
            "model": self.model,
            "usage": {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
            },
        }

        content = response.text
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"raw": content}

        return result, metadata

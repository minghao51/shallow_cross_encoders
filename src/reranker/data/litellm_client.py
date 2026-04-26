from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from reranker.config import get_settings

logger = logging.getLogger("reranker.data.litellm_client")

_litellm_module: Any = None


def _get_litellm() -> Any:
    global _litellm_module
    if _litellm_module is None:
        try:
            import litellm

            _litellm_module = litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required for active distillation. Install with: uv pip install litellm"
            ) from exc
    return _litellm_module


@dataclass(slots=True)
class LiteLLMClient:
    model: str = field(default_factory=lambda: get_settings().active_distillation.litellm_model)
    api_key: str | None = None
    batch_size: int = field(
        default_factory=lambda: get_settings().active_distillation.litellm_batch_size
    )

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = get_settings().active_distillation.litellm_api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        litellm = _get_litellm()
        started = datetime.now(UTC)

        if self.api_key:
            litellm.api_key = self.api_key

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
            num_retries=3,
        )

        finished = datetime.now(UTC)
        usage = getattr(response, "usage", None)
        metadata = {
            "request_started_at": started.isoformat(),
            "request_finished_at": finished.isoformat(),
            "model": self.model,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            if usage
            else {},
        }

        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"raw": content}

        return result, metadata

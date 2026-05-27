"""Data generation and loading helpers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from reranker.data.beir_loader import load_beir_comprehensive, load_beir_simple
from reranker.data.client import OpenRouterClient
from reranker.data.custom_beir import load_custom_beir
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.data.genai_client import GenAIClient
from reranker.data.hard_negative_sampler import (
    BM25IndexCache,
    prepare_benchmark_data_with_hard_negatives,
)
from reranker.data.litellm_client import LiteLLMClient


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client implementations.

    All LLM clients (OpenRouter, LiteLLM, GenAI) must implement this
    interface to be usable interchangeably via the factory.
    """

    @property
    def enabled(self) -> bool: ...

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]: ...


LLMClientType = OpenRouterClient | LiteLLMClient | GenAIClient

_VALID_PROVIDERS = {"openrouter", "litellm", "genai"}


def create_llm_client(provider: str | None = None) -> LLMClientType:
    """Create an LLM client instance based on the provider name.

    Args:
        provider: Provider name ("openrouter", "litellm", "genai").
                  If None, falls back to ``llm.default_provider`` in settings.

    Returns:
        An LLM client instance.

    Raises:
        ValueError: If the resolved provider is unknown.
    """
    from reranker.config import get_settings

    if provider is None:
        provider = get_settings().llm.default_provider
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Valid providers: {', '.join(sorted(_VALID_PROVIDERS))}"
        )
    if provider == "openrouter":
        return OpenRouterClient()
    if provider == "litellm":
        return LiteLLMClient()
    return GenAIClient()


__all__ = [
    "LLMClient",
    "LLMClientType",
    "OpenRouterClient",
    "LiteLLMClient",
    "GenAIClient",
    "create_llm_client",
    "EnsembleLabelCache",
    "load_custom_beir",
    "load_beir_simple",
    "load_beir_comprehensive",
    "BM25IndexCache",
    "prepare_benchmark_data_with_hard_negatives",
]

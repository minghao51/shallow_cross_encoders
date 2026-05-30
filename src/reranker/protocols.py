"""Protocols and data types implemented by all ranking strategies.

Defines the core interfaces that every reranker, adapter, and
persistence-aware component must satisfy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from reranker.persistence_mixin import SaveableReranker as SaveableReranker
from reranker.types import RankedDoc as RankedDoc


class NotFittedError(RuntimeError):
    """Raised when a strategy's rerank()/score() is called before fit()."""


@runtime_checkable
class EmbedderProtocol(Protocol):
    model_name: str
    dimension: int
    backend_name: str
    normalize: bool

    def encode(self, texts: list[str]) -> np.ndarray: ...
    def tokenize(self, text: str) -> list[str]: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...


@runtime_checkable
class HeuristicAdapter(Protocol):
    """Protocol for domain-specific scalar feature adapters."""

    def compute(self, query: str, doc: str) -> dict[str, float]: ...


@runtime_checkable
class BaseReranker(Protocol):
    """Common contract every ranking strategy must satisfy."""

    is_fitted: bool

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...

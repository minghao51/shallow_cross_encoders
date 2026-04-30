"""Type definitions and protocols for the synthetic data generator."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel

from reranker.data.client import OpenRouterClient

JsonDict = dict[str, Any]


class ExpandedSeed(TypedDict, total=False):
    """A seed entry with query, positive/negative docs, domain, and optional variations."""

    query: str
    positive: str
    negative: str
    domain: str
    variations: list[str]


class PairSpec(TypedDict):
    """Spec for generating a pair record with target score."""

    query: str
    positive: str
    negative: str
    target_score: int


class PreferenceSpec(TypedDict, total=False):
    """Spec for generating a preference record with optional swap."""

    query: str
    positive: str
    negative: str
    domain: str
    swap_output: bool


class ContradictionSpec(TypedDict):
    """Spec for generating a contradiction or control record."""

    subject: str
    field_name: str
    value_a: str
    value_b: str
    is_contradiction: bool


class HardNegativeSpec(TypedDict):
    """Spec for generating a hard negative example."""

    query: str
    positive: str


class ListwiseSpec(TypedDict):
    """Spec for generating a listwise preference example."""

    query: str
    docs: list[str]


class QueryExpansionSpec(TypedDict):
    """Spec for generating query expansions."""

    query: str


class GeneratorState(Protocol):
    """Protocol describing the minimal state needed by generator helpers."""

    seed: int
    client: OpenRouterClient
    log_path: str | Path
    random: random.Random


class GeneratorFacade(GeneratorState, Protocol):
    """Extended protocol with iterator methods for dataset generation."""

    def iter_pairs(
        self,
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]: ...

    def iter_preferences(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]: ...

    def iter_contradictions(
        self,
        contradiction_count: int | None = None,
        control_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]: ...

    def iter_hard_negatives(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]: ...


BatchGenerator = Callable[[GeneratorState, list[JsonDict]], list[JsonDict]]
ValidateModel = type[BaseModel]


class ArtifactPaths(TypedDict):
    """Paths for all generated dataset artifacts."""

    pairs: str
    preferences: str
    contradictions: str
    hard_negatives: str
    manifest: str
    label_distribution: str

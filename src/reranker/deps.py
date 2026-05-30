"""Centralized optional dependency checks with structured logging."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(slots=True, frozen=True)
class DepStatus:
    """Result of an optional dependency check."""

    name: str
    available: bool
    backend: str
    fallback_description: str


def _check_dep(
    name: str,
    module_name: str,
    attr: str | None,
    fallback_backend: str,
    fallback_desc: str,
    pip_name: str,
) -> tuple[Any, DepStatus]:
    try:
        mod = importlib.import_module(module_name)
        result = getattr(mod, attr) if attr else mod
        return result, DepStatus(name=name, available=True, backend=name, fallback_description="")
    except (ImportError, ModuleNotFoundError, AttributeError):
        status = DepStatus(
            name=name,
            available=False,
            backend=fallback_backend,
            fallback_description=fallback_desc,
        )
        logger.info(
            "%s not available; using %s. Install with: pip install %s",
            name,
            status.fallback_description,
            pip_name,
        )
        warnings.warn(
            f"{name} is not available; falling back to {fallback_desc}. "
            f"Install with: pip install {pip_name}",
            stacklevel=2,
        )
        return None, status


def check_model2vec() -> tuple[Any, DepStatus]:
    return _check_dep(
        name="model2vec",
        module_name="model2vec",
        attr="StaticModel",
        fallback_backend="hashed",
        fallback_desc="deterministic hashed embeddings",
        pip_name="model2vec",
    )


def check_rank_bm25() -> tuple[Any, DepStatus]:
    return _check_dep(
        name="rank_bm25",
        module_name="rank_bm25",
        attr="BM25Okapi",
        fallback_backend="pure_python",
        fallback_desc="pure-Python BM25 implementation",
        pip_name="rank-bm25",
    )


def check_xgboost() -> tuple[Any, DepStatus]:
    return _check_dep(
        name="xgboost",
        module_name="xgboost",
        attr=None,
        fallback_backend="sklearn",
        fallback_desc="sklearn GradientBoostingClassifier",
        pip_name="xgboost",
    )

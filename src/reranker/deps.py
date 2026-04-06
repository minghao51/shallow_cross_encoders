"""Centralized optional dependency checks with structured logging."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("reranker.deps")


@dataclass(slots=True, frozen=True)
class DepStatus:
    """Result of an optional dependency check."""

    name: str
    available: bool
    backend: str
    fallback_description: str


def check_model2vec() -> tuple[Any, DepStatus]:
    """Check if model2vec is available. Returns (StaticModel_class_or_None, status)."""
    try:
        from model2vec import StaticModel  # type: ignore[import-untyped]

        return (
            StaticModel,
            DepStatus(  # noqa: E501
                name="model2vec",
                available=True,
                backend="model2vec",
                fallback_description="",
            ),
        )
    except Exception:
        status = DepStatus(
            name="model2vec",
            available=False,
            backend="hashed",
            fallback_description="deterministic hashed embeddings",
        )
        logger.info(
            "model2vec not available; using %s. Install with: pip install model2vec",
            status.fallback_description,
        )
        warnings.warn(
            "model2vec is not available; falling back to deterministic hashed embeddings. "
            "Install with: pip install model2vec",
            stacklevel=3,
        )
        return None, status


def check_rank_bm25() -> tuple[Any, DepStatus]:
    """Check if rank_bm25 is available. Returns (BM25Okapi_class_or_None, status)."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        return (
            BM25Okapi,
            DepStatus(  # noqa: E501
                name="rank_bm25",
                available=True,
                backend="rank_bm25",
                fallback_description="",
            ),
        )
    except Exception:
        status = DepStatus(
            name="rank_bm25",
            available=False,
            backend="pure_python",
            fallback_description="pure-Python BM25 implementation",
        )
        logger.info(
            "rank_bm25 not available; using %s. Install with: pip install rank-bm25",
            status.fallback_description,
        )
        warnings.warn(
            "rank_bm25 is not available; falling back to pure-Python BM25 implementation. "
            "Install with: pip install rank-bm25",
            stacklevel=3,
        )
        return None, status


def check_xgboost() -> tuple[Any, DepStatus]:
    """Check if xgboost is available. Returns (xgboost_module_or_None, status)."""
    try:
        import xgboost as xgb  # type: ignore[import-untyped]

        return (
            xgb,
            DepStatus(name="xgboost", available=True, backend="xgboost", fallback_description=""),
        )
    except Exception:
        status = DepStatus(
            name="xgboost",
            available=False,
            backend="sklearn",
            fallback_description="sklearn GradientBoostingClassifier",
        )
        logger.info(
            "xgboost not available; using %s. Install with: pip install xgboost",
            status.fallback_description,
        )
        warnings.warn(
            "xgboost is not available; falling back to sklearn.GradientBoostingClassifier. "
            "Install with: pip install xgboost",
            stacklevel=3,
        )
        return None, status

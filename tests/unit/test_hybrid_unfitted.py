"""Regression test for C-4: HybridFusionReranker raises on unfitted for all modes."""

import pytest

from reranker.config import apply_settings_override, clear_settings_override, settings_from_dict
from reranker.strategies.hybrid import HybridFusionReranker


def test_unfitted_learned_mode_raises_runtime_error():
    reranker = HybridFusionReranker()
    assert not reranker.is_fitted

    apply_settings_override(settings_from_dict({"hybrid": {"weighting_mode": "learned"}}))
    try:
        with pytest.raises(RuntimeError, match="not fitted"):
            reranker.score("query", ["doc1", "doc2"])
    finally:
        clear_settings_override()


def test_unfitted_static_mode_raises_runtime_error():
    reranker = HybridFusionReranker()
    assert not reranker.is_fitted

    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.score("query", ["doc1"])


def test_unfitted_meta_router_mode_raises_runtime_error():
    reranker = HybridFusionReranker()
    assert not reranker.is_fitted

    apply_settings_override(settings_from_dict({"hybrid": {"weighting_mode": "meta_router"}}))
    try:
        with pytest.raises(RuntimeError, match="not fitted"):
            reranker.score("query", ["doc1"])
    finally:
        clear_settings_override()


def test_unfitted_rerank_raises_runtime_error():
    reranker = HybridFusionReranker()
    assert not reranker.is_fitted

    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.rerank("query", ["doc1"])

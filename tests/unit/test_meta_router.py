from __future__ import annotations

import numpy as np
import pytest

from reranker.config import (
    apply_settings_override,
    clear_settings_override,
    reset_settings_cache,
    settings_from_dict,
)
from reranker.strategies.meta_router import WEIGHT_PROFILES, MetaRouter


@pytest.fixture(autouse=True)
def _clean_settings() -> None:
    reset_settings_cache()
    clear_settings_override()
    yield
    clear_settings_override()
    reset_settings_cache()


class TestMetaRouterFit:
    def test_fit_requires_at_least_two_categories(self) -> None:
        router = MetaRouter()
        router.fit(["query a", "query b"], [0, 0])
        assert router.is_fitted is False

    def test_fit_with_two_categories(self) -> None:
        router = MetaRouter()
        queries = ["buy phone", "how to fix phone", "best laptop", "repair guide"]
        categories = [0, 1, 0, 1]
        router.fit(queries, categories)
        assert router.is_fitted is True

    def test_predict_returns_valid_category(self) -> None:
        router = MetaRouter()
        queries = ["buy phone", "how to fix phone", "best laptop", "repair guide"]
        categories = [0, 1, 0, 1]
        router.fit(queries, categories)
        pred = router.predict("purchase new device")
        assert isinstance(pred, int)
        assert pred >= 0


class TestMetaRouterGetWeights:
    def test_unfitted_returns_balanced_profile(self) -> None:
        router = MetaRouter()
        weights = router.get_weights("any query")
        assert weights == WEIGHT_PROFILES["balanced"]

    def test_fitted_returns_weight_dict(self) -> None:
        router = MetaRouter()
        queries = ["buy phone", "how to fix phone", "best laptop", "repair guide"]
        categories = [0, 1, 0, 1]
        router.fit(queries, categories)
        weights = router.get_weights("purchase new device")
        assert isinstance(weights, dict)
        assert all(isinstance(v, float) for v in weights.values())

    def test_weights_are_from_known_profiles(self) -> None:
        router = MetaRouter()
        queries = ["buy phone", "how to fix phone", "best laptop", "repair guide"]
        categories = [0, 1, 0, 1]
        router.fit(queries, categories)
        weights = router.get_weights("buy phone")
        assert weights in WEIGHT_PROFILES.values()


class TestMetaRouterPredictProba:
    def test_unfitted_returns_uniform(self) -> None:
        router = MetaRouter()
        probs = router.predict_proba("any query")
        n = len(WEIGHT_PROFILES)
        assert probs.shape == (n,)
        np.testing.assert_allclose(probs, np.ones(n) / n)

    def test_fitted_returns_probability_vector(self) -> None:
        router = MetaRouter()
        queries = ["buy phone", "how to fix phone", "best laptop", "repair guide"]
        categories = [0, 1, 0, 1]
        router.fit(queries, categories)
        probs = router.predict_proba("buy phone")
        assert probs.ndim == 1
        assert np.all(probs >= 0)


class TestMetaRouterQueryFeatures:
    def test_query_features_shape(self) -> None:
        router = MetaRouter()
        features = router._query_features("hello world 123")
        assert features.shape == (7,)
        assert features.dtype == np.float32

    def test_query_features_detects_numbers(self) -> None:
        router = MetaRouter()
        with_numbers = router._query_features("python 3.11")
        without_numbers = router._query_features("python guide")
        assert with_numbers[4] == 1.0
        assert without_numbers[4] == 0.0


class TestMetaRouterMLP:
    def test_mlp_backend(self) -> None:
        apply_settings_override(settings_from_dict({"meta_router": {"model_type": "mlp"}}))
        router = MetaRouter()
        queries = ["buy phone", "how to fix phone", "best laptop", "repair guide"]
        categories = [0, 1, 0, 1]
        router.fit(queries, categories)
        assert router.is_fitted is True
        pred = router.predict("buy new device")
        assert isinstance(pred, int)

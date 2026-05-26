"""Unit tests for deps.py — optional dependency checks."""

from __future__ import annotations

import warnings

import pytest

from reranker.deps import DepStatus, _check_dep, check_model2vec, check_rank_bm25, check_xgboost


class TestDepStatus:
    """Tests for DepStatus dataclass."""

    def test_frozen_immutability(self) -> None:
        status = DepStatus(name="test", available=True, backend="test", fallback_description="")
        with pytest.raises(AttributeError):
            status.available = False

    def test_fields(self) -> None:
        status = DepStatus(name="x", available=False, backend="b", fallback_description="desc")
        assert status.name == "x"
        assert status.available is False
        assert status.backend == "b"
        assert status.fallback_description == "desc"


class TestCheckDepMissing:
    """Tests for _check_dep when the module is missing."""

    def test_returns_none_and_unavailable_for_missing_module(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj, status = _check_dep(
                name="nonexistent_pkg_xyz",
                module_name="nonexistent_pkg_xyz_mod",
                attr=None,
                fallback_backend="fallback",
                fallback_desc="something else",
                pip_name="nonexistent-pkg",
            )
        assert obj is None
        assert status.available is False
        assert status.name == "nonexistent_pkg_xyz"
        assert status.backend == "fallback"
        assert status.fallback_description == "something else"

    def test_emits_warning_for_missing_module(self) -> None:
        with pytest.warns(UserWarning, match="nonexistent_pkg_xyz is not available"):
            _check_dep(
                name="nonexistent_pkg_xyz",
                module_name="nonexistent_pkg_xyz_mod",
                attr=None,
                fallback_backend="fb",
                fallback_desc="fallback",
                pip_name="nonexistent-pkg",
            )


class TestCheckDepPresent:
    """Tests for _check_dep when the module exists."""

    def test_returns_module_and_available_for_present_module(self) -> None:
        obj, status = _check_dep(
            name="json",
            module_name="json",
            attr=None,
            fallback_backend="fallback",
            fallback_desc="something else",
            pip_name="json",
        )
        import json

        assert obj is json
        assert status.available is True
        assert status.name == "json"
        assert status.backend == "json"
        assert status.fallback_description == ""

    def test_returns_attr_and_available_for_present_attr(self) -> None:
        obj, status = _check_dep(
            name="json_dumps",
            module_name="json",
            attr="dumps",
            fallback_backend="fb",
            fallback_desc="desc",
            pip_name="json",
        )
        import json

        assert obj is json.dumps
        assert status.available is True

    def test_returns_none_for_missing_attr(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj, status = _check_dep(
                name="json_bad",
                module_name="json",
                attr="nonexistent_attr_xyz",
                fallback_backend="fb",
                fallback_desc="desc",
                pip_name="json",
            )
        assert obj is None
        assert status.available is False


class TestCheckConvenience:
    """Tests for check_model2vec, check_rank_bm25, check_xgboost."""

    def test_check_model2vec_returns_tuple(self) -> None:
        result = check_model2vec()
        assert isinstance(result, tuple)
        assert len(result) == 2
        _, status = result
        assert isinstance(status, DepStatus)
        assert status.name == "model2vec"

    def test_check_model2vec_available(self) -> None:
        _, status = check_model2vec()
        assert status.available is True
        assert status.backend == "model2vec"

    def test_check_rank_bm25_returns_tuple(self) -> None:
        result = check_rank_bm25()
        assert isinstance(result, tuple)
        _, status = result
        assert isinstance(status, DepStatus)
        assert status.name == "rank_bm25"

    def test_check_rank_bm25_available(self) -> None:
        _, status = check_rank_bm25()
        assert status.available is True
        assert status.backend == "rank_bm25"

    def test_check_xgboost_returns_tuple(self) -> None:
        result = check_xgboost()
        assert isinstance(result, tuple)
        _, status = result
        assert isinstance(status, DepStatus)
        assert status.name == "xgboost"

    def test_check_xgboost_available(self) -> None:
        _, status = check_xgboost()
        assert status.available is True
        assert status.backend == "xgboost"

    def test_check_model2vec_fallback_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "model2vec":
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj, status = check_model2vec()
        assert obj is None
        assert status.available is False
        assert status.backend == "hashed"

    def test_check_rank_bm25_fallback_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "rank_bm25":
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj, status = check_rank_bm25()
        assert obj is None
        assert status.available is False
        assert status.backend == "pure_python"

    def test_check_xgboost_fallback_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "xgboost":
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj, status = check_xgboost()
        assert obj is None
        assert status.available is False
        assert status.backend == "sklearn"

"""Unit tests for src/reranker/persistence.py"""

import json
from pathlib import Path

import joblib
import pytest

from reranker.persistence import (
    legacy_pickle_loader,
    load_safe,
    save_safe,
    try_load_safe_or_warn,
)


class TestSaveSafe:
    def test_creates_meta_json_and_weights_joblib(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        save_safe(path, "test_model", {}, {"w": [1, 2, 3]})
        assert path.with_suffix(".meta.json").exists()
        assert path.with_suffix(".weights.joblib").exists()
        assert path.exists()

    def test_meta_json_contains_artifact_type(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        save_safe(path, "my_reranker", {"extra_key": "val"}, {"w": 42})
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert meta["artifact_type"] == "my_reranker"
        assert meta["format"] == "safe-joblib"
        assert meta["safe_format_version"] == 2
        assert meta["extra_key"] == "val"

    def test_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "model.pkl"
        save_safe(path, "test_model", {}, {"w": 1})
        assert path.with_suffix(".meta.json").exists()


class TestLoadSafe:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        metadata = {"n_docs": 100}
        weights = {"scores": [0.9, 0.5, 0.1]}
        save_safe(path, "test_model", metadata, weights)
        loaded_meta, loaded_weights = load_safe(path, expected_type="test_model")
        assert loaded_meta["artifact_type"] == "test_model"
        assert loaded_weights == weights

    def test_raises_on_missing_meta(self, tmp_path: Path):
        path = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            load_safe(path, expected_type="test_model")

    def test_raises_on_missing_weights(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        save_safe(path, "test_model", {}, {"w": 1})
        path.with_suffix(".weights.joblib").unlink()
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            load_safe(path, expected_type="test_model")

    def test_raises_on_type_mismatch(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        save_safe(path, "type_a", {}, {"w": 1})
        with pytest.raises(ValueError, match="Unexpected artifact type"):
            load_safe(path, expected_type="type_b")

    def test_raises_on_format_mismatch(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        save_safe(path, "test_model", {}, {"w": 1})
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["format"] = "pickle"
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Unexpected artifact format"):
            load_safe(path, expected_type="test_model")


class TestTryLoadSafeOrWarn:
    def test_loads_safe_format(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        save_safe(path, "test_model", {"key": "val"}, {"w": [1, 2]})
        result = try_load_safe_or_warn(path, expected_type="test_model", legacy_loader=None)
        assert result["w"] == [1, 2]
        assert result["key"] == "val"
        assert result["artifact_type"] == "test_model"

    def test_raises_when_no_safe_and_no_legacy(self, tmp_path: Path):
        path = tmp_path / "model.pkl"
        with pytest.raises(RuntimeError, match="Legacy pickle loading is disabled"):
            try_load_safe_or_warn(
                path,
                expected_type="test_model",
                legacy_loader=None,
                allow_legacy_pickle=False,
            )

    def test_falls_back_to_legacy_with_warning(self, tmp_path: Path):
        path = tmp_path / "legacy.pkl"
        data = {"artifact_version": 1, "artifact_type": "test_model", "format": "pickle", "w": 99}
        joblib.dump(data, path)

        def fake_legacy_loader(p: Path):
            return joblib.load(p)

        with pytest.warns(UserWarning, match="Loading legacy pickle"):
            result = try_load_safe_or_warn(
                path,
                expected_type="test_model",
                legacy_loader=fake_legacy_loader,
                allow_legacy_pickle=True,
            )
        assert result["w"] == 99


class TestLegacyPickleLoader:
    def test_factory_returns_callable(self):
        loader = legacy_pickle_loader("my_type")
        assert callable(loader)

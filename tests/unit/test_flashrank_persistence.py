"""Test FlashRankEnsemble save/load roundtrip."""

from pathlib import Path

from reranker.strategies.flashrank_ensemble import FlashRankEnsemble


def test_save_load_roundtrip(tmp_path: Path):
    ensemble = FlashRankEnsemble(models=["model-a", "model-b"])
    path = tmp_path / "ensemble.pkl"

    ensemble.save(path)
    assert path.exists()

    loaded = FlashRankEnsemble.load(path)
    assert loaded.models == ["model-a", "model-b"]


def test_save_load_single_model(tmp_path: Path):
    ensemble = FlashRankEnsemble(models=["tinybert"])
    path = tmp_path / "single.pkl"

    ensemble.save(path)
    loaded = FlashRankEnsemble.load(path)
    assert loaded.models == ["tinybert"]


def test_empty_models_raises():
    import pytest

    with pytest.raises(ValueError, match="cannot be empty"):
        FlashRankEnsemble(models=[])

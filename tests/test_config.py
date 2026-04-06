import pytest

from reranker.config import get_settings, reset_settings_cache
from reranker.embedder import Embedder


@pytest.mark.unit
def test_settings_reads_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    monkeypatch.setenv("RERANKER_SEED", "99")
    monkeypatch.setenv("RERANKER_PAIR_COUNT", "77")
    reset_settings_cache()

    settings = get_settings()

    assert settings.openrouter.model == "openai/gpt-4.1-mini"
    assert settings.synthetic_data.seed == 99
    assert settings.synthetic_data.pair_count == 77

    reset_settings_cache()


@pytest.mark.unit
def test_embedder_uses_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_EMBEDDER_MODEL", "custom/local-model")
    reset_settings_cache()

    embedder = Embedder()

    assert embedder.model_name == "custom/local-model"
    reset_settings_cache()

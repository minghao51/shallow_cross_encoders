from reranker.config import (
    apply_settings_override,
    clear_settings_override,
    get_settings,
    reset_settings_cache,
    settings_from_dict,
)
from reranker.embedder import Embedder


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


def test_embedder_uses_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_EMBEDDER_MODEL", "custom/local-model")
    reset_settings_cache()

    embedder = Embedder()

    assert embedder.model_name == "custom/local-model"
    reset_settings_cache()


def test_settings_override_is_visible_through_get_settings() -> None:
    reset_settings_cache()
    clear_settings_override()

    override = settings_from_dict({"hybrid": {"weighting_mode": "learned"}})
    apply_settings_override(override)

    try:
        assert get_settings().hybrid.weighting_mode == "learned"
    finally:
        clear_settings_override()
        reset_settings_cache()


def test_public_package_exports_keyword_match_adapter() -> None:
    namespace: dict[str, object] = {}

    exec("from reranker import *", {}, namespace)

    assert "KeywordMatchAdapter" in namespace

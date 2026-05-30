"""Configuration management for the reranker package.

All settings are defined as frozen Pydantic models with environment variable
overrides via pydantic-settings. The central :class:`Settings` model composes
all sub-configurations and is accessed via :func:`get_settings`.

Settings resolution priority (highest first):
  1. Init arguments
  2. Environment variables (RERANKER_<SECTION>__<KEY>)
  3. .env file (dotenvx encrypted)
  4. config.yaml
  5. Pydantic defaults

Legacy env var names (OPENROUTER_API_KEY, LITELLM_API_KEY) are also supported
for backward compatibility.
"""

from __future__ import annotations

import contextvars
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_settings_override: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "_settings_override", default=None
)


class OpenRouterSettings(BaseModel):
    """LLM API configuration for OpenRouter."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr | None = None
    model: str = "openrouter/@preset/basic"
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "shallow-cross-encoders"
    timeout_seconds: float = 30.0


class PathSettings(BaseModel):
    """Filesystem paths for data, models, and logs."""

    model_config = ConfigDict(frozen=True)

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_dir: Path = Path("data/models")
    api_cost_log: Path = Path("data/logs/api_costs.jsonl")


class EmbedderSettings(BaseModel):
    """Configuration for the static embedding model."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "minishlab/potion-base-32M"
    dimension: int = 256
    normalize: bool = True


class GoogleGenAISettings(BaseModel):
    """Google GenAI (Gemini) LLM provider configuration."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr | None = None
    model: str = "gemma-4-31b-it"
    temperature: float = 0.2
    max_retries: int = 3


class LiteLMSSettings(BaseModel):
    """LiteLLM provider configuration.

    LiteLLM routes requests to multiple providers via its prefix system
    (e.g. "gemini/gemini-2.5-flash", "openai/gpt-4", "vertex_ai/gemini-pro").
    """

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr | None = None
    model: str = "gemini/gemini-2.5-flash"


class LLMSettings(BaseModel):
    """Top-level LLM provider selection.

    Each consumer module can override ``default_provider`` by setting its
    own ``llm_provider`` field.
    """

    model_config = ConfigDict(frozen=True)

    default_provider: str = "genai"


class SyntheticDataSettings(BaseModel):
    """Controls for LLM-based synthetic data generation."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    teacher_batch_size: int = 20
    teacher_max_workers: int = 4
    stream_chunk_size: int = 100
    pair_count: int = 2000
    preference_count: int = 1500
    contradiction_count: int = 500
    control_count: int = 200
    llm_provider: str | None = None  # None → inherits from llm.default_provider


class WeightProfile(BaseModel):
    """Hybrid feature weight configuration."""

    model_config = ConfigDict(frozen=True)

    sem_score: float = 0.25
    bm25_score: float = 0.20
    token_overlap_ratio: float = 0.15
    query_coverage_ratio: float = 0.20
    shared_token_char_sum: float = 0.10
    exact_phrase_match: float = 0.10
    keyword_hit_rate: float = 0.05


class HybridSettings(BaseModel):
    """XGBoost/sklearn hybrid fusion reranker parameters."""

    model_config = ConfigDict(frozen=True)

    random_state: int = 42
    xgb_n_estimators: int = 120
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.08
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    weights: WeightProfile = WeightProfile()
    ensemble_mode: str = "xgboost"
    rrf_k: int = 60
    weighting_mode: str = "static"


class DistilledSettings(BaseModel):
    """Logistic regression / cross-encoder distilled reranker parameters."""

    model_config = ConfigDict(frozen=True)

    random_state: int = 42
    logistic_c: float = 1.0
    logistic_max_iter: int = 500
    full_tournament_max_docs: int = 50
    loss_type: str = "pairwise"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LateInteractionSettings(BaseModel):
    """ColBERT-style late interaction reranker parameters."""

    model_config = ConfigDict(frozen=True)

    top_k_tokens: int = 128
    use_salience: bool = True
    quantization: str = "none"


class BinaryRerankerSettings(BaseModel):
    """Binary quantization + Hamming distance reranker parameters."""

    model_config = ConfigDict(frozen=True)

    hamming_top_k: int = 500
    bilinear_top_k: int = 50
    random_state: int = 42


class MetaRouterSettings(BaseModel):
    """Query-type routing model parameters."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    model_type: str = "decision_tree"
    n_categories: int = 3
    min_samples_leaf: int = 5


class LSHSettings(BaseModel):
    """LSH-based near-duplicate detection parameters."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    ngram_size: int = 3
    num_perm: int = 128
    threshold: float = 0.5


class ActiveDistillationSettings(BaseModel):
    """Active learning distillation loop parameters."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    mode: str = "oneshot"
    mining_strategy: str = "contested"
    active_iterations: int = 3
    uncertainty_low: float = 0.4
    uncertainty_high: float = 0.6
    contested_rank_gap: int = 50
    diversity_clusters: int = 10
    llm_provider: str | None = None  # None → inherits from llm.default_provider
    batch_size: int = 20


class PipelineSettings(BaseModel):
    """Multi-stage pipeline defaults."""

    model_config = ConfigDict(frozen=True)

    default_stage_top_k: int = 200


class ConsistencySettings(BaseModel):
    """Consistency engine similarity and tolerance thresholds."""

    model_config = ConfigDict(frozen=True)

    sim_threshold: float = 0.95
    value_tolerance: float = 0.01


class EmbeddingCacheSettings(BaseModel):
    """Shared embedding cache configuration."""

    model_config = ConfigDict(frozen=True)

    max_size: int = 50000
    ttl_seconds: int = 3600


class RoiSettings(BaseModel):
    """ROI estimation parameters for cost-benefit analysis."""

    model_config = ConfigDict(frozen=True)

    llm_cost_per_judgment_usd: float = 0.0004
    projected_monthly_queries: int = 10000


class EvalSettings(BaseModel):
    """Dataset split ratios for evaluation."""

    model_config = ConfigDict(frozen=True)

    default_split: str = "test"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    @field_validator("train_ratio", "validation_ratio", "test_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {v}")
        return v


class Settings(BaseSettings):
    """Root configuration composing all sub-configurations.

    Settings are resolved from multiple sources in priority order:
    init args > env vars > .env file > config.yaml > Python defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        extra="ignore",
        frozen=True,
    )

    openrouter: OpenRouterSettings = OpenRouterSettings()
    google_genai: GoogleGenAISettings = GoogleGenAISettings()
    litellm: LiteLMSSettings = LiteLMSSettings()
    llm: LLMSettings = LLMSettings()
    paths: PathSettings = PathSettings()
    embedder: EmbedderSettings = EmbedderSettings()
    synthetic_data: SyntheticDataSettings = SyntheticDataSettings()
    hybrid: HybridSettings = HybridSettings()
    distilled: DistilledSettings = DistilledSettings()
    late_interaction: LateInteractionSettings = LateInteractionSettings()
    binary_reranker: BinaryRerankerSettings = BinaryRerankerSettings()
    pipeline: PipelineSettings = PipelineSettings()
    consistency: ConsistencySettings = ConsistencySettings()
    embedding_cache: EmbeddingCacheSettings = EmbeddingCacheSettings()
    roi: RoiSettings = RoiSettings()
    eval: EvalSettings = EvalSettings()
    meta_router: MetaRouterSettings = MetaRouterSettings()
    lsh: LSHSettings = LSHSettings()
    active_distillation: ActiveDistillationSettings = ActiveDistillationSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="before")
    @classmethod
    def inject_legacy_env_vars(cls, data: Any) -> Any:
        """Support legacy env var names (OPENROUTER_API_KEY, LITELLM_API_KEY)."""
        if not isinstance(data, dict):
            return data
        openrouter = data.get("openrouter", {})
        if isinstance(openrouter, dict) and not openrouter.get("api_key"):
            val = os.getenv("OPENROUTER_API_KEY")
            if val:
                openrouter["api_key"] = val
                data["openrouter"] = openrouter
        ad = data.get("active_distillation", {})
        if isinstance(ad, dict) and not ad.get("litellm_api_key"):
            val = os.getenv("LITELLM_API_KEY")
            if val:
                ad["litellm_api_key"] = val
                data["active_distillation"] = ad
        gg = data.get("google_genai", {})
        if isinstance(gg, dict) and not gg.get("api_key"):
            val = os.getenv("GOOGLE_GENAI_API_KEY")
            if val:
                gg["api_key"] = val
                data["google_genai"] = gg
        lt = data.get("litellm", {})
        if isinstance(lt, dict) and not lt.get("api_key"):
            val = os.getenv("LITELLM_API_KEY")
            if val:
                lt["api_key"] = val
                data["litellm"] = lt
        return data


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """Return the current global settings instance.

    Returns the context-local override if set, otherwise the cached default.
    """
    override = _settings_override.get()
    if override is not None:
        return override
    return _cached_settings()


def reset_settings_cache() -> None:
    """Clear the cached default settings so they are re-read on next access."""
    _cached_settings.cache_clear()


def apply_settings_override(settings: Settings) -> None:
    """Apply a settings override for the current context."""
    _settings_override.set(settings)


def clear_settings_override() -> None:
    """Clear any context-local settings override."""
    _settings_override.set(None)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a nested dictionary.
    """
    import yaml

    raw = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(raw) or {}


def settings_from_yaml(path: str | Path) -> Settings:
    """Load settings from a YAML file, deep-merged with current defaults.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        New Settings instance with YAML values overriding defaults.
    """
    yaml_data = load_yaml_config(path)
    current = get_settings().model_dump()
    merged = _deep_merge(current, yaml_data)
    return Settings(**merged)


def settings_from_dict(data: dict[str, Any]) -> Settings:
    """Create settings from a dictionary, deep-merged with current defaults.

    Args:
        data: Dictionary of configuration values.

    Returns:
        New Settings instance with provided values overriding defaults.
    """
    current = get_settings().model_dump()
    merged = _deep_merge(current, data)
    return Settings(**merged)

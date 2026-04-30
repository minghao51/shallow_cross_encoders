"""Configuration management for the reranker package.

All settings are defined as frozen Pydantic models with environment variable
overrides. The central :class:`Settings` model composes all sub-configurations
and is accessed via :func:`get_settings`.

Settings can be overridden at runtime via :func:`apply_settings_override`
or loaded from YAML files via :func:`settings_from_yaml`.
"""

from __future__ import annotations

import contextvars
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

_T = TypeVar("_T")
_settings_override: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "_settings_override", default=None
)


def _env(name: str, default: _T, typ: type[_T]) -> _T:
    value = os.getenv(name)
    if value is None:
        return default
    if typ is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}  # type: ignore
    return typ(value)  # type: ignore


class OpenRouterSettings(BaseModel):
    """LLM API configuration for OpenRouter.

    Controls which model is used for synthetic data generation and
    LLM-based judgments. Falls back to environment variables.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str | None = None
    model: str = "openai/gpt-4o-mini"
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
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 3600


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
    roadmap_pair_count: int = 2000
    roadmap_preference_count: int = 1500
    roadmap_contradiction_count: int = 500
    roadmap_control_count: int = 200


class HybridSettings(BaseModel):
    """XGBoost/sklearn hybrid fusion reranker parameters."""

    model_config = ConfigDict(frozen=True)

    random_state: int = 42
    xgb_n_estimators: int = 120
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.08
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    weight_sem_score: float = 0.25
    weight_bm25_score: float = 0.20
    weight_token_overlap: float = 0.15
    weight_query_coverage: float = 0.20
    weight_shared_char: float = 0.10
    weight_exact_phrase: float = 0.10
    weight_keyword_hit: float = 0.05
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


class SPLADESettings(BaseModel):
    """SPLADE sparse lexical reranker parameters."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "naver/splade-cocondenser-ensembledistil"
    top_k_terms: int = 128


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
    litellm_model: str = "openrouter/openai/gpt-4o-mini"
    litellm_api_key: str | None = None
    litellm_batch_size: int = 20


class PipelineSettings(BaseModel):
    """Multi-stage pipeline defaults."""

    model_config = ConfigDict(frozen=True)

    default_stage_top_k: int = 200


class CascadeSettings(BaseModel):
    """Cascade reranker confidence thresholding."""

    model_config = ConfigDict(frozen=True)

    confidence_threshold: float = 0.6
    fallback_strategy: str = "flashrank"


class ConsistencySettings(BaseModel):
    """Consistency engine similarity and tolerance thresholds."""

    model_config = ConfigDict(frozen=True)

    sim_threshold: float = 0.95
    value_tolerance: float = 0.01


class RoiSettings(BaseModel):
    """ROI estimation parameters for cost-benefit analysis."""

    model_config = ConfigDict(frozen=True)

    llm_cost_per_judgment_usd: float = 0.0004
    projected_monthly_queries: int = 10000


class BenchmarkSettings(BaseModel):
    """Benchmark timing targets and sample configuration."""

    model_config = ConfigDict(frozen=True)

    sample_doc: str = "This is a sample document for latency measurement."
    sample_size: int = 100
    candidate_count: int = 20
    embedding_target_ms_per_doc: float = 5.0
    rerank_target_ms_per_query: float = 50.0
    consistency_claim_count: int = 1000
    consistency_target_ms_per_1000_claims: float = 50.0


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


class Settings(BaseModel):
    """Root configuration composing all sub-configurations."""

    model_config = ConfigDict(frozen=True)

    openrouter: OpenRouterSettings
    paths: PathSettings
    embedder: EmbedderSettings
    synthetic_data: SyntheticDataSettings
    hybrid: HybridSettings
    distilled: DistilledSettings
    late_interaction: LateInteractionSettings
    binary_reranker: BinaryRerankerSettings
    pipeline: PipelineSettings
    cascade: CascadeSettings
    consistency: ConsistencySettings
    roi: RoiSettings
    benchmark: BenchmarkSettings
    eval: EvalSettings
    splade: SPLADESettings
    meta_router: MetaRouterSettings
    lsh: LSHSettings
    active_distillation: ActiveDistillationSettings


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings(
        openrouter=OpenRouterSettings(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=_env("OPENROUTER_MODEL", "openai/gpt-4o-mini", str),
            base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", str),
            app_name=_env("OPENROUTER_APP_NAME", "shallow-cross-encoders", str),
            timeout_seconds=_env("OPENROUTER_TIMEOUT_SECONDS", 30.0, float),
        ),
        paths=PathSettings(
            raw_data_dir=Path(_env("RERANKER_RAW_DATA_DIR", "data/raw", str)),
            processed_data_dir=Path(_env("RERANKER_PROCESSED_DATA_DIR", "data/processed", str)),
            model_dir=Path(_env("RERANKER_MODEL_DIR", "data/models", str)),
            api_cost_log=Path(_env("RERANKER_API_COST_LOG", "data/logs/api_costs.jsonl", str)),
        ),
        embedder=EmbedderSettings(
            model_name=_env("RERANKER_EMBEDDER_MODEL", "minishlab/potion-base-32M", str),
            dimension=_env("RERANKER_EMBEDDER_DIMENSION", 256, int),
            normalize=_env("RERANKER_EMBEDDER_NORMALIZE", True, bool),
            cache_max_size=_env("RERANKER_EMBEDDER_CACHE_MAX_SIZE", 10000, int),
            cache_ttl_seconds=_env("RERANKER_EMBEDDER_CACHE_TTL_SECONDS", 3600, int),
        ),
        synthetic_data=SyntheticDataSettings(
            seed=_env("RERANKER_SEED", 42, int),
            teacher_batch_size=_env("RERANKER_TEACHER_BATCH_SIZE", 20, int),
            teacher_max_workers=_env("RERANKER_TEACHER_MAX_WORKERS", 4, int),
            stream_chunk_size=_env("RERANKER_STREAM_CHUNK_SIZE", 100, int),
            pair_count=_env("RERANKER_PAIR_COUNT", 2000, int),
            preference_count=_env("RERANKER_PREFERENCE_COUNT", 1500, int),
            contradiction_count=_env("RERANKER_CONTRADICTION_COUNT", 500, int),
            control_count=_env("RERANKER_CONTROL_COUNT", 200, int),
            roadmap_pair_count=_env("RERANKER_ROADMAP_PAIR_COUNT", 2000, int),
            roadmap_preference_count=_env("RERANKER_ROADMAP_PREFERENCE_COUNT", 1500, int),
            roadmap_contradiction_count=_env("RERANKER_ROADMAP_CONTRADICTION_COUNT", 500, int),
            roadmap_control_count=_env("RERANKER_ROADMAP_CONTROL_COUNT", 200, int),
        ),
        hybrid=HybridSettings(
            random_state=_env("RERANKER_HYBRID_RANDOM_STATE", 42, int),
            xgb_n_estimators=_env("RERANKER_HYBRID_XGB_N_ESTIMATORS", 120, int),
            xgb_max_depth=_env("RERANKER_HYBRID_XGB_MAX_DEPTH", 4, int),
            xgb_learning_rate=_env("RERANKER_HYBRID_XGB_LEARNING_RATE", 0.08, float),
            xgb_subsample=_env("RERANKER_HYBRID_XGB_SUBSAMPLE", 0.9, float),
            xgb_colsample_bytree=_env("RERANKER_HYBRID_XGB_COLSAMPLE_BYTREE", 0.9, float),
            weight_sem_score=_env("RERANKER_HYBRID_WEIGHT_SEM_SCORE", 0.25, float),
            weight_bm25_score=_env("RERANKER_HYBRID_WEIGHT_BM25_SCORE", 0.20, float),
            weight_token_overlap=_env("RERANKER_HYBRID_WEIGHT_TOKEN_OVERLAP", 0.15, float),
            weight_query_coverage=_env("RERANKER_HYBRID_WEIGHT_QUERY_COVERAGE", 0.20, float),
            weight_shared_char=_env("RERANKER_HYBRID_WEIGHT_SHARED_CHAR", 0.10, float),
            weight_exact_phrase=_env("RERANKER_HYBRID_WEIGHT_EXACT_PHRASE", 0.10, float),
            weight_keyword_hit=_env("RERANKER_HYBRID_WEIGHT_KEYWORD_HIT", 0.05, float),
            ensemble_mode=_env("RERANKER_HYBRID_ENSEMBLE_MODE", "xgboost", str),
            rrf_k=_env("RERANKER_HYBRID_RRF_K", 60, int),
            weighting_mode=_env("RERANKER_HYBRID_WEIGHTING_MODE", "static", str),
        ),
        distilled=DistilledSettings(
            random_state=_env("RERANKER_DISTILLED_RANDOM_STATE", 42, int),
            logistic_c=_env("RERANKER_DISTILLED_LOGISTIC_C", 1.0, float),
            logistic_max_iter=_env("RERANKER_DISTILLED_LOGISTIC_MAX_ITER", 500, int),
            full_tournament_max_docs=_env("RERANKER_DISTILLED_FULL_TOURNAMENT_MAX_DOCS", 50, int),
            loss_type=_env("RERANKER_DISTILLED_LOSS_TYPE", "pairwise", str),
            cross_encoder_model=_env(
                "RERANKER_DISTILLED_CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                str,
            ),
        ),
        late_interaction=LateInteractionSettings(
            top_k_tokens=_env("RERANKER_LATE_INTERACTION_TOP_K_TOKENS", 128, int),
            use_salience=_env("RERANKER_LATE_INTERACTION_USE_SALIENCE", True, bool),
            quantization=_env("RERANKER_LATE_INTERACTION_QUANTIZATION", "none", str),
        ),
        binary_reranker=BinaryRerankerSettings(
            hamming_top_k=_env("RERANKER_BINARY_RERANKER_HAMMING_TOP_K", 500, int),
            bilinear_top_k=_env("RERANKER_BINARY_RERANKER_BILINEAR_TOP_K", 50, int),
            random_state=_env("RERANKER_BINARY_RERANKER_RANDOM_STATE", 42, int),
        ),
        pipeline=PipelineSettings(
            default_stage_top_k=_env("RERANKER_PIPELINE_DEFAULT_STAGE_TOP_K", 200, int),
        ),
        cascade=CascadeSettings(
            confidence_threshold=_env("RERANKER_CASCADE_CONFIDENCE_THRESHOLD", 0.6, float),
            fallback_strategy=_env("RERANKER_CASCADE_FALLBACK_STRATEGY", "flashrank", str),
        ),
        consistency=ConsistencySettings(
            sim_threshold=_env("RERANKER_CONSISTENCY_SIM_THRESHOLD", 0.95, float),
            value_tolerance=_env("RERANKER_CONSISTENCY_VALUE_TOLERANCE", 0.01, float),
        ),
        roi=RoiSettings(
            llm_cost_per_judgment_usd=_env("RERANKER_ROI_LLM_COST_PER_JUDGMENT_USD", 0.0004, float),
            projected_monthly_queries=_env("RERANKER_ROI_PROJECTED_MONTHLY_QUERIES", 10000, int),
        ),
        benchmark=BenchmarkSettings(
            sample_doc=_env(
                "RERANKER_BENCHMARK_SAMPLE_DOC",
                "This is a sample document for latency measurement.",
                str,
            ),
            sample_size=_env("RERANKER_BENCHMARK_SAMPLE_SIZE", 100, int),
            candidate_count=_env("RERANKER_BENCHMARK_CANDIDATE_COUNT", 20, int),
            embedding_target_ms_per_doc=_env(
                "RERANKER_BENCHMARK_EMBEDDING_TARGET_MS_PER_DOC",
                5.0,
                float,
            ),
            rerank_target_ms_per_query=_env(
                "RERANKER_BENCHMARK_RERANK_TARGET_MS_PER_QUERY",
                50.0,
                float,
            ),
            consistency_claim_count=_env(
                "RERANKER_BENCHMARK_CONSISTENCY_CLAIM_COUNT",
                1000,
                int,
            ),
            consistency_target_ms_per_1000_claims=_env(
                "RERANKER_BENCHMARK_CONSISTENCY_TARGET_MS_PER_1000_CLAIMS",
                50.0,
                float,
            ),
        ),
        eval=EvalSettings(
            default_split=_env("RERANKER_EVAL_DEFAULT_SPLIT", "test", str),
            train_ratio=_env("RERANKER_EVAL_TRAIN_RATIO", 0.7, float),
            validation_ratio=_env("RERANKER_EVAL_VALIDATION_RATIO", 0.15, float),
            test_ratio=_env("RERANKER_EVAL_TEST_RATIO", 0.15, float),
        ),
        splade=SPLADESettings(
            model_name=_env(
                "RERANKER_SPLADE_MODEL_NAME",
                "naver/splade-cocondenser-ensembledistil",
                str,
            ),
            top_k_terms=_env("RERANKER_SPLADE_TOP_K_TERMS", 128, int),
        ),
        meta_router=MetaRouterSettings(
            enabled=_env("RERANKER_META_ROUTER_ENABLED", False, bool),
            model_type=_env("RERANKER_META_ROUTER_MODEL_TYPE", "decision_tree", str),
            n_categories=_env("RERANKER_META_ROUTER_N_CATEGORIES", 3, int),
            min_samples_leaf=_env("RERANKER_META_ROUTER_MIN_SAMPLES_LEAF", 5, int),
        ),
        lsh=LSHSettings(
            enabled=_env("RERANKER_LSH_ENABLED", False, bool),
            ngram_size=_env("RERANKER_LSH_NGRAM_SIZE", 3, int),
            num_perm=_env("RERANKER_LSH_NUM_PERM", 128, int),
            threshold=_env("RERANKER_LSH_THRESHOLD", 0.5, float),
        ),
        active_distillation=ActiveDistillationSettings(
            enabled=_env("RERANKER_ACTIVE_DISTILLATION_ENABLED", False, bool),
            mode=_env("RERANKER_ACTIVE_DISTILLATION_MODE", "oneshot", str),
            mining_strategy=_env("RERANKER_ACTIVE_DISTILLATION_MINING_STRATEGY", "contested", str),
            active_iterations=_env("RERANKER_ACTIVE_DISTILLATION_ACTIVE_ITERATIONS", 3, int),
            uncertainty_low=_env("RERANKER_ACTIVE_DISTILLATION_UNCERTAINTY_LOW", 0.4, float),
            uncertainty_high=_env("RERANKER_ACTIVE_DISTILLATION_UNCERTAINTY_HIGH", 0.6, float),
            contested_rank_gap=_env("RERANKER_ACTIVE_DISTILLATION_CONTESTED_RANK_GAP", 50, int),
            diversity_clusters=_env("RERANKER_ACTIVE_DISTILLATION_DIVERSITY_CLUSTERS", 10, int),
            litellm_model=_env(
                "RERANKER_ACTIVE_DISTILLATION_LITELLM_MODEL", "openrouter/openai/gpt-4o-mini", str
            ),
            litellm_api_key=os.getenv("LITELLM_API_KEY"),
            litellm_batch_size=_env("RERANKER_ACTIVE_DISTILLATION_LITELLM_BATCH_SIZE", 20, int),
        ),
    )


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

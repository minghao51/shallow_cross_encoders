from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


class OpenRouterSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str | None = None
    model: str = "openai/gpt-4o-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "shallow-cross-encoders"
    timeout_seconds: float = 30.0


class PathSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_dir: Path = Path("data/models")
    api_cost_log: Path = Path("data/logs/api_costs.jsonl")


class EmbedderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "minishlab/potion-base-32M"
    dimension: int = 256
    normalize: bool = True


class SyntheticDataSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 42
    teacher_batch_size: int = 20
    teacher_max_workers: int = 4
    pair_count: int = 2000
    preference_count: int = 1500
    contradiction_count: int = 500
    control_count: int = 200
    roadmap_pair_count: int = 2000
    roadmap_preference_count: int = 1500
    roadmap_contradiction_count: int = 500
    roadmap_control_count: int = 200


class HybridSettings(BaseModel):
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


class DistilledSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    random_state: int = 42
    logistic_c: float = 1.0
    logistic_max_iter: int = 500
    full_tournament_max_docs: int = 50
    loss_type: str = "pairwise"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LateInteractionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k_tokens: int = 128
    use_salience: bool = True


class BinaryRerankerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    hamming_top_k: int = 500
    bilinear_top_k: int = 50
    random_state: int = 42


class SPLADESettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "naver/splade-v2-max"
    top_k_terms: int = 128


class PipelineSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_stage_top_k: int = 200


class ConsistencySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    sim_threshold: float = 0.95
    value_tolerance: float = 0.01


class RoiSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_cost_per_judgment_usd: float = 0.0004
    projected_monthly_queries: int = 10000


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_doc: str = "This is a sample document for latency measurement."
    sample_size: int = 100
    candidate_count: int = 20
    embedding_target_ms_per_doc: float = 5.0
    rerank_target_ms_per_query: float = 50.0
    consistency_claim_count: int = 1000
    consistency_target_ms_per_1000_claims: float = 50.0


class EvalSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_split: str = "test"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15


class Settings(BaseModel):
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
    consistency: ConsistencySettings
    roi: RoiSettings
    benchmark: BenchmarkSettings
    eval: EvalSettings
    splade: SPLADESettings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openrouter=OpenRouterSettings(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=_env_str("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=_env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            app_name=_env_str("OPENROUTER_APP_NAME", "shallow-cross-encoders"),
            timeout_seconds=_env_float("OPENROUTER_TIMEOUT_SECONDS", 30.0),
        ),
        paths=PathSettings(
            raw_data_dir=Path(_env_str("RERANKER_RAW_DATA_DIR", "data/raw")),
            processed_data_dir=Path(_env_str("RERANKER_PROCESSED_DATA_DIR", "data/processed")),
            model_dir=Path(_env_str("RERANKER_MODEL_DIR", "data/models")),
            api_cost_log=Path(_env_str("RERANKER_API_COST_LOG", "data/logs/api_costs.jsonl")),
        ),
        embedder=EmbedderSettings(
            model_name=_env_str("RERANKER_EMBEDDER_MODEL", "minishlab/potion-base-32M"),
            dimension=_env_int("RERANKER_EMBEDDER_DIMENSION", 256),
            normalize=_env_bool("RERANKER_EMBEDDER_NORMALIZE", True),
        ),
        synthetic_data=SyntheticDataSettings(
            seed=_env_int("RERANKER_SEED", 42),
            teacher_batch_size=_env_int("RERANKER_TEACHER_BATCH_SIZE", 20),
            teacher_max_workers=_env_int("RERANKER_TEACHER_MAX_WORKERS", 4),
            pair_count=_env_int("RERANKER_PAIR_COUNT", 60),
            preference_count=_env_int("RERANKER_PREFERENCE_COUNT", 40),
            contradiction_count=_env_int("RERANKER_CONTRADICTION_COUNT", 20),
            control_count=_env_int("RERANKER_CONTROL_COUNT", 8),
            roadmap_pair_count=_env_int("RERANKER_ROADMAP_PAIR_COUNT", 2000),
            roadmap_preference_count=_env_int("RERANKER_ROADMAP_PREFERENCE_COUNT", 1500),
            roadmap_contradiction_count=_env_int("RERANKER_ROADMAP_CONTRADICTION_COUNT", 500),
            roadmap_control_count=_env_int("RERANKER_ROADMAP_CONTROL_COUNT", 200),
        ),
        hybrid=HybridSettings(
            random_state=_env_int("RERANKER_HYBRID_RANDOM_STATE", 42),
            xgb_n_estimators=_env_int("RERANKER_HYBRID_XGB_N_ESTIMATORS", 120),
            xgb_max_depth=_env_int("RERANKER_HYBRID_XGB_MAX_DEPTH", 4),
            xgb_learning_rate=_env_float("RERANKER_HYBRID_XGB_LEARNING_RATE", 0.08),
            xgb_subsample=_env_float("RERANKER_HYBRID_XGB_SUBSAMPLE", 0.9),
            xgb_colsample_bytree=_env_float("RERANKER_HYBRID_XGB_COLSAMPLE_BYTREE", 0.9),
            weight_sem_score=_env_float("RERANKER_HYBRID_WEIGHT_SEM_SCORE", 0.25),
            weight_bm25_score=_env_float("RERANKER_HYBRID_WEIGHT_BM25_SCORE", 0.20),
            weight_token_overlap=_env_float("RERANKER_HYBRID_WEIGHT_TOKEN_OVERLAP", 0.15),
            weight_query_coverage=_env_float("RERANKER_HYBRID_WEIGHT_QUERY_COVERAGE", 0.20),
            weight_shared_char=_env_float("RERANKER_HYBRID_WEIGHT_SHARED_CHAR", 0.10),
            weight_exact_phrase=_env_float("RERANKER_HYBRID_WEIGHT_EXACT_PHRASE", 0.10),
            weight_keyword_hit=_env_float("RERANKER_HYBRID_WEIGHT_KEYWORD_HIT", 0.05),
            ensemble_mode=_env_str("RERANKER_HYBRID_ENSEMBLE_MODE", "xgboost"),
            rrf_k=_env_int("RERANKER_HYBRID_RRF_K", 60),
        ),
        distilled=DistilledSettings(
            random_state=_env_int("RERANKER_DISTILLED_RANDOM_STATE", 42),
            logistic_c=_env_float("RERANKER_DISTILLED_LOGISTIC_C", 1.0),
            logistic_max_iter=_env_int("RERANKER_DISTILLED_LOGISTIC_MAX_ITER", 500),
            full_tournament_max_docs=_env_int(
                "RERANKER_DISTILLED_FULL_TOURNAMENT_MAX_DOCS",
                50,
            ),
            loss_type=_env_str("RERANKER_DISTILLED_LOSS_TYPE", "pairwise"),
            cross_encoder_model=_env_str(
                "RERANKER_DISTILLED_CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
        ),
        late_interaction=LateInteractionSettings(
            top_k_tokens=_env_int("RERANKER_LATE_INTERACTION_TOP_K_TOKENS", 128),
            use_salience=_env_bool("RERANKER_LATE_INTERACTION_USE_SALIENCE", True),
        ),
        binary_reranker=BinaryRerankerSettings(
            hamming_top_k=_env_int("RERANKER_BINARY_RERANKER_HAMMING_TOP_K", 500),
            bilinear_top_k=_env_int("RERANKER_BINARY_RERANKER_BILINEAR_TOP_K", 50),
            random_state=_env_int("RERANKER_BINARY_RERANKER_RANDOM_STATE", 42),
        ),
        pipeline=PipelineSettings(
            default_stage_top_k=_env_int("RERANKER_PIPELINE_DEFAULT_STAGE_TOP_K", 200),
        ),
        consistency=ConsistencySettings(
            sim_threshold=_env_float("RERANKER_CONSISTENCY_SIM_THRESHOLD", 0.95),
            value_tolerance=_env_float("RERANKER_CONSISTENCY_VALUE_TOLERANCE", 0.01),
        ),
        roi=RoiSettings(
            llm_cost_per_judgment_usd=_env_float("RERANKER_ROI_LLM_COST_PER_JUDGMENT_USD", 0.0004),
            projected_monthly_queries=_env_int("RERANKER_ROI_PROJECTED_MONTHLY_QUERIES", 10000),
        ),
        benchmark=BenchmarkSettings(
            sample_doc=_env_str(
                "RERANKER_BENCHMARK_SAMPLE_DOC",
                "This is a sample document for latency measurement.",
            ),
            sample_size=_env_int("RERANKER_BENCHMARK_SAMPLE_SIZE", 100),
            candidate_count=_env_int("RERANKER_BENCHMARK_CANDIDATE_COUNT", 20),
            embedding_target_ms_per_doc=_env_float(
                "RERANKER_BENCHMARK_EMBEDDING_TARGET_MS_PER_DOC",
                5.0,
            ),
            rerank_target_ms_per_query=_env_float(
                "RERANKER_BENCHMARK_RERANK_TARGET_MS_PER_QUERY",
                50.0,
            ),
            consistency_claim_count=_env_int(
                "RERANKER_BENCHMARK_CONSISTENCY_CLAIM_COUNT",
                1000,
            ),
            consistency_target_ms_per_1000_claims=_env_float(
                "RERANKER_BENCHMARK_CONSISTENCY_TARGET_MS_PER_1000_CLAIMS",
                50.0,
            ),
        ),
        eval=EvalSettings(
            default_split=_env_str("RERANKER_EVAL_DEFAULT_SPLIT", "test"),
            train_ratio=_env_float("RERANKER_EVAL_TRAIN_RATIO", 0.7),
            validation_ratio=_env_float("RERANKER_EVAL_VALIDATION_RATIO", 0.15),
            test_ratio=_env_float("RERANKER_EVAL_TEST_RATIO", 0.15),
        ),
        splade=SPLADESettings(
            model_name=_env_str("RERANKER_SPLADE_MODEL_NAME", "naver/splade-v2-max"),
            top_k_terms=_env_int("RERANKER_SPLADE_TOP_K_TERMS", 128),
        ),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()

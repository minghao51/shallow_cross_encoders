from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

from reranker.data.hard_negative_sampler import prepare_benchmark_data_with_hard_negatives
from reranker.persistence_mixin import SaveableReranker
from reranker.types import RankedDoc
from reranker.utils import rank_docs

logger = structlog.get_logger(__name__)


class FlashRankEnsemble(SaveableReranker):
    """Ensemble of FlashRank models for multi-teacher distillation."""

    _artifact_type = "flashrank_ensemble"

    def __init__(self, models: list[str]) -> None:
        if not models:
            raise ValueError("models list cannot be empty")
        self.models = models
        self._rankers: list[Any] | None = None
        self.is_fitted = True

    def _load_rankers(self) -> list[Any]:
        if self._rankers is not None:
            return self._rankers
        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "flashrank is not installed. Install with: uv sync --extra flashrank"
            ) from e
        self._rankers = [Ranker(model_name=model) for model in self.models]
        return self._rankers

    def score_batch(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        rankers = self._load_rankers()
        from flashrank import RerankRequest

        all_scores: list[np.ndarray] = []
        for ranker in rankers:
            passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)
            scores = np.zeros(len(docs), dtype=np.float32)
            for result in results:
                doc_idx = int(result["id"])
                scores[doc_idx] = float(result.get("score", 0.0))
            all_scores.append(scores)
        return np.mean(all_scores, axis=0).astype(np.float32)

    def rerank(self, query: str, docs: list[str]) -> list[Any]:
        if not docs:
            return []
        scores = self.score_batch(query, docs)
        return rank_docs(docs, scores, "flashrank_ensemble")

    def _save_metadata(self) -> dict:
        return {"models": self.models}

    def _save_weights(self) -> dict:
        return {}

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> FlashRankEnsemble:
        payload = cls._load_payload(path, expected_type=cls._artifact_type)
        return cls(models=payload.get("models", []))


class _LazyModelWrapper(SaveableReranker):
    """Base for lazy-loading cross-encoder wrappers."""

    _artifact_type = "lazy_model_wrapper"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.is_fitted = True

    def _load_backend(self) -> Any:
        raise NotImplementedError

    def _predict_scores(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        raise NotImplementedError

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []
        indexed_scores = self._predict_scores(query, docs)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [
            RankedDoc(
                doc=docs[idx],
                score=float(score),
                rank=rank,
                metadata={"strategy": self._strategy_name},
            )
            for rank, (idx, score) in enumerate(indexed_scores, start=1)
        ]

    @property
    def _strategy_name(self) -> str:
        return type(self).__name__


class FlashRankWrapper(_LazyModelWrapper):
    """Single-model FlashRank wrapper for benchmarking baselines."""

    _artifact_type = "flashrank_wrapper"

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2") -> None:
        super().__init__(model_name)
        self._ranker: Any = None

    def _load_backend(self) -> Any:
        if self._ranker is not None:
            return self._ranker
        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "flashrank is not installed. Install with: uv sync --extra flashrank"
            ) from e
        self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def _predict_scores(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        ranker = self._load_backend()
        from flashrank import RerankRequest

        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        return [(int(result["id"]), float(result.get("score", 0.0))) for result in results]


class SentenceTransformerWrapper(_LazyModelWrapper):
    """SentenceTransformer cross-encoder adapter for benchmarking."""

    _artifact_type = "sentence_transformer_wrapper"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        super().__init__(model_name)
        self._model: Any = None

    def _load_backend(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv sync --extra sentence-transformers"
            ) from e
        self._model = CrossEncoder(self.model_name)
        return self._model

    def _predict_scores(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        model = self._load_backend()
        pairs = [[query, doc] for doc in docs]
        scores = model.predict(pairs)
        return list(enumerate(scores))


class HardNegativeFlashRankEnsemble(FlashRankEnsemble):
    """FlashRank ensemble that uses hard negative sampling for scoring."""

    def score_batch(self, query: str, docs: list[str]) -> np.ndarray:
        try:
            hard_neg_data = prepare_benchmark_data_with_hard_negatives(
                [{"query": query, "doc": d, "score": 0} for d in docs],  # type: ignore[arg-type]
                top_k=min(20, len(docs)),  # type: ignore[call-arg]
                rerank_fn=lambda q, d: super().score_batch(q, d),  # type: ignore[call-arg]
            )
            hard_scores = [row.get("hard_neg_score", 0.0) for row in hard_neg_data]
            return np.asarray(hard_scores, dtype=np.float32)
        except Exception as exc:
            logger.warning("Hard negative scoring failed, falling back to standard. Error: %s", exc)
            return super().score_batch(query, docs)

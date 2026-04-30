"""Distilled pairwise ranker using LLM-generated preference comparisons.

Trains a lightweight model (logistic regression or cross-encoder) to
approximate an expensive teacher reranker's pairwise preferences.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.protocols import RankedDoc
from reranker.utils import (
    load_pickle,
)

logger = logging.getLogger("reranker.strategies.distilled")


class DistilledPairwiseRanker:
    """Local pairwise preference model trained on LLM-generated comparisons.

    Supports multiple loss types for improved NDCG optimization:
    - "pairwise": Original LogisticRegression on pairwise features (baseline)
    - "listwise": ListMLE loss for direct listwise optimization
    - "lambdaloss": LambdaLoss with NDCG weighting for NDCG-optimized ranking
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        loss_type: Literal["pairwise", "listwise", "lambdaloss"] | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or Embedder()
        self.loss_type = loss_type or settings.distilled.loss_type
        self.model = LogisticRegression(
            C=settings.distilled.logistic_c,
            max_iter=settings.distilled.logistic_max_iter,
            random_state=settings.distilled.random_state,
        )
        self.full_tournament_max_docs = settings.distilled.full_tournament_max_docs
        self._cross_encoder = None
        self._cross_encoder_model_name = settings.distilled.cross_encoder_model
        self.is_fitted = False

    def _build_pairwise_features(self, query: str, doc_a: str, doc_b: str) -> np.ndarray:
        q_vec, a_vec, b_vec = self.embedder.encode([query, doc_a, doc_b])
        return self._pairwise_features_from_vectors(
            q_vec,
            a_vec,
            b_vec,
            len(self.embedder.tokenize(doc_a)),
            len(self.embedder.tokenize(doc_b)),
        )

    @staticmethod
    def _pairwise_features_from_vectors(
        q_vec: np.ndarray,
        a_vec: np.ndarray,
        b_vec: np.ndarray,
        a_len: int,
        b_len: int,
    ) -> np.ndarray:
        qa_sim = float(np.dot(q_vec, a_vec))
        qb_sim = float(np.dot(q_vec, b_vec))
        return np.asarray(
            [
                qa_sim,
                qb_sim,
                qa_sim - qb_sim,
                float(np.linalg.norm(a_vec - b_vec)),
                float(a_len),
                float(b_len),
                float(a_len - b_len),
            ],
            dtype=np.float32,
        )

    def _fit_pairwise(
        self,
        queries: list[str],
        doc_as: list[str],
        doc_bs: list[str],
        labels: list[int],
    ) -> None:
        samples = [
            self._build_pairwise_features(query, doc_a, doc_b)
            for query, doc_a, doc_b, _ in zip(queries, doc_as, doc_bs, labels, strict=False)
        ]
        if not samples:
            samples = [np.zeros(7, dtype=np.float32)]
            labels = [0]
        X = np.vstack(samples)
        y = np.asarray(labels[: len(samples)], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            self.model = DummyClassifier(strategy="constant", constant=int(y[0]))
            self.model.fit(X, y)
        else:
            self.model = LogisticRegression(
                C=get_settings().distilled.logistic_c,
                max_iter=get_settings().distilled.logistic_max_iter,
                random_state=get_settings().distilled.random_state,
            )
            self.model.fit(X, y)

    def _fit_listwise(
        self,
        queries: list[str],
        doc_as: list[str],
        doc_bs: list[str],
        labels: list[int],
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder, InputExample
            from sentence_transformers.cross_encoder.losses import ListMLELoss
            from torch.utils.data import DataLoader
        except ImportError as e:
            raise ImportError(
                "sentence-transformers with full training dependencies is required "
                "for listwise loss. Please run: pip install sentence-transformers "
                "datasets accelerate transformers[torch]",
            ) from e

        try:
            ce_model = CrossEncoder(
                self._cross_encoder_model_name,
                max_length=512,
            )
            ce_loss = ListMLELoss(ce_model)

            train_examples = []
            for idx, (query, doc_a, doc_b, label) in enumerate(
                zip(queries, doc_as, doc_bs, labels, strict=False)
            ):
                if label == 1:
                    train_examples.append(
                        InputExample(guid=str(idx), texts=[query, doc_a], label=idx)
                    )
                else:
                    train_examples.append(
                        InputExample(guid=str(idx), texts=[query, doc_b], label=idx)
                    )

            if len(train_examples) < 2:
                train_examples = [InputExample(guid="0", texts=[queries[0], doc_as[0]], label=0)]

            train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=32)  # type: ignore[var-annotated, arg-type]

            ce_model.fit(
                train_dataloader=train_dataloader,
                loss_fct=ce_loss,
                epochs=1,
                show_progress_bar=False,
            )
            self._cross_encoder = ce_model
        except ImportError as e:
            raise ImportError(
                f"Failed to train with listwise loss: {e}. "
                "Please ensure accelerate and transformers[torch] are installed."
            ) from e

    def _fit_lambdaloss(
        self,
        queries: list[str],
        doc_as: list[str],
        doc_bs: list[str],
        labels: list[int],
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder, InputExample
            from sentence_transformers.cross_encoder.losses import LambdaLoss
            from torch.utils.data import DataLoader
        except ImportError as e:
            raise ImportError(
                "sentence-transformers with full training dependencies is required "
                "for LambdaLoss. Please run: pip install sentence-transformers "
                "datasets accelerate transformers[torch]",
            ) from e

        try:
            ce_model = CrossEncoder(
                self._cross_encoder_model_name,
                max_length=512,
            )
            ce_loss = LambdaLoss(ce_model)

            train_examples = []
            relevance_labels = []
            for idx, (query, doc_a, doc_b, label) in enumerate(
                zip(queries, doc_as, doc_bs, labels, strict=False)
            ):
                train_examples.append(
                    InputExample(guid=f"{idx}a", texts=[query, doc_a], label=1 if label == 1 else 0)
                )
                train_examples.append(
                    InputExample(guid=f"{idx}b", texts=[query, doc_b], label=0 if label == 1 else 1)
                )
                relevance_labels.append(1 if label == 1 else 0)
                relevance_labels.append(0 if label == 1 else 1)

            if len(set(relevance_labels)) < 2:
                train_examples = [InputExample(guid="0", texts=[queries[0], doc_as[0]], label=1)]
                relevance_labels = [1]

            train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=32)  # type: ignore[var-annotated, arg-type]

            ce_model.fit(
                train_dataloader=train_dataloader,
                loss_fct=ce_loss,
                epochs=1,
                show_progress_bar=False,
            )
            self._cross_encoder = ce_model
        except ImportError as e:
            raise ImportError(
                f"Failed to train with LambdaLoss: {e}. "
                "Please ensure accelerate and transformers[torch] are installed."
            ) from e

    def fit(
        self,
        queries: list[str],
        doc_as: list[str],
        doc_bs: list[str],
        labels: list[int],
    ) -> DistilledPairwiseRanker:
        if self.loss_type == "listwise":
            self._fit_listwise(queries, doc_as, doc_bs, labels)
        elif self.loss_type == "lambdaloss":
            self._fit_lambdaloss(queries, doc_as, doc_bs, labels)
        else:
            self._fit_pairwise(queries, doc_as, doc_bs, labels)
        self.is_fitted = True
        return self

    def _predict_proba(self, features: np.ndarray) -> float:
        probs = self.model.predict_proba([features])
        if probs.ndim == 2 and probs.shape[1] > 1:
            return float(probs[0, 1])
        predicted = int(self.model.predict([features])[0])
        return float(predicted)

    def _score_cross_encoder(self, query: str, docs: list[str]) -> np.ndarray:
        if self._cross_encoder is None:
            return np.zeros(len(docs), dtype=np.float32)
        pairs = [[query, doc] for doc in docs]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    def compare(self, query: str, doc_a: str, doc_b: str) -> float:
        if not self.is_fitted:
            raise RuntimeError("DistilledPairwiseRanker must be fitted before comparison.")
        if self.loss_type in ("listwise", "lambdaloss") and self._cross_encoder is not None:
            scores = self._score_cross_encoder(query, [doc_a, doc_b])
            return float(scores[0] - scores[1] + 0.5)
        features = self._build_pairwise_features(query, doc_a, doc_b)
        return self._predict_proba(features)

    def _merge_rank(
        self,
        query: str,
        indexed_docs: list[tuple[int, str]],
    ) -> list[tuple[int, float]]:
        if len(indexed_docs) <= 1:
            return [(indexed_docs[0][0], 0.0)] if indexed_docs else []

        mid = len(indexed_docs) // 2
        left = self._merge_rank(query, indexed_docs[:mid])
        right = self._merge_rank(query, indexed_docs[mid:])
        left_lookup = dict(indexed_docs[:mid])
        right_lookup = dict(indexed_docs[mid:])
        merged: list[tuple[int, float]] = []
        while left and right:
            left_idx, left_score = left[0]
            right_idx, right_score = right[0]
            prob_left = self.compare(query, left_lookup[left_idx], right_lookup[right_idx])
            if prob_left >= 0.5:
                merged.append((left_idx, left_score + prob_left))
                right[0] = (right_idx, right_score + (1.0 - prob_left))
                left.pop(0)
            else:
                merged.append((right_idx, right_score + (1.0 - prob_left)))
                left[0] = (left_idx, left_score + prob_left)
                right.pop(0)
        merged.extend(left)
        merged.extend(right)
        return merged

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        """Rerank documents using the fitted pairwise model.

        For listwise/lambdaloss cross-encoder models, scores directly.
        For pairwise logistic, runs a full or merge-sort tournament.

        Args:
            query: Search query.
            docs: Documents to rerank.

        Returns:
            Ranked list of RankedDoc.

        Raises:
            RuntimeError: If the ranker has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("DistilledPairwiseRanker must be fitted before reranking.")
        if not docs:
            return []

        if self.loss_type in ("listwise", "lambdaloss") and self._cross_encoder is not None:
            scores = self._score_cross_encoder(query, docs)
            ranked = sorted(
                zip(docs, scores, strict=False),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [
                RankedDoc(
                    doc=doc,
                    score=float(score),
                    rank=rank,
                    metadata={"strategy": f"distilled_{self.loss_type}"},
                )
                for rank, (doc, score) in enumerate(ranked, start=1)
            ]

        q_vec = self.embedder.encode([query])[0]
        d_vecs = self.embedder.encode(docs)
        doc_lens = [len(self.embedder.tokenize(doc)) for doc in docs]
        scores = np.zeros(len(docs), dtype=np.float32)
        if len(docs) <= self.full_tournament_max_docs:
            n = len(docs)
            pair_count = n * (n - 1) // 2
            if pair_count > 0:
                all_features = np.empty((pair_count, 7), dtype=np.float32)
                pair_indices: list[tuple[int, int]] = []
                pos = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        all_features[pos] = self._pairwise_features_from_vectors(
                            q_vec,
                            d_vecs[i],
                            d_vecs[j],
                            doc_lens[i],
                            doc_lens[j],
                        )
                        pair_indices.append((i, j))
                        pos += 1
                probs = self.model.predict_proba(all_features)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    prob_values = probs[:, 1]
                else:
                    prob_values = probs[:, 0]
                for pair_idx, (i, j) in enumerate(pair_indices):
                    prob_a = float(prob_values[pair_idx])
                    scores[i] += prob_a
                    scores[j] += 1.0 - prob_a
        else:
            ranked_indices = self._merge_rank(query, list(enumerate(docs)))
            total = len(ranked_indices)
            for position, (doc_idx, merge_score) in enumerate(ranked_indices):
                scores[doc_idx] = merge_score + float(total - position)
        ranked = sorted(
            zip(docs, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RankedDoc(doc=doc, score=float(score), rank=rank, metadata={"strategy": "distilled"})
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str | Path) -> None:
        """Persist the distilled ranker to disk.

        Saves the logistic model (and cross-encoder if present).

        Args:
            path: Destination file path.
        """
        metadata = {
            "embedder_model_name": self.embedder.model_name,
            "full_tournament_max_docs": self.full_tournament_max_docs,
            "loss_type": self.loss_type,
        }
        weights = {"model": self.model}
        if self._cross_encoder is not None:
            cross_encoder_path = str(Path(path).with_suffix("")) + "_cross_encoder"
            self._cross_encoder.save(cross_encoder_path)
            metadata["cross_encoder_path"] = cross_encoder_path
        save_safe(
            path,
            artifact_type="pairwise_ranker",
            metadata=metadata,
            weights=weights,
        )

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> DistilledPairwiseRanker:
        """Load a saved DistilledPairwiseRanker from disk.

        Args:
            path: Path to the saved artifact.
            embedder: Optional embedder override.

        Returns:
            Loaded DistilledPairwiseRanker instance.
        """
        payload = try_load_safe_or_warn(
            path,
            expected_type="pairwise_ranker",
            legacy_loader=load_pickle,
        )
        loss_type = payload.get("loss_type", "pairwise")
        instance = cls(
            embedder=embedder or Embedder(payload.get("embedder_model_name")),
            loss_type=loss_type,
        )
        instance.model = payload["model"]
        instance.full_tournament_max_docs = int(
            payload.get("full_tournament_max_docs", instance.full_tournament_max_docs)
        )

        cross_encoder_path = payload.get("cross_encoder_path")
        if cross_encoder_path and loss_type in ("listwise", "lambdaloss"):
            try:
                from sentence_transformers import CrossEncoder

                instance._cross_encoder = CrossEncoder(cross_encoder_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load cross-encoder from '%s': %s. "
                    "Pairwise comparisons will fall back to the logistic model.",
                    cross_encoder_path,
                    exc,
                    exc_info=True,
                )
                instance._cross_encoder = None

        instance.is_fitted = True
        return instance

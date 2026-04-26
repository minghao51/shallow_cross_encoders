from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from reranker.config import get_settings
from reranker.data.litellm_client import LiteLLMClient
from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.utils import append_jsonl

logger = logging.getLogger("reranker.data.active_distill")

PROMPT_RELEVANCE = """You are a relevance judge. Given a query and a document, rate relevance 0-3.
0 = Irrelevant | 1 = Tangential | 2 = Relevant | 3 = Highly Relevant

Return ONLY valid JSON: {{"score": <int>, "rationale": "<one sentence>"}}

Query: {query}
Document: {doc}"""

PROMPT_PREFERENCE = """Given a query, decide which document better answers it.
Return ONLY valid JSON: {{"preferred": "A" or "B", "confidence": 0.0-1.0}}

Query: {query}
Document A: {doc_a}
Document B: {doc_b}"""


@dataclass(slots=True)
class ActiveDistillationResult:
    new_pairs: list[dict[str, Any]]
    new_preferences: list[dict[str, Any]]
    total_api_calls: int
    total_cost_estimate: float


class ActiveDistiller:
    def __init__(
        self,
        embedder: Embedder | None = None,
        client: LiteLLMClient | None = None,
    ) -> None:
        settings = get_settings().active_distillation
        self.embedder = embedder or Embedder()
        self.client = client or LiteLLMClient()
        self.mining_strategy = settings.mining_strategy
        self.active_iterations = settings.active_iterations
        self.uncertainty_low = settings.uncertainty_low
        self.uncertainty_high = settings.uncertainty_high
        self.contested_rank_gap = settings.contested_rank_gap
        self.diversity_clusters = settings.diversity_clusters
        self._seen_pairs: set[tuple[str, str]] = set()

    def mine_contested(
        self, queries: list[str], docs_list: list[list[str]]
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for query, docs in zip(queries, docs_list, strict=False):
            if len(docs) < 2:
                continue
            bm25 = BM25Engine(tokenize_fn=self.embedder.tokenize)
            bm25.fit(docs)
            bm25_scores = bm25.score(query)
            bm25_ranks = np.argsort(-bm25_scores)

            q_vec = self.embedder.encode([query])[0]
            d_vecs = self.embedder.encode(docs)
            sem_scores = d_vecs @ q_vec
            sem_ranks = np.argsort(-sem_scores)

            rank_diff = np.zeros(len(docs), dtype=np.int32)
            for pos, idx in enumerate(bm25_ranks):
                rank_diff[idx] = pos

            for pos, idx in enumerate(sem_ranks):
                diff = abs(rank_diff[idx] - pos)
                if diff > self.contested_rank_gap * min(len(docs), 100) // 100:
                    candidates.append((query, docs[idx]))

        return candidates

    def mine_max_entropy(
        self,
        queries: list[str],
        docs_list: list[list[str]],
        model_predict_fn: Any = None,
    ) -> list[tuple[str, str]]:
        if model_predict_fn is None:
            return self.mine_contested(queries, docs_list)

        candidates: list[tuple[str, str]] = []
        for query, docs in zip(queries, docs_list, strict=False):
            for doc in docs:
                try:
                    prob = model_predict_fn(query, doc)
                    if self.uncertainty_low <= prob <= self.uncertainty_high:
                        candidates.append((query, doc))
                except Exception:
                    continue
        return candidates

    def mine_diversity(
        self,
        queries: list[str],
        docs_list: list[list[str]],
    ) -> list[tuple[str, str]]:
        all_pairs: list[tuple[str, str]] = []
        for query, docs in zip(queries, docs_list, strict=False):
            for doc in docs:
                all_pairs.append((query, doc))

        if not all_pairs:
            return []

        texts = [f"{query}\n{doc}" for query, doc in all_pairs]
        vecs = self.embedder.encode(texts)
        n_clusters = min(self.diversity_clusters, len(all_pairs))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        labels = kmeans.fit_predict(vecs)

        selected: list[tuple[str, str]] = []
        for cluster_id in range(n_clusters):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) == 0:
                continue
            center = kmeans.cluster_centers_[cluster_id]
            dists = np.linalg.norm(vecs[indices] - center, axis=1)
            best = indices[np.argmin(dists)]
            selected.append(all_pairs[best])

        return selected

    def label_with_teacher(
        self, pairs: list[tuple[str, str]], cost_log_path: Path | None = None
    ) -> list[dict[str, Any]]:
        if not self.client.enabled:
            logger.warning("LiteLLM client not enabled; returning empty labels")
            return []

        results: list[dict[str, Any]] = []
        for query, doc in pairs:
            pair = (query, doc)
            if pair in self._seen_pairs:
                continue
            prompt = PROMPT_RELEVANCE.format(query=query, doc=doc)
            try:
                parsed, metadata = self.client.complete_json(prompt)
                record = {
                    "query": query,
                    "doc": doc,
                    "score": int(parsed.get("score", 0)),
                    "rationale": parsed.get("rationale", ""),
                    **metadata,
                }
                results.append(record)
                self._seen_pairs.add(pair)
                if cost_log_path is not None:
                    append_jsonl(cost_log_path, record)
            except Exception as e:
                logger.warning("Teacher labeling failed: %s", e)
                continue
        return results

    def run(
        self,
        queries: list[str],
        docs_list: list[list[str]],
        model_predict_fn: Any = None,
        cost_log_path: Path | None = None,
    ) -> ActiveDistillationResult:
        settings = get_settings().active_distillation
        all_pairs: list[dict[str, Any]] = []
        all_prefs: list[dict[str, Any]] = []
        total_calls = 0

        for iteration in range(settings.active_iterations):
            logger.info(
                "Active distillation iteration %d/%d",
                iteration + 1,
                settings.active_iterations,
            )

            if self.mining_strategy == "contested":
                mined = self.mine_contested(queries, docs_list)
            elif self.mining_strategy == "max_entropy":
                mined = self.mine_max_entropy(queries, docs_list, model_predict_fn)
            elif self.mining_strategy == "diversity":
                mined = self.mine_diversity(queries, docs_list)
            else:
                mined = self.mine_contested(queries, docs_list)

            logger.info("Mined %d candidate pairs", len(mined))

            mined = [pair for pair in mined if pair not in self._seen_pairs]

            if not mined:
                break

            labeled = self.label_with_teacher(mined, cost_log_path)
            total_calls += len(labeled)
            all_pairs.extend(labeled)

        estimated_cost = total_calls * 0.0004

        return ActiveDistillationResult(
            new_pairs=all_pairs,
            new_preferences=all_prefs,
            total_api_calls=total_calls,
            total_cost_estimate=estimated_cost,
        )

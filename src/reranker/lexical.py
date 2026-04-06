from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np

from reranker.deps import check_rank_bm25


class BM25Engine:
    """BM25 wrapper with a pure-Python fallback when rank_bm25 is unavailable."""

    def __init__(self, tokenize_fn: Callable[[str], list[str]] | None = None) -> None:
        self._corpus: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None
        self._doc_freqs: Counter[str] = Counter()
        self._avgdl = 0.0
        self.backend_name = "pure_python"
        self._tokenize_fn = tokenize_fn or (lambda text: text.lower().split())

    def fit(self, corpus: list[str]) -> None:
        self._corpus = corpus
        self._tokenized = [self._tokenize_fn(doc) for doc in corpus]
        self._avgdl = float(sum(len(tokens) for tokens in self._tokenized)) / max(
            len(self._tokenized), 1
        )
        self._doc_freqs = Counter()
        for tokens in self._tokenized:
            self._doc_freqs.update(set(tokens))
        bm25_cls, status = check_rank_bm25()
        if bm25_cls is not None:
            self._bm25 = bm25_cls(self._tokenized)
            self.backend_name = status.backend
        else:
            self._bm25 = None
            self.backend_name = status.backend

    def _fallback_scores(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize_fn(query)
        n_docs = len(self._tokenized)
        scores = np.zeros(n_docs, dtype=np.float32)
        k1 = 1.5
        b = 0.75
        for idx, tokens in enumerate(self._tokenized):
            tf = Counter(tokens)
            doc_len = len(tokens) or 1
            score = 0.0
            for token in query_tokens:
                df = self._doc_freqs.get(token, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                freq = tf.get(token, 0)
                denom = freq + k1 * (1 - b + b * doc_len / max(self._avgdl, 1.0))
                if denom > 0:
                    score += idf * (freq * (k1 + 1)) / denom
            scores[idx] = score
        return scores

    def score(self, query: str, normalize: bool = True) -> np.ndarray:
        if not self._corpus:
            return np.zeros(0, dtype=np.float32)
        scores = (
            np.asarray(self._bm25.get_scores(self._tokenize_fn(query)), dtype=np.float32)
            if self._bm25 is not None
            else self._fallback_scores(query)
        )
        if not scores.size:
            return scores
        if float(scores.max()) <= 0.0:
            scores = self._fallback_scores(query)
        scores = np.maximum(scores, 0.0)
        if normalize and scores.size and float(scores.max()) > 0:
            scores = scores / float(scores.max())
        return scores

    def rerank(self, query: str, docs: list[str]) -> list:
        from reranker.protocols import RankedDoc

        self.fit(docs)
        scores = self.score(query)
        ranked = sorted(
            zip(docs, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RankedDoc(doc=doc, score=float(score), rank=rank, metadata={"strategy": "bm25"})
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]

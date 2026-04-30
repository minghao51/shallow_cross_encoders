"""BM25 lexical scoring engine.

Wraps rank_bm25 when available with a pure-Python BM25Okapi fallback
so that lexical scoring always works, even without optional dependencies.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np

from reranker.deps import check_rank_bm25


class BM25Engine:
    """BM25 wrapper with a pure-Python fallback when rank_bm25 is unavailable."""

    def __init__(self, tokenize_fn: Callable[[str], list[str]] | None = None) -> None:
        """Initialize the BM25 engine.

        Args:
            tokenize_fn: Custom tokenizer. Defaults to whitespace splitting.
        """
        self._corpus: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None
        self._doc_freqs: Counter[str] = Counter()
        self._avgdl = 0.0
        self.backend_name = "pure_python"
        self._tokenize_fn = tokenize_fn or (lambda text: text.lower().split())

    def fit(self, corpus: list[str]) -> None:
        """Index a corpus for BM25 scoring.

        Args:
            corpus: List of document strings to index.
        """
        self._corpus = corpus
        self._tokenized = [self._tokenize_fn(doc) for doc in corpus]
        self._avgdl = float(sum(len(tokens) for tokens in self._tokenized)) / max(
            len(self._tokenized), 1
        )
        self._doc_freqs = Counter()
        for tokens in self._tokenized:
            self._doc_freqs.update(set(tokens))
        bm25_cls, status = check_rank_bm25()
        if bm25_cls is not None and self._tokenized:
            self._bm25 = bm25_cls(self._tokenized)
            self.backend_name = status.backend
        else:
            self._bm25 = None
            self.backend_name = "pure_python"

    def _fallback_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores using pure-Python implementation.

        Uses BM25Okapi formulation with k1=1.5, b=0.75.
        Used when rank_bm25 is not available.
        """
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
        """Compute BM25 scores for all indexed documents.

        Args:
            query: Search query string.
            normalize: Whether to L2-normalize scores. Defaults to True.

        Returns:
            Array of BM25 scores, one per document in the corpus.
        """
        if not self._corpus:
            return np.zeros(0, dtype=np.float32)
        scores = (
            np.asarray(self._bm25.get_scores(self._tokenize_fn(query)), dtype=np.float32)  # type: ignore[attr-defined]
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
        """Score and rank documents by BM25 relevance.

        Implements the BaseReranker protocol for drop-in compatibility.

        Args:
            query: Search query string.
            docs: Documents to rank.

        Returns:
            List of RankedDoc sorted by BM25 score descending.
        """
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

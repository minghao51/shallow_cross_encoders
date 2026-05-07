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

    def _rebuild_bm25(self) -> None:
        """Rebuild the rank_bm25 backend from current tokenized corpus."""
        bm25_cls, _ = check_rank_bm25()
        if bm25_cls is not None and self._tokenized:
            self._bm25 = bm25_cls(self._tokenized)
            self.backend_name = "rank_bm25"
        else:
            self._bm25 = None
            self.backend_name = "pure_python"

    def update(self, docs: list[str]) -> None:
        """Incrementally add documents without full rebuild.

        Appends new documents to the existing index, updating document
        frequencies and average document length incrementally. Faster than
        calling ``fit()`` again when only adding a few documents.

        Args:
            docs: Document strings to add to the index.
        """
        if not docs:
            return
        new_tokenized = [self._tokenize_fn(doc) for doc in docs]
        new_total_len = sum(len(tokens) for tokens in new_tokenized)
        old_total_len = int(self._avgdl * len(self._tokenized)) if self._tokenized else 0
        self._corpus.extend(docs)
        self._tokenized.extend(new_tokenized)
        n = len(self._tokenized)
        self._avgdl = (old_total_len + new_total_len) / max(n, 1)
        for tokens in new_tokenized:
            self._doc_freqs.update(set(tokens))
        self._rebuild_bm25()

    def remove(self, doc_ids: list[int]) -> None:
        """Remove documents by index from the index.

        Removes documents at the specified indices, updating all internal
        statistics. Indices refer to the *current* corpus ordering.

        Args:
            doc_ids: 0-based indices of documents to remove.

        Raises:
            IndexError: If any index is out of range.
        """
        if not doc_ids:
            return
        n = len(self._corpus)
        for idx in doc_ids:
            if idx < 0 or idx >= n:
                raise IndexError(f"doc_id {idx} out of range [0, {n})")
        remove_set = set(doc_ids)
        surviving_corpus = [d for i, d in enumerate(self._corpus) if i not in remove_set]
        surviving_tokenized = [t for i, t in enumerate(self._tokenized) if i not in remove_set]
        self._corpus = surviving_corpus
        self._tokenized = surviving_tokenized
        self._doc_freqs = Counter()
        for tokens in self._tokenized:
            self._doc_freqs.update(set(tokens))
        n = len(self._tokenized)
        self._avgdl = float(sum(len(t) for t in self._tokenized)) / max(n, 1)
        self._rebuild_bm25()

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

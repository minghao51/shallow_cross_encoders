from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reranker.config import get_settings


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    cleaned = text.lower().strip()
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def _minhash_signature(ngrams: set[str], num_perm: int = 128) -> np.ndarray:
    if not ngrams:
        return np.zeros(num_perm, dtype=np.int64)
    signature = np.full(num_perm, np.iinfo(np.int64).max, dtype=np.int64)
    for gram in ngrams:
        for i in range(num_perm):
            h = int(hashlib.md5(f"{gram}|{i}".encode()).hexdigest(), 16)
            signature[i] = min(signature[i], h)
    return signature


def _jaccard_from_signatures(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    if sig_a.shape[0] == 0 or sig_b.shape[0] == 0:
        return 0.0
    return float(np.mean(sig_a == sig_b))


@dataclass(slots=True)
class LSHAdapter:
    """HeuristicAdapter implementing MinHash-based fuzzy matching for typo rescue."""

    ngram_size: int = 3
    num_perm: int = 128
    tokenize_fn: Callable[[str], list[str]] | None = None

    def __post_init__(self) -> None:
        settings = get_settings().lsh
        if self.ngram_size == 3 and self.num_perm == 128:
            self.ngram_size = settings.ngram_size
            self.num_perm = settings.num_perm

    def compute(self, query: str, doc: str) -> dict[str, float]:
        q_ngrams = _char_ngrams(query, self.ngram_size)
        d_ngrams = _char_ngrams(doc, self.ngram_size)
        if not q_ngrams or not d_ngrams:
            return {"lsh_score": 0.0, "lsh_jaccard": 0.0}

        q_sig = _minhash_signature(q_ngrams, self.num_perm)
        d_sig = _minhash_signature(d_ngrams, self.num_perm)
        approx_jaccard = _jaccard_from_signatures(q_sig, d_sig)
        exact_jaccard = len(q_ngrams & d_ngrams) / max(len(q_ngrams | d_ngrams), 1)

        return {
            "lsh_score": approx_jaccard,
            "lsh_jaccard": exact_jaccard,
        }

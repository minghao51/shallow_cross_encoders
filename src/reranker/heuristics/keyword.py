"""Keyword hit-rate heuristic adapter for hybrid reranking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class KeywordMatchAdapter:
    """HeuristicAdapter that emits a simple term hit-rate signal."""

    tokenize_fn: Callable[[str], list[str]] | None = None

    def compute(self, query: str, doc: str) -> dict[str, float]:
        """Compute keyword hit-rate between query and document.

        Args:
            query: Query text.
            doc: Document text.

        Returns:
            Dict with "keyword_hit_rate" (fraction of query terms in doc).
        """
        tokenize = self.tokenize_fn or (lambda t: t.lower().split())
        terms = tokenize(query.lower())
        doc_lower = doc.lower()
        hit_rate = sum(1 for term in terms if term in doc_lower) / max(len(terms), 1)
        return {"keyword_hit_rate": hit_rate}

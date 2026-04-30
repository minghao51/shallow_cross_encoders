"""Consistency engine that detects contradictions across document claims.

Extracts structured claims from documents using pattern matching, then
compares semantically aligned claims to find contradictions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.protocols import RankedDoc
from reranker.strategies.patterns import (
    ENTITY_PATTERN,
    SENTENCE_SPLIT_PATTERN,
    STRUCTURED_PATTERN_HINTS,
    STRUCTURED_PATTERNS,
    VALUE_PATTERN,
)


class Claim(BaseModel):
    entity: str = Field(description="The subject entity this claim is about")
    attribute: str = Field(description="The attribute or field being stated")
    value: Any = Field(description="The stated value")
    source_doc_id: str


class ClaimSet(BaseModel):
    claims: list[Claim]


@dataclass(slots=True)
class Contradiction:
    claim_a: Claim
    claim_b: Claim
    reason: str


class ConsistencyEngine:
    """Detect contradictions across semantically aligned claims."""

    _ENTITY_PATTERN = ENTITY_PATTERN
    _VALUE_PATTERN = VALUE_PATTERN
    _SENTENCE_SPLIT_PATTERN = SENTENCE_SPLIT_PATTERN

    def __init__(
        self,
        sim_threshold: float | None = None,
        value_tolerance: float | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or Embedder()
        self.sim_threshold = (
            settings.consistency.sim_threshold if sim_threshold is None else sim_threshold
        )
        self.value_tolerance = (
            settings.consistency.value_tolerance if value_tolerance is None else value_tolerance
        )
        self._structured_patterns = STRUCTURED_PATTERNS
        self._structured_pattern_hints = STRUCTURED_PATTERN_HINTS

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _normalize_entity(cls, value: str) -> str:
        cleaned = cls._normalize_whitespace(value.strip(" .,:;()[]{}\"'"))
        narrative_prefix = (
            r"^(?:the|contrarily|in reality|recent tests show that|however|overall|"
            r"in contrast|recent evaluations indicate that|latest tests indicate that|"
            r"recent evaluations show that|recent tests indicate that|latest results reveal that|"
            r"testing shows that|testing of|new benchmarks show that|in fact|"
            r"contrary to popular belief|on the other hand|it has been reported that|"
            r"the latest results indicate that|the latest tests indicate that|"
            r"latest results indicate that|latest updates indicate that|"
            r"new data reveals that|some reports indicate that|reports indicate that|"
            r"recent updates indicate that|despite this|contradicting this|"
            r"on the contrary|in testing|in tests|yet)\s+"
        )
        while True:
            updated = re.sub(narrative_prefix, "", cleaned, flags=re.IGNORECASE)
            if updated == cleaned:
                break
            cleaned = updated
        cleaned = re.sub(r"\s+(?:is|has|was|will|again)\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+actually$", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @classmethod
    def _normalize_attribute(cls, value: str) -> str:
        cleaned = cls._normalize_whitespace(value.strip(" .,:;()[]{}\"'")).lower()
        cleaned = re.sub(r"[\s\-]+", "_", cleaned)
        cleaned = re.sub(r"^(?:a|an|the)_", "", cleaned)
        cleaned = re.sub(r"^(?:actual)_", "", cleaned)
        cleaned = re.sub(r"^(?:low|high)_", "", cleaned)
        cleaned = re.sub(
            r"^(?:recent_tests_show_that|latest_results_indicate_that|latest_tests_indicate_that)_",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^(?:the_)?", "", cleaned)
        if "latency" in cleaned:
            return "latency"
        if cleaned == "release":
            return "release_year"
        return cleaned.strip("_")

    @classmethod
    def _normalize_value(cls, value: str) -> str:
        cleaned = cls._normalize_whitespace(value.strip(" .,:;()[]{}\"'"))
        cleaned = re.sub(
            r"^(?:reported|confirmed|measured|recorded)\s+(?:to\s+be|as|at)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:only|just|actual(?:ly)?|approximately|around)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:an?|the)\s+impressive\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^now\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\s+(?:in|during|which|making|ensuring|indicating)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned

    def _claim_from_match(self, match: re.Match[str], source_doc_id: str) -> Claim | None:
        entity = self._normalize_entity(match.group("entity"))
        attribute = self._normalize_attribute(match.groupdict().get("attribute", ""))
        value = self._normalize_value(match.group("value"))
        if not attribute:
            lowered = match.group(0).lower()
            if "release" in lowered:
                attribute = "release_year"
            elif "screening status" in lowered:
                attribute = "screening_status"
            elif "best metric" in lowered:
                attribute = "best_metric"
        if attribute in {"which", "that"}:
            return None
        if not entity or not attribute or not value:
            return None
        return Claim(
            entity=entity,
            attribute=attribute,
            value=value,
            source_doc_id=source_doc_id,
        )

    def _extract_structured_claims(self, doc: str, source_doc_id: str) -> list[Claim]:
        claims: list[Claim] = []
        seen: set[tuple[str, str, str]] = set()
        segments = [
            segment.strip()
            for segment in self._SENTENCE_SPLIT_PATTERN.split(doc)
            if segment.strip()
        ]
        for segment in segments:
            segment_lower = f" {segment.lower()} "
            candidate_patterns = [
                pattern
                for hints, pattern in zip(
                    self._structured_pattern_hints,
                    self._structured_patterns,
                    strict=False,
                )
                if any(hint in segment_lower for hint in hints)
            ]
            for pattern in candidate_patterns or self._structured_patterns:
                for match in pattern.finditer(segment):
                    claim = self._claim_from_match(match, source_doc_id)
                    if claim is None:
                        continue
                    key = (claim.entity.lower(), claim.attribute, str(claim.value).lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    claims.append(claim)
        return claims

    def extract_claims(self, docs: list[str], doc_ids: list[str] | None = None) -> list[ClaimSet]:
        """Extract structured claims from documents.

        Uses pattern matching to identify entity-attribute-value triples.

        Args:
            docs: List of document strings.
            doc_ids: Optional list of document IDs; defaults to doc_0, doc_1, ...

        Returns:
            List of ClaimSets, one per document.
        """
        doc_ids = doc_ids or [f"doc_{idx}" for idx in range(len(docs))]
        sets: list[ClaimSet] = []
        pattern = re.compile(
            r"(?P<entity>[A-Z][A-Za-z0-9\-\s]+?)\s+(?:reports|lists|states|shows)\s+"
            r"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]+?)\s+(?:as|is)\s+(?P<value>[A-Za-z0-9@._%-]+)"
        )
        for doc, source_doc_id in zip(docs, doc_ids, strict=False):
            claims = self._extract_structured_claims(doc, source_doc_id)
            seen = {
                (claim.entity.lower(), claim.attribute, str(claim.value).lower())
                for claim in claims
            }
            for match in pattern.finditer(doc):
                claim = self._claim_from_match(match, source_doc_id)
                if claim is None:
                    continue
                key = (claim.entity.lower(), claim.attribute, str(claim.value).lower())
                if key not in seen:
                    seen.add(key)
                    claims.append(claim)
            if not claims:
                claims.append(
                    Claim(
                        entity=source_doc_id,
                        attribute="raw_text",
                        value=doc.strip(),
                        source_doc_id=source_doc_id,
                    )
                )
            sets.append(ClaimSet(claims=claims))
        return sets

    _VALUE_SYNONYMS: dict[str, str] = {
        "yes": "true",
        "no": "false",
        "enabled": "true",
        "disabled": "false",
        "on": "true",
        "off": "false",
        "active": "true",
        "inactive": "false",
        "approved": "true",
        "rejected": "false",
        "pending": "unknown",
    }
    _UNIT_MAP: dict[str, str] = {
        "ms": "",
        "milliseconds": "",
        "millisecond": "",
        "s": "",
        "seconds": "",
        "second": "",
        "mb": "",
        "gb": "",
        "tb": "",
        "kb": "",
        "bytes": "",
        "byte": "",
        "%": "",
        "percent": "",
    }

    @classmethod
    def _strip_units(cls, value: str) -> str:
        cleaned = value.strip().lower()
        for unit in sorted(cls._UNIT_MAP, key=len, reverse=True):
            cleaned = re.sub(rf"\b{re.escape(unit)}\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @classmethod
    def _canonical_value(cls, value: Any) -> str:
        raw = str(value).strip().lower()
        synonym = cls._VALUE_SYNONYMS.get(raw)
        if synonym is not None:
            return synonym
        return cls._strip_units(raw)

    def _values_conflict(self, value_a: Any, value_b: Any) -> bool:
        try:
            return abs(float(value_a) - float(value_b)) > self.value_tolerance
        except (TypeError, ValueError):
            canon_a = self._canonical_value(value_a)
            canon_b = self._canonical_value(value_b)
            if canon_a == canon_b:
                return False
            try:
                return abs(float(canon_a) - float(canon_b)) > self.value_tolerance
            except (TypeError, ValueError):
                return canon_a != canon_b

    def check(self, claim_sets: list[ClaimSet]) -> list[Contradiction]:
        """Check claim sets for contradictions.

        Compares claims with the same entity and attribute (fast path)
        and semantically similar attributes (fuzzy path), flagging
        conflicting values.

        Args:
            claim_sets: ClaimSets to check for contradictions.

        Returns:
            List of Contradictions found.
        """
        all_claims = [claim for claim_set in claim_sets for claim in claim_set.claims]
        if len(all_claims) < 2:
            return []
        contradictions: list[Contradiction] = []
        structured_groups: dict[str, list[Claim]] = {}
        fuzzy_claims: list[Claim] = []
        for claim in all_claims:
            if claim.attribute != "raw_text":
                key = claim.entity.strip().lower()
                structured_groups.setdefault(key, []).append(claim)
            else:
                fuzzy_claims.append(claim)

        for claims in structured_groups.values():
            if len(claims) < 2:
                continue
            threshold = 1.0 - self.sim_threshold

            # Phase 1: Group by exact attribute for fast path
            by_attr: dict[str, list[int]] = {}
            for idx, claim in enumerate(claims):
                by_attr.setdefault(claim.attribute, []).append(idx)

            # Fast path: compare within same-attribute groups
            for indices in by_attr.values():
                for a_pos in range(len(indices)):
                    for b_pos in range(a_pos + 1, len(indices)):
                        i, j = indices[a_pos], indices[b_pos]
                        if claims[i].source_doc_id == claims[j].source_doc_id:
                            continue
                        if self._values_conflict(claims[i].value, claims[j].value):
                            contradictions.append(
                                Contradiction(
                                    claim_a=claims[i],
                                    claim_b=claims[j],
                                    reason=(
                                        "Structured claims report conflicting values for the same "
                                        "entity and attribute."
                                    ),
                                )
                            )

            # Fuzzy path: compare across semantically similar attribute groups
            unique_attrs = list(by_attr.keys())
            if len(unique_attrs) > 1:
                attr_vectors = self.embedder.encode(unique_attrs)
                attr_dists = cdist(attr_vectors, attr_vectors, metric="cosine")
                for ai in range(len(unique_attrs)):
                    for aj in range(ai + 1, len(unique_attrs)):
                        if attr_dists[ai, aj] > threshold:
                            continue
                        indices_a = by_attr[unique_attrs[ai]]
                        indices_b = by_attr[unique_attrs[aj]]
                        for i in indices_a:
                            for j in indices_b:
                                if claims[i].source_doc_id == claims[j].source_doc_id:
                                    continue
                                if self._values_conflict(claims[i].value, claims[j].value):
                                    contradictions.append(
                                        Contradiction(
                                            claim_a=claims[i],
                                            claim_b=claims[j],
                                            reason=(
                                                "Structured claims report conflicting values "
                                                "for the same entity and attribute."
                                            ),
                                        )
                                    )

        if len(fuzzy_claims) < 2:
            return contradictions

        texts = [f"{claim.entity} {claim.attribute}" for claim in fuzzy_claims]
        vectors = self.embedder.encode(texts)
        dist_matrix = cdist(vectors, vectors, metric="cosine")
        threshold = 1.0 - self.sim_threshold
        for i in range(len(fuzzy_claims)):
            for j in range(i + 1, len(fuzzy_claims)):
                if dist_matrix[i, j] > threshold:
                    continue
                if fuzzy_claims[i].source_doc_id == fuzzy_claims[j].source_doc_id:
                    continue
                if self._values_conflict(fuzzy_claims[i].value, fuzzy_claims[j].value):
                    contradictions.append(
                        Contradiction(
                            claim_a=fuzzy_claims[i],
                            claim_b=fuzzy_claims[j],
                            reason="Semantically aligned claims report conflicting values.",
                        )
                    )
        return contradictions

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        """Rerank documents by contradiction penalty.

        Documents with more contradictions receive lower scores.

        Args:
            query: Search query (unused, kept for interface compatibility).
            docs: Documents to rerank.

        Returns:
            Ranked list of RankedDoc.
        """
        claim_sets = self.extract_claims(docs)
        contradictions = self.check(claim_sets)
        penalties = {f"doc_{idx}": 0.0 for idx in range(len(docs))}
        for contradiction in contradictions:
            penalties[contradiction.claim_a.source_doc_id] += 1.0
            penalties[contradiction.claim_b.source_doc_id] += 1.0
        ranked = sorted(
            enumerate(docs),
            key=lambda item: penalties[f"doc_{item[0]}"],
        )
        return [
            RankedDoc(
                doc=doc,
                score=float(-penalties[f"doc_{idx}"]),
                rank=rank,
                metadata={
                    "strategy": "consistency",
                    "contradiction_penalty": penalties[f"doc_{idx}"],
                },
            )
            for rank, (idx, doc) in enumerate(ranked, start=1)
        ]

    def diagnose_misses(self, contradictions_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze which contradictions were missed and why.

        Args:
            contradictions_data: List of dicts with keys:
                subject, doc_a, doc_b, is_contradiction, contradicted_field,
                value_a, value_b.

        Returns:
            List of dicts with diagnosis info for each contradiction.
        """
        results: list[dict[str, Any]] = []
        for item in contradictions_data:
            doc_a = str(item["doc_a"])
            doc_b = str(item["doc_b"])
            expected = bool(item.get("is_contradiction", True))
            claim_sets = self.extract_claims([doc_a, doc_b], ["doc_a", "doc_b"])
            detected = self.check(claim_sets)
            was_detected = len(detected) > 0

            claims_a = [c for c in claim_sets[0].claims if c.attribute != "raw_text"]
            claims_b = [c for c in claim_sets[1].claims if c.attribute != "raw_text"]
            has_structured_a = len(claims_a) > 0
            has_structured_b = len(claims_b) > 0

            field = str(item.get("contradicted_field", ""))
            value_a = str(item.get("value_a", ""))
            value_b = str(item.get("value_b", ""))

            miss_reason = None
            if expected and not was_detected:
                if not has_structured_a and not has_structured_b:
                    miss_reason = "no_structured_claims_extracted"
                elif not has_structured_a:
                    miss_reason = "no_claims_from_doc_a"
                elif not has_structured_b:
                    miss_reason = "no_claims_from_doc_b"
                else:
                    attr_match = False
                    for ca in claims_a:
                        for cb in claims_b:
                            if (
                                ca.attribute == cb.attribute
                                or ca.attribute in cb.attribute
                                or cb.attribute in ca.attribute
                            ):
                                attr_match = True
                                if not self._values_conflict(ca.value, cb.value):
                                    miss_reason = (
                                        f"values_did_not_conflict({ca.value!r} vs {cb.value!r})"
                                    )
                                break
                        if attr_match:
                            break
                    if not attr_match:
                        miss_reason = "no_matching_attributes_between_docs"
            elif not expected and was_detected:
                miss_reason = "false_positive_detected"

            if miss_reason or (expected != was_detected):
                results.append(
                    {
                        "subject": item.get("subject", ""),
                        "is_contradiction": expected,
                        "was_detected": was_detected,
                        "miss_reason": miss_reason,
                        "contradicted_field": field,
                        "value_a": value_a,
                        "value_b": value_b,
                        "claims_doc_a": len(claims_a),
                        "claims_doc_b": len(claims_b),
                        "doc_a_snippet": doc_a[:120],
                        "doc_b_snippet": doc_b[:120],
                    }
                )
        return results

    def save(self, path: str | Path) -> None:
        """Persist the consistency engine to disk.

        Args:
            path: Destination file path.
        """
        save_safe(
            path,
            artifact_type="consistency_engine",
            metadata={
                "sim_threshold": self.sim_threshold,
                "value_tolerance": self.value_tolerance,
                "embedder_model_name": self.embedder.model_name,
            },
            weights={},
        )

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> ConsistencyEngine:
        """Load a saved ConsistencyEngine from disk.

        Args:
            path: Path to the saved artifact.
            embedder: Optional embedder override.

        Returns:
            Loaded ConsistencyEngine instance.
        """
        payload = try_load_safe_or_warn(
            path,
            expected_type="consistency_engine",
            legacy_loader=_legacy_load,
        )
        return cls(
            sim_threshold=payload["sim_threshold"],
            value_tolerance=payload["value_tolerance"],
            embedder=embedder or Embedder(payload["embedder_model_name"]),
        )


def _legacy_load(path: Path) -> dict[str, Any]:
    from reranker.utils import load_pickle

    return load_pickle(path)

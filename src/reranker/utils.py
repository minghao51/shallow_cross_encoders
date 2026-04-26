from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1

try:
    import cloudpickle  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cloudpickle = None


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_parent(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_pickle(path: str | Path, obj: Any) -> None:
    target = ensure_parent(path)
    with target.open("wb") as handle:
        if cloudpickle is not None:
            cloudpickle.dump(obj, handle)
            return
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    # SECURITY NOTE: pickle can execute arbitrary code on deserialization.
    # Only load pickles from trusted sources. Consider using joblib for sklearn models
    # or implementing a schema-validated alternative for untrusted input.
    import warnings

    warnings.warn(
        f"Loading pickle from '{path}' can execute arbitrary code. "
        "Only load artifacts from trusted sources.",
        RuntimeWarning,
        stacklevel=2,
    )
    with Path(path).open("rb") as handle:
        if cloudpickle is not None:
            return cloudpickle.load(handle)
        return pickle.load(handle)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_serializable(v) for v in value]
    return value


def build_artifact_metadata(
    artifact_type: str,
    *,
    format_name: str,
    embedder_model_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": artifact_type,
        "format": format_name,
    }
    if embedder_model_name is not None:
        payload["embedder_model_name"] = embedder_model_name
    if extra:
        payload.update(extra)
    return payload


def validate_artifact_metadata(
    payload: dict[str, Any],
    *,
    expected_type: str,
    expected_formats: set[str],
) -> None:
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported artifact version: {payload.get('artifact_version')!r}.")
    if payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"Unexpected artifact type: {payload.get('artifact_type')!r}, "
            f"expected {expected_type!r}."
        )
    if payload.get("format") not in expected_formats:
        raise ValueError(
            f"Unexpected artifact format: {payload.get('format')!r}, "
            f"expected one of {sorted(expected_formats)!r}."
        )


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i) where rank_i is the position in list i.

    Args:
        ranked_lists: List of ranked documents, each as [(doc_id, score), ...]
                      Documents are sorted by score descending (rank 0 = best).
        k: Constant that controls how much low-ranked documents are penalized.
           Higher k = more weight to high ranks. Default: 60 (from Hakuinn et al.)

    Returns:
        Fused ranked list as [(doc_id, rrf_score), ...] sorted by RRF score descending.

    Example:
        >>> list1 = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
        >>> list2 = [("B", 1.0), ("C", 0.9), ("D", 0.8)]
        >>> fused = reciprocal_rank_fusion([list1, list2], k=60)
        >>> # B appears in both lists and will rank high
    """
    if not ranked_lists:
        return []

    all_doc_ids: set[str] = set()
    for ranked_list in ranked_lists:
        for doc_id, _ in ranked_list:
            all_doc_ids.add(doc_id)

    rrf_scores: dict[str, float] = {doc_id: 0.0 for doc_id in all_doc_ids}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return fused


def rrf_from_scores(score_arrays: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Apply RRF fusion directly from score arrays.

    Args:
        score_arrays: List of score arrays, one per ranker.
                      Each array has scores for the same documents in the same order.
        k: RRF constant. Default: 60.

    Returns:
        Fused scores array with same length as input score arrays.
    """
    if not score_arrays:
        return np.array([])

    n_docs = len(score_arrays[0])
    doc_ids = list(range(n_docs))

    ranked_lists: list[list[tuple[int, float]]] = []
    for scores in score_arrays:
        ranked = sorted(doc_ids, key=lambda i: scores[i], reverse=True)
        ranked_lists.append([(doc_id, scores[doc_id]) for doc_id in ranked])

    str_ranked_lists = [[(f"doc_{i}", s) for i, s in ranked_list] for ranked_list in ranked_lists]

    fused = reciprocal_rank_fusion(str_ranked_lists, k=k)

    fused_scores = np.zeros(n_docs, dtype=np.float32)
    str_to_int = {f"doc_{i}": i for i in doc_ids}

    for doc_id_str, score in fused:
        fused_scores[str_to_int[doc_id_str]] = score

    return fused_scores

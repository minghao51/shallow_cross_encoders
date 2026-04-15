from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from reranker.config import get_settings
from reranker.data.synth.generator.types import (
    BatchGenerator,
    GeneratorState,
    JsonDict,
    ValidateModel,
)
from reranker.utils import append_jsonl


def should_use_teacher(gen: GeneratorState, use_teacher: bool | None) -> bool:
    return gen.client.enabled if use_teacher is None else use_teacher


def require_teacher(gen: GeneratorState) -> None:
    if not gen.client.enabled:
        raise RuntimeError("Teacher mode requires OPENROUTER_API_KEY to be set.")


def log_cost(gen: GeneratorState, metadata: JsonDict, dataset_name: str) -> None:
    usage = metadata.get("usage", {})
    if not usage:
        return
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "model": metadata.get("model", gen.client.model),
        "provider": metadata.get("provider"),
        "response_id": metadata.get("response_id"),
        "request_started_at": metadata.get("request_started_at"),
        "request_finished_at": metadata.get("request_finished_at"),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost_usd": usage.get("cost", 0.0),
    }
    append_jsonl(gen.log_path, record)


def validate_record(model_cls: ValidateModel, payload: JsonDict) -> JsonDict:
    try:
        return model_cls.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise ValueError(f"Generated payload failed schema validation: {exc}") from exc


def normalize_generated_value(value: Any) -> str:
    return str(value).strip().lower()


def render_contradiction_docs(
    subject: str,
    field_name: str,
    value_a: str,
    value_b: str,
) -> tuple[str, str]:
    field_label = field_name.replace("_", " ")
    return (
        f"{subject} reports {field_label} as {value_a}. Supporting context remains stable.",
        f"{subject} reports {field_label} as {value_b}. Supporting context remains stable.",
    )


def render_control_docs(subject: str, field_name: str, value: str) -> tuple[str, str]:
    field_label = field_name.replace("_", " ")
    return (
        f"{subject} reports {field_label} as {value}. Supporting context remains stable.",
        f"{subject} again reports {field_label} as {value}. Supporting context remains stable.",
    )


def stabilize_contradiction_record(
    gen: GeneratorState,
    spec: JsonDict,
    record: JsonDict,
    teacher_model: str,
) -> JsonDict:
    subject = str(spec["subject"])
    field_name = str(spec["field_name"])
    value_a = str(spec["value_a"])
    expected_value_b = str(spec["value_b"]) if bool(spec["is_contradiction"]) else value_a
    normalized_value_a = normalize_generated_value(value_a)
    normalized_value_b = normalize_generated_value(expected_value_b)
    doc_a = str(record.get("doc_a", ""))
    doc_b = str(record.get("doc_b", ""))
    doc_a_lower = doc_a.lower()
    doc_b_lower = doc_b.lower()

    if bool(spec["is_contradiction"]):
        if (
            normalized_value_a == normalized_value_b
            or normalized_value_a not in doc_a_lower
            or normalized_value_b not in doc_b_lower
        ):
            doc_a, doc_b = render_contradiction_docs(subject, field_name, value_a, expected_value_b)
    elif (
        normalized_value_a != normalize_generated_value(record.get("value_b", value_a))
        or normalized_value_a not in doc_a_lower
        or normalized_value_a not in doc_b_lower
    ):
        doc_a, doc_b = render_control_docs(subject, field_name, value_a)

    return {
        **record,
        "subject": subject,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "contradicted_field": field_name,
        "value_a": value_a,
        "value_b": expected_value_b,
        "is_contradiction": bool(spec["is_contradiction"]),
        "generation_seed": gen.seed,
        "generation_mode": "teacher",
        "teacher_model": teacher_model,
    }


def chunk_specs(specs: list[JsonDict], batch_size: int) -> list[list[JsonDict]]:
    return [specs[idx : idx + batch_size] for idx in range(0, len(specs), batch_size)]


def parallel_teacher_batches(
    gen: GeneratorState,
    specs: list[JsonDict],
    *,
    batch_size: int,
    fn: BatchGenerator,
) -> list[JsonDict]:
    settings = get_settings()
    chunks = chunk_specs(specs, batch_size)
    if len(chunks) <= 1:
        return fn(gen, specs)
    max_workers = max(1, settings.synthetic_data.teacher_max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(executor.map(lambda chunk: fn(gen, chunk), chunks))
    return [record for records in chunk_results for record in records]


def collect_records(records: Iterable[JsonDict]) -> list[JsonDict]:
    return list(records)


def batch_prompt_payload(specs: list[JsonDict]) -> str:
    return json.dumps(specs, indent=2)

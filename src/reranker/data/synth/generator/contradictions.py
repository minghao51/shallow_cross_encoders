"""Contradiction generation helpers for the synthetic data generator."""

from __future__ import annotations

from collections.abc import Iterator

from reranker.config import get_settings
from reranker.data.synth._models import ContradictionRecord
from reranker.data.synth._prompts import (
    CONTRADICTION_BATCH_PROMPT,
    CONTRADICTION_PROMPT,
)
from reranker.data.synth.generator import core
from reranker.data.synth.generator.types import (
    ContradictionSpec,
    GeneratorState,
    JsonDict,
)

OFFLINE_SUBJECTS: list[tuple[str, str, str, str]] = [
    ("Model2Vec Potion-8M", "latency_ms", "2", "7"),
    ("HybridFusionReranker", "best_metric", "NDCG@10", "MRR"),
    ("Project Atlas", "release_year", "2025", "2026"),
    ("Northwind Clinic", "screening_status", "approved", "pending"),
    ("Potion-32M Embedder", "embedding_dim", "256", "512"),
    ("BM25 Engine", "k1_parameter", "1.2", "2.0"),
    ("Kubernetes Cluster", "max_pods_per_node", "110", "250"),
    ("Type 1 Diabetes", "onset_age", "childhood", "adulthood"),
    ("OAuth2 Flow", "token_expiry", "3600", "7200"),
    ("Bloom Taxonomy", "total_levels", "5", "6"),
    ("Spark DAG Optimizer", "stage_count", "3", "7"),
    ("Fair Use Doctrine", "max_quotation_words", "100", "400"),
    ("Star Schema", "fact_table_count", "1", "3"),
    ("Bayes Classifier", "independence_assumption", "strict", "relaxed"),
    ("React useEffect", "cleanup_timing", "before_unmount", "after_unmount"),
]


def teacher_contradiction_record(
    gen: GeneratorState,
    subject: str,
    field_name: str,
    value_a: str,
    value_b: str,
    is_contradiction: bool,
) -> JsonDict:
    """Generate a single contradiction record using the teacher model.

    Args:
        gen: Generator state with client access.
        subject: Entity subject for the contradiction.
        field_name: Contradicted field name.
        value_a: Value claimed in document A.
        value_b: Value claimed in document B.
        is_contradiction: Whether this is a contradiction or control.

    Returns:
        Validated contradiction record dict.
    """
    core.require_teacher(gen)
    payload, metadata = gen.client.complete_json(
        CONTRADICTION_PROMPT.format(
            subject=subject,
            field=field_name,
            value_a=value_a,
            value_b=value_b,
            target_label=str(is_contradiction).lower(),
        )
    )
    payload.update(
        core.stabilize_contradiction_record(
            gen,
            {
                "subject": subject,
                "field_name": field_name,
                "value_a": value_a,
                "value_b": value_b,
                "is_contradiction": is_contradiction,
            },
            payload,
            str(metadata.get("model", gen.client.model)),
        )
    )
    core.log_cost(gen, metadata, "contradictions")
    return core.validate_record(ContradictionRecord, payload)


def teacher_contradiction_records(
    gen: GeneratorState,
    batch_specs: list[JsonDict],
) -> list[JsonDict]:
    """Generate contradiction records in batch via the teacher model.

    Falls back to single-record generation and binary splitting on failure.

    Args:
        gen: Generator state with client access.
        batch_specs: List of contradiction spec dicts.

    Returns:
        List of validated contradiction record dicts.
    """
    if len(batch_specs) == 1:
        spec = batch_specs[0]
        return [
            teacher_contradiction_record(
                gen,
                str(spec["subject"]),
                str(spec["field_name"]),
                str(spec["value_a"]),
                str(spec["value_b"]),
                bool(spec["is_contradiction"]),
            )
        ]
    core.require_teacher(gen)
    try:
        payload, metadata = gen.client.complete_json(
            CONTRADICTION_BATCH_PROMPT.format(
                count=len(batch_specs),
                items_json=core.batch_prompt_payload(batch_specs),
            )
        )
    except Exception:
        midpoint = len(batch_specs) // 2
        return teacher_contradiction_records(
            gen,
            batch_specs[:midpoint],
        ) + teacher_contradiction_records(gen, batch_specs[midpoint:])
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) != len(batch_specs):
        midpoint = len(batch_specs) // 2
        return teacher_contradiction_records(
            gen,
            batch_specs[:midpoint],
        ) + teacher_contradiction_records(gen, batch_specs[midpoint:])
    core.log_cost(gen, metadata, "contradictions")
    try:
        return [
            core.validate_record(
                ContradictionRecord,
                core.stabilize_contradiction_record(
                    gen,
                    spec,
                    record,
                    str(metadata.get("model", gen.client.model)),
                ),
            )
            for spec, record in zip(batch_specs, records, strict=False)
        ]
    except ValueError:
        midpoint = len(batch_specs) // 2
        return teacher_contradiction_records(
            gen,
            batch_specs[:midpoint],
        ) + teacher_contradiction_records(gen, batch_specs[midpoint:])


def iter_contradictions(
    gen: GeneratorState,
    contradiction_count: int | None = None,
    control_count: int | None = None,
    use_teacher: bool | None = None,
) -> Iterator[JsonDict]:
    """Yield contradiction and control records.

    Args:
        gen: Generator state.
        contradiction_count: Number of contradiction records.
        control_count: Number of control records.
        use_teacher: Whether to use teacher model.

    Yields:
        Validated contradiction or control record dicts.
    """
    """Yield contradiction and control records."""
    settings = get_settings()
    resolved_contradictions = (
        settings.synthetic_data.contradiction_count
        if contradiction_count is None
        else contradiction_count
    )
    resolved_controls = (
        settings.synthetic_data.control_count if control_count is None else control_count
    )
    teacher_mode = core.should_use_teacher(gen, use_teacher)

    if teacher_mode:
        batch_size = max(1, settings.synthetic_data.teacher_batch_size)
        contradiction_specs: list[ContradictionSpec] = [
            {
                "subject": subject,
                "field_name": field_name,
                "value_a": value_a,
                "value_b": value_b,
                "is_contradiction": True,
            }
            for idx in range(resolved_contradictions)
            for subject, field_name, value_a, value_b in [
                OFFLINE_SUBJECTS[idx % len(OFFLINE_SUBJECTS)]
            ]
        ]
        control_specs: list[ContradictionSpec] = [
            {
                "subject": subject,
                "field_name": field_name,
                "value_a": value,
                "value_b": value,
                "is_contradiction": False,
            }
            for idx in range(resolved_controls)
            for subject, field_name, value, _ in [OFFLINE_SUBJECTS[idx % len(OFFLINE_SUBJECTS)]]
        ]
        yield from core.parallel_teacher_batches(
            gen,
            [
                *(dict(spec) for spec in contradiction_specs),
                *(dict(spec) for spec in control_specs),
            ],
            batch_size=batch_size,
            fn=teacher_contradiction_records,
        )
        return

    records: list[JsonDict] = []
    for idx in range(resolved_contradictions):
        subject, field_name, value_a, value_b = OFFLINE_SUBJECTS[idx % len(OFFLINE_SUBJECTS)]
        records.append(
            core.validate_record(
                ContradictionRecord,
                {
                    "subject": subject,
                    "doc_a": (
                        f"{subject} reports {field_name} as {value_a}. "
                        "The rest of the setup is unchanged."
                    ),
                    "doc_b": (
                        f"{subject} reports {field_name} as {value_b}. "
                        "The rest of the setup is unchanged."
                    ),
                    "contradicted_field": field_name,
                    "value_a": value_a,
                    "value_b": value_b,
                    "is_contradiction": True,
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )
    for idx in range(resolved_controls):
        subject, field_name, value, _ = OFFLINE_SUBJECTS[idx % len(OFFLINE_SUBJECTS)]
        records.append(
            core.validate_record(
                ContradictionRecord,
                {
                    "subject": subject,
                    "doc_a": (
                        f"{subject} lists {field_name} as {value}. Supporting context is similar."
                    ),
                    "doc_b": (
                        f"{subject} again lists {field_name} as {value}. "
                        "Supporting context is similar."
                    ),
                    "contradicted_field": field_name,
                    "value_a": value,
                    "value_b": value,
                    "is_contradiction": False,
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )
    yield from records

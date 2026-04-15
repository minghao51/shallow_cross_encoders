"""Preference generation helpers for the synthetic data generator."""

from __future__ import annotations

from collections.abc import Iterator

from reranker.config import get_settings
from reranker.data.synth._models import PreferenceRecord
from reranker.data.synth._prompts import PREFERENCE_BATCH_PROMPT, PREFERENCE_PROMPT
from reranker.data.synth.generator import core
from reranker.data.synth.generator.pairs import get_expanded_seeds
from reranker.data.synth.generator.types import GeneratorState, JsonDict, PreferenceSpec


def apply_preference_swap(record: JsonDict, swap_output: bool) -> JsonDict:
    if not swap_output:
        return record
    preferred = "B" if record["preferred"] == "A" else "A"
    return {
        **record,
        "doc_a": record["doc_b"],
        "doc_b": record["doc_a"],
        "preferred": preferred,
    }


def teacher_preference_record(gen: GeneratorState, seed: JsonDict) -> JsonDict:
    core.require_teacher(gen)
    payload, metadata = gen.client.complete_json(
        PREFERENCE_PROMPT.format(
            query=seed["query"],
            positive=seed["positive"],
            negative=seed["negative"],
        )
    )
    payload.update(
        {
            "generation_seed": gen.seed,
            "generation_mode": "teacher",
            "teacher_model": metadata.get("model", gen.client.model),
        }
    )
    core.log_cost(gen, metadata, "preferences")
    return core.validate_record(PreferenceRecord, payload)


def teacher_preference_records(gen: GeneratorState, batch_specs: list[JsonDict]) -> list[JsonDict]:
    if len(batch_specs) == 1:
        spec = batch_specs[0]
        record = teacher_preference_record(gen, spec)
        return [apply_preference_swap(record, bool(spec.get("swap_output", False)))]
    core.require_teacher(gen)
    try:
        payload, metadata = gen.client.complete_json(
            PREFERENCE_BATCH_PROMPT.format(
                count=len(batch_specs),
                items_json=core.batch_prompt_payload(batch_specs),
            )
        )
    except Exception:
        midpoint = len(batch_specs) // 2
        return teacher_preference_records(gen, batch_specs[:midpoint]) + teacher_preference_records(
            gen, batch_specs[midpoint:]
        )
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) != len(batch_specs):
        midpoint = len(batch_specs) // 2
        return teacher_preference_records(gen, batch_specs[:midpoint]) + teacher_preference_records(
            gen, batch_specs[midpoint:]
        )
    core.log_cost(gen, metadata, "preferences")
    try:
        return [
            core.validate_record(
                PreferenceRecord,
                apply_preference_swap(
                    {
                        **record,
                        "generation_seed": gen.seed,
                        "generation_mode": "teacher",
                        "teacher_model": metadata.get("model", gen.client.model),
                    },
                    bool(spec.get("swap_output", False)),
                ),
            )
            for spec, record in zip(batch_specs, records, strict=False)
        ]
    except ValueError:
        midpoint = len(batch_specs) // 2
        return teacher_preference_records(gen, batch_specs[:midpoint]) + teacher_preference_records(
            gen, batch_specs[midpoint:]
        )


def iter_preferences(
    gen: GeneratorState,
    pairs: list[JsonDict],
    target_count: int | None = None,
    use_teacher: bool | None = None,
) -> Iterator[JsonDict]:
    """Yield preference records derived from pairs or the teacher prompts."""
    resolved_target_count = (
        get_settings().synthetic_data.preference_count if target_count is None else target_count
    )
    teacher_mode = core.should_use_teacher(gen, use_teacher)
    expanded_seeds = get_expanded_seeds()

    if teacher_mode:
        batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
        if resolved_target_count <= 4:
            batch_size = 1
        specs: list[PreferenceSpec] = []
        for idx in range(resolved_target_count):
            seed = expanded_seeds[idx % len(expanded_seeds)]
            spec: PreferenceSpec = {
                "query": str(seed["query"]),
                "positive": str(seed["positive"]),
                "negative": str(seed["negative"]),
                "domain": str(seed.get("domain", "general")),
            }
            spec["swap_output"] = bool(idx % 2)
            specs.append(spec)
        yield from core.parallel_teacher_batches(
            gen,
            [dict(spec) for spec in specs],
            batch_size=batch_size,
            fn=teacher_preference_records,
        )
        return

    by_query: dict[str, list[JsonDict]] = {}
    for pair in pairs:
        by_query.setdefault(str(pair["query"]), []).append(pair)

    records: list[JsonDict] = []
    for query, examples in by_query.items():
        if len(records) >= resolved_target_count:
            break
        ordered = sorted(examples, key=lambda row: int(row["score"]))
        if len(ordered) < 2:
            continue
        lo = ordered[0]
        hi = ordered[-1]
        records.append(
            core.validate_record(
                PreferenceRecord,
                {
                    "query": query,
                    "doc_a": hi["doc"],
                    "doc_b": lo["doc"],
                    "preferred": "A",
                    "confidence": 0.95,
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )

    while len(records) < resolved_target_count and by_query:
        query = gen.random.choice(list(by_query))
        examples = by_query[query]
        doc_a, doc_b = gen.random.sample(examples, 2)
        preferred = "A" if int(doc_a["score"]) >= int(doc_b["score"]) else "B"
        confidence = min(1.0, 0.55 + abs(int(doc_a["score"]) - int(doc_b["score"])) * 0.15)
        records.append(
            core.validate_record(
                PreferenceRecord,
                {
                    "query": query,
                    "doc_a": doc_a["doc"],
                    "doc_b": doc_b["doc"],
                    "preferred": preferred,
                    "confidence": round(confidence, 2),
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )

    yield from records

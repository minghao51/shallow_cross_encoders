"""Pair generation helpers for the synthetic data generator."""

from __future__ import annotations

from collections.abc import Iterator

from reranker.config import get_settings
from reranker.data.synth._models import PairRecord
from reranker.data.synth._prompts import PAIR_BATCH_PROMPT, PAIR_PROMPT
from reranker.data.synth._seeds import DEFAULT_PAIR_SEEDS
from reranker.data.synth.generator import core
from reranker.data.synth.generator.types import ExpandedSeed, GeneratorState, JsonDict, PairSpec


def get_expanded_seeds() -> list[ExpandedSeed]:
    """Expand seed queries with query variations for better offline diversity."""
    expanded: list[ExpandedSeed] = []
    for seed in DEFAULT_PAIR_SEEDS:
        expanded.append(
            {
                "query": str(seed["query"]),
                "positive": str(seed["positive"]),
                "negative": str(seed["negative"]),
                "domain": str(seed.get("domain", "general")),
            }
        )
        for variation in seed.get("variations", []):
            expanded.append(
                {
                    "query": str(variation),
                    "positive": str(seed["positive"]),
                    "negative": str(seed["negative"]),
                    "domain": str(seed.get("domain", "general")),
                }
            )
    return expanded


def teacher_pair_record(gen: GeneratorState, seed: JsonDict, target_score: int) -> JsonDict:
    core.require_teacher(gen)
    payload, metadata = gen.client.complete_json(
        PAIR_PROMPT.format(
            query=seed["query"],
            positive=seed["positive"],
            negative=seed["negative"],
            target_score=target_score,
        )
    )
    payload.update(
        {
            "generation_seed": gen.seed,
            "generation_mode": "teacher",
            "teacher_model": metadata.get("model", gen.client.model),
        }
    )
    core.log_cost(gen, metadata, "pairs")
    return core.validate_record(PairRecord, payload)


_MAX_BATCH_RECURSION_DEPTH = 10


def teacher_pair_records(
    gen: GeneratorState,
    batch_specs: list[JsonDict],
    _depth: int = 0,
) -> list[JsonDict]:
    if _depth >= _MAX_BATCH_RECURSION_DEPTH:
        raise RecursionError(
            f"Max recursion depth ({_MAX_BATCH_RECURSION_DEPTH}) exceeded in "
            f"teacher_pair_records. Batch size may be too small or API responses malformed."
        )
    if len(batch_specs) == 1:
        spec = batch_specs[0]
        return [teacher_pair_record(gen, spec, int(spec["target_score"]))]
    core.require_teacher(gen)
    try:
        payload, metadata = gen.client.complete_json(
            PAIR_BATCH_PROMPT.format(
                count=len(batch_specs),
                items_json=core.batch_prompt_payload(batch_specs),
            )
        )
    except Exception:
        midpoint = len(batch_specs) // 2
        return teacher_pair_records(gen, batch_specs[:midpoint], _depth + 1) + teacher_pair_records(
            gen, batch_specs[midpoint:], _depth + 1
        )
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) != len(batch_specs):
        midpoint = len(batch_specs) // 2
        return teacher_pair_records(gen, batch_specs[:midpoint], _depth + 1) + teacher_pair_records(
            gen, batch_specs[midpoint:], _depth + 1
        )
    core.log_cost(gen, metadata, "pairs")
    try:
        return [
            core.validate_record(
                PairRecord,
                {
                    **record,
                    "generation_seed": gen.seed,
                    "generation_mode": "teacher",
                    "teacher_model": metadata.get("model", gen.client.model),
                },
            )
            for record in records
        ]
    except ValueError:
        midpoint = len(batch_specs) // 2
        return teacher_pair_records(gen, batch_specs[:midpoint], _depth + 1) + teacher_pair_records(
            gen, batch_specs[midpoint:], _depth + 1
        )


def iter_pairs(
    gen: GeneratorState,
    target_count: int | None = None,
    use_teacher: bool | None = None,
) -> Iterator[JsonDict]:
    """Yield pair records for offline or teacher-backed generation."""
    resolved_target_count = (
        get_settings().synthetic_data.pair_count if target_count is None else target_count
    )
    teacher_mode = core.should_use_teacher(gen, use_teacher)
    expanded_seeds = get_expanded_seeds()

    if teacher_mode:
        score_cycle = [0, 1, 2, 3]
        batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
        if resolved_target_count <= 4:
            batch_size = 1
        specs: list[PairSpec] = []
        for idx in range(resolved_target_count):
            seed = expanded_seeds[idx % len(expanded_seeds)]
            specs.append(
                {
                    "query": seed["query"],
                    "positive": seed["positive"],
                    "negative": seed["negative"],
                    "target_score": score_cycle[idx % len(score_cycle)],
                }
            )
        yield from core.parallel_teacher_batches(
            gen,
            [dict(spec) for spec in specs],
            batch_size=batch_size,
            fn=teacher_pair_records,
        )
        return

    records: list[JsonDict] = []
    shuffled_seeds = list(expanded_seeds)
    gen.random.shuffle(shuffled_seeds)

    for seed in shuffled_seeds:
        if len(records) >= resolved_target_count:
            break
        records.append(
            core.validate_record(
                PairRecord,
                {
                    "query": seed["query"],
                    "doc": seed["positive"],
                    "score": 3,
                    "rationale": "Directly addresses the query with the relevant concept.",
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )
        if len(records) >= resolved_target_count:
            break
        records.append(
            core.validate_record(
                PairRecord,
                {
                    "query": seed["query"],
                    "doc": seed["negative"],
                    "score": 0,
                    "rationale": "Discusses a different topic from the query.",
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )

    score_cycle = [0, 1, 2, 3]
    suffixes = [
        "This note adds practical detail.",
        "This explanation stays high level.",
        "This example is intentionally compact.",
        "This statement includes a nearby concept.",
    ]
    while len(records) < resolved_target_count:
        seed = gen.random.choice(shuffled_seeds)
        score = score_cycle[len(records) % len(score_cycle)]
        text = seed["positive"] if score >= 2 else seed["negative"]
        suffix = suffixes[gen.random.randrange(len(suffixes))]
        records.append(
            core.validate_record(
                PairRecord,
                {
                    "query": seed["query"],
                    "doc": f"{text} {suffix}",
                    "score": score,
                    "rationale": "Offline synthetic label generated from seeded examples.",
                    "generation_seed": gen.seed,
                    "generation_mode": "offline",
                    "teacher_model": None,
                },
            )
        )

    yield from records

"""Enhanced synthetic dataset helpers."""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Iterator

from reranker.config import get_settings
from reranker.data.synth._models import (
    HardNegativeRecord,
    ListwisePreferenceRecord,
    QueryExpansionRecord,
)
from reranker.data.synth._prompts import (
    HARD_NEGATIVE_BATCH_PROMPT,
    LISTWISE_BATCH_PROMPT,
    QUERY_EXPANSION_BATCH_PROMPT,
)
from reranker.data.synth.generator import core
from reranker.data.synth.generator.pairs import get_expanded_seeds
from reranker.data.synth.generator.types import (
    GeneratorState,
    HardNegativeSpec,
    JsonDict,
    ListwiseSpec,
    QueryExpansionSpec,
)

DOMAIN_HARD_NEGATIVES: dict[str, list[str]] = {
    "python": [
        "Python decorators use the @ syntax to wrap functions with additional behavior at definition time.",
        "The __init__ method in Python classes initializes instance attributes when objects are created.",
        "Python's GIL prevents true parallel thread execution for CPU-bound tasks.",
    ],
    "information_retrieval": [
        "Vector space models represent documents as points in a high-dimensional term space.",
        "Lucene uses a combination of inverted indexes and stored fields for efficient retrieval.",
        "Query expansion techniques like synonym replacement can improve recall at the cost of precision.",
    ],
    "singapore_real_estate": [
        "Singapore's property cooling measures include additional buyer's stamp duty and loan-to-value limits.",
        "URA master plan designates land use zones for residential, commercial, and industrial purposes.",
        "Singapore property market trends are influenced by interest rates and government housing policies.",
    ],
    "machine_learning": [
        "Random forests build multiple decision trees and aggregate their predictions for better accuracy.",
        "Batch normalization normalizes layer inputs to accelerate deep network training convergence.",
        "Transfer learning reuses pretrained model weights to improve performance on smaller target datasets.",
    ],
    "devops": [
        "Docker containers package applications with their dependencies for consistent deployment across environments.",
        "Infrastructure as code tools like Terraform manage cloud resources through declarative configuration files.",
        "Monitoring and alerting systems track system health metrics and notify teams of anomalies.",
    ],
}

EASY_NEGATIVES = [
    "The migration patterns of arctic terns span hemispheres each breeding season.",
    "Traditional pottery techniques vary significantly across different cultural traditions.",
    "Classical music composition follows established harmonic and structural conventions.",
    "Marine biodiversity in coral reef ecosystems supports thousands of interdependent species.",
    "The architectural evolution of gothic cathedrals reflects medieval engineering advances.",
]


def iter_hard_negatives(
    gen: GeneratorState,
    pairs: list[JsonDict],
    target_count: int | None = None,
    use_teacher: bool | None = None,
) -> Iterator[JsonDict]:
    resolved_target_count = (
        get_settings().synthetic_data.pair_count if target_count is None else target_count
    )
    teacher_mode = core.should_use_teacher(gen, use_teacher)

    if teacher_mode:
        batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
        if resolved_target_count <= 4:
            batch_size = 1
        specs: list[HardNegativeSpec] = []
        for idx in range(resolved_target_count):
            pair = pairs[idx % len(pairs)]
            specs.append({"query": str(pair["query"]), "positive": str(pair["doc"])})
        for chunk in core.chunk_specs([dict(spec) for spec in specs], batch_size):
            try:
                payload, metadata = gen.client.complete_json(
                    HARD_NEGATIVE_BATCH_PROMPT.format(
                        count=len(chunk),
                        items_json=core.batch_prompt_payload(chunk),
                    )
                )
                batch_records = payload.get("records", [])
                for record in batch_records:
                    record.update(
                        {
                            "generation_seed": gen.seed,
                            "generation_mode": "teacher",
                            "teacher_model": metadata.get("model", gen.client.model),
                        }
                    )
                    yield core.validate_record(HardNegativeRecord, record)
                core.log_cost(gen, metadata, "hard_negatives")
            except Exception:
                midpoint = len(chunk) // 2
                if midpoint > 0:
                    yield from iter_hard_negatives(
                        gen,
                        [
                            {"query": spec["query"], "doc": spec["positive"]}
                            for spec in chunk[:midpoint]
                        ],
                        target_count=midpoint,
                        use_teacher=True,
                    )
                    yield from iter_hard_negatives(
                        gen,
                        [
                            {"query": spec["query"], "doc": spec["positive"]}
                            for spec in chunk[midpoint:]
                        ],
                        target_count=len(chunk) - midpoint,
                        use_teacher=True,
                    )
        return

    expanded_seeds = get_expanded_seeds()
    for idx in range(resolved_target_count):
        pair = pairs[idx % len(pairs)]
        query = str(pair["query"])
        positive = str(pair["doc"])
        domain = "general"
        for seed in expanded_seeds:
            variations = seed.get("variations", [])
            if seed["query"] == query or query in variations:
                domain = seed.get("domain", "general")
                break
        hn_pool = DOMAIN_HARD_NEGATIVES.get(domain, DOMAIN_HARD_NEGATIVES.get("python", []))
        yield core.validate_record(
            HardNegativeRecord,
            {
                "query": query,
                "positive": positive,
                "hard_negative": hn_pool[idx % len(hn_pool)],
                "easy_negative": EASY_NEGATIVES[idx % len(EASY_NEGATIVES)],
                "generation_seed": gen.seed,
                "generation_mode": "offline",
                "teacher_model": None,
            },
        )


def iter_listwise_preferences(
    gen: GeneratorState,
    pairs: list[JsonDict],
    target_count: int | None = None,
    use_teacher: bool | None = None,
) -> Iterator[JsonDict]:
    resolved_target_count = (
        get_settings().synthetic_data.preference_count if target_count is None else target_count
    )
    teacher_mode = core.should_use_teacher(gen, use_teacher)

    if teacher_mode:
        batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
        query_docs: dict[str, list[str]] = {}
        for pair in pairs:
            query_docs.setdefault(str(pair["query"]), []).append(str(pair["doc"]))
        specs: list[ListwiseSpec] = []
        for idx in range(resolved_target_count):
            query = list(query_docs.keys())[idx % len(query_docs)]
            docs = query_docs[query][:4]
            if len(docs) >= 3:
                specs.append({"query": query, "docs": docs})
        for chunk in core.chunk_specs([dict(spec) for spec in specs], batch_size):
            try:
                payload, metadata = gen.client.complete_json(
                    LISTWISE_BATCH_PROMPT.format(
                        count=len(chunk),
                        items_json=core.batch_prompt_payload(chunk),
                    )
                )
                for record in payload.get("records", []):
                    record.update(
                        {
                            "generation_seed": gen.seed,
                            "generation_mode": "teacher",
                            "teacher_model": metadata.get("model", gen.client.model),
                        }
                    )
                    yield core.validate_record(ListwisePreferenceRecord, record)
                core.log_cost(gen, metadata, "listwise_preferences")
            except Exception:
                continue
        return

    query_pairs: dict[str, list[JsonDict]] = {}
    for pair in pairs:
        query_pairs.setdefault(str(pair["query"]), []).append(pair)

    count = 0
    for query, examples in query_pairs.items():
        if count >= resolved_target_count:
            break
        if len(examples) < 3:
            continue
        ordered = sorted(examples, key=lambda x: int(x.get("score", 0)), reverse=True)
        docs = [str(example["doc"]) for example in ordered[:4]]
        scores = [0.4, 0.3, 0.2, 0.1][: len(docs)]
        yield core.validate_record(
            ListwisePreferenceRecord,
            {
                "query": query,
                "docs": docs,
                "scores": scores,
                "generation_seed": gen.seed,
                "generation_mode": "offline",
                "teacher_model": None,
            },
        )
        count += 1


def iter_query_expansions(
    gen: GeneratorState,
    pairs: list[JsonDict],
    target_count: int | None = None,
    use_teacher: bool | None = None,
) -> Iterator[JsonDict]:
    resolved_target_count = (
        get_settings().synthetic_data.pair_count if target_count is None else target_count
    )
    teacher_mode = core.should_use_teacher(gen, use_teacher)

    if teacher_mode:
        batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
        seen_queries = list({str(pair["query"]) for pair in pairs})
        specs: list[QueryExpansionSpec] = [
            {"query": query} for query in seen_queries[:resolved_target_count]
        ]
        for chunk in core.chunk_specs([dict(spec) for spec in specs], batch_size):
            try:
                payload, metadata = gen.client.complete_json(
                    QUERY_EXPANSION_BATCH_PROMPT.format(
                        count=len(chunk),
                        items_json=core.batch_prompt_payload(chunk),
                    )
                )
                for record in payload.get("records", []):
                    record.update(
                        {
                            "generation_seed": gen.seed,
                            "generation_mode": "teacher",
                            "teacher_model": metadata.get("model", gen.client.model),
                        }
                    )
                    yield core.validate_record(QueryExpansionRecord, record)
                core.log_cost(gen, metadata, "query_expansions")
            except Exception:
                continue
        return

    seen_queries = list({str(pair["query"]) for pair in pairs})
    for idx in range(min(resolved_target_count, len(seen_queries))):
        query = seen_queries[idx]
        expansions = [
            f"how to {query}",
            f"{query} best practices",
            f"{query} example",
            f"understanding {query}",
            f"{query} guide",
        ]
        yield core.validate_record(
            QueryExpansionRecord,
            {
                "original_query": query,
                "expanded_queries": expansions[:3],
                "generation_seed": gen.seed,
                "generation_mode": "offline",
                "teacher_model": None,
            },
        )

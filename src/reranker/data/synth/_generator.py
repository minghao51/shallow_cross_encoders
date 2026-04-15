from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from reranker.config import get_settings
from reranker.data.client import OpenRouterClient
from reranker.data.synth import generator
from reranker.data.synth.generator.types import BatchGenerator, JsonDict


@dataclass(slots=True)
class SyntheticDataGenerator:
    """Facade for synthetic dataset generation.

    Public methods keep the historical API stable while delegating to smaller,
    typed helper modules under `reranker.data.synth.generator`.
    """

    seed: int = field(default_factory=lambda: get_settings().synthetic_data.seed)
    client: OpenRouterClient = field(default_factory=OpenRouterClient)
    log_path: str | Path = field(default_factory=lambda: get_settings().paths.api_cost_log)
    random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.random = random.Random(self.seed)

    def _should_use_teacher(self, use_teacher: bool | None) -> bool:
        return generator.should_use_teacher(self, use_teacher)

    def _require_teacher(self) -> None:
        generator.require_teacher(self)

    def _log_cost(self, metadata: dict[str, Any], dataset_name: str) -> None:
        generator.log_cost(self, metadata, dataset_name)

    def _validate_record(self, model_cls: type[Any], payload: JsonDict) -> JsonDict:
        return generator.validate_record(model_cls, payload)

    def _stabilize_contradiction_record(
        self,
        spec: dict[str, Any],
        record: dict[str, Any],
        teacher_model: str,
    ) -> JsonDict:
        return generator.stabilize_contradiction_record(self, spec, record, teacher_model)

    def _chunk_specs(self, specs: list[JsonDict], batch_size: int) -> list[list[JsonDict]]:
        return generator.chunk_specs(specs, batch_size)

    def _parallel_teacher_batches(
        self,
        specs: list[JsonDict],
        batch_size: int,
        fn: BatchGenerator,
    ) -> list[JsonDict]:
        return generator.parallel_teacher_batches(self, specs, batch_size=batch_size, fn=fn)

    def _get_expanded_seeds(self) -> list[dict[str, Any]]:
        return [dict(seed) for seed in generator.get_expanded_seeds()]

    def iter_pairs(
        self,
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]:
        """Stream pair records for large generation jobs."""
        return generator.iter_pairs(self, target_count=target_count, use_teacher=use_teacher)

    def generate_pairs(
        self,
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[JsonDict]:
        """Return graded query-document pairs as a convenience wrapper over `iter_pairs`."""
        return list(self.iter_pairs(target_count=target_count, use_teacher=use_teacher))

    def iter_preferences(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]:
        """Stream preference records derived from existing pair examples."""
        return generator.iter_preferences(
            self,
            pairs,
            target_count=target_count,
            use_teacher=use_teacher,
        )

    def generate_preferences(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[JsonDict]:
        """Return pairwise preference examples for training or evaluation."""
        return list(
            self.iter_preferences(
                pairs,
                target_count=target_count,
                use_teacher=use_teacher,
            )
        )

    def iter_contradictions(
        self,
        contradiction_count: int | None = None,
        control_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]:
        """Stream contradiction/control examples with deterministic offline fallbacks."""
        return generator.iter_contradictions(
            self,
            contradiction_count=contradiction_count,
            control_count=control_count,
            use_teacher=use_teacher,
        )

    def generate_contradictions(
        self,
        contradiction_count: int | None = None,
        control_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[JsonDict]:
        """Return contradiction and control examples as JSON-serializable records."""
        return list(
            self.iter_contradictions(
                contradiction_count=contradiction_count,
                control_count=control_count,
                use_teacher=use_teacher,
            )
        )

    def _distribution_report(self, dataset_name: str, records: list[JsonDict]) -> JsonDict:
        return generator.distribution_report(dataset_name, records)

    def _write_distribution_summary(
        self,
        raw_root: Path,
        pairs: list[JsonDict],
        preferences: list[JsonDict],
        contradictions: list[JsonDict],
    ) -> None:
        generator.write_distribution_summary(self, raw_root, pairs, preferences, contradictions)

    def _write_manifest(
        self,
        raw_root: Path,
        pairs: list[JsonDict],
        preferences: list[JsonDict],
        contradictions: list[JsonDict],
    ) -> None:
        generator.write_manifest(self, raw_root, pairs, preferences, contradictions)

    def materialize_all(
        self,
        root: str | Path | None = None,
        pair_count: int | None = None,
        preference_count: int | None = None,
        contradiction_count: int | None = None,
        control_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> dict[str, str]:
        """Generate all core datasets and write them out in chunked JSONL form."""
        outputs = generator.materialize_all(
            self,
            root=root,
            pair_count=pair_count,
            preference_count=preference_count,
            contradiction_count=contradiction_count,
            control_count=control_count,
            use_teacher=use_teacher,
        )
        return cast(dict[str, str], outputs)

    def iter_hard_negatives(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]:
        """Stream hard-negative examples for reranker training."""
        return generator.iter_hard_negatives(
            self,
            pairs,
            target_count=target_count,
            use_teacher=use_teacher,
        )

    def generate_hard_negatives(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[JsonDict]:
        return list(
            self.iter_hard_negatives(
                pairs,
                target_count=target_count,
                use_teacher=use_teacher,
            )
        )

    def iter_listwise_preferences(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]:
        """Stream listwise preferences built from pair data grouped by query."""
        return generator.iter_listwise_preferences(
            self,
            pairs,
            target_count=target_count,
            use_teacher=use_teacher,
        )

    def generate_listwise_preferences(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[JsonDict]:
        return list(
            self.iter_listwise_preferences(
                pairs,
                target_count=target_count,
                use_teacher=use_teacher,
            )
        )

    def iter_query_expansions(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> Iterator[JsonDict]:
        """Stream query expansion suggestions for retrieval augmentation tasks."""
        return generator.iter_query_expansions(
            self,
            pairs,
            target_count=target_count,
            use_teacher=use_teacher,
        )

    def generate_query_expansions(
        self,
        pairs: list[JsonDict],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[JsonDict]:
        return list(
            self.iter_query_expansions(
                pairs,
                target_count=target_count,
                use_teacher=use_teacher,
            )
        )

    def refresh_metadata(self, root: str | Path | None = None) -> dict[str, str]:
        """Rebuild the manifest and label-distribution outputs from existing JSONL files."""
        return generator.refresh_metadata(self, root=root)

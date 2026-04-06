from __future__ import annotations

# ruff: noqa: E501
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError

from reranker.config import get_settings
from reranker.data.client import OpenRouterClient
from reranker.data.synth._models import (
    ContradictionRecord,
    DatasetManifest,
    HardNegativeRecord,
    ListwisePreferenceRecord,
    PairRecord,
    PreferenceRecord,
    QueryExpansionRecord,
)
from reranker.data.synth._prompts import (
    CONTRADICTION_BATCH_PROMPT,
    CONTRADICTION_PROMPT,
    HARD_NEGATIVE_BATCH_PROMPT,
    LISTWISE_BATCH_PROMPT,
    PAIR_BATCH_PROMPT,
    PAIR_PROMPT,
    PREFERENCE_BATCH_PROMPT,
    PREFERENCE_PROMPT,
    QUERY_EXPANSION_BATCH_PROMPT,
)
from reranker.data.synth._seeds import DEFAULT_PAIR_SEEDS
from reranker.utils import append_jsonl, ensure_parent, read_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class SyntheticDataGenerator:
    seed: int = field(default_factory=lambda: get_settings().synthetic_data.seed)
    client: OpenRouterClient = field(default_factory=OpenRouterClient)
    log_path: str | Path = field(default_factory=lambda: get_settings().paths.api_cost_log)
    random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.random = random.Random(self.seed)

    def _should_use_teacher(self, use_teacher: bool | None) -> bool:
        return self.client.enabled if use_teacher is None else use_teacher

    def _require_teacher(self) -> None:
        if not self.client.enabled:
            raise RuntimeError("Teacher mode requires OPENROUTER_API_KEY to be set.")

    def _log_cost(self, metadata: dict[str, Any], dataset_name: str) -> None:
        usage = metadata.get("usage", {})
        if not usage:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": dataset_name,
            "model": metadata.get("model", self.client.model),
            "provider": metadata.get("provider"),
            "response_id": metadata.get("response_id"),
            "request_started_at": metadata.get("request_started_at"),
            "request_finished_at": metadata.get("request_finished_at"),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": usage.get("cost", 0.0),
        }
        append_jsonl(self.log_path, record)

    def _validate_record(
        self,
        model_cls: type[BaseModel],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return model_cls.model_validate(payload).model_dump()
        except ValidationError as exc:
            raise ValueError(f"Generated payload failed schema validation: {exc}") from exc

    @staticmethod
    def _normalize_generated_value(value: Any) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _render_contradiction_docs(
        subject: str,
        field_name: str,
        value_a: str,
        value_b: str,
    ) -> tuple[str, str]:
        field_label = field_name.replace("_", " ")
        doc_a = f"{subject} reports {field_label} as {value_a}. Supporting context remains stable."
        doc_b = f"{subject} reports {field_label} as {value_b}. Supporting context remains stable."
        return doc_a, doc_b

    @staticmethod
    def _render_control_docs(subject: str, field_name: str, value: str) -> tuple[str, str]:
        field_label = field_name.replace("_", " ")
        doc_a = f"{subject} reports {field_label} as {value}. Supporting context remains stable."
        doc_b = (
            f"{subject} again reports {field_label} as {value}. Supporting context remains stable."
        )
        return doc_a, doc_b

    def _stabilize_contradiction_record(
        self,
        spec: dict[str, Any],
        record: dict[str, Any],
        teacher_model: str,
    ) -> dict[str, Any]:
        subject = str(spec["subject"])
        field_name = str(spec["field_name"])
        value_a = str(spec["value_a"])
        expected_value_b = str(spec["value_b"]) if bool(spec["is_contradiction"]) else value_a
        normalized_value_a = self._normalize_generated_value(value_a)
        normalized_value_b = self._normalize_generated_value(expected_value_b)
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
                doc_a, doc_b = self._render_contradiction_docs(
                    subject,
                    field_name,
                    value_a,
                    expected_value_b,
                )
        elif (
            normalized_value_a != self._normalize_generated_value(record.get("value_b", value_a))
            or normalized_value_a not in doc_a_lower
            or normalized_value_a not in doc_b_lower
        ):
            doc_a, doc_b = self._render_control_docs(subject, field_name, value_a)

        return {
            **record,
            "subject": subject,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "contradicted_field": field_name,
            "value_a": value_a,
            "value_b": expected_value_b,
            "is_contradiction": bool(spec["is_contradiction"]),
            "generation_seed": self.seed,
            "generation_mode": "teacher",
            "teacher_model": teacher_model,
        }

    def _teacher_pair_record(self, seed: dict[str, str], target_score: int) -> dict[str, Any]:
        self._require_teacher()
        payload, metadata = self.client.complete_json(
            PAIR_PROMPT.format(
                query=seed["query"],
                positive=seed["positive"],
                negative=seed["negative"],
                target_score=target_score,
            )
        )
        payload.update(
            {
                "generation_seed": self.seed,
                "generation_mode": "teacher",
                "teacher_model": metadata.get("model", self.client.model),
            }
        )
        self._log_cost(metadata, "pairs")
        return self._validate_record(PairRecord, payload)

    def _teacher_pair_records(self, batch_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(batch_specs) == 1:
            spec = batch_specs[0]
            return [self._teacher_pair_record(spec, int(spec["target_score"]))]
        self._require_teacher()
        try:
            payload, metadata = self.client.complete_json(
                PAIR_BATCH_PROMPT.format(
                    count=len(batch_specs),
                    items_json=json.dumps(batch_specs, indent=2),
                )
            )
        except Exception:
            midpoint = len(batch_specs) // 2
            return self._teacher_pair_records(batch_specs[:midpoint]) + self._teacher_pair_records(
                batch_specs[midpoint:]
            )
        records = payload.get("records", [])
        if not isinstance(records, list) or len(records) != len(batch_specs):
            midpoint = len(batch_specs) // 2
            return self._teacher_pair_records(batch_specs[:midpoint]) + self._teacher_pair_records(
                batch_specs[midpoint:]
            )
        self._log_cost(metadata, "pairs")
        try:
            return [
                self._validate_record(
                    PairRecord,
                    {
                        **record,
                        "generation_seed": self.seed,
                        "generation_mode": "teacher",
                        "teacher_model": metadata.get("model", self.client.model),
                    },
                )
                for spec, record in zip(batch_specs, records, strict=False)
            ]
        except ValueError:
            midpoint = len(batch_specs) // 2
            return self._teacher_pair_records(batch_specs[:midpoint]) + self._teacher_pair_records(
                batch_specs[midpoint:]
            )

    @staticmethod
    def _chunk_specs(
        specs: list[dict[str, Any]],
        batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        return [specs[idx : idx + batch_size] for idx in range(0, len(specs), batch_size)]

    def _parallel_teacher_batches(
        self,
        specs: list[dict[str, Any]],
        batch_size: int,
        fn,
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        chunks = self._chunk_specs(specs, batch_size)
        if len(chunks) <= 1:
            return fn(specs)
        max_workers = max(1, settings.synthetic_data.teacher_max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(fn, chunks))
        return [record for records in chunk_results for record in records]

    def _teacher_preference_record(self, seed: dict[str, str]) -> dict[str, Any]:
        self._require_teacher()
        payload, metadata = self.client.complete_json(
            PREFERENCE_PROMPT.format(
                query=seed["query"],
                positive=seed["positive"],
                negative=seed["negative"],
            )
        )
        payload.update(
            {
                "generation_seed": self.seed,
                "generation_mode": "teacher",
                "teacher_model": metadata.get("model", self.client.model),
            }
        )
        self._log_cost(metadata, "preferences")
        return self._validate_record(PreferenceRecord, payload)

    @staticmethod
    def _apply_preference_swap(
        record: dict[str, Any],
        swap_output: bool,
    ) -> dict[str, Any]:
        if not swap_output:
            return record
        preferred = "B" if record["preferred"] == "A" else "A"
        return {
            **record,
            "doc_a": record["doc_b"],
            "doc_b": record["doc_a"],
            "preferred": preferred,
        }

    def _teacher_preference_records(
        self,
        batch_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(batch_specs) == 1:
            spec = batch_specs[0]
            record = self._teacher_preference_record(spec)
            return [self._apply_preference_swap(record, bool(spec.get("swap_output", False)))]
        self._require_teacher()
        try:
            payload, metadata = self.client.complete_json(
                PREFERENCE_BATCH_PROMPT.format(
                    count=len(batch_specs),
                    items_json=json.dumps(batch_specs, indent=2),
                )
            )
        except Exception:
            midpoint = len(batch_specs) // 2
            return self._teacher_preference_records(
                batch_specs[:midpoint]
            ) + self._teacher_preference_records(batch_specs[midpoint:])
        records = payload.get("records", [])
        if not isinstance(records, list) or len(records) != len(batch_specs):
            midpoint = len(batch_specs) // 2
            return self._teacher_preference_records(
                batch_specs[:midpoint]
            ) + self._teacher_preference_records(batch_specs[midpoint:])
        self._log_cost(metadata, "preferences")
        try:
            return [
                self._validate_record(
                    PreferenceRecord,
                    self._apply_preference_swap(
                        {
                            **record,
                            "generation_seed": self.seed,
                            "generation_mode": "teacher",
                            "teacher_model": metadata.get("model", self.client.model),
                        },
                        bool(spec.get("swap_output", False)),
                    ),
                )
                for spec, record in zip(batch_specs, records, strict=False)
            ]
        except ValueError:
            midpoint = len(batch_specs) // 2
            return self._teacher_preference_records(
                batch_specs[:midpoint]
            ) + self._teacher_preference_records(batch_specs[midpoint:])

    def _teacher_contradiction_record(
        self,
        subject: str,
        field_name: str,
        value_a: str,
        value_b: str,
        is_contradiction: bool,
    ) -> dict[str, Any]:
        self._require_teacher()
        payload, metadata = self.client.complete_json(
            CONTRADICTION_PROMPT.format(
                subject=subject,
                field=field_name,
                value_a=value_a,
                value_b=value_b,
                target_label=str(is_contradiction).lower(),
            )
        )
        payload.update(
            self._stabilize_contradiction_record(
                {
                    "subject": subject,
                    "field_name": field_name,
                    "value_a": value_a,
                    "value_b": value_b,
                    "is_contradiction": is_contradiction,
                },
                payload,
                str(metadata.get("model", self.client.model)),
            )
        )
        self._log_cost(metadata, "contradictions")
        return self._validate_record(ContradictionRecord, payload)

    def _teacher_contradiction_records(
        self,
        batch_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(batch_specs) == 1:
            spec = batch_specs[0]
            return [
                self._teacher_contradiction_record(
                    str(spec["subject"]),
                    str(spec["field_name"]),
                    str(spec["value_a"]),
                    str(spec["value_b"]),
                    bool(spec["is_contradiction"]),
                )
            ]
        self._require_teacher()
        try:
            payload, metadata = self.client.complete_json(
                CONTRADICTION_BATCH_PROMPT.format(
                    count=len(batch_specs),
                    items_json=json.dumps(batch_specs, indent=2),
                )
            )
        except Exception:
            midpoint = len(batch_specs) // 2
            return self._teacher_contradiction_records(
                batch_specs[:midpoint]
            ) + self._teacher_contradiction_records(batch_specs[midpoint:])
        records = payload.get("records", [])
        if not isinstance(records, list) or len(records) != len(batch_specs):
            midpoint = len(batch_specs) // 2
            return self._teacher_contradiction_records(
                batch_specs[:midpoint]
            ) + self._teacher_contradiction_records(batch_specs[midpoint:])
        self._log_cost(metadata, "contradictions")
        try:
            return [
                self._validate_record(
                    ContradictionRecord,
                    self._stabilize_contradiction_record(
                        spec,
                        record,
                        str(metadata.get("model", self.client.model)),
                    ),
                )
                for spec, record in zip(batch_specs, records, strict=False)
            ]
        except ValueError:
            midpoint = len(batch_specs) // 2
            return self._teacher_contradiction_records(
                batch_specs[:midpoint]
            ) + self._teacher_contradiction_records(batch_specs[midpoint:])

    def _get_expanded_seeds(self) -> list[dict[str, Any]]:
        """Expand seeds with query variations for diversity.

        Each seed produces (1 + len(variations)) unique queries.
        With 30 seeds x 5 variations each = 180 unique queries.
        """
        expanded: list[dict[str, Any]] = []
        for seed in DEFAULT_PAIR_SEEDS:
            expanded.append(
                {
                    "query": seed["query"],
                    "positive": seed["positive"],
                    "negative": seed["negative"],
                    "domain": seed.get("domain", "general"),
                }
            )
            for variation in seed.get("variations", []):
                expanded.append(
                    {
                        "query": variation,
                        "positive": seed["positive"],
                        "negative": seed["negative"],
                        "domain": seed.get("domain", "general"),
                    }
                )
        return expanded

    def generate_pairs(
        self,
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[dict[str, Any]]:
        target_count = (
            get_settings().synthetic_data.pair_count if target_count is None else target_count
        )
        records: list[dict[str, Any]] = []
        teacher_mode = self._should_use_teacher(use_teacher)
        expanded_seeds = self._get_expanded_seeds()

        if teacher_mode:
            score_cycle = [0, 1, 2, 3]
            batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
            if target_count <= 4:
                batch_size = 1
            specs = []
            for idx in range(target_count):
                seed = expanded_seeds[idx % len(expanded_seeds)]
                specs.append(
                    {
                        "query": seed["query"],
                        "positive": seed["positive"],
                        "negative": seed["negative"],
                        "target_score": score_cycle[idx % len(score_cycle)],
                    }
                )
            records = self._parallel_teacher_batches(
                specs,
                batch_size=batch_size,
                fn=self._teacher_pair_records,
            )
            return records

        shuffled_seeds = list(expanded_seeds)
        self.random.shuffle(shuffled_seeds)

        for seed in shuffled_seeds:
            if len(records) >= target_count:
                break
            records.append(
                self._validate_record(
                    PairRecord,
                    {
                        "query": seed["query"],
                        "doc": seed["positive"],
                        "score": 3,
                        "rationale": "Directly addresses the query with the relevant concept.",
                        "generation_seed": self.seed,
                        "generation_mode": "offline",
                        "teacher_model": None,
                    },
                )
            )
            if len(records) >= target_count:
                break
            records.append(
                self._validate_record(
                    PairRecord,
                    {
                        "query": seed["query"],
                        "doc": seed["negative"],
                        "score": 0,
                        "rationale": "Discusses a different topic from the query.",
                        "generation_seed": self.seed,
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
        while len(records) < target_count:
            seed = self.random.choice(shuffled_seeds)
            score = score_cycle[len(records) % len(score_cycle)]
            text = seed["positive"] if score >= 2 else seed["negative"]
            suffix = suffixes[self.random.randrange(len(suffixes))]
            records.append(
                self._validate_record(
                    PairRecord,
                    {
                        "query": seed["query"],
                        "doc": f"{text} {suffix}",
                        "score": score,
                        "rationale": "Offline synthetic label generated from seeded examples.",
                        "generation_seed": self.seed,
                        "generation_mode": "offline",
                        "teacher_model": None,
                    },
                )
            )
        return records

    def generate_preferences(
        self,
        pairs: list[dict[str, Any]],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[dict[str, Any]]:
        target_count = (
            get_settings().synthetic_data.preference_count if target_count is None else target_count
        )
        teacher_mode = self._should_use_teacher(use_teacher)
        expanded_seeds = self._get_expanded_seeds()
        if teacher_mode:
            batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
            if target_count <= 4:
                batch_size = 1
            specs = []
            for idx in range(target_count):
                seed: dict[str, Any] = dict(expanded_seeds[idx % len(expanded_seeds)])
                seed["swap_output"] = bool(idx % 2)
                specs.append(seed)
            return self._parallel_teacher_batches(
                specs,
                batch_size=batch_size,
                fn=self._teacher_preference_records,
            )

        by_query: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            by_query.setdefault(pair["query"], []).append(pair)
        records: list[dict[str, Any]] = []
        for query, examples in by_query.items():
            if len(records) >= target_count:
                break
            ordered = sorted(examples, key=lambda row: row["score"])
            if len(ordered) < 2:
                continue
            lo = ordered[0]
            hi = ordered[-1]
            records.append(
                self._validate_record(
                    PreferenceRecord,
                    {
                        "query": query,
                        "doc_a": hi["doc"],
                        "doc_b": lo["doc"],
                        "preferred": "A",
                        "confidence": 0.95,
                        "generation_seed": self.seed,
                        "generation_mode": "offline",
                        "teacher_model": None,
                    },
                )
            )
        while len(records) < target_count and by_query:
            query = self.random.choice(list(by_query))
            examples = by_query[query]
            doc_a, doc_b = self.random.sample(examples, 2)
            preferred = "A" if doc_a["score"] >= doc_b["score"] else "B"
            confidence = min(1.0, 0.55 + abs(doc_a["score"] - doc_b["score"]) * 0.15)
            records.append(
                self._validate_record(
                    PreferenceRecord,
                    {
                        "query": query,
                        "doc_a": doc_a["doc"],
                        "doc_b": doc_b["doc"],
                        "preferred": preferred,
                        "confidence": round(confidence, 2),
                        "generation_seed": self.seed,
                        "generation_mode": "offline",
                        "teacher_model": None,
                    },
                )
            )
        return records

    def generate_contradictions(
        self,
        contradiction_count: int | None = None,
        control_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        contradiction_count = (
            settings.synthetic_data.contradiction_count
            if contradiction_count is None
            else contradiction_count
        )
        control_count = (
            settings.synthetic_data.control_count if control_count is None else control_count
        )
        subjects = [
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
        teacher_mode = self._should_use_teacher(use_teacher)
        if teacher_mode:
            batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
            contradiction_specs = [
                {
                    "subject": subject,
                    "field_name": field_name,
                    "value_a": value_a,
                    "value_b": value_b,
                    "is_contradiction": True,
                }
                for idx in range(contradiction_count)
                for subject, field_name, value_a, value_b in [subjects[idx % len(subjects)]]
            ]
            control_specs = [
                {
                    "subject": subject,
                    "field_name": field_name,
                    "value_a": value,
                    "value_b": value,
                    "is_contradiction": False,
                }
                for idx in range(control_count)
                for subject, field_name, value, _ in [subjects[idx % len(subjects)]]
            ]
            all_specs = contradiction_specs + control_specs
            return self._parallel_teacher_batches(
                all_specs,
                batch_size=batch_size,
                fn=self._teacher_contradiction_records,
            )

        records = []
        for idx in range(contradiction_count):
            subject, field_name, value_a, value_b = subjects[idx % len(subjects)]
            records.append(
                self._validate_record(
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
                        "generation_seed": self.seed,
                        "generation_mode": "offline",
                        "teacher_model": None,
                    },
                )
            )
        for idx in range(control_count):
            subject, field_name, value, _ = subjects[idx % len(subjects)]
            records.append(
                self._validate_record(
                    ContradictionRecord,
                    {
                        "subject": subject,
                        "doc_a": (
                            f"{subject} lists {field_name} as {value}. Supporting context is"
                            " similar."
                        ),
                        "doc_b": (
                            f"{subject} again lists {field_name} as {value}. "
                            "Supporting context is similar."
                        ),
                        "contradicted_field": field_name,
                        "value_a": value,
                        "value_b": value,
                        "is_contradiction": False,
                        "generation_seed": self.seed,
                        "generation_mode": "offline",
                        "teacher_model": None,
                    },
                )
            )
        return records

    def _distribution_report(
        self,
        dataset_name: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if dataset_name == "pairs":
            labels = [str(record["score"]) for record in records]
        elif dataset_name == "preferences":
            labels = [str(record["preferred"]) for record in records]
        else:
            labels = [
                "contradiction" if record["is_contradiction"] else "control" for record in records
            ]

        counts = Counter(labels)
        total = max(len(records), 1)
        proportions = {label: round(count / total, 4) for label, count in sorted(counts.items())}
        min_count = min(counts.values()) if counts else 0
        max_count = max(counts.values()) if counts else 0
        imbalance_ratio = round(max_count / max(min_count, 1), 4) if counts else 0.0
        return {
            "count": len(records),
            "labels": dict(sorted(counts.items())),
            "proportions": proportions,
            "imbalance_ratio": imbalance_ratio,
            "is_balanced_enough": imbalance_ratio <= 3.0 if counts else True,
        }

    def _write_distribution_summary(
        self,
        raw_root: Path,
        pairs: list[dict[str, Any]],
        preferences: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
    ) -> None:
        processed_root = raw_root.parent / "processed"
        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "seed": self.seed,
            "pairs": self._distribution_report("pairs", pairs),
            "preferences": self._distribution_report("preferences", preferences),
            "contradictions": self._distribution_report("contradictions", contradictions),
        }
        write_json(processed_root / "label_distribution_summary.json", summary)

        lines = ["# Label Distribution Summary", f"seed={self.seed}", ""]
        for dataset_name in ("pairs", "preferences", "contradictions"):
            dataset_summary = cast(dict[str, Any], summary[dataset_name])
            lines.append(f"[{dataset_name}]")
            ds_count: int = dataset_summary["count"]
            imbalance_ratio: float = dataset_summary["imbalance_ratio"]
            is_balanced_enough: bool = dataset_summary["is_balanced_enough"]
            lines.append(f"count={ds_count}")
            lines.append(f"imbalance_ratio={imbalance_ratio}")
            lines.append(f"is_balanced_enough={is_balanced_enough}")
            ds_labels: dict[str, int] = dataset_summary["labels"]
            for label, count in ds_labels.items():
                bar = "#" * max(1, int(round((count / max(ds_count, 1)) * 40)))
                lines.append(f"{label}: {count} {bar}")
            lines.append("")
        text_path = ensure_parent(raw_root.parent / "processed" / "label_distribution.txt")
        text_path.write_text("\n".join(lines), encoding="utf-8")

        self._write_distribution_chart(raw_root.parent / "processed", summary)

    def _write_distribution_chart(
        self,
        processed_root: Path,
        summary: dict[str, Any],
    ) -> None:
        try:
            import matplotlib  # type: ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Synthetic Dataset Label Distributions", fontsize=14, fontweight="bold")

        dataset_configs = [
            ("pairs", "Relevance Pairs", "score", ["0", "1", "2", "3"]),
            ("preferences", "Pairwise Preferences", "preferred", ["A", "B"]),
            ("contradictions", "Contradictions", "is_contradiction", ["contradiction", "control"]),
        ]

        for ax, (dataset_name, title, _label_key, _expected_labels) in zip(
            axes, dataset_configs, strict=False
        ):
            ds = summary[dataset_name]
            labels = list(ds["labels"].keys())
            counts = list(ds["labels"].values())

            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
            bars = ax.bar(range(len(labels)), counts, color=colors[: len(labels)])

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_title(
                f"{title}\n(n={ds['count']}, imbalance={ds['imbalance_ratio']:.2f})", fontsize=11
            )
            ax.set_ylabel("Count", fontsize=10)

            for bar, count in zip(bars, counts, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(count * 0.01, 0.5),
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        chart_path = processed_root / "label_distribution.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _write_manifest(
        self,
        raw_root: Path,
        pairs: list[dict[str, Any]],
        preferences: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
    ) -> None:
        all_records = [*pairs, *preferences, *contradictions]
        modes = {
            *(record["generation_mode"] for record in pairs),
            *(record["generation_mode"] for record in preferences),
            *(record["generation_mode"] for record in contradictions),
        }
        generation_mode: Literal["offline", "teacher", "mixed"]
        if len(modes) == 1:
            generation_mode = next(iter(modes))
        else:
            generation_mode = "mixed"

        teacher_models = sorted(
            {
                str(record["teacher_model"]).strip()
                for record in all_records
                if record.get("generation_mode") == "teacher" and record.get("teacher_model")
            }
        )
        if not teacher_models or "teacher" not in modes:
            manifest_teacher_model: str | None = None
        elif len(teacher_models) == 1:
            manifest_teacher_model = teacher_models[0]
        else:
            manifest_teacher_model = ", ".join(teacher_models)

        manifest = DatasetManifest(
            generated_at=datetime.now(UTC).isoformat(),
            root=str(raw_root),
            seed=self.seed,
            generation_mode=generation_mode,
            teacher_model=manifest_teacher_model,
            datasets={
                "pairs": self._distribution_report("pairs", pairs),
                "preferences": self._distribution_report("preferences", preferences),
                "contradictions": self._distribution_report("contradictions", contradictions),
            },
        )
        write_json(raw_root / "manifest.json", manifest.model_dump())

    def materialize_all(
        self,
        root: str | Path | None = None,
        pair_count: int | None = None,
        preference_count: int | None = None,
        contradiction_count: int | None = None,
        control_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> dict[str, str]:
        settings = get_settings()
        root = settings.paths.raw_data_dir if root is None else root
        raw_root = Path(root)
        raw_root.mkdir(parents=True, exist_ok=True)
        pairs = self.generate_pairs(target_count=pair_count, use_teacher=use_teacher)
        hard_neg_target = (pair_count or settings.synthetic_data.pair_count) // 5
        hard_negs = self.generate_hard_negatives(
            pairs, target_count=hard_neg_target, use_teacher=use_teacher
        )
        preferences = self.generate_preferences(
            pairs,
            target_count=preference_count,
            use_teacher=use_teacher,
        )
        contradictions = self.generate_contradictions(
            contradiction_count=contradiction_count,
            control_count=control_count,
            use_teacher=use_teacher,
        )
        outputs = {
            "pairs": str(raw_root / "pairs.jsonl"),
            "preferences": str(raw_root / "preferences.jsonl"),
            "contradictions": str(raw_root / "contradictions.jsonl"),
            "hard_negatives": str(raw_root / "hard_negatives.jsonl"),
            "manifest": str(raw_root / "manifest.json"),
            "label_distribution": str(
                raw_root.parent / "processed" / "label_distribution_summary.json"
            ),
        }
        write_jsonl(outputs["pairs"], pairs)
        write_jsonl(outputs["preferences"], preferences)
        write_jsonl(outputs["contradictions"], contradictions)
        write_jsonl(outputs["hard_negatives"], hard_negs)
        self._write_manifest(raw_root, pairs, preferences, contradictions)
        self._write_distribution_summary(raw_root, pairs, preferences, contradictions)
        return outputs

    def generate_hard_negatives(
        self,
        pairs: list[dict[str, Any]],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[dict[str, Any]]:
        target_count = (
            get_settings().synthetic_data.pair_count if target_count is None else target_count
        )
        teacher_mode = self._should_use_teacher(use_teacher)
        records: list[dict[str, Any]] = []

        if teacher_mode:
            batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
            if target_count <= 4:
                batch_size = 1
            specs = []
            for idx in range(target_count):
                pair = pairs[idx % len(pairs)]
                specs.append(
                    {
                        "query": pair["query"],
                        "positive": pair["doc"],
                    }
                )
            chunks = self._chunk_specs(specs, batch_size)
            for chunk in chunks:
                try:
                    payload, metadata = self.client.complete_json(
                        HARD_NEGATIVE_BATCH_PROMPT.format(
                            count=len(chunk),
                            items_json=json.dumps(chunk, indent=2),
                        )
                    )
                    batch_records = payload.get("records", [])
                    for _spec, record in zip(chunk, batch_records, strict=False):
                        record.update(
                            {
                                "generation_seed": self.seed,
                                "generation_mode": "teacher",
                                "teacher_model": metadata.get("model", self.client.model),
                            }
                        )
                        records.append(self._validate_record(HardNegativeRecord, record))
                    self._log_cost(metadata, "hard_negatives")
                except Exception:
                    midpoint = len(chunk) // 2
                    if midpoint > 0:
                        first_chunk_pairs = [
                            {"query": s["query"], "doc": s["positive"]} for s in chunk[:midpoint]
                        ]
                        records.extend(
                            self.generate_hard_negatives(
                                first_chunk_pairs,
                                target_count=midpoint,
                                use_teacher=True,
                            )
                        )
                        second_chunk_pairs = [
                            {"query": s["query"], "doc": s["positive"]} for s in chunk[midpoint:]
                        ]
                        records.extend(
                            self.generate_hard_negatives(
                                second_chunk_pairs,
                                target_count=len(chunk) - midpoint,
                                use_teacher=True,
                            )
                        )
        else:
            expanded_seeds = self._get_expanded_seeds()
            domain_hard_negatives: dict[str, list[str]] = {
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
                "finance": [
                    "Diversification reduces portfolio risk by spreading investments across uncorrelated assets.",
                    "The efficient market hypothesis states that asset prices reflect all available information.",
                    "Bond yields move inversely to bond prices, reflecting changes in interest rate expectations.",
                ],
                "legal": [
                    "Common law systems rely on judicial precedent and case law to interpret statutes and regulations.",
                    "Arbitration clauses in contracts require disputes to be resolved by a private arbitrator rather than courts.",
                    "Intellectual property licensing agreements grant permission to use protected works under specific terms.",
                ],
                "education": [
                    "Formative assessment provides ongoing feedback during the learning process to guide instruction.",
                    "Constructivist learning theory emphasizes active knowledge construction through experience and reflection.",
                    "Differentiated instruction adapts teaching methods to accommodate diverse student learning needs.",
                ],
                "healthcare": [
                    "Preventive healthcare focuses on early detection and risk factor management to avoid disease onset.",
                    "Electronic health records centralize patient information across healthcare providers for coordinated care.",
                    "Clinical trials follow phased protocols to evaluate drug safety and efficacy before regulatory approval.",
                ],
                "climate_science": [
                    "Carbon capture technologies aim to remove CO2 from industrial emissions before atmospheric release.",
                    "Ocean acidification results from increased atmospheric CO2 dissolving in seawater forming carbonic acid.",
                    "Renewable energy sources like solar and wind are becoming cost-competitive with fossil fuels.",
                ],
                "web_development": [
                    "Server-side rendering generates HTML on the server before sending it to the client browser.",
                    "HTTP/2 multiplexing allows multiple requests over a single TCP connection for faster page loads.",
                    "Content delivery networks cache static assets at edge locations to reduce latency for global users.",
                ],
                "data_engineering": [
                    "Data lakes store raw structured and unstructured data at scale for future analytical processing.",
                    "Change data capture tracks database modifications in real time for downstream system synchronization.",
                    "Data quality frameworks validate completeness, accuracy, and consistency of pipeline outputs.",
                ],
                "security": [
                    "Zero-trust architecture verifies every request as though it originates from an untrusted network.",
                    "Encryption at rest protects stored data from unauthorized access even if physical media is compromised.",
                    "Multi-factor authentication requires two or more verification methods to confirm user identity.",
                ],
                "mathematics": [
                    "Eigenvalues and eigenvectors characterize linear transformations and are fundamental to PCA.",
                    "Markov chains model systems where the next state depends only on the current state.",
                    "Fourier transforms decompose signals into constituent frequencies for analysis and filtering.",
                ],
                "biology": [
                    "Natural selection favors organisms with traits that improve survival and reproductive success.",
                    "Gene expression regulation controls which proteins a cell produces in response to environmental signals.",
                    "Ecological succession describes the gradual process of species composition change in an ecosystem.",
                ],
                "nlp_search": [
                    "Named entity recognition identifies and classifies proper nouns into predefined categories.",
                    "Word sense disambiguation determines the correct meaning of a word based on surrounding context.",
                    "Document similarity metrics like cosine distance measure semantic proximity between text vectors.",
                ],
                "systems": [
                    "Connection pooling reuses database connections to reduce overhead from repeated connection setup.",
                    "Message queues decouple producers and consumers for asynchronous communication in distributed systems.",
                    "Circuit breakers prevent cascading failures by stopping requests to unhealthy downstream services.",
                ],
            }
            easy_negatives = [
                "The migration patterns of arctic terns span hemispheres each breeding season.",
                "Traditional pottery techniques vary significantly across different cultural traditions.",
                "Classical music composition follows established harmonic and structural conventions.",
                "Marine biodiversity in coral reef ecosystems supports thousands of interdependent species.",
                "The architectural evolution of gothic cathedrals reflects medieval engineering advances.",
            ]
            for idx in range(target_count):
                pair = pairs[idx % len(pairs)]
                query = pair["query"]
                positive = pair["doc"]
                domain = None
                for seed in expanded_seeds:
                    if seed["query"] == query or query in seed.get("variations", []):
                        domain = seed.get("domain", "general")
                        break
                if domain is None:
                    for seed in expanded_seeds:
                        if any(word in query.lower() for word in seed["query"].lower().split()[:2]):
                            domain = seed.get("domain", "general")
                            break
                domain = domain or "general"
                hn_pool = domain_hard_negatives.get(domain, domain_hard_negatives.get("python", []))
                records.append(
                    self._validate_record(
                        HardNegativeRecord,
                        {
                            "query": query,
                            "positive": positive,
                            "hard_negative": hn_pool[idx % len(hn_pool)],
                            "easy_negative": easy_negatives[idx % len(easy_negatives)],
                            "generation_seed": self.seed,
                            "generation_mode": "offline",
                            "teacher_model": None,
                        },
                    )
                )
        return records

    def generate_listwise_preferences(
        self,
        pairs: list[dict[str, Any]],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[dict[str, Any]]:
        target_count = (
            get_settings().synthetic_data.preference_count if target_count is None else target_count
        )
        teacher_mode = self._should_use_teacher(use_teacher)
        records: list[dict[str, Any]] = []

        if teacher_mode:
            batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
            query_docs: dict[str, list[str]] = {}
            for pair in pairs:
                query_docs.setdefault(pair["query"], []).append(pair["doc"])

            specs = []
            for idx in range(target_count):
                query = list(query_docs.keys())[idx % len(query_docs)]
                docs = query_docs[query][:4]
                if len(docs) >= 3:
                    specs.append({"query": query, "docs": docs})

            chunks = self._chunk_specs(specs, batch_size)
            for chunk in chunks:
                try:
                    payload, metadata = self.client.complete_json(
                        LISTWISE_BATCH_PROMPT.format(
                            count=len(chunk),
                            items_json=json.dumps(chunk, indent=2),
                        )
                    )
                    batch_records = payload.get("records", [])
                    for _spec, record in zip(chunk, batch_records, strict=False):
                        record.update(
                            {
                                "generation_seed": self.seed,
                                "generation_mode": "teacher",
                                "teacher_model": metadata.get("model", self.client.model),
                            }
                        )
                        records.append(self._validate_record(ListwisePreferenceRecord, record))
                    self._log_cost(metadata, "listwise_preferences")
                except Exception:
                    pass
        else:
            query_pairs: dict[str, list[dict[str, Any]]] = {}
            for pair in pairs:
                query_pairs.setdefault(pair["query"], []).append(pair)

            for _idx, (query, examples) in enumerate(query_pairs.items()):
                if len(records) >= target_count:
                    break
                if len(examples) < 3:
                    continue
                ordered = sorted(examples, key=lambda x: x.get("score", 0), reverse=True)
                docs = [ex["doc"] for ex in ordered[:4]]
                scores = [0.4, 0.3, 0.2, 0.1][: len(docs)]
                records.append(
                    self._validate_record(
                        ListwisePreferenceRecord,
                        {
                            "query": query,
                            "docs": docs,
                            "scores": scores,
                            "generation_seed": self.seed,
                            "generation_mode": "offline",
                            "teacher_model": None,
                        },
                    )
                )
        return records

    def generate_query_expansions(
        self,
        pairs: list[dict[str, Any]],
        target_count: int | None = None,
        use_teacher: bool | None = None,
    ) -> list[dict[str, Any]]:
        target_count = (
            get_settings().synthetic_data.pair_count if target_count is None else target_count
        )
        teacher_mode = self._should_use_teacher(use_teacher)
        records: list[dict[str, Any]] = []

        if teacher_mode:
            batch_size = max(1, get_settings().synthetic_data.teacher_batch_size)
            seen_queries = list({pair["query"] for pair in pairs})
            specs = [{"query": q} for q in seen_queries[:target_count]]
            chunks = self._chunk_specs(specs, batch_size)
            for chunk in chunks:
                try:
                    payload, metadata = self.client.complete_json(
                        QUERY_EXPANSION_BATCH_PROMPT.format(
                            count=len(chunk),
                            items_json=json.dumps(chunk, indent=2),
                        )
                    )
                    batch_records = payload.get("records", [])
                    for _spec, record in zip(chunk, batch_records, strict=False):
                        record.update(
                            {
                                "generation_seed": self.seed,
                                "generation_mode": "teacher",
                                "teacher_model": metadata.get("model", self.client.model),
                            }
                        )
                        records.append(self._validate_record(QueryExpansionRecord, record))
                    self._log_cost(metadata, "query_expansions")
                except Exception:
                    pass
        else:
            seen_queries = list({pair["query"] for pair in pairs})
            for idx in range(min(target_count, len(seen_queries))):
                query = seen_queries[idx]
                expansions = [
                    f"how to {query}",
                    f"{query} best practices",
                    f"{query} example",
                    f"understanding {query}",
                    f"{query} guide",
                ]
                records.append(
                    self._validate_record(
                        QueryExpansionRecord,
                        {
                            "original_query": query,
                            "expanded_queries": expansions[:3],
                            "generation_seed": self.seed,
                            "generation_mode": "offline",
                            "teacher_model": None,
                        },
                    )
                )
        return records

    def refresh_metadata(self, root: str | Path | None = None) -> dict[str, str]:
        settings = get_settings()
        raw_root = Path(settings.paths.raw_data_dir if root is None else root)
        raw_root.mkdir(parents=True, exist_ok=True)
        pairs = read_jsonl(raw_root / "pairs.jsonl")
        preferences = read_jsonl(raw_root / "preferences.jsonl")
        contradictions = read_jsonl(raw_root / "contradictions.jsonl")
        self._write_manifest(raw_root, pairs, preferences, contradictions)
        self._write_distribution_summary(raw_root, pairs, preferences, contradictions)
        return {
            "manifest": str(raw_root / "manifest.json"),
            "label_distribution": str(
                raw_root.parent / "processed" / "label_distribution_summary.json"
            ),
        }

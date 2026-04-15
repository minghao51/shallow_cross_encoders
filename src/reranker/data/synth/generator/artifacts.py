"""Artifacts, summaries, and streaming materialization helpers."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from reranker.config import get_settings
from reranker.data.synth._models import DatasetManifest
from reranker.data.synth.generator.types import ArtifactPaths, GeneratorFacade, JsonDict
from reranker.utils import append_jsonl, ensure_parent, read_jsonl, write_json


def distribution_report(dataset_name: str, records: list[JsonDict]) -> JsonDict:
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


def write_distribution_chart(processed_root: Path, summary: JsonDict) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Synthetic Dataset Label Distributions", fontsize=14, fontweight="bold")
    dataset_configs = [
        ("pairs", "Relevance Pairs"),
        ("preferences", "Pairwise Preferences"),
        ("contradictions", "Contradictions"),
    ]
    for ax, (dataset_name, title) in zip(axes, dataset_configs, strict=False):
        ds = cast(JsonDict, summary[dataset_name])
        labels = list(cast(dict[str, int], ds["labels"]).keys())
        counts = list(cast(dict[str, int], ds["labels"]).values())
        bars = ax.bar(
            range(len(labels)),
            counts,
            color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(labels)],
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(
            f"{title}\n(n={ds['count']}, imbalance={ds['imbalance_ratio']:.2f})",
            fontsize=11,
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
    fig.savefig(processed_root / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_distribution_summary(
    gen: GeneratorFacade,
    raw_root: Path,
    pairs: list[JsonDict],
    preferences: list[JsonDict],
    contradictions: list[JsonDict],
) -> None:
    processed_root = raw_root.parent / "processed"
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": gen.seed,
        "pairs": distribution_report("pairs", pairs),
        "preferences": distribution_report("preferences", preferences),
        "contradictions": distribution_report("contradictions", contradictions),
    }
    write_json(processed_root / "label_distribution_summary.json", summary)
    lines = ["# Label Distribution Summary", f"seed={gen.seed}", ""]
    for dataset_name in ("pairs", "preferences", "contradictions"):
        dataset_summary = cast(JsonDict, summary[dataset_name])
        lines.append(f"[{dataset_name}]")
        ds_count = int(dataset_summary["count"])
        lines.append(f"count={ds_count}")
        lines.append(f"imbalance_ratio={dataset_summary['imbalance_ratio']}")
        lines.append(f"is_balanced_enough={dataset_summary['is_balanced_enough']}")
        for label, count in cast(dict[str, int], dataset_summary["labels"]).items():
            bar = "#" * max(1, int(round((count / max(ds_count, 1)) * 40)))
            lines.append(f"{label}: {count} {bar}")
        lines.append("")
    ensure_parent(raw_root.parent / "processed" / "label_distribution.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    write_distribution_chart(processed_root, summary)


def write_manifest(
    gen: GeneratorFacade,
    raw_root: Path,
    pairs: list[JsonDict],
    preferences: list[JsonDict],
    contradictions: list[JsonDict],
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
        seed=gen.seed,
        generation_mode=generation_mode,
        teacher_model=manifest_teacher_model,
        datasets={
            "pairs": distribution_report("pairs", pairs),
            "preferences": distribution_report("preferences", preferences),
            "contradictions": distribution_report("contradictions", contradictions),
        },
    )
    write_json(raw_root / "manifest.json", manifest.model_dump())


def _reset_output(path: Path) -> None:
    if path.exists():
        path.unlink()


def _stream_to_jsonl(path: Path, records: list[JsonDict], *, chunk_size: int) -> None:
    _reset_output(path)
    buffer: list[JsonDict] = []
    for record in records:
        buffer.append(record)
        if len(buffer) >= chunk_size:
            for item in buffer:
                append_jsonl(path, item)
            buffer.clear()
    for item in buffer:
        append_jsonl(path, item)


def materialize_all(
    gen: GeneratorFacade,
    root: str | Path | None = None,
    pair_count: int | None = None,
    preference_count: int | None = None,
    contradiction_count: int | None = None,
    control_count: int | None = None,
    use_teacher: bool | None = None,
) -> ArtifactPaths:
    settings = get_settings()
    raw_root = Path(settings.paths.raw_data_dir if root is None else root)
    raw_root.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, settings.synthetic_data.stream_chunk_size)

    pairs = list(gen.iter_pairs(target_count=pair_count, use_teacher=use_teacher))
    hard_neg_target = (pair_count or settings.synthetic_data.pair_count) // 5
    hard_negs = list(
        gen.iter_hard_negatives(
            pairs,
            target_count=hard_neg_target,
            use_teacher=use_teacher,
        )
    )
    preferences = list(
        gen.iter_preferences(
            pairs,
            target_count=preference_count,
            use_teacher=use_teacher,
        )
    )
    contradictions = list(
        gen.iter_contradictions(
            contradiction_count=contradiction_count,
            control_count=control_count,
            use_teacher=use_teacher,
        )
    )

    outputs: ArtifactPaths = {
        "pairs": str(raw_root / "pairs.jsonl"),
        "preferences": str(raw_root / "preferences.jsonl"),
        "contradictions": str(raw_root / "contradictions.jsonl"),
        "hard_negatives": str(raw_root / "hard_negatives.jsonl"),
        "manifest": str(raw_root / "manifest.json"),
        "label_distribution": str(
            raw_root.parent / "processed" / "label_distribution_summary.json"
        ),
    }
    _stream_to_jsonl(Path(outputs["pairs"]), pairs, chunk_size=chunk_size)
    _stream_to_jsonl(Path(outputs["preferences"]), preferences, chunk_size=chunk_size)
    _stream_to_jsonl(Path(outputs["contradictions"]), contradictions, chunk_size=chunk_size)
    _stream_to_jsonl(Path(outputs["hard_negatives"]), hard_negs, chunk_size=chunk_size)
    write_manifest(gen, raw_root, pairs, preferences, contradictions)
    write_distribution_summary(gen, raw_root, pairs, preferences, contradictions)
    return outputs


def refresh_metadata(gen: GeneratorFacade, root: str | Path | None = None) -> dict[str, str]:
    settings = get_settings()
    raw_root = Path(settings.paths.raw_data_dir if root is None else root)
    raw_root.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(raw_root / "pairs.jsonl")
    preferences = read_jsonl(raw_root / "preferences.jsonl")
    contradictions = read_jsonl(raw_root / "contradictions.jsonl")
    write_manifest(gen, raw_root, pairs, preferences, contradictions)
    write_distribution_summary(gen, raw_root, pairs, preferences, contradictions)
    return {
        "manifest": str(raw_root / "manifest.json"),
        "label_distribution": str(
            raw_root.parent / "processed" / "label_distribution_summary.json"
        ),
    }

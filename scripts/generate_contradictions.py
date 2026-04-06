from __future__ import annotations

import argparse

from reranker.config import get_settings
from reranker.data.synth import SyntheticDataGenerator
from reranker.utils import write_jsonl


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate contradiction and control pairs.")
    parser.add_argument(
        "--contradictions",
        type=int,
        default=settings.synthetic_data.roadmap_contradiction_count,
        help="Number of contradiction examples to generate.",
    )
    parser.add_argument(
        "--controls",
        type=int,
        default=settings.synthetic_data.roadmap_control_count,
        help="Number of non-contradicting control examples to generate.",
    )
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="Require OpenRouter teacher generation instead of offline fallback.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.synthetic_data.seed,
        help="Reproducibility seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    output_path = settings.paths.raw_data_dir / "contradictions.jsonl"
    generator = SyntheticDataGenerator(seed=args.seed)
    contradictions = generator.generate_contradictions(
        contradiction_count=args.contradictions,
        control_count=args.controls,
        use_teacher=args.teacher,
    )
    write_jsonl(output_path, contradictions)
    metadata_paths = generator.refresh_metadata(settings.paths.raw_data_dir)
    print(f"Wrote {len(contradictions)} rows to {output_path}")
    print(f"Updated manifest at {metadata_paths['manifest']}")


if __name__ == "__main__":
    main()

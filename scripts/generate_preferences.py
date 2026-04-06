from __future__ import annotations

import argparse

from reranker.config import get_settings
from reranker.data.synth import SyntheticDataGenerator
from reranker.utils import read_jsonl, write_jsonl


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate pairwise preference examples.")
    parser.add_argument(
        "--count",
        type=int,
        default=settings.synthetic_data.roadmap_preference_count,
        help="Number of preferences to generate.",
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
    generator = SyntheticDataGenerator(seed=args.seed)
    pairs_path = settings.paths.raw_data_dir / "pairs.jsonl"
    output_path = settings.paths.raw_data_dir / "preferences.jsonl"
    pairs = read_jsonl(pairs_path) or generator.generate_pairs()
    preferences = generator.generate_preferences(
        pairs,
        target_count=args.count,
        use_teacher=args.teacher,
    )
    write_jsonl(output_path, preferences)
    metadata_paths = generator.refresh_metadata(settings.paths.raw_data_dir)
    print(f"Wrote {len(preferences)} rows to {output_path}")
    print(f"Updated manifest at {metadata_paths['manifest']}")


if __name__ == "__main__":
    main()

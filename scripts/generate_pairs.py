from __future__ import annotations

import argparse

from reranker.config import get_settings
from reranker.data.synth import SyntheticDataGenerator
from reranker.utils import write_jsonl


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate graded query-document pairs.")
    parser.add_argument(
        "--count",
        type=int,
        default=settings.synthetic_data.roadmap_pair_count,
        help="Number of pairs to generate.",
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
    generator = SyntheticDataGenerator(seed=args.seed)
    pairs = generator.generate_pairs(target_count=args.count, use_teacher=args.teacher)
    settings = get_settings()
    output_path = settings.paths.raw_data_dir / "pairs.jsonl"
    write_jsonl(output_path, pairs)
    metadata_paths = generator.refresh_metadata(settings.paths.raw_data_dir)
    print(f"Wrote {len(pairs)} rows to {output_path}")
    print(f"Updated manifest at {metadata_paths['manifest']}")


if __name__ == "__main__":
    main()

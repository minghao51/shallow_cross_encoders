from reranker.config import get_settings
from reranker.data.synth import SyntheticDataGenerator


def main() -> None:
    settings = get_settings()
    outputs = SyntheticDataGenerator().materialize_all(
        settings.paths.raw_data_dir,
        pair_count=settings.synthetic_data.pair_count,
        preference_count=settings.synthetic_data.preference_count,
        contradiction_count=settings.synthetic_data.contradiction_count,
        control_count=settings.synthetic_data.control_count,
        use_teacher=False,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()

"""Persistent caching for expensive teacher label generation."""

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from reranker.utils import read_json, write_json


class EnsembleLabelCache:
    """Provides persistent caching for expensive teacher label generation.

    This cache avoids re-running FlashRank inference by storing generated labels
    on disk. Cache keys are derived from dataset name and teacher model list.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize cache with directory, creating it if needed.

        Args:
            cache_dir: Directory path for cache storage.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_hash(self, dataset: str, teachers: list[str]) -> str:
        """Generate deterministic 16-character hash from inputs.

        Args:
            dataset: Dataset identifier.
            teachers: List of teacher model names.

        Returns:
            16-character hexadecimal string (first 16 chars of SHA256).
        """
        # Sort teachers for deterministic hashing regardless of input order
        sorted_teachers = sorted(teachers)
        combined = f"{dataset}:{','.join(sorted_teachers)}"
        return sha256(combined.encode()).hexdigest()[:16]

    def _convert_tuples_to_lists(self, labels: dict) -> dict:
        """Convert tuple keys to list representation for JSON serialization.

        Args:
            labels: Dictionary with tuple keys.

        Returns:
            Dictionary with string keys representing tuples as "idx1_idx2".
        """
        return {f"{k[0]}_{k[1]}": v for k, v in labels.items()}

    def _convert_lists_to_tuples(self, labels: dict) -> dict:
        """Convert list representation back to tuple keys.

        Args:
            labels: Dictionary with string keys like "idx1_idx2".

        Returns:
            Dictionary with tuple keys (idx1, idx2).
        """
        result = {}
        for k, v in labels.items():
            parts = k.split("_")
            if len(parts) == 2:
                try:
                    result[(int(parts[0]), int(parts[1]))] = v
                except ValueError:
                    result[k] = v
            else:
                result[k] = v
        return result

    def load_or_generate(
        self,
        dataset: str,
        teachers: list[str],
        generator_fn: Callable[[], dict],
        force_regenerate: bool = False,
    ) -> dict:
        """Load labels from cache or generate using provided function.

        Args:
            dataset: Dataset identifier for cache key.
            teachers: List of teacher model names for cache key.
            generator_fn: Function that generates labels dict (called on cache miss).
            force_regenerate: If True, bypass cache and regenerate.

        Returns:
            Labels dictionary from cache or generator.
        """
        hash_key = self.get_hash(dataset, teachers)
        cache_file = self.cache_dir / f"{hash_key}.json"

        # Return cached labels if available and not forcing regeneration
        if not force_regenerate and cache_file.exists():
            try:
                cached = read_json(cache_file)
                return self._convert_lists_to_tuples(cached)
            except Exception:
                # Corrupted cache - fall through to regeneration
                pass

        # Generate new labels
        labels = generator_fn()
        serializable_labels = self._convert_tuples_to_lists(labels)
        write_json(cache_file, serializable_labels)
        return labels

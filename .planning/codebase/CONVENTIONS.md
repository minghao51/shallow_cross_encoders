# Code Conventions

## Overview

This document describes the coding conventions, patterns, and error handling strategies used throughout the codebase.

## Code Style

### Python Version and Type Hints

- **Python Version**: Python 3.11+
- **Type Hints**: Mandatory for all function signatures and class attributes
- **Future Annotations**: All files start with `from __future__ import annotations`

```python
from __future__ import annotations

def example_function(query: str, docs: list[str]) -> list[RankedDoc]:
    ...
```

### Formatting

- **Line Length**: 100 characters (enforced by ruff)
- **Indentation**: 4 spaces
- **Imports**: Standard library → third-party → local (ruff I ordering)

### Naming Conventions

- **Classes**: PascalCase (`HybridFusionReranker`, `Embedder`)
- **Functions/Variables**: snake_case (`encode_docs`, `model_name`)
- **Constants**: UPPER_SNAKE_CASE (`ARTIFACT_VERSION`, `POTION_MODELS`)
- **Private Methods**: Leading underscore (`_normalize_rows`, `_encode_hashed`)
- **Protected Attributes**: Leading underscore in dataclasses (`_backend`, `_encode_cache`)

## Code Patterns

### Dataclasses

Use `@dataclass(slots=True)` for performance-critical classes:

```python
@dataclass(slots=True)
class Embedder:
    model_name: str
    dimension: int
    normalize: bool = True
    _backend: Any = field(init=False, default=None, repr=False)
```

For immutable configuration, use Pydantic with frozen config:

```python
class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "default-model"
    dimension: int = 256
```

### Protocols and Interfaces

Use `@runtime_checkable` Protocol for interface definitions:

```python
@runtime_checkable
class BaseReranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...

@runtime_checkable
class TrainableReranker(Protocol):
    def fit(self, queries: list[str], docs: list[str], labels: list[int]) -> Any: ...
```

### Configuration Management

All configuration uses Pydantic models with environment variable support:

```python
from reranker.config import get_settings

settings = get_settings()
model = settings.embedder.model_name
```

Environment variables follow naming convention: `RERANKER_<SECTION>_<KEY>` (e.g., `RERANKER_EMBEDDER_MODEL_NAME`).

## Error Handling

### Optional Dependencies

Use try/except with structured status returns for optional dependencies:

```python
def check_model2vec() -> tuple[Any, DepStatus]:
    try:
        from model2vec import StaticModel
        return StaticModel, DepStatus(name="model2vec", available=True, backend="model2vec", fallback_description="")
    except Exception:
        return None, DepStatus(
            name="model2vec",
            available=False,
            backend="hashed",
            fallback_description="deterministic hashed embeddings"
        )
```

### Graceful Degradation

Implement fallback mechanisms when optional dependencies are unavailable:

```python
def __post_init__(self) -> None:
    backend_cls, status = check_model2vec()
    if backend_cls is not None:
        try:
            self._backend = backend_cls.from_pretrained(self.model_name)
        except Exception:
            warnings.warn(f"Unable to load model; using fallback")
            self._backend = None
            self.backend_name = "hashed"
```

### Validation Errors

Use Pydantic validators for configuration validation:

```python
class EvalSettings(BaseModel):
    train_ratio: float = 0.7
    test_ratio: float = 0.15

    @field_validator("train_ratio", "test_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {v}")
        return v
```

## Logging and Warnings

### Logging

Use standard logging with module-level loggers:

```python
import logging

logger = logging.getLogger("reranker.deps")

logger.info("model2vec not available; using fallback")
```

### Warnings

Use warnings for deprecation and optional dependency messages:

```python
import warnings

warnings.warn(
    "model2vec is not available; falling back to deterministic hashed embeddings. "
    "Install with: pip install model2vec",
    stacklevel=3,
)
```

### Print Statements

Print statements are only used in CLI/benchmark contexts for user-facing output:

```python
print(f"NDCG@10: {metrics['ndcg@10']:.4f} ± {metrics['ndcg@10_std']:.4f}")
```

## File I/O Patterns

### Utility Functions

Use centralized utility functions for file operations:

```python
from reranker.utils import read_json, write_json, read_jsonl, write_jsonl, dump_pickle, load_pickle

# JSON
data = read_json("config.json")
write_json("output.json", payload)

# JSONL (line-delimited JSON)
records = read_jsonl("data.jsonl")
write_jsonl("output.jsonl", records)

# Pickle (with cloudpickle if available)
obj = load_pickle("model.pkl")
dump_pickle("model.pkl", obj)
```

### Path Handling

Always use `Path` objects for file paths:

```python
from pathlib import Path

data_dir = Path("data/processed")
model_path = model_dir / "reranker.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
```

## Performance Patterns

### Caching

- Use `functools.lru_cache` for expensive function calls
- Use `TTLCache` from cachetools for time-based caching (optional dependency)
- Manual caching in classes for repeated operations

```python
from functools import lru_cache
from cachetools import TTLCache

@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings(...)

class Embedder:
    def __init__(self):
        self._encode_cache = TTLCache(maxsize=10000, ttl=3600)
```

### NumPy Type Hints

Always specify NumPy array types for clarity:

```python
import numpy as np

def encode(self, texts: list[str]) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors
```

## Constants and Magic Numbers

Define constants at module level for clarity:

```python
ARTIFACT_VERSION = 1
POTION_MODELS = ["minishlab/potion-base-8M", "minishlab/potion-base-32M"]
DIMENSIONS = [64, 128, 256, 512]
```

## Testing Conventions

See `TESTING.md` for detailed testing patterns and conventions.

## Linting and Type Checking

### Ruff Configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

### Mypy Configuration

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
check_untyped_defs = true
disallow_untyped_defs = true
```

Run linting and type checking:
```bash
uv run ruff check src/
uv run mypy src/reranker/data
```

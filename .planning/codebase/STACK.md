# Tech Stack

## Languages & Runtime
- **Python**: 3.11+ (required)
- **Package Manager**: uv
- **Build System**: setuptools (pyproject.toml)
- **Virtual Environment**: .venv

## Core Frameworks & Libraries

### Core Dependencies
- **numpy**: Numerical operations and array handling
- **scikit-learn**: ML algorithms (MiniBatchKMeans for clustering)
- **scipy**: Scientific computing utilities
- **pydantic**: Data validation and settings management (v2)
- **pyyaml**: YAML configuration parsing
- **httpx**: Async HTTP client for API calls
- **cachetools**: Caching utilities
- **tenacity**: Retry logic
- **cloudpickle**: Serialization

### Optional Runtimes

#### Model Embeddings
- **model2vec** (optional): Static embeddings for CPU-native inference
- **sentence-transformers** (optional): PyTorch-based sentence embeddings

#### Reranking & ML
- **flashrank** (optional): ONNX cross-encoder models
- **xgboost** (optional): Gradient boosting for ensemble methods
- **rank-bm25** (optional): BM25 lexical ranking

#### LLM Integration
- **litellm** (optional): Multi-provider LLM client (OpenRouter)

#### Evaluation
- **beir** (optional): Benchmarking information retrieval datasets

## Development Tools

### Code Quality
- **ruff**: Linter and formatter (Python 3.11 target, 100 char line length)
  - Rules: E, F, I, B, UP
- **mypy**: Static type checking (focused on src/reranker/data)

### Testing
- **pytest**: Test framework with markers:
  - `unit`: Fast, isolated unit tests
  - `integration`: Tests loading local models or mocked services
  - `e2e`: Full workflow end-to-end tests
  - `llm`: Tests making real LLM API calls (requires OPENROUTER_API_KEY)
  - `llm_mock`: Tests mocking LLM API calls
  - `slow`: Tests taking >1s
- **pytest-benchmark**: Performance benchmarking
- **pytest-cov**: Coverage reporting (85% minimum threshold)
- **pytest-mock**: Mocking utilities

### Coverage Configuration
- Source: src/
- Omitted: __init__.py, __main__.py
- Excluded lines: pragma: no cover, def __repr__, raise NotImplementedError, TYPE_CHECKING, __name__ == .__main__.

## Configuration

### Project Structure
- **Layout**: src layout (packages in src/)
- **Entry Points**: src/reranker/__init__.py, src/reranker/eval/__main__.py

### Settings Management
- **Primary**: src/reranker/config.py (Pydantic models)
- **Overrides**: Environment variables (.env)
- **YAML Support**: load_yaml_config(), settings_from_yaml()

### Key Config Areas
- OpenRouter settings (API key, model, base URL, timeout)
- Reranker hyperparameters (ensemble mode, fallback strategy)
- Data generation settings (batch sizes, sample counts)
- Model paths and caching
